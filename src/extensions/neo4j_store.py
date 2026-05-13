"""Neo4j persistence layer for the conceptual network.

This module is optional. It is only imported when the pipeline is invoked
with ``--neo4j`` or the Streamlit dashboard flips the "Use Neo4j backend"
toggle. It fails gracefully if the ``neo4j`` driver or a reachable server
are not available.

Responsibilities:

* Manage a single ``neo4j.Driver`` lifetime.
* Ensure unique constraint on ``:Concept(id)`` and a vector index on
  ``:Concept(embedding)`` so the chatbot vector store can query it.
* Batched upserts of the NetworkX graph (nodes + edges).
* Optional embedding backfill using ``sentence-transformers``.

The graph model is the canonical schema for the conceptual network in the
Neo4j-only pipeline. ``Neo4jGraphContext`` reads from it directly via
Cypher and ``Neo4jGraphWriter`` writes into it without ever materialising
an in-memory graph.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import networkx as nx
import pandas as pd

from src import config as _cfg

if TYPE_CHECKING:
    from neo4j import Driver, Session


logger = logging.getLogger(__name__)

_NODE_BATCH = 500
_EDGE_BATCH = 500
_EMBED_BATCH = 64


class Neo4jUnavailableError(RuntimeError):
    """Raised when the driver can't be imported or the server is unreachable."""


class Neo4jStore:
    """Thin wrapper around the Neo4j Python driver.

    Use as a context manager so the driver is always closed::

        with Neo4jStore.from_config() as store:
            store.ensure_schema()
            store.push_graph(G, partition, centrality_df, "combined")
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        vector_index: str = "concept_embedding",
        edge_vector_index: str = "related_embedding",
        embedding_dim: int = 384,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.vector_index = vector_index
        self.edge_vector_index = edge_vector_index
        self.embedding_dim = embedding_dim
        self._driver: "Driver | None" = None

    # ── Construction ───────────────────────────────────────────────
    @classmethod
    def from_config(cls) -> "Neo4jStore":
        return cls(
            uri=_cfg.NEO4J_URI,
            user=_cfg.NEO4J_USER,
            password=_cfg.NEO4J_PASSWORD,
            database=_cfg.NEO4J_DATABASE,
            vector_index=_cfg.NEO4J_VECTOR_INDEX,
            edge_vector_index=_cfg.NEO4J_EDGE_VECTOR_INDEX,
            embedding_dim=_cfg.NEO4J_EMBEDDING_DIM,
        )

    # ── Lifecycle ──────────────────────────────────────────────────
    def connect(self) -> "Driver":
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as e:
            raise Neo4jUnavailableError(
                "neo4j driver is not installed. Run `pip install neo4j` to enable."
            ) from e
        if not self.password:
            raise Neo4jUnavailableError(
                "NEO4J_PASSWORD is empty. Set it in your environment (.env)."
            )
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            driver.verify_connectivity()
        except Exception as e:
            raise Neo4jUnavailableError(
                f"Unable to connect to Neo4j at {self.uri}: {e}"
            ) from e
        self._driver = driver
        return driver

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            finally:
                self._driver = None

    def __enter__(self) -> "Neo4jStore":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def session(self) -> Iterator["Session"]:
        driver = self.connect()
        sess = driver.session(database=self.database)
        try:
            yield sess
        finally:
            sess.close()

    # ── Availability helpers ───────────────────────────────────────
    @staticmethod
    def ping() -> bool:
        """Return True if a configured Neo4j instance is reachable."""
        try:
            store = Neo4jStore.from_config()
            store.connect()
            store.close()
            return True
        except Exception as e:
            logger.debug("Neo4j ping failed: %s", e)
            return False

    # ── Schema ─────────────────────────────────────────────────────
    def ensure_schema(self) -> None:
        """Create id uniqueness constraint, lookup indexes and vector indexes."""
        with self.session() as s:
            s.run(
                "CREATE CONSTRAINT concept_id IF NOT EXISTS "
                "FOR (c:Concept) REQUIRE c.id IS UNIQUE"
            )
            # Composite index speeds up the per-source / per-slice scoping
            # used by GDS projections, the writer and the dashboard.
            s.run(
                "CREATE INDEX concept_source_slice IF NOT EXISTS "
                "FOR (c:Concept) ON (c.source_label, c.slice_id)"
            )
            s.run(
                "CREATE INDEX concept_label IF NOT EXISTS "
                "FOR (c:Concept) ON (c.label)"
            )
            s.run(
                "CREATE INDEX concept_community IF NOT EXISTS "
                "FOR (c:Concept) ON (c.community)"
            )
            s.run(
                "CREATE INDEX related_source_slice IF NOT EXISTS "
                "FOR ()-[r:RELATED]-() ON (r.source_label, r.slice_id)"
            )
            s.run(
                "CREATE VECTOR INDEX $name IF NOT EXISTS "
                "FOR (c:Concept) ON (c.embedding) "
                "OPTIONS {indexConfig: {"
                "  `vector.dimensions`: $dim, "
                "  `vector.similarity_function`: 'cosine'"
                "}}",
                name=self.vector_index,
                dim=self.embedding_dim,
            )
            # Relationship vector index for edge-level semantic search.
            # Neo4j 5.18+ supports `FOR ()-[r:TYPE]-() ON (r.prop)`.
            s.run(
                "CREATE VECTOR INDEX $name IF NOT EXISTS "
                "FOR ()-[r:RELATED]-() ON (r.embedding) "
                "OPTIONS {indexConfig: {"
                "  `vector.dimensions`: $dim, "
                "  `vector.similarity_function`: 'cosine'"
                "}}",
                name=self.edge_vector_index,
                dim=self.embedding_dim,
            )

    # ── Ingestion ──────────────────────────────────────────────────
    def wipe_source(
        self, source_label: str, slice_id: str | None = None
    ) -> int:
        """Delete all Concept nodes scoped to a given source_label / slice.

        Returns the number of nodes removed. Used for deterministic re-runs.

        Implemented as two statements so the COUNT row is unambiguous —
        a single MATCH ... WITH c, count(c) ... DETACH DELETE form
        groups per-node and trips a "multiple records" warning when the
        result is consumed via ``.single()``.
        """
        with self.session() as s:
            if slice_id is None:
                count_q = (
                    "MATCH (c:Concept {source_label: $s}) RETURN count(c) AS n"
                )
                delete_q = (
                    "MATCH (c:Concept {source_label: $s}) DETACH DELETE c"
                )
                rec = s.run(count_q, s=source_label).single()
                s.run(delete_q, s=source_label).consume()
            else:
                count_q = (
                    "MATCH (c:Concept {source_label: $s, slice_id: $sid}) "
                    "RETURN count(c) AS n"
                )
                delete_q = (
                    "MATCH (c:Concept {source_label: $s, slice_id: $sid}) "
                    "DETACH DELETE c"
                )
                rec = s.run(count_q, s=source_label, sid=slice_id).single()
                s.run(
                    delete_q, s=source_label, sid=slice_id
                ).consume()
            return int(rec["n"]) if rec else 0

    def push_graph(
        self,
        G: nx.DiGraph,
        partition: dict[str, int] | None,
        centrality_df: pd.DataFrame | None,
        source_label: str = "combined",
        reset: bool = False,
    ) -> dict[str, int]:
        """Upsert nodes + edges from a NetworkX graph into Neo4j.

        Parameters
        ----------
        G:
            The annotated ``nx.DiGraph`` from Stage 4/5.
        partition:
            Optional community partition produced by GraphAnalyser.
        centrality_df:
            Optional centrality DataFrame (expects ``node_id`` + ``pagerank``
            columns); used to enrich node properties.
        source_label:
            Tag written on every node for later scoping (e.g. ``policy``,
            ``yelp``, ``combined``).
        reset:
            If True, delete existing nodes with this ``source_label`` first.
        """
        self.ensure_schema()
        if reset:
            removed = self.wipe_source(source_label)
            logger.info("Neo4j: cleared %d existing nodes for %s", removed, source_label)

        pagerank_lookup: dict[str, float] = {}
        betweenness_lookup: dict[str, float] = {}
        if centrality_df is not None and not centrality_df.empty:
            if "node_id" in centrality_df.columns:
                if "pagerank" in centrality_df.columns:
                    pagerank_lookup = dict(
                        zip(centrality_df["node_id"], centrality_df["pagerank"])
                    )
                if "betweenness" in centrality_df.columns:
                    betweenness_lookup = dict(
                        zip(centrality_df["node_id"], centrality_df["betweenness"])
                    )

        part = partition or {}

        nodes_payload: list[dict[str, Any]] = []
        for nid, data in G.nodes(data=True):
            pr = data.get("pagerank")
            if pr is None:
                pr = pagerank_lookup.get(nid)
            btwn = data.get("betweenness")
            if btwn is None:
                btwn = betweenness_lookup.get(nid)
            community = part.get(nid, data.get("community"))

            nodes_payload.append(
                {
                    "id": str(nid),
                    "label": str(data.get("label", nid)),
                    "concept_type": str(data.get("concept_type") or data.get("type") or "concept"),
                    "source_type": str(data.get("source_type", "unknown")),
                    "frequency": int(data.get("frequency", 0) or 0),
                    "community": int(community) if community is not None else -1,
                    "pagerank": float(pr) if pr is not None else 0.0,
                    "betweenness": float(btwn) if btwn is not None else 0.0,
                    "source_label": source_label,
                }
            )

        edges_payload: list[dict[str, Any]] = []
        for u, v, data in G.edges(data=True):
            types = data.get("types") or data.get("type") or ""
            if isinstance(types, (set, list, tuple)):
                types_str = ",".join(sorted(str(t) for t in types))
            else:
                types_str = str(types)
            sent = data.get("sentiment")

            verbs = data.get("verbs")
            top_verb: str | None = None
            verb_count: int = 0
            verb_list: list[str] = []
            if verbs:
                try:
                    ordered = verbs.most_common()
                except AttributeError:
                    ordered = sorted(
                        (verbs.items() if hasattr(verbs, "items") else []),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                if ordered:
                    top_verb = str(ordered[0][0])
                    verb_count = int(ordered[0][1])
                    verb_list = [str(v) for v, _ in ordered]

            edges_payload.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "weight": float(data.get("weight", 1) or 1),
                    "types": types_str,
                    "sentiment": float(sent) if sent is not None else None,
                    "source_label": source_label,
                    "top_verb": top_verb,
                    "verb_count": verb_count,
                    "verb_list": verb_list,
                }
            )

        node_count = self._upsert_nodes(nodes_payload)
        edge_count = self._upsert_edges(edges_payload)

        logger.info(
            "Neo4j: upserted %d nodes and %d edges for source_label=%s",
            node_count,
            edge_count,
            source_label,
        )
        return {"nodes": node_count, "edges": edge_count}

    def _upsert_nodes(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        query = (
            "UNWIND $rows AS row "
            "MERGE (c:Concept {id: row.id}) "
            "SET c.label = row.label, "
            "    c.concept_type = row.concept_type, "
            "    c.source_type = row.source_type, "
            "    c.frequency = row.frequency, "
            "    c.community = row.community, "
            "    c.pagerank = row.pagerank, "
            "    c.betweenness = row.betweenness, "
            "    c.source_label = row.source_label"
        )
        with self.session() as s:
            for batch in _chunks(rows, _NODE_BATCH):
                s.run(query, rows=batch)
        return len(rows)

    def _upsert_edges(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        query = (
            "UNWIND $rows AS row "
            "MATCH (a:Concept {id: row.source}) "
            "MATCH (b:Concept {id: row.target}) "
            "MERGE (a)-[r:RELATED {source_label: row.source_label}]->(b) "
            "SET r.weight = row.weight, "
            "    r.types = row.types, "
            "    r.sentiment = row.sentiment, "
            "    r.top_verb = row.top_verb, "
            "    r.verb_count = row.verb_count, "
            "    r.verb_list = row.verb_list"
        )
        with self.session() as s:
            for batch in _chunks(rows, _EDGE_BATCH):
                s.run(query, rows=batch)
        return len(rows)

    # ── Embeddings ─────────────────────────────────────────────────
    def embed_and_store(
        self,
        source_label: str = "combined",
        slice_id: str | None = None,
        model_name: str | None = None,
        G: nx.DiGraph | None = None,
    ) -> int:
        """Compute sentence-transformer embeddings for Concept nodes.

        Reads node + neighbour data directly from Neo4j (so it works in the
        Neo4j-only pipeline where no in-memory ``nx.DiGraph`` ever exists).
        The legacy ``G`` argument is still accepted for backward
        compatibility with older call sites; if provided, descriptions come
        from the in-memory graph.

        Returns the number of nodes embedded; 0 silently if
        ``sentence-transformers`` is unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; skipping Neo4j embedding backfill."
            )
            return 0

        model_name = model_name or _cfg.EMBEDDING_MODEL

        if G is not None:
            node_ids = [str(n) for n in G.nodes()]
            descriptions = [_short_description(G, n) for n in G.nodes()]
        else:
            node_ids, descriptions = self._read_node_descriptions(
                source_label, slice_id
            )

        if not node_ids:
            return 0

        model = SentenceTransformer(model_name)
        vectors = model.encode(
            descriptions, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

        with self.session() as s:
            for batch_start in range(0, len(node_ids), _EMBED_BATCH):
                batch_ids = node_ids[batch_start : batch_start + _EMBED_BATCH]
                batch_vecs = vectors[batch_start : batch_start + _EMBED_BATCH]
                payload = [
                    {"id": nid, "vec": vec.tolist()}
                    for nid, vec in zip(batch_ids, batch_vecs)
                ]
                s.run(
                    "UNWIND $rows AS row "
                    "MATCH (c:Concept {id: row.id}) "
                    "CALL db.create.setNodeVectorProperty(c, 'embedding', row.vec) "
                    "RETURN count(*)",
                    rows=payload,
                )

        logger.info("Neo4j: stored %d embeddings (%s)", len(node_ids), model_name)
        return len(node_ids)

    def _read_node_descriptions(
        self, source_label: str, slice_id: str | None
    ) -> tuple[list[str], list[str]]:
        """Pull (id, short_description) pairs for every Concept in scope.

        The description string mimics ``_short_description`` but is built
        purely from Cypher results so we never need an in-memory graph.
        """
        if slice_id is None:
            params = {"sl": source_label}
            cypher = (
                "MATCH (c:Concept {source_label: $sl}) "
                "OPTIONAL MATCH (c)-[r:RELATED]-(n:Concept {source_label: $sl}) "
                "WITH c, n, r ORDER BY r.weight DESC "
                "WITH c, collect({label: n.label, weight: r.weight})[..5] AS nbs "
                "RETURN c.id AS id, c.label AS label, "
                "       coalesce(c.concept_type, 'concept') AS ctype, "
                "       coalesce(c.source_type, 'n/a') AS src, nbs"
            )
        else:
            params = {"sl": source_label, "sid": slice_id}
            cypher = (
                "MATCH (c:Concept {source_label: $sl, slice_id: $sid}) "
                "OPTIONAL MATCH (c)-[r:RELATED]-(n:Concept "
                "                              {source_label: $sl, slice_id: $sid}) "
                "WITH c, n, r ORDER BY r.weight DESC "
                "WITH c, collect({label: n.label, weight: r.weight})[..5] AS nbs "
                "RETURN c.id AS id, c.label AS label, "
                "       coalesce(c.concept_type, 'concept') AS ctype, "
                "       coalesce(c.source_type, 'n/a') AS src, nbs"
            )
        ids: list[str] = []
        descs: list[str] = []
        with self.session() as s:
            for rec in s.run(cypher, **params):
                ids.append(str(rec["id"]))
                top = ", ".join(
                    n["label"] for n in rec["nbs"] or [] if n and n.get("label")
                )
                desc = (
                    f"{rec['label']}. A {rec['ctype']} concept from "
                    f"{rec['src']} data."
                    + (f" Related to: {top}." if top else "")
                )
                descs.append(desc)
        return ids, descs

    # ── Edge embeddings ────────────────────────────────────────────
    def embed_edges(
        self,
        source_label: str = "combined",
        slice_id: str | None = None,
        top_n_association: int = 2000,
        model_name: str | None = None,
        G: nx.DiGraph | None = None,
    ) -> int:
        """Compute and store embeddings for a curated subset of edges.

        Strategy (aligned with the Edge Embedding plan):

        * Always embed every edge whose ``top_verb`` is non-empty (i.e.
          ACTION + CAUSATION edges from the dependency parser).
        * Additionally embed the top ``top_n_association`` association-only
          edges (no verb) by weight.

        Reads edge data directly from Neo4j (so it works in the Neo4j-only
        pipeline). Legacy ``G`` argument still accepted for backward
        compatibility.

        Returns the number of edges embedded. Returns 0 silently if
        ``sentence-transformers`` is unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; skipping Neo4j edge embedding."
            )
            return 0

        if G is not None:
            selected = self._select_edges_from_graph(G, top_n_association)
            if not selected:
                return 0
            descriptions = [
                _edge_description(G, u, v, data) for (u, v, data) in selected
            ]
            payload_keys = [
                {
                    "source": str(u),
                    "target": str(v),
                    "sl": source_label,
                    "sid": slice_id,
                }
                for (u, v, _d) in selected
            ]
        else:
            descriptions, payload_keys = self._select_edges_from_neo4j(
                source_label, slice_id, top_n_association
            )
            if not payload_keys:
                return 0

        model_name = model_name or _cfg.EMBEDDING_MODEL
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            descriptions, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

        # Cypher branches differ on whether we filter by slice_id.
        with_slice = slice_id is not None
        if with_slice:
            cypher = (
                "UNWIND $rows AS row "
                "MATCH (a:Concept {id: row.source})"
                "-[r:RELATED {source_label: row.sl, slice_id: row.sid}]->"
                "(b:Concept {id: row.target}) "
                "CALL db.create.setRelationshipVectorProperty(r, 'embedding', row.vec) "
                "RETURN count(*)"
            )
        else:
            cypher = (
                "UNWIND $rows AS row "
                "MATCH (a:Concept {id: row.source})"
                "-[r:RELATED {source_label: row.sl}]->"
                "(b:Concept {id: row.target}) "
                "CALL db.create.setRelationshipVectorProperty(r, 'embedding', row.vec) "
                "RETURN count(*)"
            )

        with self.session() as s:
            for batch_start in range(0, len(payload_keys), _EMBED_BATCH):
                batch = payload_keys[batch_start : batch_start + _EMBED_BATCH]
                batch_vecs = vectors[batch_start : batch_start + _EMBED_BATCH]
                payload = []
                for key, vec in zip(batch, batch_vecs):
                    item = dict(key)
                    item["vec"] = vec.tolist()
                    payload.append(item)
                s.run(cypher, rows=payload)

        logger.info(
            "Neo4j: stored %d edge embeddings for source_label=%s slice_id=%s",
            len(payload_keys),
            source_label,
            slice_id,
        )
        return len(payload_keys)

    @staticmethod
    def _select_edges_from_graph(
        G: nx.DiGraph, top_n_association: int
    ) -> list[tuple[Any, Any, dict[str, Any]]]:
        verb_edges: list[tuple[Any, Any, dict[str, Any]]] = []
        assoc_edges: list[tuple[Any, Any, dict[str, Any]]] = []
        for u, v, data in G.edges(data=True):
            verbs = data.get("verbs")
            has_verb = False
            if verbs:
                try:
                    has_verb = sum(verbs.values()) > 0
                except AttributeError:
                    has_verb = bool(verbs)
            if has_verb:
                verb_edges.append((u, v, data))
            else:
                assoc_edges.append((u, v, data))
        assoc_edges.sort(
            key=lambda tup: float(tup[2].get("weight", 0) or 0),
            reverse=True,
        )
        return verb_edges + assoc_edges[: max(0, int(top_n_association))]

    def _select_edges_from_neo4j(
        self,
        source_label: str,
        slice_id: str | None,
        top_n_association: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Stream candidate edges out of Neo4j and build descriptions.

        Returns (descriptions, payload_keys) where payload_keys carry the
        match parameters (source/target/source_label/slice_id) the writer
        needs to relocate each edge for the embedding update.
        """
        params: dict[str, Any] = {"sl": source_label, "n": int(top_n_association)}
        slice_filter = ""
        if slice_id is not None:
            slice_filter = " AND r.slice_id = $sid"
            params["sid"] = slice_id

        # Verb edges first (always included).
        verb_q = (
            "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            "WHERE r.source_label = $sl AND coalesce(r.top_verb, '') <> ''"
            + slice_filter
            + " RETURN a.id AS sid, b.id AS tid, "
            "        a.label AS slabel, b.label AS tlabel, "
            "        r.top_verb AS verb, r.sentiment AS sentiment, "
            "        r.weight AS weight"
        )
        # Then assoc edges by descending weight, capped.
        assoc_q = (
            "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            "WHERE r.source_label = $sl "
            "  AND (r.top_verb IS NULL OR r.top_verb = '')"
            + slice_filter
            + " RETURN a.id AS sid, b.id AS tid, "
            "        a.label AS slabel, b.label AS tlabel, "
            "        null AS verb, r.sentiment AS sentiment, r.weight AS weight "
            "ORDER BY r.weight DESC LIMIT $n"
        )

        descriptions: list[str] = []
        keys: list[dict[str, Any]] = []
        with self.session() as s:
            for rec in s.run(verb_q, **params):
                descriptions.append(_edge_description_from_row(rec))
                keys.append(
                    {
                        "source": str(rec["sid"]),
                        "target": str(rec["tid"]),
                        "sl": source_label,
                        "sid": slice_id,
                    }
                )
            if int(top_n_association) > 0:
                for rec in s.run(assoc_q, **params):
                    descriptions.append(_edge_description_from_row(rec))
                    keys.append(
                        {
                            "source": str(rec["sid"]),
                            "target": str(rec["tid"]),
                            "sl": source_label,
                            "sid": slice_id,
                        }
                    )
        return descriptions, keys


# ── Helpers ────────────────────────────────────────────────────────
def _chunks(seq: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _edge_description_from_row(rec: Any) -> str:
    """Verb-aware short description from a Cypher row.

    Mirrors :func:`_edge_description` but consumes a Neo4j ``Record``
    directly (so the embedder works without an in-memory graph).
    """
    src = rec["slabel"]
    tgt = rec["tlabel"]
    verb = rec.get("verb") if hasattr(rec, "get") else rec["verb"]
    sentiment = (
        rec.get("sentiment") if hasattr(rec, "get") else rec["sentiment"]
    )
    if verb:
        tail = ""
        if sentiment is not None:
            try:
                sentf = float(sentiment)
                if sentf > 0.15:
                    tail = " (positive tone)"
                elif sentf < -0.15:
                    tail = " (negative tone)"
            except (TypeError, ValueError):
                pass
        return f"{src} {verb} {tgt}{tail}."
    return f"{src} co-occurs with {tgt}."


def _edge_description(G: nx.DiGraph, u: Any, v: Any, data: dict[str, Any]) -> str:
    """Verb-aware one-line description of an edge, used for embedding.

    Mirrors the style of ``_short_description`` but focuses on the
    ``src verb tgt`` triple so vector similarity captures relation semantics
    (e.g. "locals recommend milktooth" vs "food co-occurs with staff").
    """
    u_lbl = G.nodes[u].get("label", u)
    v_lbl = G.nodes[v].get("label", v)
    verbs = data.get("verbs")
    top_verb: str | None = None
    if verbs:
        try:
            ordered = verbs.most_common(1)
        except AttributeError:
            ordered = []
        if ordered:
            top_verb = str(ordered[0][0])
    if top_verb:
        sent = data.get("sentiment")
        tail = ""
        if sent is not None:
            try:
                sentf = float(sent)
                if sentf > 0.15:
                    tail = " (positive tone)"
                elif sentf < -0.15:
                    tail = " (negative tone)"
            except (TypeError, ValueError):
                tail = ""
        return f"{u_lbl} {top_verb} {v_lbl}{tail}."
    return f"{u_lbl} co-occurs with {v_lbl}."


def _short_description(G: nx.DiGraph, nid: Any) -> str:
    """One-line node blurb good for embedding. Mirrors GraphVectorStore."""
    data = G.nodes[nid]
    label = data.get("label", nid)
    ctype = data.get("concept_type") or data.get("type") or "concept"
    src = data.get("source_type", "n/a")
    neighbours: list[tuple[str, float]] = []
    for _, v, edata in G.out_edges(nid, data=True):
        neighbours.append((G.nodes[v].get("label", v), edata.get("weight", 1)))
    for u, _, edata in G.in_edges(nid, data=True):
        neighbours.append((G.nodes[u].get("label", u), edata.get("weight", 1)))
    neighbours.sort(key=lambda x: x[1], reverse=True)
    top_nbs = ", ".join(lbl for lbl, _ in neighbours[:5])
    return (
        f"{label}. A {ctype} concept from {src} data."
        + (f" Related to: {top_nbs}." if top_nbs else "")
    )
