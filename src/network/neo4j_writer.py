"""Direct Concept[]/Relationship[] -> Neo4j writer.

This module replaces :class:`src.network.graph_builder.GraphBuilder`. Instead
of materialising an in-memory ``nx.DiGraph`` and then pushing it via
:class:`src.extensions.neo4j_store.Neo4jStore`, it writes the extracted
concepts and relationships straight into Neo4j with batched ``UNWIND``
upserts.

The schema is identical to what :class:`Neo4jStore` already creates:

* ``(:Concept {id, label, concept_type, source_type, frequency, source_label,
                slice_id})``
* ``-[:RELATED {weight, types, source_label, slice_id, top_verb, verb_count,
                 verb_list, directed}]->``

Centrality / community properties (``pagerank``, ``community``, ``betweenness``,
``degree``, ``wcc_component``) are written later by
:class:`src.network.gds_analyser.GdsAnalysisRunner` running GDS server-side.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Iterable

from src.extraction.concept_extractor import Concept
from src.extraction.relationship_extractor import Relationship

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


_NODE_BATCH = 500
_EDGE_BATCH = 500


class Neo4jGraphWriter:
    """Persist a Concept[] + Relationship[] pair into Neo4j.

    Use as::

        with Neo4jStore.from_config() as store:
            writer = Neo4jGraphWriter(store, source_label="combined")
            writer.reset_source()                    # optional
            writer.write_concepts(concepts)
            writer.write_relationships(rels)
    """

    def __init__(
        self,
        store: "Neo4jStore",
        source_label: str = "combined",
        slice_id: str | None = None,
        min_edge_weight: int = 1,
    ):
        self.store = store
        self.source_label = source_label
        self.slice_id = slice_id
        self.min_edge_weight = max(1, int(min_edge_weight))
        self.store.ensure_schema()

    # ── Reset ──────────────────────────────────────────────────────
    def reset_source(self) -> int:
        """Wipe all Concept nodes (and their edges) for this source/slice."""
        if self.slice_id is None:
            return self.store.wipe_source(self.source_label)
        with self.store.session() as s:
            rec = s.run(
                "MATCH (c:Concept {source_label: $sl, slice_id: $sid}) "
                "WITH c, count(c) AS n "
                "DETACH DELETE c RETURN n",
                sl=self.source_label,
                sid=self.slice_id,
            ).single()
            return int(rec["n"]) if rec else 0

    # ── Concepts ───────────────────────────────────────────────────
    def write_concepts(self, concepts: list[Concept]) -> int:
        """Upsert Concept rows into Neo4j. Returns number of nodes written."""
        if not concepts:
            return 0
        rows = [self._concept_payload(c) for c in concepts]
        query = (
            "UNWIND $rows AS row "
            "MERGE (c:Concept {id: row.id}) "
            "SET c.label = row.label, "
            "    c.concept_type = row.concept_type, "
            "    c.source_type = row.source_type, "
            "    c.frequency = row.frequency, "
            "    c.source_label = row.source_label, "
            "    c.slice_id = row.slice_id"
        )
        with self.store.session() as s:
            for batch in _chunks(rows, _NODE_BATCH):
                s.run(query, rows=batch)
        logger.info(
            "Neo4jGraphWriter: wrote %d concepts (source_label=%s, slice_id=%s)",
            len(rows),
            self.source_label,
            self.slice_id,
        )
        return len(rows)

    # ── Relationships ──────────────────────────────────────────────
    def write_relationships(self, relationships: list[Relationship]) -> int:
        """Aggregate parallel edges and upsert them as :RELATED.

        Mirrors what GraphBuilder used to do in-memory: identical (src,tgt)
        pairs across multiple types are merged onto one edge whose ``types``
        property carries the union of types and whose ``verbs`` Counter is
        summed. Undirected (ASSOCIATION) edges produce two complementary
        rows so the graph stays symmetric.
        """
        if not relationships:
            return 0

        # Aggregate per (src, tgt) so MERGE doesn't create duplicate edges.
        edges: dict[tuple[str, str], dict[str, Any]] = {}
        for r in relationships:
            if r.weight < self.min_edge_weight:
                continue
            self._add_edge(edges, r.source_id, r.target_id, r)
            if not r.directed:
                self._add_edge(edges, r.target_id, r.source_id, r)

        if not edges:
            return 0

        rows: list[dict[str, Any]] = []
        for (src, tgt), data in edges.items():
            rows.append(self._edge_payload(src, tgt, data))

        # Neo4j 5+ refuses ``MERGE`` patterns with null property values
        # ("Cannot merge the following relationship because of null
        # property value for 'slice_id'"), so the MERGE pattern is
        # branched: temporal runs key on (source_label, slice_id) to
        # keep parallel slice views distinct, non-temporal runs key on
        # source_label only and persist slice_id as a regular property.
        if self.slice_id is None:
            query = (
                "UNWIND $rows AS row "
                "MATCH (a:Concept {id: row.source}) "
                "MATCH (b:Concept {id: row.target}) "
                "MERGE (a)-[r:RELATED {source_label: row.source_label}]->(b) "
                "SET r.weight = row.weight, "
                "    r.types = row.types, "
                "    r.directed = row.directed, "
                "    r.top_verb = row.top_verb, "
                "    r.verb_count = row.verb_count, "
                "    r.verb_list = row.verb_list, "
                "    r.slice_id = row.slice_id"
            )
        else:
            query = (
                "UNWIND $rows AS row "
                "MATCH (a:Concept {id: row.source}) "
                "MATCH (b:Concept {id: row.target}) "
                "MERGE (a)-[r:RELATED {source_label: row.source_label, "
                "                      slice_id: row.slice_id}]->(b) "
                "SET r.weight = row.weight, "
                "    r.types = row.types, "
                "    r.directed = row.directed, "
                "    r.top_verb = row.top_verb, "
                "    r.verb_count = row.verb_count, "
                "    r.verb_list = row.verb_list"
            )
        written = 0
        with self.store.session() as s:
            for batch in _chunks(rows, _EDGE_BATCH):
                s.run(query, rows=batch)
                written += len(batch)
        logger.info(
            "Neo4jGraphWriter: wrote %d relationships (source_label=%s, slice_id=%s)",
            written,
            self.source_label,
            self.slice_id,
        )
        return written

    # ── Convenience: write everything in one call ─────────────────
    def write(
        self,
        concepts: list[Concept],
        relationships: list[Relationship],
        reset: bool = False,
    ) -> dict[str, int]:
        """Write concepts + relationships and return counts."""
        if reset:
            removed = self.reset_source()
            logger.info(
                "Neo4jGraphWriter: cleared %d existing nodes (source_label=%s, "
                "slice_id=%s)",
                removed,
                self.source_label,
                self.slice_id,
            )
        n = self.write_concepts(concepts)
        e = self.write_relationships(relationships)
        return {"nodes": n, "edges": e}

    # ── Internals ──────────────────────────────────────────────────
    def _concept_payload(self, c: Concept) -> dict[str, Any]:
        return {
            "id": str(c.id),
            "label": str(c.label),
            "concept_type": str(c.type or "concept"),
            "source_type": str(c.source_type or "unknown"),
            "frequency": int(c.frequency or 0),
            "source_label": self.source_label,
            "slice_id": self.slice_id,
        }

    @staticmethod
    def _add_edge(
        edges: dict[tuple[str, str], dict[str, Any]],
        src: str,
        tgt: str,
        r: Relationship,
    ) -> None:
        key = (str(src), str(tgt))
        bucket = edges.get(key)
        if bucket is None:
            bucket = {
                "weight": float(r.weight),
                "types": {r.type},
                "directed": bool(r.directed),
                "verbs": Counter(r.verbs or {}),
            }
            edges[key] = bucket
        else:
            bucket["weight"] = float(bucket["weight"]) + float(r.weight)
            bucket["types"].add(r.type)
            bucket["directed"] = bool(bucket["directed"]) or bool(r.directed)
            bucket["verbs"].update(r.verbs or {})

    def _edge_payload(
        self, src: str, tgt: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        verbs: Counter = data.get("verbs") or Counter()
        ordered = verbs.most_common() if verbs else []
        top_verb = str(ordered[0][0]) if ordered else None
        verb_count = int(ordered[0][1]) if ordered else 0
        verb_list = [str(v) for v, _ in ordered]
        types = ",".join(sorted(str(t) for t in data.get("types", set())))
        return {
            "source": src,
            "target": tgt,
            "weight": float(data.get("weight", 1) or 1),
            "types": types,
            "directed": bool(data.get("directed", False)),
            "source_label": self.source_label,
            "slice_id": self.slice_id,
            "top_verb": top_verb,
            "verb_count": verb_count,
            "verb_list": verb_list,
        }


def _chunks(seq: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
