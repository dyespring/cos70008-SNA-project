"""Server-side SNA via the Neo4j Graph Data Science (GDS) plugin.

Replaces :class:`src.network.graph_analysis.GraphAnalyser`. Every centrality,
community-detection and path query is expressed as a Cypher / GDS call against
the running Neo4j instance, with results either written back to node
properties (``c.pagerank``, ``c.community`` etc.) or streamed for one-shot
queries (shortest path, broker score, cross-source aggregations).

Typical lifecycle::

    with Neo4jStore.from_config() as store:
        gds = GdsAnalysisRunner(store, source_label="combined")
        gds.run_all()                              # all write-back algos
        gds.shortest_path("climate change", "resilience")
        gds.top_concepts(metric="pagerank", n=10)

Algorithms used (Community / Apache 2-licensed tier so it works with the
Neo4j 5.x community Docker image already pinned in ``docker-compose.yml``):

* ``gds.degree.write``                   → ``c.degree``
* ``gds.pageRank.write``                 → ``c.pagerank``
* ``gds.betweenness.write``              → ``c.betweenness``
* ``gds.closeness.write``                → ``c.closeness``
* ``gds.eigenvector.write``              → ``c.eigenvector``
* ``gds.louvain.write``                  → ``c.community``
* ``gds.wcc.write``                      → ``c.wcc_component``
* ``gds.localClusteringCoefficient.write``→``c.local_clustering``
* ``gds.shortestPath.dijkstra.stream``    (per-call, no write-back)

Each ``run_*`` method projects an in-memory graph scoped to ``source_label``
(and optional ``slice_id``), executes the algorithm, then drops the
projection so subsequent runs start clean.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


class GdsUnavailableError(RuntimeError):
    """Raised when the GDS plugin isn't installed in the connected Neo4j instance."""


class GdsAnalysisRunner:
    """Run SNA algorithms server-side via Neo4j GDS, scoped per source_label."""

    def __init__(
        self,
        store: "Neo4jStore",
        source_label: str = "combined",
        slice_id: str | None = None,
    ):
        self.store = store
        self.source_label = source_label
        self.slice_id = slice_id

    # ── Lifecycle helpers ──────────────────────────────────────────
    def _projection_name(self) -> str:
        token = uuid.uuid4().hex[:8]
        return f"proj_{self.source_label}_{token}"

    def _ensure_gds(self) -> None:
        with self.store.session() as s:
            try:
                rec = s.run("CALL gds.version() YIELD gdsVersion").single()
            except Exception as e:
                raise GdsUnavailableError(
                    "Neo4j Graph Data Science plugin is not installed. "
                    "Restart docker-compose with the `graph-data-science` "
                    "plugin enabled (already wired in docker-compose.yml)."
                ) from e
            logger.info("GDS version: %s", rec["gdsVersion"] if rec else "?")

    def _node_filter(self) -> str:
        if self.slice_id is None:
            return "c.source_label = $sl"
        return "c.source_label = $sl AND c.slice_id = $sid"

    def _params(self) -> dict[str, Any]:
        if self.slice_id is None:
            return {"sl": self.source_label}
        return {"sl": self.source_label, "sid": self.slice_id}

    def _project(self, name: str) -> None:
        node_filter = self._node_filter()
        edge_filter = node_filter.replace("c.", "a.")
        edge_filter2 = node_filter.replace("c.", "b.")
        node_query = f"MATCH (c:Concept) WHERE {node_filter} RETURN id(c) AS id"
        edge_query = (
            f"MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            f"WHERE {edge_filter} AND {edge_filter2} "
            "RETURN id(a) AS source, id(b) AS target, "
            "       coalesce(r.weight, 1.0) AS weight"
        )
        with self.store.session() as s:
            s.run(
                "CALL gds.graph.project.cypher($name, $nq, $eq, "
                "{parameters: $params})",
                name=name,
                nq=node_query,
                eq=edge_query,
                params=self._params(),
            )

    def _drop(self, name: str) -> None:
        with self.store.session() as s:
            try:
                s.run("CALL gds.graph.drop($name, false)", name=name)
            except Exception as e:
                logger.debug("gds.graph.drop(%s) failed: %s", name, e)

    # ── Top-level: write every metric we know about ────────────────
    def run_all(self) -> dict[str, int]:
        """Run every write-back algorithm sequentially.

        Returns a dict {metric -> nodes_written}. Errors in individual
        algorithms are logged and the loop continues so one missing algo
        (e.g. GDS edition mismatch) doesn't kill the whole analysis pass.
        """
        self._ensure_gds()
        results: dict[str, int] = {}
        # Each call below is self-contained (projects, runs, drops).
        for fn_name, fn in (
            ("degree", self.run_degree),
            ("pagerank", self.run_pagerank),
            ("betweenness", self.run_betweenness),
            ("closeness", self.run_closeness),
            ("eigenvector", self.run_eigenvector),
            ("local_clustering", self.run_local_clustering),
            ("wcc", self.run_wcc),
            ("louvain", self.run_louvain),
        ):
            try:
                t0 = time.time()
                results[fn_name] = fn()
                logger.info("GDS %s done in %.2fs", fn_name, time.time() - t0)
            except Exception as e:
                logger.warning("GDS %s failed: %s", fn_name, e)
                results[fn_name] = 0
        return results

    # ── Centralities ───────────────────────────────────────────────
    def run_degree(self, write_property: str = "degree") -> int:
        return self._run_write(
            "gds.degree.write",
            write_property=write_property,
            extra_config={
                "relationshipWeightProperty": "weight",
                "writeProperty": write_property,
            },
        )

    def run_pagerank(
        self,
        write_property: str = "pagerank",
        damping_factor: float = 0.85,
        max_iterations: int = 30,
    ) -> int:
        return self._run_write(
            "gds.pageRank.write",
            write_property=write_property,
            extra_config={
                "writeProperty": write_property,
                "dampingFactor": damping_factor,
                "maxIterations": max_iterations,
                "relationshipWeightProperty": "weight",
            },
        )

    def run_betweenness(self, write_property: str = "betweenness") -> int:
        return self._run_write(
            "gds.betweenness.write",
            write_property=write_property,
            extra_config={"writeProperty": write_property},
        )

    def run_closeness(self, write_property: str = "closeness") -> int:
        return self._run_write(
            "gds.closeness.write",
            write_property=write_property,
            extra_config={"writeProperty": write_property},
        )

    def run_eigenvector(self, write_property: str = "eigenvector") -> int:
        return self._run_write(
            "gds.eigenvector.write",
            write_property=write_property,
            extra_config={
                "writeProperty": write_property,
                "relationshipWeightProperty": "weight",
            },
        )

    def run_local_clustering(self, write_property: str = "local_clustering") -> int:
        return self._run_write(
            "gds.localClusteringCoefficient.write",
            write_property=write_property,
            extra_config={"writeProperty": write_property},
        )

    # ── Communities & components ───────────────────────────────────
    def run_louvain(self, write_property: str = "community") -> int:
        return self._run_write(
            "gds.louvain.write",
            write_property=write_property,
            extra_config={
                "writeProperty": write_property,
                "relationshipWeightProperty": "weight",
                "includeIntermediateCommunities": False,
            },
        )

    def run_wcc(self, write_property: str = "wcc_component") -> int:
        return self._run_write(
            "gds.wcc.write",
            write_property=write_property,
            extra_config={"writeProperty": write_property},
        )

    # ── Generic write-back wrapper ─────────────────────────────────
    def _run_write(
        self,
        proc: str,
        write_property: str,
        extra_config: dict[str, Any] | None = None,
    ) -> int:
        name = self._projection_name()
        self._project(name)
        try:
            cfg = {"writeProperty": write_property}
            if extra_config:
                cfg.update(extra_config)
            with self.store.session() as s:
                rec = s.run(
                    f"CALL {proc}($name, $cfg) "
                    "YIELD nodePropertiesWritten",
                    name=name,
                    cfg=cfg,
                ).single()
                return int(rec["nodePropertiesWritten"]) if rec else 0
        finally:
            self._drop(name)

    # ── One-shot queries (no write-back) ───────────────────────────
    def shortest_path(
        self, source_label: str, target_label: str
    ) -> list[str] | None:
        """Find the labels along the shortest path between two concepts.

        Uses :func:`MATCH p = shortestPath(...)` directly — no GDS projection
        needed for a single short query, and works on weakly-connected hops.
        """
        sl = self.source_label
        with self.store.session() as s:
            row = s.run(
                "MATCH (s:Concept {source_label: $sl, label: $a}) "
                "MATCH (t:Concept {source_label: $sl, label: $b}) "
                "MATCH p = shortestPath((s)-[:RELATED*..10]-(t)) "
                "RETURN [n IN nodes(p) | n.label] AS labels",
                sl=sl,
                a=source_label,
                b=target_label,
            ).single()
        if not row:
            return None
        return list(row["labels"])

    def all_centralities(self) -> pd.DataFrame:
        """Read centrality properties out of Neo4j into a DataFrame.

        Equivalent to the old :meth:`GraphAnalyser.all_centralities`. Assumes
        :meth:`run_all` (or the relevant individual ``run_*`` methods) has
        already been executed so the properties exist.
        """
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "RETURN c.id AS node_id, c.label AS label, "
                "       coalesce(c.degree, 0.0)        AS degree, "
                "       coalesce(c.betweenness, 0.0)   AS betweenness, "
                "       coalesce(c.closeness, 0.0)     AS closeness, "
                "       coalesce(c.eigenvector, 0.0)   AS eigenvector, "
                "       coalesce(c.pagerank, 0.0)      AS pagerank, "
                "       coalesce(c.community, -1)      AS community "
                "ORDER BY pagerank DESC",
                **self._params(),
            ).data()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "node_id", "label", "degree", "betweenness",
                    "closeness", "eigenvector", "pagerank", "community",
                ]
            )
        return pd.DataFrame(rows)

    def partition(self) -> dict[str, int]:
        """Return {node_id -> community} from the louvain write-back result."""
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "RETURN c.id AS id, coalesce(c.community, -1) AS community",
                **self._params(),
            ).data()
        return {r["id"]: int(r["community"]) for r in rows}

    def community_summary(self) -> pd.DataFrame:
        """Per-community size + the highest-frequency / highest-PageRank members.

        Returns columns:

        * ``community`` — community id
        * ``size`` — number of concepts in the community
        * ``top_concepts`` — five highest-frequency labels (legacy column,
          kept for backward compatibility with older callers)
        * ``top_pagerank`` — five highest-PageRank labels (better aligns
          with the Hub / SPOF cards)
        * ``avg_pagerank`` — mean PageRank across the community
        """
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "WITH coalesce(c.community, -1) AS community, "
                "     collect({label: c.label, "
                "              freq: coalesce(c.frequency, 0), "
                "              pr:   coalesce(c.pagerank, 0.0)}) AS members "
                "RETURN community, size(members) AS size, members "
                "ORDER BY community",
                **self._params(),
            ).data()
        out = []
        for r in rows:
            members = list(r["members"] or [])
            if not members:
                continue
            top_freq = sorted(
                members, key=lambda m: m.get("freq") or 0, reverse=True,
            )[:5]
            top_pr = sorted(
                members, key=lambda m: m.get("pr") or 0.0, reverse=True,
            )[:5]
            avg_pr = (
                sum(float(m.get("pr") or 0.0) for m in members) / len(members)
            )
            out.append(
                {
                    "community": int(r["community"]),
                    "size": int(r["size"]),
                    "top_concepts": ", ".join(
                        m["label"] for m in top_freq
                    ),
                    "top_pagerank": ", ".join(
                        m["label"] for m in top_pr
                    ),
                    "avg_pagerank": float(avg_pr),
                }
            )
        return pd.DataFrame(out)

    def community_linkage_stats(self) -> pd.DataFrame:
        """Per-community internal vs outward bridge mass (directed from cluster).

        Each row is the community id of the **source** node of ``RELATED``
        edges. ``cohesion`` is ``internal_weight / (internal + bridge_out)``
        so low values flag porous / externally wired themes.
        """
        nf_a = self._node_filter().replace("c.", "a.")
        nf_b = self._node_filter().replace("c.", "b.")
        query = (
            f"MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            f"WHERE {nf_a} AND {nf_b} "
            "AND a.community IS NOT NULL AND b.community IS NOT NULL "
            "WITH a.community AS home, b.community AS other, "
            "     coalesce(r.weight, 1.0) AS w, r.sentiment AS sent "
            "WHERE home >= 0 AND other >= 0 "
            "WITH home AS community, "
            "     sum(CASE WHEN home = other THEN 1 ELSE 0 END) AS internal_edges, "
            "     sum(CASE WHEN home = other THEN w ELSE 0.0 END) AS internal_weight, "
            "     avg(CASE WHEN home = other THEN toFloat(sent) END) "
            "       AS avg_internal_sentiment, "
            "     sum(CASE WHEN home <> other THEN 1 ELSE 0 END) AS bridge_out_edges, "
            "     sum(CASE WHEN home <> other THEN w ELSE 0.0 END) AS bridge_out_weight "
            "RETURN community, internal_edges, internal_weight, "
            "       avg_internal_sentiment, bridge_out_edges, bridge_out_weight, "
            "       internal_weight / (internal_weight + bridge_out_weight + 1e-9) "
            "         AS cohesion "
            "ORDER BY community"
        )
        with self.store.session() as s:
            rows = s.run(query, **self._params()).data()
        return pd.DataFrame(rows)

    def community_members(self, community_id: int, top_n: int = 25) -> pd.DataFrame:
        """Return the top members of a single community for a drill-down view.

        Sorted by PageRank (with frequency as tie-breaker). Includes the
        most useful per-node columns the dashboard needs.
        """
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "  AND c.community = $cid "
                "RETURN c.label AS label, "
                "       coalesce(c.concept_type, 'concept') AS concept_type, "
                "       coalesce(c.source_type, 'unknown')  AS source_type, "
                "       coalesce(c.frequency, 0)            AS frequency, "
                "       coalesce(c.pagerank, 0.0)           AS pagerank, "
                "       coalesce(c.betweenness, 0.0)        AS betweenness "
                "ORDER BY pagerank DESC, frequency DESC LIMIT $n",
                cid=int(community_id), n=int(top_n), **self._params(),
            ).data()
        return pd.DataFrame(rows)

    def community_edges(
        self, community_id: int, top_n: int = 15,
    ) -> pd.DataFrame:
        """Sample the strongest intra-community edges.

        Returns one row per edge, ordered by weight. Includes sentiment
        and verb info so the dashboard can show *what* connects the
        community — "X causes Y", "A loves B", etc.
        """
        with self.store.session() as s:
            rows = s.run(
                "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
                f"WHERE {self._node_filter().replace('c.', 'a.')} "
                f"  AND {self._node_filter().replace('c.', 'b.')} "
                "  AND a.community = $cid AND b.community = $cid "
                "RETURN a.label AS source, b.label AS target, "
                "       coalesce(r.weight, 1.0) AS weight, "
                "       coalesce(r.types, 'ASSOCIATION') AS types, "
                "       r.top_verb AS top_verb, "
                "       r.sentiment AS sentiment "
                "ORDER BY weight DESC LIMIT $n",
                cid=int(community_id), n=int(top_n), **self._params(),
            ).data()
        return pd.DataFrame(rows)

    def community_neighbours(
        self, community_id: int, top_n: int = 10,
    ) -> pd.DataFrame:
        """Adjacent communities ranked by edge weight crossing into them.

        Used by the Brokers / Communities drill-downs to show which
        clusters are most strongly bridged from a given community.
        """
        with self.store.session() as s:
            rows = s.run(
                "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
                f"WHERE {self._node_filter().replace('c.', 'a.')} "
                f"  AND {self._node_filter().replace('c.', 'b.')} "
                "  AND a.community = $cid AND b.community <> $cid "
                "  AND b.community IS NOT NULL "
                "RETURN b.community AS neighbour_community, "
                "       count(r) AS n_edges, "
                "       sum(coalesce(r.weight, 1.0)) AS total_weight "
                "ORDER BY total_weight DESC LIMIT $n",
                cid=int(community_id), n=int(top_n), **self._params(),
            ).data()
        return pd.DataFrame(rows)

    def broker_drilldown(
        self, label: str, top_n_edges: int = 12,
    ) -> dict[str, Any]:
        """Return everything the Brokers drill-down needs for one concept.

        Output keys:

        * ``home_community``  — the broker's own community id (or -1)
        * ``reached_communities`` — DataFrame of (community, n_edges, total_weight)
        * ``cross_edges``    — DataFrame of (target_label, neighbour_community, weight)
        """
        with self.store.session() as s:
            home_rec = s.run(
                "MATCH (c:Concept {label: $lbl}) "
                f"WHERE {self._node_filter()} "
                "RETURN coalesce(c.community, -1) AS home_community LIMIT 1",
                lbl=str(label), **self._params(),
            ).single()
            home = int(home_rec["home_community"]) if home_rec else -1

            reach_rows = s.run(
                "MATCH (c:Concept {label: $lbl})-[r:RELATED]-(nb:Concept) "
                f"WHERE {self._node_filter()} "
                f"  AND {self._node_filter().replace('c.', 'nb.')} "
                "  AND nb.community IS NOT NULL "
                "  AND nb.community <> coalesce(c.community, -2) "
                "RETURN nb.community AS neighbour_community, "
                "       count(r) AS n_edges, "
                "       sum(coalesce(r.weight, 1.0)) AS total_weight "
                "ORDER BY total_weight DESC",
                lbl=str(label), **self._params(),
            ).data()

            edge_rows = s.run(
                "MATCH (c:Concept {label: $lbl})-[r:RELATED]-(nb:Concept) "
                f"WHERE {self._node_filter()} "
                f"  AND {self._node_filter().replace('c.', 'nb.')} "
                "  AND nb.community IS NOT NULL "
                "  AND nb.community <> coalesce(c.community, -2) "
                "RETURN nb.label AS neighbour_label, "
                "       coalesce(nb.community, -1) AS neighbour_community, "
                "       coalesce(r.weight, 1.0) AS weight, "
                "       coalesce(r.types, 'ASSOCIATION') AS types, "
                "       r.top_verb AS top_verb, "
                "       r.sentiment AS sentiment "
                "ORDER BY weight DESC LIMIT $n",
                lbl=str(label), n=int(top_n_edges), **self._params(),
            ).data()

        return {
            "home_community": home,
            "reached_communities": pd.DataFrame(reach_rows),
            "cross_edges": pd.DataFrame(edge_rows),
        }

    def find_brokers(self, top_n: int = 10) -> pd.DataFrame:
        """High-betweenness nodes whose neighbours span multiple communities.

        Equivalent to the old NX heuristic, expressed in pure Cypher.
        Requires both ``c.betweenness`` and ``c.community`` to be populated
        (run :meth:`run_betweenness` + :meth:`run_louvain` first).
        """
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "OPTIONAL MATCH (c)-[r:RELATED]-(nb:Concept) "
                f"WHERE {self._node_filter().replace('c.', 'nb.')} "
                "WITH c, "
                "     count(DISTINCT nb) AS total_edges, "
                "     count(DISTINCT CASE WHEN nb.community <> c.community "
                "                          THEN nb END) AS cross_comm "
                "RETURN c.id AS node_id, c.label AS label, "
                "       coalesce(c.community, -1) AS community, "
                "       coalesce(c.betweenness, 0.0) AS betweenness, "
                "       cross_comm AS cross_community_edges, "
                "       total_edges, "
                "       coalesce(c.betweenness, 0.0) * cross_comm AS broker_score "
                "ORDER BY broker_score DESC LIMIT $top",
                top=int(top_n),
                **self._params(),
            ).data()
        return pd.DataFrame(rows)

    # ── Cross-source aggregations (pure Cypher, no GDS) ────────────
    def source_distribution(self) -> pd.DataFrame:
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "RETURN coalesce(c.source_type, 'unknown') AS source_type, "
                "       count(*) AS node_count "
                "ORDER BY source_type",
                **self._params(),
            ).data()
        return pd.DataFrame(rows)

    def cross_source_edges(self) -> pd.DataFrame:
        with self.store.session() as s:
            rows = s.run(
                "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
                f"WHERE {self._node_filter().replace('c.', 'a.')} "
                f"  AND {self._node_filter().replace('c.', 'b.')} "
                "  AND coalesce(a.source_type, 'unknown') <> "
                "      coalesce(b.source_type, 'unknown') "
                "RETURN a.label AS source_node, b.label AS target_node, "
                "       a.source_type AS source_type_from, "
                "       b.source_type AS source_type_to, "
                "       r.weight AS weight, r.types AS types "
                "ORDER BY weight DESC",
                **self._params(),
            ).data()
        return pd.DataFrame(rows)

    def bridging_concepts(self, top_n: int = 20) -> pd.DataFrame:
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "OPTIONAL MATCH (c)-[r:RELATED]-(nb:Concept) "
                f"WHERE {self._node_filter().replace('c.', 'nb.')} "
                "WITH c, count(DISTINCT nb) AS total_nb, "
                "     count(DISTINCT CASE WHEN coalesce(nb.source_type, 'unknown') "
                "                              <> coalesce(c.source_type, 'unknown') "
                "                          THEN nb END) AS cross_nb "
                "RETURN c.id AS node_id, c.label AS label, "
                "       coalesce(c.source_type, 'unknown') AS source_type, "
                "       coalesce(c.betweenness, 0.0) AS betweenness, "
                "       cross_nb AS cross_source_neighbors, "
                "       total_nb AS total_neighbors, "
                "       CASE WHEN total_nb = 0 THEN 0.0 "
                "            ELSE toFloat(cross_nb) / total_nb END AS cross_ratio, "
                "       coalesce(c.betweenness, 0.0) "
                "         * (1 + CASE WHEN total_nb = 0 THEN 0 "
                "                     ELSE toFloat(cross_nb) / total_nb END) "
                "         + (CASE WHEN c.source_type = 'both' THEN 0.1 ELSE 0 END) "
                "         AS bridge_score "
                "ORDER BY bridge_score DESC LIMIT $top",
                top=int(top_n),
                **self._params(),
            ).data()
        return pd.DataFrame(rows)

    def source_comparison_summary(self) -> dict[str, Any]:
        with self.store.session() as s:
            rec = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "WITH count(c) AS total, "
                "     count(CASE c.source_type WHEN 'policy' THEN 1 END) AS policy, "
                "     count(CASE c.source_type WHEN 'yelp' THEN 1 END) AS yelp, "
                "     count(CASE c.source_type WHEN 'both' THEN 1 END) AS both "
                "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
                f"WHERE {self._node_filter().replace('c.', 'a.')} "
                f"  AND {self._node_filter().replace('c.', 'b.')} "
                "WITH total, policy, yelp, both, count(r) AS edges "
                "RETURN total, policy, yelp, both, edges",
                **self._params(),
            ).single()
        if rec is None:
            return {
                "total_nodes": 0, "total_edges": 0,
                "policy_only_concepts": 0, "yelp_only_concepts": 0,
                "shared_concepts": 0, "overlap_pct": 0.0,
            }
        total = int(rec["total"] or 0)
        return {
            "total_nodes": total,
            "total_edges": int(rec["edges"] or 0),
            "policy_only_concepts": int(rec["policy"] or 0),
            "yelp_only_concepts": int(rec["yelp"] or 0),
            "shared_concepts": int(rec["both"] or 0),
            "overlap_pct": (
                100.0 * int(rec["both"] or 0) / total if total else 0.0
            ),
        }

    # ── Summary ────────────────────────────────────────────────────
    def summary_stats(self) -> dict[str, Any]:
        with self.store.session() as s:
            rec = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "WITH count(c) AS n "
                "OPTIONAL MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
                f"WHERE {self._node_filter().replace('c.', 'a.')} "
                f"  AND {self._node_filter().replace('c.', 'b.')} "
                "RETURN n AS nodes, count(r) AS edges",
                **self._params(),
            ).single()
        n = int(rec["nodes"] or 0) if rec else 0
        e = int(rec["edges"] or 0) if rec else 0
        density = (e / (n * (n - 1))) if n > 1 else 0.0

        # Average local clustering needs the GDS write to have happened.
        with self.store.session() as s:
            cc_rec = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "  AND c.local_clustering IS NOT NULL "
                "RETURN avg(c.local_clustering) AS avg_cc",
                **self._params(),
            ).single()
        avg_cc = float(cc_rec["avg_cc"] or 0.0) if cc_rec else 0.0

        with self.store.session() as s:
            comp_rec = s.run(
                "MATCH (c:Concept) "
                f"WHERE {self._node_filter()} "
                "  AND c.wcc_component IS NOT NULL "
                "RETURN count(DISTINCT c.wcc_component) AS k",
                **self._params(),
            ).single()
        components = int(comp_rec["k"] or 0) if comp_rec else 0

        return {
            "nodes": n,
            "edges": e,
            "density": density,
            "avg_clustering": avg_cc,
            "weakly_connected_components": components,
        }
