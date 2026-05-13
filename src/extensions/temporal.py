"""Temporal comparison: write per-slice subgraphs into Neo4j and compare them.

In the Neo4j-only pipeline each "slice" is just a value of the
``slice_id`` property on the same ``(:Concept)`` / ``[:RELATED]`` schema.
That keeps everything queryable from Cypher and lets the dashboard /
chatbot see snapshots side-by-side without rebuilding any in-memory graph.

Comparison metrics (overlap, jaccard) are computed via Cypher set
operations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from src.extraction.concept_extractor import Concept, ConceptExtractor
from src.extraction.relationship_extractor import RelationshipExtractor
from src.network.neo4j_writer import Neo4jGraphWriter
from src.preprocessing.tokeniser import ProcessedDocument

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


@dataclass
class TemporalSlice:
    """A network snapshot for one time period or document section."""
    label: str
    slice_id: str
    source_label: str
    concepts: list[Concept]
    node_count: int
    edge_count: int


# Yelp per-year slices use ``yelp_reviews_to_year_slices`` labels like "2019";
# :meth:`TemporalAnalyser._safe_slice_id` turns them into ``slice_XX_YYYY``.
_YELP_CALENDAR_SLICE_ID = re.compile(r"^slice_\d{2}_(\d{4})$")


def is_yelp_calendar_slice_id(slice_id: str) -> bool:
    """True when ``slice_id`` is a Yelp calendar-year snapshot (not policy page chunks)."""
    m = _YELP_CALENDAR_SLICE_ID.match((slice_id or "").strip())
    if not m:
        return False
    y = int(m.group(1))
    return 1900 <= y <= 2100


def yelp_calendar_year_from_slice_id(slice_id: str) -> int | None:
    """Extract four-digit year from a Yelp calendar slice id, or ``None``."""
    m = _YELP_CALENDAR_SLICE_ID.match((slice_id or "").strip())
    return int(m.group(1)) if m else None


_POLICY_PAGES_SUFFIX = re.compile(r"_Pages_(\d+)_(\d+)$", re.IGNORECASE)


def slice_display_label(slice_id: str) -> str:
    """Short label for UI and narrative cards (``slice_id`` stays the Neo4j key)."""
    sid = (slice_id or "").strip()
    y = yelp_calendar_year_from_slice_id(sid)
    if y is not None:
        return f"Yelp reviews ({y})"
    m = _POLICY_PAGES_SUFFIX.search(sid)
    if m:
        return f"Policy text (pp. {m.group(1)}–{m.group(2)})"
    return sid


def dashboard_temporal_slices(
    source_label: str, slices: list[TemporalSlice],
) -> list[TemporalSlice]:
    """Slices shown on the dashboard Temporal tab.

    For ``combined`` graphs, ``pipeline.py temporal`` may attach both policy
    page-chunk slices and Yelp year slices. Only the latter are calendar-time
    comparable, so we hide policy chunks in the UI while leaving Neo4j data
    untouched.
    """
    if source_label != "combined":
        return list(slices)
    filtered = [s for s in slices if is_yelp_calendar_slice_id(s.slice_id)]
    return sorted(
        filtered,
        key=lambda s: yelp_calendar_year_from_slice_id(s.slice_id) or 0,
    )


def canonical_concept_overlap_key(label: str | None, node_id: str | None) -> str:
    """Stable lemma token for cross-slice overlap (aligns label vs ``Concept.id``).

    When both ``label`` and ``id`` exist but normalise differently (e.g.
    ``Carrot Cake`` vs ``carrot_cake``), we prefer the ``id`` side because it
    is the MERGE key in Neo4j; when only one is present, we normalise it.
    """
    def _norm(x: object) -> str:
        if x is None:
            return ""
        return str(x).strip().lower().replace("-", "_").replace(" ", "_")

    la, iid = _norm(label), _norm(node_id)
    if la and iid:
        return iid if la != iid else la
    return la or iid


class TemporalAnalyser:
    """Build per-slice subgraphs in Neo4j and compare them with Cypher."""

    def __init__(
        self,
        store: "Neo4jStore",
        source_label: str,
        concept_extractor: ConceptExtractor | None = None,
        rel_extractor: RelationshipExtractor | None = None,
        min_edge_weight: int = 1,
    ):
        self.store = store
        self.source_label = source_label
        self.concept_extractor = concept_extractor or ConceptExtractor()
        self.rel_extractor = rel_extractor or RelationshipExtractor()
        self.min_edge_weight = min_edge_weight

    # ── Build slices ───────────────────────────────────────────────
    def build_slices(
        self,
        slices: list[tuple[str, list[ProcessedDocument]]],
        reset: bool = True,
    ) -> list[TemporalSlice]:
        """Extract + write one subgraph per slice, returning a TemporalSlice each.

        Each slice is stored under ``(source_label, slice_id)`` where
        ``slice_id = "slice_{i}"`` (and the human ``label`` is preserved
        on the dataclass for display purposes).
        """
        if reset:
            self._wipe_all_slices()

        results: list[TemporalSlice] = []
        for i, (label, docs) in enumerate(slices):
            slice_id = self._safe_slice_id(label, i)
            concepts = self.concept_extractor.extract(docs)
            rels = self.rel_extractor.extract(docs, concepts)
            writer = Neo4jGraphWriter(
                self.store,
                source_label=self.source_label,
                slice_id=slice_id,
                min_edge_weight=self.min_edge_weight,
            )
            counts = writer.write(concepts, rels, reset=False)
            results.append(
                TemporalSlice(
                    label=label,
                    slice_id=slice_id,
                    source_label=self.source_label,
                    concepts=concepts,
                    node_count=counts["nodes"],
                    edge_count=counts["edges"],
                )
            )
            logger.info(
                "TemporalSlice '%s' (slice_id=%s): %d nodes, %d edges",
                label,
                slice_id,
                counts["nodes"],
                counts["edges"],
            )
        return results

    def _wipe_all_slices(self) -> None:
        """Delete every slice subgraph for this source_label."""
        with self.store.session() as s:
            s.run(
                "MATCH (c:Concept {source_label: $sl}) "
                "WHERE c.slice_id IS NOT NULL "
                "DETACH DELETE c",
                sl=self.source_label,
            )

    @staticmethod
    def _safe_slice_id(label: str, idx: int) -> str:
        token = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label
        )
        return f"slice_{idx:02d}_{token}"[:60]

    def _canonical_lemmas_for_slice(self, slice_id: str) -> set[str]:
        """Distinct normalised lemmas in a slice (MATCH on ``slice_id``)."""
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept {source_label: $sl, slice_id: $sid}) "
                "RETURN c.label AS label, c.id AS id",
                sl=self.source_label,
                sid=slice_id,
            ).data()
        out: set[str] = set()
        for r in rows:
            k = canonical_concept_overlap_key(r.get("label"), r.get("id"))
            if k:
                out.add(k)
        return out

    def _canonical_edge_pair_keys_for_slice(self, slice_id: str) -> set[str]:
        """Undirected canonical (lemma_a, lemma_b) keys for RELATED in a slice."""
        with self.store.session() as s:
            rows = s.run(
                "MATCH (a:Concept {source_label: $sl, slice_id: $sid})"
                "-[:RELATED {source_label: $sl, slice_id: $sid}]->"
                "(b:Concept {source_label: $sl, slice_id: $sid}) "
                "RETURN a.label AS la, a.id AS ia, b.label AS lb, b.id AS ib",
                sl=self.source_label,
                sid=slice_id,
            ).data()
        keys: set[str] = set()
        for r in rows:
            ka = canonical_concept_overlap_key(r.get("la"), r.get("ia"))
            kb = canonical_concept_overlap_key(r.get("lb"), r.get("ib"))
            if not ka or not kb:
                continue
            if ka <= kb:
                keys.add(f"{ka}||{kb}")
            else:
                keys.add(f"{kb}||{ka}")
        return keys

    # ── Compare ────────────────────────────────────────────────────
    def compare_slices(self, a: TemporalSlice, b: TemporalSlice) -> dict:
        """Jaccard / overlap between two stored slices.

        Computed in **Python** from ``MATCH`` rows so we read ``label`` and
        ``id`` exactly as the driver returns them (avoids brittle Cypher-only
        string pipelines). Lemma keys use :func:`canonical_concept_overlap_key`
        so ``Carrot Cake`` lines up with ``carrot_cake`` like ``MERGE`` on ``id``.
        """
        lemmas_a = self._canonical_lemmas_for_slice(a.slice_id)
        lemmas_b = self._canonical_lemmas_for_slice(b.slice_id)
        shared_nodes = len(lemmas_a & lemmas_b)
        nodes_a, nodes_b = len(lemmas_a), len(lemmas_b)
        union_nodes = nodes_a + nodes_b - shared_nodes

        edges_a = self._canonical_edge_pair_keys_for_slice(a.slice_id)
        edges_b = self._canonical_edge_pair_keys_for_slice(b.slice_id)
        shared_edges = len(edges_a & edges_b)
        edges_a_n, edges_b_n = len(edges_a), len(edges_b)
        union_edges = edges_a_n + edges_b_n - shared_edges

        return {
            "slice_a": a.label,
            "slice_b": b.label,
            "nodes_added": nodes_b - shared_nodes,
            "nodes_removed": nodes_a - shared_nodes,
            "nodes_shared": shared_nodes,
            "edges_added": edges_b_n - shared_edges,
            "edges_removed": edges_a_n - shared_edges,
            "edges_shared": shared_edges,
            "jaccard_nodes": (
                shared_nodes / union_nodes if union_nodes else 0.0
            ),
            "jaccard_edges": (
                shared_edges / union_edges if union_edges else 0.0
            ),
        }

    def comparison_table(self, slices: list[TemporalSlice]) -> pd.DataFrame:
        rows = [
            self.compare_slices(slices[i], slices[i + 1])
            for i in range(len(slices) - 1)
        ]
        return pd.DataFrame(rows)

    def slice_summary(self, slices: list[TemporalSlice]) -> pd.DataFrame:
        rows = []
        for sl in slices:
            density = self._density(sl.slice_id)
            top = sl.concepts[0].label if sl.concepts else "n/a"
            rows.append(
                {
                    "slice": sl.label,
                    "slice_id": sl.slice_id,
                    "nodes": sl.node_count,
                    "edges": sl.edge_count,
                    "density": density,
                    "top_concept": top,
                }
            )
        return pd.DataFrame(rows)

    def _density(self, slice_id: str) -> float:
        with self.store.session() as s:
            row = s.run(
                "OPTIONAL MATCH (c:Concept {source_label: $sl, slice_id: $sid}) "
                "WITH count(c) AS n "
                "OPTIONAL MATCH (a:Concept {source_label: $sl, slice_id: $sid})"
                "-[r:RELATED {source_label: $sl, slice_id: $sid}]->"
                "(b:Concept {source_label: $sl, slice_id: $sid}) "
                "RETURN n, count(r) AS e",
                sl=self.source_label,
                sid=slice_id,
            ).single()
        row = row or {"n": 0, "e": 0}
        n = int(row["n"] or 0)
        e = int(row["e"] or 0)
        return e / (n * (n - 1)) if n > 1 else 0.0

    # ── Server-side analytics per slice ────────────────────────────
    def run_gds_per_slice(self, slices: list[TemporalSlice]) -> dict[str, dict]:
        """Run GDS centrality/community write-back for every slice.

        Each slice gets its own ``c.pagerank`` / ``c.community`` /
        ``c.betweenness`` … so the per-slice subgraph behaves like a
        first-class network — community drift, centrality drift, and
        every Insight Engine query can be re-aimed at a slice.
        """
        from src.network.gds_analyser import GdsAnalysisRunner

        out: dict[str, dict] = {}
        for sl in slices:
            runner = GdsAnalysisRunner(
                self.store,
                source_label=self.source_label,
                slice_id=sl.slice_id,
            )
            try:
                out[sl.slice_id] = runner.run_all()
            except Exception as e:
                logger.warning(
                    "GDS run failed for slice %s: %s", sl.slice_id, e
                )
                out[sl.slice_id] = {"_error": str(e)}
        return out

    # ── Richer comparison metrics ──────────────────────────────────
    def top_concepts_for_slice(
        self, slice_id: str, k: int = 10, metric: str = "frequency",
    ) -> list[dict]:
        """Return the top-K concepts in a slice ordered by ``metric``.

        Used for human-readable diff text in the dashboard
        ("Slice 2024 added 'sustainability', 'net zero'; dropped …").
        Falls back to ``frequency`` when the requested metric isn't
        populated (no GDS run, or GDS run failed for that slice).
        """
        if metric not in {"frequency", "pagerank", "betweenness", "degree"}:
            raise ValueError(f"Unsupported metric: {metric}")
        rank_expr = f"coalesce(c.{metric}, toFloat(coalesce(c.frequency, 0)))"
        with self.store.session() as s:
            rows = s.run(
                f"MATCH (c:Concept {{source_label: $sl, slice_id: $sid}}) "
                f"RETURN c.label AS label, "
                f"       coalesce(c.frequency, 0) AS frequency, "
                f"       coalesce(c.{metric}, 0.0) AS metric_value "
                f"ORDER BY {rank_expr} DESC LIMIT $k",
                sl=self.source_label, sid=slice_id, k=int(k),
            ).data()
        return rows

    def avg_sentiment_for_slice(self, slice_id: str) -> dict:
        """Average ``r.sentiment`` across all edges of a slice (NaN-safe)."""
        with self.store.session() as s:
            rec = s.run(
                "MATCH (a:Concept {source_label: $sl, slice_id: $sid})"
                "-[r:RELATED {source_label: $sl, slice_id: $sid}]->"
                "(b:Concept {source_label: $sl, slice_id: $sid}) "
                "WHERE r.sentiment IS NOT NULL "
                "RETURN avg(r.sentiment) AS avg_s, count(r) AS n",
                sl=self.source_label, sid=slice_id,
            ).single()
        if not rec or rec["n"] is None:
            return {"avg": None, "n": 0}
        n = int(rec["n"] or 0)
        return {
            "avg": float(rec["avg_s"]) if rec["avg_s"] is not None else None,
            "n": n,
        }

    def enriched_summary(self, slices: list[TemporalSlice]) -> pd.DataFrame:
        """Per-slice summary including density, top concept, avg sentiment.

        Superset of :meth:`slice_summary` — kept separate so legacy
        callers still get the lighter shape.
        """
        rows = []
        for sl in slices:
            density = self._density(sl.slice_id)
            sent = self.avg_sentiment_for_slice(sl.slice_id)
            top = self.top_concepts_for_slice(sl.slice_id, k=1)
            top_label = top[0]["label"] if top else "n/a"
            rows.append(
                {
                    "slice": sl.label,
                    "slice_id": sl.slice_id,
                    "nodes": sl.node_count,
                    "edges": sl.edge_count,
                    "density": density,
                    "avg_sentiment": sent["avg"],
                    "sentiment_edges": sent["n"],
                    "top_concept": top_label,
                }
            )
        return pd.DataFrame(rows)

    def detailed_compare(
        self, a: TemporalSlice, b: TemporalSlice, top_k: int = 8,
    ) -> dict:
        """Compare two slices: jaccard + symmetric diff on top concepts.

        ``compare_slices`` returns counts only; this method goes further
        and lists the *labels* that newly appeared / disappeared between
        ``a`` and ``b``, plus how the average sentiment shifted. The
        dashboard uses the result to write a one-paragraph drift story.
        """
        base = self.compare_slices(a, b)

        with self.store.session() as s:
            row = s.run(
                "OPTIONAL MATCH (na:Concept {source_label: $sl, slice_id: $a}) "
                "WITH collect(DISTINCT na.label) AS raw_a "
                "OPTIONAL MATCH (nb:Concept {source_label: $sl, slice_id: $b}) "
                "WITH raw_a, collect(DISTINCT nb.label) AS raw_b "
                "RETURN [x IN raw_a WHERE x IS NOT NULL] AS a_labels, "
                "       [x IN raw_b WHERE x IS NOT NULL] AS b_labels",
                sl=self.source_label, a=a.slice_id, b=b.slice_id,
            ).single()
        row = row or {"a_labels": [], "b_labels": []}
        a_labels = set(row["a_labels"] or [])
        b_labels = set(row["b_labels"] or [])

        appeared = sorted(b_labels - a_labels)
        disappeared = sorted(a_labels - b_labels)

        # Add a frequency-weighted top-K view of the diffs so the
        # dashboard can highlight which new concepts actually mattered.
        top_a = {
            r["label"]: r["frequency"]
            for r in self.top_concepts_for_slice(a.slice_id, k=top_k * 4)
        }
        top_b = {
            r["label"]: r["frequency"]
            for r in self.top_concepts_for_slice(b.slice_id, k=top_k * 4)
        }
        appeared_top = sorted(
            (lbl for lbl in appeared if lbl in top_b),
            key=lambda l: top_b.get(l, 0), reverse=True,
        )[:top_k]
        disappeared_top = sorted(
            (lbl for lbl in disappeared if lbl in top_a),
            key=lambda l: top_a.get(l, 0), reverse=True,
        )[:top_k]

        sent_a = self.avg_sentiment_for_slice(a.slice_id)
        sent_b = self.avg_sentiment_for_slice(b.slice_id)
        sentiment_delta = None
        if sent_a["avg"] is not None and sent_b["avg"] is not None:
            sentiment_delta = sent_b["avg"] - sent_a["avg"]

        return {
            **base,
            "appeared": appeared,
            "disappeared": disappeared,
            "appeared_top": appeared_top,
            "disappeared_top": disappeared_top,
            "sentiment_a": sent_a["avg"],
            "sentiment_b": sent_b["avg"],
            "sentiment_delta": sentiment_delta,
        }

    def existing_slices(self) -> list[TemporalSlice]:
        """Reconstruct lightweight :class:`TemporalSlice` rows from Neo4j.

        Used by the dashboard / TemporalInsightEngine when no in-memory
        ``TemporalSlice`` list is available (e.g. after a Streamlit
        restart). Concepts list is left empty — only ``slice_id``,
        ``label``, and node / edge counts are reconstructed.
        """
        with self.store.session() as s:
            rows = s.run(
                "MATCH (c:Concept {source_label: $sl}) "
                "WHERE c.slice_id IS NOT NULL "
                "WITH c.slice_id AS slice_id, count(c) AS nodes "
                "OPTIONAL MATCH "
                "  (a:Concept {source_label: $sl, slice_id: slice_id})"
                "  -[r:RELATED {source_label: $sl, slice_id: slice_id}]->"
                "  (b:Concept {source_label: $sl, slice_id: slice_id}) "
                "RETURN slice_id, nodes, count(r) AS edges "
                "ORDER BY slice_id",
                sl=self.source_label,
            ).data()
        return [
            TemporalSlice(
                label=slice_display_label(r["slice_id"]),
                slice_id=r["slice_id"],
                source_label=self.source_label,
                concepts=[],
                node_count=int(r["nodes"] or 0),
                edge_count=int(r["edges"] or 0),
            )
            for r in rows
        ]
