"""Temporal insights — turn slice-over-slice deltas into narrative cards.

Mirrors the design of :class:`src.extensions.insight_engine.InsightEngine`
but operates on a sequence of :class:`TemporalSlice` snapshots produced
by :class:`TemporalAnalyser`.

Three card categories
---------------------
* **Trend Insights**     — shape of the network across time
  (size growth, density, sentiment drift).
* **Drift Insights**     — what entered / left the discourse
  (new top concepts, retired concepts, biggest jaccard shift).
* **Comparison Insights**— side-by-side latest vs earliest snapshot.

Each insight is computed from one Cypher / DataFrame summary; the same
optional :class:`LLMProvider` polish layer used by ``InsightEngine`` is
reused so cards can be re-phrased into executive language on demand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from src.extensions.insight_engine import Insight, _polish_text

if TYPE_CHECKING:
    from src.extensions.chatbot import LLMProvider
    from src.extensions.neo4j_store import Neo4jStore
    from src.extensions.temporal import TemporalAnalyser, TemporalSlice


logger = logging.getLogger(__name__)


# Same severity-driven cap as InsightEngine for visual parity in the UI.
_TOP_PER_CATEGORY = 3
_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}


@dataclass
class TemporalInsightBundle:
    """Three-bucket bundle returned by :meth:`TemporalInsightEngine.all_insights`."""

    trend: list[Insight]
    drift: list[Insight]
    comparison: list[Insight]

    def as_dict(self) -> dict[str, list[Insight]]:
        return {"trend": self.trend, "drift": self.drift, "comparison": self.comparison}


class TemporalInsightEngine:
    """Compute time-aware insights across a list of :class:`TemporalSlice`."""

    def __init__(
        self,
        store: "Neo4jStore",
        source_label: str,
        analyser: Optional["TemporalAnalyser"] = None,
        llm: Optional["LLMProvider"] = None,
    ):
        from src.extensions.temporal import TemporalAnalyser

        self.store = store
        self.source_label = source_label
        self.analyser = analyser or TemporalAnalyser(
            store, source_label=source_label
        )
        self.llm = llm
        self._polish_cache: dict[str, str] = {}

    # ── Public API ─────────────────────────────────────────────────
    def all_insights(
        self,
        slices: list["TemporalSlice"] | None = None,
        polish: bool = False,
    ) -> TemporalInsightBundle:
        """Compute the full bundle and (optionally) LLM-polish each card."""
        if slices is None:
            slices = self.analyser.existing_slices()

        if len(slices) < 2:
            note = self._not_enough_slices(slices)
            return TemporalInsightBundle(trend=[note], drift=[], comparison=[])

        trend = self._top(self._collect_trend(slices))
        drift = self._top(self._collect_drift(slices))
        comparison = self._top(self._collect_comparison(slices))

        bundle = TemporalInsightBundle(trend=trend, drift=drift, comparison=comparison)
        if polish and self.llm is not None:
            for cards in bundle.as_dict().values():
                for ins in cards:
                    if ins.available:
                        ins.body_polished = _polish_text(
                            self.llm, ins, self._polish_cache
                        )
        return bundle

    # ── Card collectors ────────────────────────────────────────────
    def _collect_trend(self, slices: list["TemporalSlice"]) -> list[Insight]:
        out: list[Insight] = []
        try:
            out.append(self._trend_size(slices))
        except Exception as e:
            logger.warning("trend_size failed: %s", e)
        try:
            out.append(self._trend_density(slices))
        except Exception as e:
            logger.warning("trend_density failed: %s", e)
        try:
            ins = self._trend_sentiment(slices)
            if ins is not None:
                out.append(ins)
        except Exception as e:
            logger.warning("trend_sentiment failed: %s", e)
        return out

    def _collect_drift(self, slices: list["TemporalSlice"]) -> list[Insight]:
        out: list[Insight] = []
        for prev, curr in zip(slices, slices[1:]):
            try:
                out.append(self._drift_pair(prev, curr))
            except Exception as e:
                logger.warning("drift pair %s→%s failed: %s",
                               prev.slice_id, curr.slice_id, e)
        return out

    def _collect_comparison(self, slices: list["TemporalSlice"]) -> list[Insight]:
        first, last = slices[0], slices[-1]
        out: list[Insight] = []
        try:
            out.append(self._compare_first_last(first, last))
        except Exception as e:
            logger.warning("compare_first_last failed: %s", e)
        try:
            ins = self._compare_top_concepts(first, last)
            if ins is not None:
                out.append(ins)
        except Exception as e:
            logger.warning("compare_top_concepts failed: %s", e)
        return out

    # ── Trend cards ────────────────────────────────────────────────
    def _trend_size(self, slices: list["TemporalSlice"]) -> Insight:
        first, last = slices[0], slices[-1]
        delta_n = last.node_count - first.node_count
        delta_e = last.edge_count - first.edge_count
        ratio_n = (last.node_count / first.node_count) if first.node_count else 0
        direction = "grew" if delta_n > 0 else ("shrank" if delta_n < 0 else "stayed flat")
        body = (
            f"The network {direction} from {first.label} ({first.node_count} nodes, "
            f"{first.edge_count} edges) to {last.label} "
            f"({last.node_count} nodes, {last.edge_count} edges). That is a "
            f"{ratio_n:.2f}× change in node count and a delta of "
            f"{delta_e:+d} edges over {len(slices)} slices."
        )
        severity = "info" if abs(delta_n) < first.node_count * 0.2 else "medium"
        return Insight(
            id="trend.size",
            category="trend",
            severity=severity,
            title="Network Size Trend",
            body=body,
            metric="node_count,edge_count",
            data={
                "first_nodes": first.node_count,
                "last_nodes": last.node_count,
                "delta_nodes": delta_n,
                "delta_edges": delta_e,
                "n_slices": len(slices),
            },
        )

    def _trend_density(self, slices: list["TemporalSlice"]) -> Insight:
        densities = [
            (sl.label, self.analyser._density(sl.slice_id))
            for sl in slices
        ]
        first_d = densities[0][1]
        last_d = densities[-1][1]
        delta = last_d - first_d
        direction = (
            "denser" if delta > 0 else
            ("sparser" if delta < 0 else "unchanged")
        )
        body = (
            f"Density moved from {first_d:.4f} ({densities[0][0]}) to "
            f"{last_d:.4f} ({densities[-1][0]}) — the network became "
            f"{direction} ({delta:+.4f}). A higher value means concepts "
            "co-occur more tightly within the slice."
        )
        return Insight(
            id="trend.density",
            category="trend",
            severity="info" if abs(delta) < 0.001 else "low",
            title="Density Trend",
            body=body,
            metric="density",
            data={"densities": [{"slice": s, "density": d} for s, d in densities]},
        )

    def _trend_sentiment(self, slices: list["TemporalSlice"]) -> Insight | None:
        per_slice: list[dict[str, Any]] = []
        for sl in slices:
            row = self.analyser.avg_sentiment_for_slice(sl.slice_id)
            per_slice.append({
                "slice": sl.label, "slice_id": sl.slice_id,
                "avg": row["avg"], "n": row["n"],
            })
        scored = [r for r in per_slice if r["avg"] is not None]
        if len(scored) < 2:
            return None  # not enough to talk about a trend

        first_s = scored[0]
        last_s = scored[-1]
        delta = last_s["avg"] - first_s["avg"]
        direction = (
            "more positive" if delta > 0.02 else
            ("more negative" if delta < -0.02 else "essentially flat")
        )
        body = (
            f"Average edge sentiment shifted from {first_s['avg']:+.2f} "
            f"({first_s['slice']}, n={first_s['n']:,}) to "
            f"{last_s['avg']:+.2f} ({last_s['slice']}, n={last_s['n']:,}) — "
            f"an overall delta of {delta:+.2f}, i.e. the discourse became "
            f"{direction}."
        )
        severity = "high" if delta < -0.10 else ("medium" if delta < -0.02 else "info")
        return Insight(
            id="trend.sentiment",
            category="trend",
            severity=severity,
            title="Sentiment Trend",
            body=body,
            metric="avg_sentiment",
            data={"per_slice": per_slice, "delta": delta},
        )

    # ── Drift cards ────────────────────────────────────────────────
    def _drift_pair(
        self, prev: "TemporalSlice", curr: "TemporalSlice",
    ) -> Insight:
        cmp = self.analyser.detailed_compare(prev, curr, top_k=8)
        appeared = cmp["appeared_top"]
        disappeared = cmp["disappeared_top"]
        jacc = cmp["jaccard_nodes"]

        new_bits = ", ".join(f"'{lbl}'" for lbl in appeared[:3]) or "no new top concepts"
        gone_bits = ", ".join(f"'{lbl}'" for lbl in disappeared[:3]) or "no top concept dropped"
        body = (
            f"From {prev.label} to {curr.label} the slice retained "
            f"{cmp['nodes_shared']} of its concepts (jaccard {jacc:.2f}). "
            f"New top concepts: {new_bits}. Notable disappearances: {gone_bits}."
        )
        severity = "medium" if jacc < 0.3 else ("low" if jacc < 0.6 else "info")
        return Insight(
            id=f"drift.{prev.slice_id}__{curr.slice_id}",
            category="drift",
            severity=severity,
            title=f"Drift: {prev.label} → {curr.label}",
            body=body,
            concepts=appeared[:5],
            metric="jaccard_nodes",
            data=cmp,
        )

    # ── Comparison cards ───────────────────────────────────────────
    def _compare_first_last(
        self, first: "TemporalSlice", last: "TemporalSlice",
    ) -> Insight:
        cmp = self.analyser.detailed_compare(first, last, top_k=8)
        body = (
            f"Comparing {first.label} with {last.label}: "
            f"{cmp['nodes_shared']} concepts persist, "
            f"{len(cmp['appeared'])} new ones appear, "
            f"{len(cmp['disappeared'])} disappear, "
            f"jaccard on edges {cmp['jaccard_edges']:.2f}."
        )
        if cmp.get("sentiment_delta") is not None:
            body += (
                f" Average sentiment delta: {cmp['sentiment_delta']:+.2f}."
            )
        return Insight(
            id="comparison.first_last",
            category="comparison",
            severity="info",
            title=f"{first.label} vs {last.label}",
            body=body,
            metric="jaccard_nodes,jaccard_edges",
            data=cmp,
        )

    def _compare_top_concepts(
        self, first: "TemporalSlice", last: "TemporalSlice",
    ) -> Insight | None:
        top_first = self.analyser.top_concepts_for_slice(first.slice_id, k=5)
        top_last = self.analyser.top_concepts_for_slice(last.slice_id, k=5)
        if not top_first or not top_last:
            return None
        first_str = ", ".join(f"'{r['label']}'" for r in top_first)
        last_str = ", ".join(f"'{r['label']}'" for r in top_last)
        body = (
            f"Top concepts shifted from [{first_str}] in {first.label} to "
            f"[{last_str}] in {last.label}. Use this to see what's "
            f"newly dominating the discourse."
        )
        return Insight(
            id="comparison.top_concepts",
            category="comparison",
            severity="info",
            title="Top Concepts: First vs Last",
            body=body,
            concepts=[r["label"] for r in top_last],
            metric="top_concepts",
            data={"first": top_first, "last": top_last},
        )

    # ── Helpers ────────────────────────────────────────────────────
    def _top(self, items: list[Insight]) -> list[Insight]:
        ranked = sorted(
            items,
            key=lambda i: (
                1 if i.available else 0,
                _SEVERITY_RANK.get(i.severity, 0),
                len(i.data) if i.data else 0,
            ),
            reverse=True,
        )
        return ranked[:_TOP_PER_CATEGORY]

    def _not_enough_slices(self, slices: list["TemporalSlice"]) -> Insight:
        return Insight(
            id="trend.not_enough",
            category="trend",
            severity="info",
            title="Not enough slices",
            body=(
                f"Found {len(slices)} slice(s) for source_label="
                f"'{self.source_label}'. Run "
                f"`pipeline.py temporal --source {self.source_label} "
                "--temporal 3` (or higher) to enable temporal insights."
            ),
            available=False,
            unavailable_reason="fewer than two slices stored in Neo4j",
        )
