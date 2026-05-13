"""Offline tests for TemporalInsightEngine.

We don't rely on a live Neo4j here — a fake :class:`TemporalAnalyser`
replaces the Cypher I/O so the engine logic (severity, top-3 cap,
not-enough-slices fallback, drift cards per pair) is tested in
isolation.
"""

from __future__ import annotations

import pytest

from src.extensions.temporal import TemporalSlice
from src.extensions.temporal_insights import (
    TemporalInsightBundle,
    TemporalInsightEngine,
)


# ── Fakes ──────────────────────────────────────────────────────────


class _FakeAnalyser:
    """Stand-in for :class:`TemporalAnalyser` with deterministic data."""

    def __init__(self, slices, density_map, sentiment_map, top_concepts_map,
                 detailed_map):
        self._slices = slices
        self._density_map = density_map
        self._sentiment_map = sentiment_map
        self._top_concepts_map = top_concepts_map
        self._detailed_map = detailed_map

    def existing_slices(self):
        return list(self._slices)

    def _density(self, slice_id: str) -> float:
        return float(self._density_map.get(slice_id, 0.0))

    def avg_sentiment_for_slice(self, slice_id: str) -> dict:
        return self._sentiment_map.get(slice_id, {"avg": None, "n": 0})

    def top_concepts_for_slice(
        self, slice_id: str, k: int = 5, metric: str = "frequency",
    ) -> list[dict]:
        return self._top_concepts_map.get(slice_id, [])[:k]

    def detailed_compare(self, a, b, top_k: int = 8) -> dict:
        return self._detailed_map.get((a.slice_id, b.slice_id), {})


def _slice(slice_id: str, label: str | None = None,
           nodes: int = 100, edges: int = 200) -> TemporalSlice:
    return TemporalSlice(
        label=label or slice_id,
        slice_id=slice_id,
        source_label="combined",
        concepts=[],
        node_count=nodes,
        edge_count=edges,
    )


def _engine_with(analyser):
    """Build engine without touching a real Neo4j connection."""
    eng = TemporalInsightEngine.__new__(TemporalInsightEngine)
    eng.store = None
    eng.source_label = "combined"
    eng.analyser = analyser
    eng.llm = None
    eng._polish_cache = {}
    return eng


# ── Tests ──────────────────────────────────────────────────────────


def test_not_enough_slices_emits_unavailable_card():
    eng = _engine_with(_FakeAnalyser([], {}, {}, {}, {}))
    bundle = eng.all_insights()
    assert isinstance(bundle, TemporalInsightBundle)
    assert len(bundle.trend) == 1
    assert bundle.trend[0].available is False
    assert "fewer than two" in bundle.trend[0].unavailable_reason


def test_two_slices_produces_trend_drift_comparison():
    s1, s2 = _slice("s1", nodes=100, edges=300), _slice("s2", nodes=160, edges=480)
    fake = _FakeAnalyser(
        slices=[s1, s2],
        density_map={"s1": 0.0030, "s2": 0.0042},
        sentiment_map={
            "s1": {"avg": 0.20, "n": 1000},
            "s2": {"avg": 0.30, "n": 2000},
        },
        top_concepts_map={
            "s1": [{"label": "service", "frequency": 50, "metric_value": 0.5}],
            "s2": [{"label": "climate", "frequency": 80, "metric_value": 0.6}],
        },
        detailed_map={
            ("s1", "s2"): {
                "slice_a": "s1", "slice_b": "s2",
                "nodes_added": 20, "nodes_removed": 5, "nodes_shared": 80,
                "edges_added": 30, "edges_removed": 10, "edges_shared": 200,
                "jaccard_nodes": 0.7, "jaccard_edges": 0.65,
                "appeared": ["climate"], "disappeared": [],
                "appeared_top": ["climate"], "disappeared_top": [],
                "sentiment_a": 0.20, "sentiment_b": 0.30,
                "sentiment_delta": 0.10,
            },
        },
    )
    eng = _engine_with(fake)
    bundle = eng.all_insights()

    assert len(bundle.trend) >= 2
    # Each category capped at 3
    assert len(bundle.trend) <= 3
    assert len(bundle.drift) <= 3
    assert len(bundle.comparison) <= 3
    # Drift card exists for the only pair
    assert any("drift." in i.id for i in bundle.drift)
    # Comparison: first vs last must be present
    assert any(i.id == "comparison.first_last" for i in bundle.comparison)


def test_top_cap_is_three_per_category():
    """Even with many drift pairs we never exceed three drift cards."""
    slices = [_slice(f"s{i}") for i in range(6)]
    detailed = {}
    for prev, curr in zip(slices, slices[1:]):
        detailed[(prev.slice_id, curr.slice_id)] = {
            "slice_a": prev.label, "slice_b": curr.label,
            "nodes_added": 1, "nodes_removed": 1, "nodes_shared": 50,
            "edges_added": 1, "edges_removed": 1, "edges_shared": 100,
            "jaccard_nodes": 0.2,  # low jaccard → medium severity
            "jaccard_edges": 0.2,
            "appeared": [], "disappeared": [],
            "appeared_top": [], "disappeared_top": [],
            "sentiment_a": None, "sentiment_b": None,
            "sentiment_delta": None,
        }
    fake = _FakeAnalyser(
        slices=slices,
        density_map={s.slice_id: 0.001 for s in slices},
        sentiment_map={s.slice_id: {"avg": None, "n": 0} for s in slices},
        top_concepts_map={s.slice_id: [] for s in slices},
        detailed_map=detailed,
    )
    eng = _engine_with(fake)
    bundle = eng.all_insights()

    assert len(bundle.drift) == 3
    # The three with the lowest jaccard / highest severity should win;
    # since they're all the same severity, we just verify count + ordering
    # is stable (all have severity "medium" given jaccard < 0.3).
    assert all(i.severity == "medium" for i in bundle.drift)


def test_sentiment_trend_is_skipped_when_no_data():
    s1, s2 = _slice("a"), _slice("b")
    fake = _FakeAnalyser(
        slices=[s1, s2],
        density_map={"a": 0.001, "b": 0.001},
        sentiment_map={
            "a": {"avg": None, "n": 0},
            "b": {"avg": None, "n": 0},
        },
        top_concepts_map={"a": [], "b": []},
        detailed_map={
            ("a", "b"): {
                "slice_a": "a", "slice_b": "b",
                "nodes_shared": 0, "appeared_top": [], "disappeared_top": [],
                "jaccard_nodes": 1.0, "jaccard_edges": 1.0,
                "appeared": [], "disappeared": [],
                "sentiment_a": None, "sentiment_b": None,
                "sentiment_delta": None,
            },
        },
    )
    eng = _engine_with(fake)
    bundle = eng.all_insights()
    assert all(i.id != "trend.sentiment" for i in bundle.trend)


def test_negative_sentiment_drift_marked_high_severity():
    s1, s2 = _slice("year_2022"), _slice("year_2024")
    fake = _FakeAnalyser(
        slices=[s1, s2],
        density_map={"year_2022": 0.002, "year_2024": 0.002},
        sentiment_map={
            "year_2022": {"avg": 0.30, "n": 1000},
            "year_2024": {"avg": 0.10, "n": 1500},
        },
        top_concepts_map={"year_2022": [], "year_2024": []},
        detailed_map={
            ("year_2022", "year_2024"): {
                "slice_a": "year_2022", "slice_b": "year_2024",
                "nodes_shared": 100, "appeared_top": [], "disappeared_top": [],
                "jaccard_nodes": 0.5, "jaccard_edges": 0.5,
                "appeared": [], "disappeared": [],
                "sentiment_a": 0.30, "sentiment_b": 0.10,
                "sentiment_delta": -0.20,
            },
        },
    )
    eng = _engine_with(fake)
    bundle = eng.all_insights()
    sent_card = next(
        (i for i in bundle.trend if i.id == "trend.sentiment"), None
    )
    assert sent_card is not None
    assert sent_card.severity == "high"
    assert "more negative" in sent_card.body
