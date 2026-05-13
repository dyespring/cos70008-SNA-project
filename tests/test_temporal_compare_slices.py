"""Regression: compare_slices / _density must survive empty Neo4j slices."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.extensions.temporal import (
    TemporalAnalyser,
    TemporalSlice,
    canonical_concept_overlap_key,
    dashboard_temporal_slices,
    is_yelp_calendar_slice_id,
    slice_display_label,
    yelp_calendar_year_from_slice_id,
)


def _slice(slice_id: str, label: str = "L") -> TemporalSlice:
    return TemporalSlice(
        label=label,
        slice_id=slice_id,
        source_label="combined",
        concepts=[],
        node_count=0,
        edge_count=0,
    )


def test_compare_slices_falls_back_when_single_returns_none():
    """If the driver returns no record, treat counts as zero (defensive)."""

    class _Result:
        def single(self):
            return None

        def data(self):
            return []

    class _Session:
        def run(self, *a, **kw):
            return _Result()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    store = MagicMock()
    store.session.return_value = _Session()
    ta = TemporalAnalyser(store, source_label="combined")

    out = ta.compare_slices(_slice("slice_00_a"), _slice("slice_01_b"))
    assert out["nodes_shared"] == 0
    assert out["jaccard_nodes"] == 0.0
    assert out["jaccard_edges"] == 0.0


def test_is_yelp_calendar_slice_id():
    assert is_yelp_calendar_slice_id("slice_00_2019") is True
    assert is_yelp_calendar_slice_id("slice_12_2024") is True
    assert is_yelp_calendar_slice_id("slice_00_Pages_1_50") is False
    assert is_yelp_calendar_slice_id("") is False
    assert yelp_calendar_year_from_slice_id("slice_03_2018") == 2018
    assert yelp_calendar_year_from_slice_id("slice_00_Pages_1") is None


def test_canonical_concept_overlap_key_unifies_label_and_id():
    assert canonical_concept_overlap_key("Carrot Cake", None) == "carrot_cake"
    assert canonical_concept_overlap_key(None, "carrot_cake") == "carrot_cake"
    assert canonical_concept_overlap_key("self-service", None) == "self_service"
    assert canonical_concept_overlap_key("Carrot Cake", "carrot_cake") == "carrot_cake"
    assert canonical_concept_overlap_key("foo", "bar_x") == "bar_x"


def test_slice_display_label_yelp_year_and_policy_pages():
    assert slice_display_label("slice_05_2009") == "Yelp reviews (2009)"
    assert slice_display_label("slice_00_2018") == "Yelp reviews (2018)"
    assert slice_display_label("slice_00_Pages_12_34") == "Policy text (pp. 12–34)"
    assert slice_display_label("opaque_id") == "opaque_id"


def test_dashboard_temporal_slices_combined_filters_policy_chunks():
    s_y = _slice("slice_02_2020", label="x")
    s_p = _slice("slice_00_Pages_1_50", label="y")
    s_y2 = _slice("slice_01_2018", label="z")
    out = dashboard_temporal_slices("combined", [s_y, s_p, s_y2])
    assert [x.slice_id for x in out] == ["slice_01_2018", "slice_02_2020"]
    out_policy = dashboard_temporal_slices("policy", [s_p, s_y])
    assert len(out_policy) == 2