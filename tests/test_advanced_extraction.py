"""Tests for the optional substring-pair concept refiner."""

from __future__ import annotations

from src.extraction.concept_extractor import Concept
from src.extraction.advanced import (
    candidate_substring_pairs,
    refine_concepts,
)


def _c(label: str, freq: int = 1, origins=None, docs=None) -> Concept:
    return Concept(
        id=label.replace(" ", "_"),
        label=label,
        type="noun_phrase",
        frequency=freq,
        source_docs=set(docs or {f"doc_{label}"}),
        source_sentences=[],
        source_origins=set(origins or {"yelp"}),
    )


def test_substring_merge_absorbs_short_into_long():
    concepts = [
        _c("climate change", freq=10),
        _c("climate", freq=5),
        _c("policy", freq=3),
    ]
    out = refine_concepts(concepts, merge_substring_pairs=True,
                         min_substring_ratio=1.0)
    labels = {c.label for c in out}
    assert "climate change" in labels
    assert "climate" not in labels
    assert "policy" in labels
    survivor = next(c for c in out if c.label == "climate change")
    # Frequencies are summed.
    assert survivor.frequency == 15


def test_substring_merge_preserves_origin_set_union():
    long_c = _c("net zero target", freq=4, origins={"policy"})
    short_c = _c("net zero", freq=2, origins={"yelp"})
    out = refine_concepts(
        [long_c, short_c], merge_substring_pairs=True, min_substring_ratio=1.0,
    )
    survivor = next(c for c in out if c.label == "net zero target")
    assert survivor.source_origins == {"policy", "yelp"}


def test_no_merge_when_no_token_overlap():
    """Single-token concepts must never absorb each other."""
    concepts = [_c("alpha"), _c("beta"), _c("gamma")]
    out = refine_concepts(concepts, merge_substring_pairs=True)
    assert len(out) == 3


def test_no_merge_when_long_is_single_token():
    """A short single-token concept must not absorb anything."""
    concepts = [_c("food"), _c("foodie", freq=2)]
    out = refine_concepts(concepts, merge_substring_pairs=True)
    # Neither should be touched (no multi-token long concept).
    assert {c.label for c in out} == {"food", "foodie"}


def test_partial_overlap_respects_min_ratio():
    """Token overlap below the ratio gate must not merge."""
    long_c = _c("renewable energy policy", freq=8)
    short_c = _c("nuclear policy", freq=4)
    # Overlap is {"policy"} → ratio 1/2 = 0.5
    out_lax = refine_concepts(
        [long_c, short_c], merge_substring_pairs=True, min_substring_ratio=0.4,
    )
    out_strict = refine_concepts(
        [long_c, short_c], merge_substring_pairs=True, min_substring_ratio=0.7,
    )
    assert len(out_lax) == 1
    assert len(out_strict) == 2


def test_disabled_flag_is_no_op():
    concepts = [_c("climate change", freq=10), _c("climate", freq=5)]
    out = refine_concepts(concepts, merge_substring_pairs=False)
    assert {c.label for c in out} == {"climate change", "climate"}


def test_candidate_pairs_preview_does_not_mutate():
    concepts = [
        _c("climate change", freq=10),
        _c("climate", freq=5),
        _c("net zero target", freq=8),
        _c("net zero", freq=3),
    ]
    pairs = candidate_substring_pairs(concepts, min_ratio=1.0)
    # Both long → short pairs should be reported once each.
    pair_labels = {(p[0], p[1]) for p in pairs}
    assert ("climate change", "climate") in pair_labels
    assert ("net zero target", "net zero") in pair_labels
    # Ratios are 1.0 since short ⊂ long.
    assert all(p[2] == 1.0 for p in pairs)
    # Original list isn't mutated.
    assert all(c.frequency in {10, 5, 8, 3} for c in concepts)


def test_chain_merging_handles_three_levels():
    """A 3-token concept absorbs a 2-token one, which would otherwise
    have absorbed a single-token one."""
    a = _c("renewable energy policy", freq=5)
    b = _c("renewable energy", freq=3)
    c = _c("renewable", freq=2)
    out = refine_concepts(
        [a, b, c], merge_substring_pairs=True, min_substring_ratio=1.0,
    )
    # Only the 3-token long survives; both shorter folded in.
    assert {x.label for x in out} == {"renewable energy policy"}
    survivor = out[0]
    # frequency = 5 + 3 + 2
    assert survivor.frequency == 10


def test_empty_list_returns_empty_list():
    assert refine_concepts([], merge_substring_pairs=True) == []


def test_refine_returns_new_list_object():
    """The function must not return the same list instance (callers
    may mutate)."""
    concepts = [_c("alpha"), _c("beta")]
    out = refine_concepts(concepts, merge_substring_pairs=True)
    assert out is not concepts
