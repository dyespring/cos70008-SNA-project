"""Optional, dependency-free post-processing for extracted concepts.

This module sits *behind a flag* (``pipeline.py etl --advanced-extraction``)
so it never affects the default pipeline. Its job is to reduce noise that
slips past the per-extractor filters by acting on the concept list as a
whole — currently:

* **Substring-pair merge** — when one concept's lemma is a strict
  superstring of another (e.g. ``"climate"`` and ``"climate change"``
  share the token ``"climate"``), and they share enough tokens, fold the
  shorter into the longer to avoid the pair forming a fake hub→spoke
  edge in the network.

The function is pure (no I/O), so it's trivial to test in isolation.
"""

from __future__ import annotations

from typing import Iterable

from src.extraction.concept_extractor import Concept


def refine_concepts(
    concepts: list[Concept],
    *,
    merge_substring_pairs: bool = True,
    min_substring_ratio: float = 0.6,
) -> list[Concept]:
    """Apply optional advanced refinements to a concept list.

    Parameters
    ----------
    concepts:
        Output of :class:`ConceptExtractor.extract`.
    merge_substring_pairs:
        Enable the substring-pair merger (currently the only refinement).
    min_substring_ratio:
        Token-overlap ratio gate for substring-pair merges. Computed as
        ``len(common_tokens) / len(short_tokens)``. ``1.0`` means *every*
        token of the short concept must appear in the long one;
        ``0.6`` allows a slightly looser match.

    Returns
    -------
    A new list of concepts. Frequencies, source docs, source origins,
    and source sentences of merged concepts are summed into the survivor.
    """
    if not concepts:
        return list(concepts)

    out = list(concepts)
    if merge_substring_pairs:
        out = _merge_substring_pairs(out, min_ratio=min_substring_ratio)
    return out


# ── Substring merging ────────────────────────────────────────────


def _merge_substring_pairs(
    concepts: list[Concept],
    *,
    min_ratio: float,
) -> list[Concept]:
    """Fold short concepts into longer ones when they share enough tokens.

    Algorithm (deterministic, single pass):

    1. Sort concepts by ``label`` length descending — longer phrases get
       to absorb the shorter ones, never the reverse.
    2. For each "long" concept, look at every shorter, still-active
       concept and check if its lemma tokens form a sufficient subset
       of the long one's lemma tokens. If yes, merge.
    3. Multi-token concepts only — single-token concepts don't merge
       into each other (they're often distinct, e.g. ``"sentiment"``
       vs ``"sentence"``).

    The survivor inherits all per-doc / per-sentence / origin tracking
    from its absorbed shorter twins.
    """
    if not concepts:
        return concepts

    # Index by id for ``O(1)`` survivor lookup.
    by_id: dict[str, Concept] = {c.id: c for c in concepts}

    # Sort by label-token count desc, then label length desc — most
    # specific phrase first.
    ordered = sorted(
        concepts,
        key=lambda c: (len(c.label.split()), len(c.label)),
        reverse=True,
    )

    absorbed: set[str] = set()

    for long_c in ordered:
        if long_c.id in absorbed:
            continue
        long_tokens = set(long_c.label.split())
        if len(long_tokens) < 2:
            # Only multi-token concepts can absorb shorter ones.
            continue
        for short_c in ordered:
            if short_c.id == long_c.id or short_c.id in absorbed:
                continue
            short_tokens = set(short_c.label.split())
            if len(short_tokens) >= len(long_tokens):
                continue
            if not short_tokens.issubset(long_tokens):
                # Strict substring: every short-token must appear in long.
                # ``min_ratio`` is the *fraction* of short_tokens we need,
                # so 1.0 == strict subset, anything less is partial.
                ratio = (
                    len(short_tokens & long_tokens) / len(short_tokens)
                )
                if ratio < min_ratio:
                    continue
            # Merge: the long concept absorbs the short one.
            _absorb(by_id[long_c.id], by_id[short_c.id])
            absorbed.add(short_c.id)

    return [c for c in concepts if c.id not in absorbed]


def _absorb(long_c: Concept, short_c: Concept) -> None:
    """Pour ``short_c``'s evidence into ``long_c`` in place."""
    long_c.frequency += short_c.frequency
    long_c.source_docs |= short_c.source_docs
    long_c.source_sentences = list(long_c.source_sentences) + list(
        short_c.source_sentences
    )
    long_c.source_origins |= short_c.source_origins


# ── Convenience helpers ──────────────────────────────────────────


def candidate_substring_pairs(
    concepts: Iterable[Concept],
    *,
    min_ratio: float = 0.6,
) -> list[tuple[str, str, float]]:
    """Return ``(long_label, short_label, ratio)`` pairs that *would* merge.

    Useful for previewing the impact of ``refine_concepts`` without
    actually applying it (e.g. in a dashboard preview).
    """
    pairs: list[tuple[str, str, float]] = []
    items = list(concepts)
    for i, long_c in enumerate(items):
        long_tokens = set(long_c.label.split())
        if len(long_tokens) < 2:
            continue
        for j, short_c in enumerate(items):
            if i == j:
                continue
            short_tokens = set(short_c.label.split())
            if len(short_tokens) >= len(long_tokens):
                continue
            if not short_tokens & long_tokens:
                continue
            ratio = len(short_tokens & long_tokens) / len(short_tokens)
            if ratio >= min_ratio:
                pairs.append((long_c.label, short_c.label, ratio))
    return pairs
