"""Apply a user-defined concept dictionary (allow-list and/or alias merging)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.extraction.concept_extractor import Concept


def _normalise(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _make_id(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_")


def load_dictionary_yaml(path: str | Path) -> dict[str, Any]:
    """Load dictionary config from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Concept dictionary not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_label_map(data: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Return (mode, normalised_label -> normalised_canonical)."""
    mode = (data.get("mode") or "allowlist").strip().lower()
    entries = data.get("entries") or []
    label_to_canonical: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_c = entry.get("canonical")
        if not raw_c:
            continue
        canon = _normalise(str(raw_c))
        label_to_canonical[canon] = canon
        for a in entry.get("aliases") or []:
            label_to_canonical[_normalise(str(a))] = canon
    return mode, label_to_canonical


def apply_concept_dictionary(
    concepts: list[Concept],
    dictionary_path: str | Path | None,
) -> list[Concept]:
    """Filter and/or merge concepts using config/concept_dictionary.yaml-style file.

    Returns a new list of Concept instances. If dictionary_path is None, returns concepts unchanged.
    """
    if not dictionary_path:
        return concepts

    data = load_dictionary_yaml(dictionary_path)
    mode, label_to_canonical = build_label_map(data)
    if not label_to_canonical:
        return concepts

    # Map each concept to a canonical key (or None if dropped)
    buckets: dict[str, list[Concept]] = defaultdict(list)
    for c in concepts:
        n = _normalise(c.label)
        if n in label_to_canonical:
            canon = label_to_canonical[n]
            buckets[canon].append(c)
        elif mode == "merge_only":
            buckets[n].append(c)
        # allowlist: drop unknown

    merged: list[Concept] = []
    for canon, group in buckets.items():
        if not group:
            continue
        freq = sum(x.frequency for x in group)
        types = [x.type for x in group]
        ctype = "entity" if "entity" in types else ("noun_phrase" if "noun_phrase" in types else group[0].type)
        source_docs: set[str] = set()
        source_sents: list[int] = []
        source_origins: set[str] = set()
        for x in group:
            source_docs |= x.source_docs
            source_sents.extend(x.source_sentences)
            source_origins |= x.source_origins
        merged.append(
            Concept(
                id=_make_id(canon),
                label=canon,
                type=ctype,
                frequency=freq,
                source_docs=source_docs,
                source_sentences=source_sents,
                source_origins=source_origins,
            )
        )
    merged.sort(key=lambda x: x.frequency, reverse=True)
    return merged
