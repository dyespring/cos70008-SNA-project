"""Unit tests for Neo4jGraphWriter.

These cover the in-memory aggregation + payload-shaping logic that does NOT
require a live Neo4j instance. Round-trip tests against a real database
live in ``test_neo4j_store.py`` and ``test_neo4j_pipeline.py``.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src.extraction.concept_extractor import Concept
from src.extraction.relationship_extractor import Relationship
from src.network.neo4j_writer import Neo4jGraphWriter


# ── Sample data ────────────────────────────────────────────────────


@pytest.fixture
def sample_concepts() -> list[Concept]:
    return [
        Concept(
            id="climate_change", label="climate change", type="noun_phrase",
            frequency=10, source_origins={"policy"},
        ),
        Concept(
            id="adaptation", label="adaptation", type="keyword",
            frequency=8, source_origins={"policy", "yelp"},
        ),
    ]


@pytest.fixture
def sample_rels() -> list[Relationship]:
    return [
        Relationship(
            source_id="climate_change", target_id="adaptation",
            type="ACTION", weight=2, directed=True,
            verbs=Counter({"drive": 2}),
        ),
        Relationship(
            source_id="climate_change", target_id="adaptation",
            type="CAUSATION", weight=1, directed=True,
            verbs=Counter({"cause": 1, "drive": 1}),
        ),
        Relationship(
            source_id="climate_change", target_id="adaptation",
            type="ASSOCIATION", weight=3, directed=False,
        ),
    ]


# ── _add_edge / aggregation ────────────────────────────────────────


class TestEdgeAggregation:
    def test_add_edge_merges_weights_and_types(self, sample_rels):
        bucket: dict = {}
        for r in sample_rels:
            Neo4jGraphWriter._add_edge(bucket, r.source_id, r.target_id, r)
        key = ("climate_change", "adaptation")
        assert key in bucket
        merged = bucket[key]
        assert merged["weight"] == 6  # 2 + 1 + 3
        assert merged["types"] == {"ACTION", "CAUSATION", "ASSOCIATION"}
        # Verbs combine across rows.
        assert merged["verbs"]["drive"] == 3
        assert merged["verbs"]["cause"] == 1

    def test_add_edge_directed_flag_sticky(self):
        bucket: dict = {}
        # First, an undirected row.
        Neo4jGraphWriter._add_edge(
            bucket, "a", "b",
            Relationship(source_id="a", target_id="b", type="ASSOCIATION",
                         weight=1, directed=False),
        )
        assert bucket[("a", "b")]["directed"] is False
        # Then a directed row on the same pair flips it to True.
        Neo4jGraphWriter._add_edge(
            bucket, "a", "b",
            Relationship(source_id="a", target_id="b", type="ACTION",
                         weight=1, directed=True),
        )
        assert bucket[("a", "b")]["directed"] is True


# ── Payload shape ──────────────────────────────────────────────────


class _StoreStub:
    """No-op stand-in for Neo4jStore so we can construct a writer in unit tests."""

    def ensure_schema(self) -> None:
        pass


class TestPayloads:
    def test_concept_payload_shape(self, sample_concepts):
        w = Neo4jGraphWriter(_StoreStub(), source_label="combined")
        payload = w._concept_payload(sample_concepts[0])
        assert payload["id"] == "climate_change"
        assert payload["label"] == "climate change"
        assert payload["concept_type"] == "noun_phrase"
        assert payload["frequency"] == 10
        assert payload["source_label"] == "combined"
        assert payload["slice_id"] is None

    def test_edge_payload_extracts_top_verb(self):
        w = Neo4jGraphWriter(_StoreStub(), source_label="combined")
        bucket = {
            "weight": 4.0,
            "types": {"ACTION", "CAUSATION"},
            "directed": True,
            "verbs": Counter({"cause": 3, "drive": 1}),
        }
        payload = w._edge_payload("a", "b", bucket)
        assert payload["source"] == "a"
        assert payload["target"] == "b"
        assert payload["weight"] == 4.0
        # types serialised as a sorted comma-joined string.
        assert payload["types"] == "ACTION,CAUSATION"
        assert payload["top_verb"] == "cause"
        assert payload["verb_count"] == 3
        assert "cause" in payload["verb_list"]
        assert "drive" in payload["verb_list"]

    def test_edge_payload_no_verbs_yields_none_top_verb(self):
        w = Neo4jGraphWriter(_StoreStub(), source_label="combined")
        bucket = {
            "weight": 1.0,
            "types": {"ASSOCIATION"},
            "directed": False,
            "verbs": Counter(),
        }
        payload = w._edge_payload("a", "b", bucket)
        assert payload["top_verb"] is None
        assert payload["verb_count"] == 0
        assert payload["verb_list"] == []
        assert payload["directed"] is False

    def test_min_edge_weight_filter(self, sample_concepts):
        """Relationships below min_edge_weight are dropped before aggregation."""
        w = Neo4jGraphWriter(
            _StoreStub(), source_label="combined", min_edge_weight=3
        )
        rels = [
            Relationship(source_id="a", target_id="b", type="ACTION",
                         weight=2, directed=True),
            Relationship(source_id="a", target_id="b", type="ACTION",
                         weight=4, directed=True),
        ]
        # Walk the same logic as write_relationships's pre-filter.
        bucket: dict = {}
        for r in rels:
            if r.weight < w.min_edge_weight:
                continue
            Neo4jGraphWriter._add_edge(bucket, r.source_id, r.target_id, r)
        assert ("a", "b") in bucket
        assert bucket[("a", "b")]["weight"] == 4


# ── Regression: MERGE pattern must not embed null slice_id ─────────


class _CaptureSession:
    """Records every Cypher query passed to ``s.run``."""

    def __init__(self, captured: list[tuple[str, dict]]):
        self.captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params):
        self.captured.append((query, params))
        return _NullResult()


class _NullResult:
    def consume(self):
        return None

    def single(self):
        return None


class _CaptureStore:
    """Store stub whose ``session()`` yields a query-capturing session."""

    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def session(self):
        return _CaptureSession(self.queries)

    def ensure_schema(self) -> None:
        pass


class TestRelationshipMergePattern:
    """Regression tests for the slice_id-null MERGE bug.

    Neo4j 5+ refuses ``MERGE (a)-[:RELATED {slice_id: null}]->(b)``
    with a SemanticError. ``write_relationships`` therefore branches
    its MERGE pattern based on whether ``self.slice_id`` is None.
    """

    def _rels(self) -> list[Relationship]:
        return [
            Relationship(source_id="a", target_id="b", type="ACTION",
                         weight=3, directed=True),
            Relationship(source_id="b", target_id="c", type="ASSOCIATION",
                         weight=4, directed=False),
        ]

    def test_merge_omits_slice_id_when_none(self):
        store = _CaptureStore()
        w = Neo4jGraphWriter(store, source_label="combined", slice_id=None)
        n = w.write_relationships(self._rels())
        assert n > 0
        # Every captured query must avoid the null-slice MERGE form.
        for query, _ in store.queries:
            if "MERGE" in query and "RELATED" in query:
                assert "slice_id: row.slice_id}]->" not in query, (
                    "MERGE pattern must not embed slice_id when slice_id "
                    "is None — Neo4j 5+ rejects null property values inside "
                    "a MERGE pattern."
                )
                assert "MERGE (a)-[r:RELATED {source_label: row.source_label}]->" in query

    def test_merge_includes_slice_id_when_set(self):
        store = _CaptureStore()
        w = Neo4jGraphWriter(
            store, source_label="combined", slice_id="2024-Q1"
        )
        w.write_relationships(self._rels())
        merge_qs = [
            q for q, _ in store.queries
            if "MERGE" in q and "RELATED" in q
        ]
        assert merge_qs
        for q in merge_qs:
            # Temporal runs key the relationship on (source_label, slice_id)
            # to keep parallel slice views distinct.
            assert "slice_id: row.slice_id" in q

    def test_payload_carries_slice_id_in_both_modes(self):
        """The payload always includes slice_id; only the MERGE pattern
        differs. Non-temporal runs persist slice_id via SET, not MERGE."""
        for sid in (None, "2024-Q1"):
            w = Neo4jGraphWriter(
                _StoreStub(), source_label="combined", slice_id=sid
            )
            payload = w._edge_payload(
                "a", "b",
                {"weight": 1.0, "types": {"ACTION"},
                 "directed": True, "verbs": Counter()},
            )
            assert payload["slice_id"] == sid
