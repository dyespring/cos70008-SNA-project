"""Integration test: end-to-end Neo4j-only pipeline.

Skips automatically when no Neo4j is reachable (same gating logic as
``test_neo4j_store.py``). Exercises the new modules:

* :class:`Neo4jGraphWriter` — direct Concept[]/Relationship[] -> Neo4j.
* :class:`GdsAnalysisRunner` — server-side centrality + community.
* :class:`Neo4jGraphContext` — Cypher-backed read API used by the chatbot.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

pytest.importorskip("neo4j")

from src.extensions.neo4j_store import Neo4jStore, Neo4jUnavailableError
from src.extraction.concept_extractor import Concept
from src.extraction.relationship_extractor import Relationship
from src.network.neo4j_writer import Neo4jGraphWriter


_SOURCE_LABEL = "pytest_neo4j_pipeline"


@pytest.fixture(scope="module")
def store():
    if not os.getenv("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD not set; skipping Neo4j integration test.")
    try:
        s = Neo4jStore.from_config()
        s.connect()
    except Neo4jUnavailableError as e:
        pytest.skip(f"Neo4j not reachable: {e}")
    yield s
    try:
        s.wipe_source(_SOURCE_LABEL)
    finally:
        s.close()


@pytest.fixture
def sample_data():
    concepts = [
        Concept(
            id="climate_change", label="climate change", type="noun_phrase",
            frequency=10, source_origins={"policy"},
        ),
        Concept(
            id="adaptation", label="adaptation", type="keyword",
            frequency=8, source_origins={"policy", "yelp"},
        ),
        Concept(
            id="resilience", label="resilience", type="keyword",
            frequency=6, source_origins={"policy"},
        ),
    ]
    rels = [
        Relationship(
            source_id="climate_change", target_id="adaptation",
            type="CAUSATION", weight=5, directed=True,
            verbs=Counter({"cause": 4, "drive": 1}),
        ),
        Relationship(
            source_id="adaptation", target_id="resilience",
            type="ASSOCIATION", weight=3, directed=False,
        ),
    ]
    return concepts, rels


class TestNeo4jWriterRoundTrip:
    def test_write_creates_nodes_and_edges(self, store, sample_data):
        concepts, rels = sample_data
        writer = Neo4jGraphWriter(store, source_label=_SOURCE_LABEL)
        counts = writer.write(concepts, rels, reset=True)
        assert counts["nodes"] == 3
        # 1 directed + 1 undirected (which becomes 2 :RELATED rows) = 3.
        assert counts["edges"] == 3

        with store.session() as s:
            n = s.run(
                "MATCH (c:Concept {source_label: $sl}) RETURN count(c) AS n",
                sl=_SOURCE_LABEL,
            ).single()["n"]
            assert n == 3
            e = s.run(
                "MATCH (:Concept {source_label: $sl})"
                "-[r:RELATED {source_label: $sl}]->"
                "(:Concept {source_label: $sl}) RETURN count(r) AS e",
                sl=_SOURCE_LABEL,
            ).single()["e"]
            assert e == 3

    def test_verb_metadata_preserved(self, store, sample_data):
        concepts, rels = sample_data
        Neo4jGraphWriter(store, source_label=_SOURCE_LABEL).write(
            concepts, rels, reset=True
        )
        with store.session() as s:
            rec = s.run(
                "MATCH (:Concept {source_label: $sl, id: 'climate_change'})"
                "-[r:RELATED {source_label: $sl}]->"
                "(:Concept {source_label: $sl, id: 'adaptation'}) "
                "RETURN r.top_verb AS top_verb, r.verb_count AS verb_count",
                sl=_SOURCE_LABEL,
            ).single()
        assert rec is not None
        assert rec["top_verb"] == "cause"
        assert int(rec["verb_count"]) == 4


class TestGdsAnalysisRunner:
    def test_run_pagerank_writes_back(self, store, sample_data):
        concepts, rels = sample_data
        Neo4jGraphWriter(store, source_label=_SOURCE_LABEL).write(
            concepts, rels, reset=True
        )

        from src.network.gds_analyser import (
            GdsAnalysisRunner,
            GdsUnavailableError,
        )

        gds = GdsAnalysisRunner(store, source_label=_SOURCE_LABEL)
        try:
            written = gds.run_pagerank()
        except GdsUnavailableError:
            pytest.skip("Neo4j GDS plugin not installed.")
        assert written == 3

        with store.session() as s:
            rows = s.run(
                "MATCH (c:Concept {source_label: $sl}) "
                "RETURN c.id AS id, c.pagerank AS pagerank",
                sl=_SOURCE_LABEL,
            ).data()
        prs = {r["id"]: r["pagerank"] for r in rows}
        assert all(v is not None and v > 0 for v in prs.values())

    def test_centrality_dataframe(self, store, sample_data):
        concepts, rels = sample_data
        Neo4jGraphWriter(store, source_label=_SOURCE_LABEL).write(
            concepts, rels, reset=True
        )

        from src.network.gds_analyser import (
            GdsAnalysisRunner,
            GdsUnavailableError,
        )

        gds = GdsAnalysisRunner(store, source_label=_SOURCE_LABEL)
        try:
            gds.run_pagerank()
            gds.run_louvain()
        except GdsUnavailableError:
            pytest.skip("Neo4j GDS plugin not installed.")
        df = gds.all_centralities()
        assert not df.empty
        assert "pagerank" in df.columns
        assert "community" in df.columns
        assert set(df["node_id"]) == {"climate_change", "adaptation", "resilience"}


class TestNeo4jGraphContext:
    def test_node_meta_dict(self, store, sample_data):
        concepts, rels = sample_data
        Neo4jGraphWriter(store, source_label=_SOURCE_LABEL).write(
            concepts, rels, reset=True
        )
        from src.extensions.neo4j_graph_context import Neo4jGraphContext

        gc = Neo4jGraphContext(store, source_label=_SOURCE_LABEL)
        assert set(gc.all_labels()) == {"climate change", "adaptation", "resilience"}
        assert gc.label_for("climate_change") == "climate change"
        assert gc.label_for("missing_id") == "missing_id"
        assert "climate_change" in gc.node_meta
        assert gc.node_meta["climate_change"]["concept_type"] == "noun_phrase"
