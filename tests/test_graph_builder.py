"""Tests for the graph builder module."""

import pytest
import networkx as nx

from src.extraction.concept_extractor import Concept
from src.extraction.relationship_extractor import Relationship
from src.network.graph_builder import GraphBuilder


@pytest.fixture
def sample_concepts() -> list[Concept]:
    return [
        Concept(id="climate_change", label="climate change", type="noun_phrase", frequency=10, source_docs={"doc1"}),
        Concept(id="adaptation", label="adaptation", type="keyword", frequency=8, source_docs={"doc1"}),
        Concept(id="resilience", label="resilience", type="keyword", frequency=6, source_docs={"doc1"}),
        Concept(id="australia", label="australia", type="entity", frequency=5, source_docs={"doc1"}),
    ]


@pytest.fixture
def sample_relationships() -> list[Relationship]:
    return [
        Relationship(source_id="climate_change", target_id="adaptation", type="CAUSATION", weight=5, directed=True, source_docs={"doc1"}),
        Relationship(source_id="climate_change", target_id="resilience", type="ASSOCIATION", weight=3, directed=False, source_docs={"doc1"}),
        Relationship(source_id="adaptation", target_id="australia", type="ACTION", weight=2, directed=True, source_docs={"doc1"}),
    ]


class TestGraphBuilder:
    def test_build_creates_graph(self, sample_concepts, sample_relationships):
        builder = GraphBuilder()
        G = builder.build(sample_concepts, sample_relationships)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() > 0

    def test_node_attributes(self, sample_concepts, sample_relationships):
        G = GraphBuilder().build(sample_concepts, sample_relationships)
        node = G.nodes["climate_change"]
        assert node["label"] == "climate change"
        assert node["type"] == "noun_phrase"
        assert node["frequency"] == 10

    def test_edge_attributes(self, sample_concepts, sample_relationships):
        G = GraphBuilder().build(sample_concepts, sample_relationships)
        assert G.has_edge("climate_change", "adaptation")
        edge = G["climate_change"]["adaptation"]
        assert edge["weight"] == 5
        assert "CAUSATION" in edge["types"]

    def test_undirected_edges_for_association(self, sample_concepts, sample_relationships):
        G = GraphBuilder().build(sample_concepts, sample_relationships)
        assert G.has_edge("climate_change", "resilience")
        assert G.has_edge("resilience", "climate_change")

    def test_min_edge_weight_filter(self, sample_concepts, sample_relationships):
        builder = GraphBuilder(min_edge_weight=4)
        G = builder.build(sample_concepts, sample_relationships)
        for u, v, data in G.edges(data=True):
            assert data["weight"] >= 4

    def test_summary(self, sample_concepts, sample_relationships):
        G = GraphBuilder().build(sample_concepts, sample_relationships)
        stats = GraphBuilder.summary(G)
        assert "nodes" in stats
        assert "edges" in stats
        assert "density" in stats
        assert stats["nodes"] == 4

    def test_to_undirected(self, sample_concepts, sample_relationships):
        G = GraphBuilder().build(sample_concepts, sample_relationships)
        U = GraphBuilder.to_undirected(G)
        assert isinstance(U, nx.Graph)
        assert not U.is_directed()

    def test_export_edge_csv(self, sample_concepts, sample_relationships, tmp_path):
        G = GraphBuilder().build(sample_concepts, sample_relationships)
        csv_path = tmp_path / "edges.csv"
        GraphBuilder.export_edge_csv(G, csv_path)
        assert csv_path.exists()
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) > 0
        assert "source" in df.columns

    def test_empty_graph(self):
        G = GraphBuilder().build([], [])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
