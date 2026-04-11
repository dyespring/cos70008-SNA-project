"""Temporal comparison: build and compare networks across time slices or document sections."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

from src.extraction.concept_extractor import ConceptExtractor, Concept
from src.extraction.relationship_extractor import RelationshipExtractor
from src.network.graph_builder import GraphBuilder
from src.preprocessing.tokeniser import ProcessedDocument


@dataclass
class TemporalSlice:
    """A network snapshot for one time period or document section."""
    label: str
    graph: nx.DiGraph
    concepts: list[Concept]
    node_count: int
    edge_count: int


class TemporalAnalyser:
    """Compare networks built from different document segments or time periods."""

    def __init__(
        self,
        concept_extractor: ConceptExtractor | None = None,
        rel_extractor: RelationshipExtractor | None = None,
        graph_builder: GraphBuilder | None = None,
    ):
        self.concept_extractor = concept_extractor or ConceptExtractor()
        self.rel_extractor = rel_extractor or RelationshipExtractor()
        self.graph_builder = graph_builder or GraphBuilder()

    def build_slices(
        self,
        slices: list[tuple[str, list[ProcessedDocument]]],
    ) -> list[TemporalSlice]:
        """Build a network for each named slice of documents.

        slices: list of (label, [ProcessedDocument, ...])
        """
        results = []
        for label, docs in slices:
            concepts = self.concept_extractor.extract(docs)
            rels = self.rel_extractor.extract(docs, concepts)
            G = self.graph_builder.build(concepts, rels)
            results.append(TemporalSlice(
                label=label,
                graph=G,
                concepts=concepts,
                node_count=G.number_of_nodes(),
                edge_count=G.number_of_edges(),
            ))
        return results

    @staticmethod
    def compare_slices(a: TemporalSlice, b: TemporalSlice) -> dict:
        """Compare two temporal slices and report differences."""
        nodes_a = set(a.graph.nodes())
        nodes_b = set(b.graph.nodes())
        edges_a = set(a.graph.edges())
        edges_b = set(b.graph.edges())

        return {
            "slice_a": a.label,
            "slice_b": b.label,
            "nodes_added": len(nodes_b - nodes_a),
            "nodes_removed": len(nodes_a - nodes_b),
            "nodes_shared": len(nodes_a & nodes_b),
            "edges_added": len(edges_b - edges_a),
            "edges_removed": len(edges_a - edges_b),
            "edges_shared": len(edges_a & edges_b),
            "jaccard_nodes": len(nodes_a & nodes_b) / len(nodes_a | nodes_b) if (nodes_a | nodes_b) else 0,
            "jaccard_edges": len(edges_a & edges_b) / len(edges_a | edges_b) if (edges_a | edges_b) else 0,
        }

    def comparison_table(self, slices: list[TemporalSlice]) -> pd.DataFrame:
        """Pairwise comparison of all consecutive slice pairs."""
        rows = []
        for i in range(len(slices) - 1):
            rows.append(self.compare_slices(slices[i], slices[i + 1]))
        return pd.DataFrame(rows)

    @staticmethod
    def slice_summary(slices: list[TemporalSlice]) -> pd.DataFrame:
        """Summary table of all slices."""
        rows = []
        for s in slices:
            rows.append({
                "slice": s.label,
                "nodes": s.node_count,
                "edges": s.edge_count,
                "density": nx.density(s.graph),
                "top_concept": s.concepts[0].label if s.concepts else "n/a",
            })
        return pd.DataFrame(rows)
