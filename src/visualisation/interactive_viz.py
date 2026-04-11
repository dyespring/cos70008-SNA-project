"""Interactive pyvis network visualisations."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from pyvis.network import Network

from src.config import TOP_N_DISPLAY

COMMUNITY_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]


class InteractiveVisualiser:
    """Generate interactive HTML network visualisations using pyvis."""

    def __init__(self, output_dir: str | Path = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    SOURCE_COLORS = {"policy": "#4363d8", "yelp": "#e6194b", "both": "#3cb44b", "unknown": "#808080"}

    def create_interactive_network(
        self,
        G: nx.DiGraph,
        partition: dict[str, int] | None = None,
        top_n: int = TOP_N_DISPLAY,
        title: str = "Conceptual Network Explorer",
        filename: str = "interactive_network.html",
        height: str = "800px",
        width: str = "100%",
        colour_by_source: bool = False,
        colour_edges_by_sentiment: bool = False,
    ) -> Path:
        """Build a pyvis interactive network and save as HTML.

        When colour_edges_by_sentiment is True, edges with a ``sentiment`` attribute are
        coloured green (positive), red (negative), or grey (neutral) and width scales with |sentiment|.
        """
        if top_n and G.number_of_nodes() > top_n:
            pr = nx.pagerank(G, weight="weight")
            top_nodes = sorted(pr, key=pr.get, reverse=True)[:top_n]
            G = G.subgraph(top_nodes).copy()

        net = Network(height=height, width=width, directed=True,
                      notebook=False, bgcolor="#ffffff", font_color="#333333")
        net.heading = title

        pr = nx.pagerank(G, weight="weight")
        max_pr = max(pr.values()) if pr else 1

        for node in G.nodes():
            attrs = G.nodes[node]
            label = attrs.get("label", node)
            size = 10 + 40 * (pr.get(node, 0) / max_pr)
            community = partition.get(node, 0) if partition else 0
            source_type = attrs.get("source_type", "unknown")

            if colour_by_source:
                color = self.SOURCE_COLORS.get(source_type, "#808080")
            else:
                color = COMMUNITY_COLORS[community % len(COMMUNITY_COLORS)]

            tooltip = (
                f"<b>{label}</b><br>"
                f"Type: {attrs.get('type', 'n/a')}<br>"
                f"Source: {source_type}<br>"
                f"Frequency: {attrs.get('frequency', 0)}<br>"
                f"PageRank: {pr.get(node, 0):.4f}<br>"
                f"Community: {community}"
            )

            net.add_node(node, label=label[:30], title=tooltip,
                         size=size, color=color, font={"size": 10})

        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1)
            types = data.get("types", set())
            type_str = ", ".join(types) if isinstance(types, set) else str(types)
            edge_tooltip = f"Weight: {weight}<br>Types: {type_str}"
            ewidth = 0.5 + 2 * min(weight / 5, 3)
            ecolor = None

            if colour_edges_by_sentiment and "sentiment" in data:
                s = float(data["sentiment"])
                sl = data.get("sentiment_label", "neutral")
                edge_tooltip += f"<br>Sentiment: {s:.3f} ({sl})"
                if s > 0.05:
                    ecolor = "#2ca02c"
                elif s < -0.05:
                    ecolor = "#d62728"
                else:
                    ecolor = "#a8a8a8"
                ewidth = max(0.5, 0.5 + 3.0 * min(abs(s), 1.0))

            kwargs = dict(value=weight, title=edge_tooltip, width=ewidth)
            if ecolor is not None:
                kwargs["color"] = ecolor
            net.add_edge(u, v, **kwargs)

        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "zoomView": true
          }
        }
        """)

        output_path = self.output_dir / filename
        net.save_graph(str(output_path))
        return output_path
