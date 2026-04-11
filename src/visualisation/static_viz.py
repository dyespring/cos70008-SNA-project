"""Static matplotlib/seaborn visualisations for conceptual networks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import TOP_N_DISPLAY


class StaticVisualiser:
    """Generate publication-quality static plots."""

    def __init__(self, output_dir: str | Path = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="husl")

    def plot_network(
        self,
        G: nx.DiGraph,
        partition: dict[str, int] | None = None,
        centrality_metric: str = "pagerank",
        top_n: int = TOP_N_DISPLAY,
        title: str = "Conceptual Network",
        filename: str = "network_graph.png",
    ) -> None:
        """Draw the network with nodes sized by centrality and coloured by community."""
        if top_n and G.number_of_nodes() > top_n:
            metric = nx.pagerank(G, weight="weight") if centrality_metric == "pagerank" else nx.degree_centrality(G)
            top_nodes = sorted(metric, key=metric.get, reverse=True)[:top_n]
            G = G.subgraph(top_nodes).copy()

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42, weight="weight")

        centrality = nx.pagerank(G, weight="weight") if centrality_metric == "pagerank" else nx.degree_centrality(G)
        node_sizes = [3000 * centrality.get(n, 0.01) + 100 for n in G.nodes()]

        if partition:
            node_colors = [partition.get(n, 0) for n in G.nodes()]
            n_communities = max(node_colors) + 1 if node_colors else 1
            cmap = cm.get_cmap("tab10", n_communities)
        else:
            node_colors = list(centrality.values())
            cmap = "YlOrRd"

        weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [0.5 + 2.0 * (w / max_w) for w in weights]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=edge_widths, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                node_color=node_colors, cmap=cmap, alpha=0.8)

        labels = {n: G.nodes[n].get("label", n)[:20] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight="bold")

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_top_concepts(
        self,
        centrality_df: pd.DataFrame,
        metric: str = "pagerank",
        top_n: int = 20,
        title: str = "Top Concepts by PageRank",
        filename: str = "top_concepts.png",
    ) -> None:
        """Bar chart of top-N concepts by a centrality measure."""
        df = centrality_df.nlargest(top_n, metric)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(df)), df[metric].values, color=sns.color_palette("viridis", len(df)))
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["label"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_community_distribution(
        self,
        partition: dict[str, int],
        title: str = "Community Size Distribution",
        filename: str = "community_distribution.png",
    ) -> None:
        """Bar chart showing the size of each detected community."""
        from collections import Counter
        counts = Counter(partition.values())
        comm_ids = sorted(counts.keys())
        sizes = [counts[c] for c in comm_ids]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([str(c) for c in comm_ids], sizes, color=sns.color_palette("tab10", len(comm_ids)))
        ax.set_xlabel("Community ID", fontsize=12)
        ax.set_ylabel("Number of Concepts", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_network_by_source(
        self,
        G: nx.DiGraph,
        top_n: int = TOP_N_DISPLAY,
        title: str = "Network by Source Type",
        filename: str = "network_by_source.png",
    ) -> None:
        """Draw the network with nodes coloured by source_type (policy/yelp/both)."""
        if top_n and G.number_of_nodes() > top_n:
            pr = nx.pagerank(G, weight="weight")
            top_nodes = sorted(pr, key=pr.get, reverse=True)[:top_n]
            G = G.subgraph(top_nodes).copy()

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42, weight="weight")

        pr = nx.pagerank(G, weight="weight")
        node_sizes = [3000 * pr.get(n, 0.01) + 100 for n in G.nodes()]

        source_color_map = {"policy": "#4363d8", "yelp": "#e6194b", "both": "#3cb44b", "unknown": "#808080"}
        node_colors = [source_color_map.get(G.nodes[n].get("source_type", "unknown"), "#808080") for n in G.nodes()]

        weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [0.5 + 2.0 * (w / max_w) for w in weights]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=edge_widths, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                node_color=node_colors, alpha=0.8)

        labels = {n: G.nodes[n].get("label", n)[:20] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight="bold")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#4363d8", label="Policy"),
            Patch(facecolor="#e6194b", label="Yelp"),
            Patch(facecolor="#3cb44b", label="Both"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_source_overlap(
        self,
        G: nx.DiGraph,
        title: str = "Concept Source Distribution",
        filename: str = "source_overlap.png",
    ) -> None:
        """Bar chart showing concept counts per source_type."""
        from collections import Counter
        src_counts = Counter(data.get("source_type", "unknown") for _, data in G.nodes(data=True))
        categories = ["policy", "yelp", "both"]
        counts = [src_counts.get(c, 0) for c in categories]
        colors = ["#4363d8", "#e6194b", "#3cb44b"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_xlabel("Source Type", fontsize=12)
        ax.set_ylabel("Number of Concepts", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        for i, v in enumerate(counts):
            ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_cooccurrence_heatmap(
        self,
        G: nx.DiGraph,
        top_n: int = 20,
        title: str = "Concept Co-occurrence Heatmap",
        filename: str = "cooccurrence_heatmap.png",
    ) -> None:
        """Heatmap of co-occurrence weights between top concepts."""
        pr = nx.pagerank(G, weight="weight")
        top_nodes = sorted(pr, key=pr.get, reverse=True)[:top_n]
        sub = G.subgraph(top_nodes)
        labels = [G.nodes[n].get("label", n)[:25] for n in top_nodes]

        adj = nx.to_numpy_array(sub, nodelist=top_nodes, weight="weight")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(adj, xticklabels=labels, yticklabels=labels,
                    cmap="YlOrRd", ax=ax, square=True, linewidths=0.5)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
