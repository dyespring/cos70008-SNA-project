"""Social network analysis on conceptual networks."""

from __future__ import annotations

import networkx as nx
import pandas as pd

try:
    from community import community_louvain
except ImportError:
    community_louvain = None


class GraphAnalyser:
    """Compute centrality, community structure, brokerage and paths."""

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.G_undirected = G.to_undirected()

    # ── Centrality ─────────────────────────────────────────────────

    def degree_centrality(self) -> dict[str, float]:
        return nx.degree_centrality(self.G)

    def betweenness_centrality(self) -> dict[str, float]:
        return nx.betweenness_centrality(self.G, weight="weight")

    def closeness_centrality(self) -> dict[str, float]:
        return nx.closeness_centrality(self.G)

    def eigenvector_centrality(self, max_iter: int = 500) -> dict[str, float]:
        try:
            return nx.eigenvector_centrality(self.G, max_iter=max_iter, weight="weight")
        except nx.PowerIterationFailedConvergence:
            return {}

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        return nx.pagerank(self.G, alpha=alpha, weight="weight")

    def all_centralities(self) -> pd.DataFrame:
        """Compute all centrality measures and return as a DataFrame.

        Each measure is computed exactly once and cached for the loop.
        For large graphs (>5000 edges) betweenness uses a k-sample
        approximation for speed.
        """
        nodes = list(self.G.nodes())
        labels = {n: self.G.nodes[n].get("label", n) for n in nodes}

        deg = self.degree_centrality()
        pr = self.pagerank()
        eig = self.eigenvector_centrality()

        n_edges = self.G.number_of_edges()
        if n_edges > 5000:
            k = min(200, self.G.number_of_nodes())
            btwn = nx.betweenness_centrality(self.G, weight="weight", k=k)
        else:
            btwn = self.betweenness_centrality()

        close = self.closeness_centrality()

        data = {
            "node_id": nodes,
            "label": [labels[n] for n in nodes],
            "degree": [deg.get(n, 0) for n in nodes],
            "betweenness": [btwn.get(n, 0) for n in nodes],
            "closeness": [close.get(n, 0) for n in nodes],
            "eigenvector": [eig.get(n, 0) for n in nodes],
            "pagerank": [pr.get(n, 0) for n in nodes],
        }
        df = pd.DataFrame(data)
        df = df.sort_values("pagerank", ascending=False).reset_index(drop=True)
        return df

    # ── Community Detection ────────────────────────────────────────

    def detect_communities_louvain(self) -> dict[str, int]:
        """Louvain community detection on the undirected projection."""
        if community_louvain is None:
            raise ImportError("Install python-louvain: pip install python-louvain")
        partition = community_louvain.best_partition(self.G_undirected, weight="weight")
        return partition

    def detect_communities_label_propagation(self) -> dict[str, int]:
        """Label propagation community detection."""
        communities = nx.community.label_propagation_communities(self.G_undirected)
        partition = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                partition[node] = idx
        return partition

    def community_summary(self, partition: dict[str, int]) -> pd.DataFrame:
        """Summarise communities: size, key members."""
        from collections import defaultdict
        comm_nodes: dict[int, list] = defaultdict(list)
        for node, comm_id in partition.items():
            label = self.G.nodes[node].get("label", node)
            freq = self.G.nodes[node].get("frequency", 0)
            comm_nodes[comm_id].append((label, freq))

        rows = []
        for comm_id, members in sorted(comm_nodes.items()):
            members.sort(key=lambda x: x[1], reverse=True)
            top_members = [m[0] for m in members[:5]]
            rows.append({
                "community": comm_id,
                "size": len(members),
                "top_concepts": ", ".join(top_members),
            })
        return pd.DataFrame(rows)

    # ── Brokerage ──────────────────────────────────────────────────

    def find_brokers(self, partition: dict[str, int], top_n: int = 10) -> pd.DataFrame:
        """Identify nodes that bridge different communities (high betweenness + cross-community edges)."""
        betweenness = self.betweenness_centrality()
        rows = []
        for node in self.G.nodes():
            node_comm = partition.get(node, -1)
            neighbors = set(self.G_undirected.neighbors(node))
            cross_comm = sum(1 for nb in neighbors if partition.get(nb, -1) != node_comm)
            rows.append({
                "node_id": node,
                "label": self.G.nodes[node].get("label", node),
                "community": node_comm,
                "betweenness": betweenness.get(node, 0),
                "cross_community_edges": cross_comm,
                "total_edges": len(neighbors),
            })
        df = pd.DataFrame(rows)
        df["broker_score"] = df["betweenness"] * df["cross_community_edges"]
        return df.sort_values("broker_score", ascending=False).head(top_n).reset_index(drop=True)

    # ── Path Analysis ──────────────────────────────────────────────

    def shortest_path(self, source_label: str, target_label: str) -> list[str] | None:
        """Find shortest path between two concepts by label."""
        label_to_id = {self.G.nodes[n].get("label", n): n for n in self.G.nodes()}
        src = label_to_id.get(source_label)
        tgt = label_to_id.get(target_label)
        if src is None or tgt is None:
            return None
        try:
            path_ids = nx.shortest_path(self.G_undirected, src, tgt)
            return [self.G.nodes[n].get("label", n) for n in path_ids]
        except nx.NetworkXNoPath:
            return None

    # ── Summary Statistics ─────────────────────────────────────────

    def summary_stats(self) -> dict:
        """Compute high-level graph statistics."""
        stats = {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "density": nx.density(self.G),
            "weakly_connected_components": nx.number_weakly_connected_components(self.G),
            "avg_clustering": nx.average_clustering(self.G_undirected),
        }
        if nx.is_weakly_connected(self.G):
            stats["diameter"] = nx.diameter(self.G_undirected)
        return stats

    # ── Cross-Source Analysis ─────────────────────────────────────

    def source_distribution(self) -> pd.DataFrame:
        """Count nodes and edges by source_type (policy / yelp / both)."""
        from collections import Counter
        src_counts = Counter()
        for _, data in self.G.nodes(data=True):
            src_counts[data.get("source_type", "unknown")] += 1
        rows = [{"source_type": k, "node_count": v} for k, v in sorted(src_counts.items())]
        return pd.DataFrame(rows)

    def cross_source_edges(self) -> pd.DataFrame:
        """Find edges that connect nodes from different source_types (bridging edges)."""
        rows = []
        for u, v, data in self.G.edges(data=True):
            u_src = self.G.nodes[u].get("source_type", "unknown")
            v_src = self.G.nodes[v].get("source_type", "unknown")
            if u_src != v_src:
                rows.append({
                    "source_node": self.G.nodes[u].get("label", u),
                    "target_node": self.G.nodes[v].get("label", v),
                    "source_type_from": u_src,
                    "source_type_to": v_src,
                    "weight": data.get("weight", 1),
                    "types": ",".join(data.get("types", set())) if isinstance(data.get("types"), set) else str(data.get("types", "")),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        return df

    def bridging_concepts(self, top_n: int = 20) -> pd.DataFrame:
        """Identify concepts that connect policy and yelp domains.

        A bridging concept has source_type='both' or has neighbours in
        a different source_type. Ranked by a bridge_score that combines
        betweenness centrality and cross-source neighbour ratio.
        """
        betweenness = self.betweenness_centrality()
        rows = []
        for node in self.G.nodes():
            node_src = self.G.nodes[node].get("source_type", "unknown")
            neighbors = set(self.G_undirected.neighbors(node))
            cross_source_nb = sum(
                1 for nb in neighbors
                if self.G.nodes[nb].get("source_type", "unknown") != node_src
            )
            total_nb = len(neighbors)
            cross_ratio = cross_source_nb / total_nb if total_nb > 0 else 0
            is_shared = 1.0 if node_src == "both" else 0.0
            bridge_score = betweenness.get(node, 0) * (1 + cross_ratio) + 0.1 * is_shared

            rows.append({
                "node_id": node,
                "label": self.G.nodes[node].get("label", node),
                "source_type": node_src,
                "betweenness": betweenness.get(node, 0),
                "cross_source_neighbors": cross_source_nb,
                "total_neighbors": total_nb,
                "cross_ratio": cross_ratio,
                "bridge_score": bridge_score,
            })
        df = pd.DataFrame(rows)
        return df.sort_values("bridge_score", ascending=False).head(top_n).reset_index(drop=True)

    def source_comparison_summary(self) -> dict:
        """High-level summary comparing policy vs yelp sub-networks."""
        policy_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get("source_type") == "policy"]
        yelp_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get("source_type") == "yelp"]
        both_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get("source_type") == "both"]

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "policy_only_concepts": len(policy_nodes),
            "yelp_only_concepts": len(yelp_nodes),
            "shared_concepts": len(both_nodes),
            "overlap_pct": len(both_nodes) / max(self.G.number_of_nodes(), 1) * 100,
        }

    def annotate_graph(self, partition: dict[str, int]) -> nx.DiGraph:
        """Add centrality and community attributes to graph nodes."""
        centralities = {
            "degree": self.degree_centrality(),
            "betweenness": self.betweenness_centrality(),
            "pagerank": self.pagerank(),
        }
        for node in self.G.nodes():
            self.G.nodes[node]["community"] = partition.get(node, -1)
            for metric, values in centralities.items():
                self.G.nodes[node][metric] = values.get(node, 0)
        return self.G
