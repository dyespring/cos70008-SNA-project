"""FAISS-backed semantic index over the graph's concepts.

Optional dependency. If ``sentence-transformers`` or ``faiss-cpu`` are not
installed, :class:`GraphVectorStore` degrades gracefully: ``available``
is ``False`` and :meth:`search` returns an empty list, letting the
chatbot fall back to structured queries / fuzzy label matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.extensions.graph_context import GraphContext


class GraphVectorStore:
    """Semantic search over node descriptions.

    Parameters
    ----------
    graph_context : GraphContext
        Provides concept descriptions for each node.
    model_name : str
        ``sentence-transformers`` model to use.
    """

    def __init__(
        self,
        graph_context: "GraphContext",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.graph_context = graph_context
        self.model_name = model_name
        self.available = False
        self._model = None
        self._index = None
        self._node_ids: list[str] = []
        self._descriptions: list[str] = []

        try:
            self._build()
        except Exception as e:
            # Any failure (missing deps, network, etc.) -> disable cleanly
            self._last_error = str(e)
            self.available = False

    # ── Build index ────────────────────────────────────────────────
    def _build(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` to enable semantic search."
            ) from e
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Run `pip install faiss-cpu` to enable semantic search."
            ) from e

        self._model = SentenceTransformer(self.model_name)

        G = self.graph_context.G
        node_ids: list[str] = list(G.nodes())
        descriptions: list[str] = []
        for nid in node_ids:
            label = G.nodes[nid].get("label", nid)
            descriptions.append(self._short_description(label))
        if not descriptions:
            raise ValueError("No nodes in graph to index.")

        vectors = self._model.encode(
            descriptions, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        # normalise for cosine/IP equivalence
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self._index = index
        self._node_ids = node_ids
        self._descriptions = descriptions
        self.available = True

    def _short_description(self, label: str) -> str:
        """Compact one-line node blurb good for embedding."""
        G = self.graph_context.G
        nid = self.graph_context.resolve(label) or label
        if nid not in G.nodes():
            return label
        data = G.nodes[nid]
        ctype = data.get("concept_type") or data.get("type") or "concept"
        src = data.get("source_type", "n/a")
        neighbours = []
        for _, v, edata in G.out_edges(nid, data=True):
            neighbours.append((G.nodes[v].get("label", v), edata.get("weight", 1)))
        for u, _, edata in G.in_edges(nid, data=True):
            neighbours.append((G.nodes[u].get("label", u), edata.get("weight", 1)))
        neighbours.sort(key=lambda x: x[1], reverse=True)
        top_nbs = ", ".join(lbl for lbl, _ in neighbours[:5])
        return (
            f"{label}. A {ctype} concept from {src} data."
            + (f" Related to: {top_nbs}." if top_nbs else "")
        )

    # ── Query ──────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (node_id, similarity) pairs, or [] if unavailable."""
        if not self.available or self._model is None or self._index is None:
            return []
        try:
            import faiss  # type: ignore

            vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(vec)
            k = min(k, len(self._node_ids))
            scores, idx = self._index.search(vec, k)
            out: list[tuple[str, float]] = []
            for score, i in zip(scores[0], idx[0]):
                if i < 0:
                    continue
                out.append((self._node_ids[i], float(score)))
            return out
        except Exception:
            return []

    def __len__(self) -> int:
        return len(self._node_ids)
