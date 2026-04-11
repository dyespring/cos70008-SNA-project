"""Sentiment-weighted relationships: attach polarity scores to network edges."""

from __future__ import annotations

import networkx as nx

from src.preprocessing.tokeniser import ProcessedDocument

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False


class SentimentAnnotator:
    """Score sentence sentiment and propagate to network edges."""

    def __init__(self, method: str = "vader"):
        """method: 'vader' (best for informal text) or 'textblob' (simpler polarity)."""
        self.method = method
        if method == "vader" and not _VADER_AVAILABLE:
            raise ImportError("Install vaderSentiment: pip install vaderSentiment")
        if method == "textblob" and not _TEXTBLOB_AVAILABLE:
            raise ImportError("Install textblob: pip install textblob")
        if method == "vader":
            self._analyser = SentimentIntensityAnalyzer()

    def score_sentence(self, text: str) -> float:
        """Return a sentiment polarity score in [-1, 1]."""
        if self.method == "vader":
            return self._analyser.polarity_scores(text)["compound"]
        else:
            return TextBlob(text).sentiment.polarity

    def annotate_documents(self, documents: list[ProcessedDocument]) -> dict[str, list[float]]:
        """Score every sentence in each document. Returns {doc_id: [scores]}."""
        results = {}
        for doc in documents:
            scores = [self.score_sentence(s.text) for s in doc.sentences]
            results[doc.doc_id] = scores
        return results

    def annotate_graph(
        self,
        G: nx.DiGraph,
        documents: list[ProcessedDocument],
        concept_labels: set[str],
    ) -> nx.DiGraph:
        """Add average sentiment to each edge based on sentences where both endpoints co-occur."""
        doc_scores = self.annotate_documents(documents)

        for u, v, data in G.edges(data=True):
            u_label = G.nodes[u].get("label", u)
            v_label = G.nodes[v].get("label", v)
            edge_sentiments = []

            for doc in documents:
                scores = doc_scores.get(doc.doc_id, [])
                for i, sent in enumerate(doc.sentences):
                    text_lower = sent.text.lower()
                    if u_label in text_lower and v_label in text_lower and i < len(scores):
                        edge_sentiments.append(scores[i])

            if edge_sentiments:
                avg = sum(edge_sentiments) / len(edge_sentiments)
                data["sentiment"] = round(avg, 4)
                data["sentiment_label"] = (
                    "positive" if avg > 0.05 else "negative" if avg < -0.05 else "neutral"
                )
            else:
                data["sentiment"] = 0.0
                data["sentiment_label"] = "neutral"

        return G
