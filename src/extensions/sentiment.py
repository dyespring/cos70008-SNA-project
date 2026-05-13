"""Sentiment-weighted relationships: write polarity scores onto :RELATED edges.

In the Neo4j-only pipeline this module reads the edge list straight out of
Neo4j, scores the relevant sentences, and writes the average sentiment back
onto each edge as ``r.sentiment`` / ``r.sentiment_label``.

The sentence corpus comes from the Stage-2 :class:`ProcessedDocument` objects
that the same pipeline run already produced — this avoids needing a copy of
the raw text inside Neo4j.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.preprocessing.tokeniser import ProcessedDocument

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore

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


logger = logging.getLogger(__name__)
_EDGE_BATCH = 500


class SentimentAnnotator:
    """Score sentence sentiment and write average values back onto Neo4j edges."""

    def __init__(self, method: str = "vader"):
        """method: 'vader' (best for informal text) or 'textblob' (simpler polarity)."""
        self.method = method
        if method == "vader" and not _VADER_AVAILABLE:
            raise ImportError("Install vaderSentiment: pip install vaderSentiment")
        if method == "textblob" and not _TEXTBLOB_AVAILABLE:
            raise ImportError("Install textblob: pip install textblob")
        if method == "vader":
            self._analyser = SentimentIntensityAnalyzer()

    # ── Sentence-level scoring ─────────────────────────────────────
    def score_sentence(self, text: str) -> float:
        """Return a sentiment polarity score in [-1, 1]."""
        if self.method == "vader":
            return self._analyser.polarity_scores(text)["compound"]
        return TextBlob(text).sentiment.polarity

    def annotate_documents(
        self, documents: list[ProcessedDocument]
    ) -> dict[str, list[float]]:
        """Score every sentence in each document. Returns {doc_id: [scores]}."""
        return {
            doc.doc_id: [self.score_sentence(s.text) for s in doc.sentences]
            for doc in documents
        }

    # ── Neo4j edge annotation ──────────────────────────────────────
    def annotate_edges_in_neo4j(
        self,
        store: "Neo4jStore",
        documents: list[ProcessedDocument],
        source_label: str,
        slice_id: str | None = None,
    ) -> int:
        """Compute sentiment per :RELATED edge and write it back to Neo4j.

        Returns the number of edges updated.
        """
        if not documents:
            return 0
        edges = self._fetch_edges(store, source_label, slice_id)
        if not edges:
            return 0

        doc_scores = self.annotate_documents(documents)

        rows = []
        for src_id, tgt_id, src_label, tgt_label in edges:
            ul = (src_label or "").lower()
            vl = (tgt_label or "").lower()
            edge_sentiments: list[float] = []
            for doc in documents:
                scores = doc_scores.get(doc.doc_id, [])
                for i, sent in enumerate(doc.sentences):
                    text_lower = sent.text.lower()
                    if ul in text_lower and vl in text_lower and i < len(scores):
                        edge_sentiments.append(scores[i])
            if edge_sentiments:
                avg = sum(edge_sentiments) / len(edge_sentiments)
            else:
                avg = 0.0
            label = (
                "positive" if avg > 0.05
                else "negative" if avg < -0.05
                else "neutral"
            )
            rows.append(
                {
                    "source": src_id,
                    "target": tgt_id,
                    "sl": source_label,
                    "sid": slice_id,
                    "sentiment": round(avg, 4),
                    "sentiment_label": label,
                }
            )

        with_slice = slice_id is not None
        if with_slice:
            cypher = (
                "UNWIND $rows AS row "
                "MATCH (a:Concept {id: row.source})"
                "-[r:RELATED {source_label: row.sl, slice_id: row.sid}]->"
                "(b:Concept {id: row.target}) "
                "SET r.sentiment = row.sentiment, "
                "    r.sentiment_label = row.sentiment_label"
            )
        else:
            cypher = (
                "UNWIND $rows AS row "
                "MATCH (a:Concept {id: row.source})"
                "-[r:RELATED {source_label: row.sl}]->"
                "(b:Concept {id: row.target}) "
                "SET r.sentiment = row.sentiment, "
                "    r.sentiment_label = row.sentiment_label"
            )

        with store.session() as s:
            for i in range(0, len(rows), _EDGE_BATCH):
                s.run(cypher, rows=rows[i : i + _EDGE_BATCH])

        logger.info(
            "Sentiment: annotated %d edges (source_label=%s, slice_id=%s)",
            len(rows),
            source_label,
            slice_id,
        )
        return len(rows)

    @staticmethod
    def _fetch_edges(
        store: "Neo4jStore",
        source_label: str,
        slice_id: str | None,
    ) -> list[tuple[str, str, str, str]]:
        params = {"sl": source_label}
        slice_filter = ""
        if slice_id is not None:
            params["sid"] = slice_id
            slice_filter = " AND r.slice_id = $sid AND a.slice_id = $sid AND b.slice_id = $sid"
        cypher = (
            "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            "WHERE r.source_label = $sl AND a.source_label = $sl "
            "  AND b.source_label = $sl"
            + slice_filter
            + " RETURN a.id AS sid, b.id AS tid, "
            "        a.label AS slabel, b.label AS tlabel"
        )
        with store.session() as s:
            return [
                (rec["sid"], rec["tid"], rec["slabel"], rec["tlabel"])
                for rec in s.run(cypher, **params)
            ]
