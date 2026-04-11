"""Extract key concepts from processed documents using multiple strategies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import NER_LABELS, MIN_CONCEPT_FREQ, TFIDF_TOP_N, CUSTOM_STOP_WORDS
from src.preprocessing.tokeniser import ProcessedDocument


@dataclass
class Concept:
    """A key concept extracted from text."""
    id: str
    label: str
    type: str                         # 'entity', 'noun_phrase', 'keyword'
    frequency: int = 0
    source_docs: set[str] = field(default_factory=set)
    source_sentences: list[int] = field(default_factory=list)
    source_origins: set[str] = field(default_factory=set)  # 'policy', 'yelp', or both

    @property
    def source_type(self) -> str:
        """Return 'both' if concept appears in multiple sources, else the single source."""
        if len(self.source_origins) > 1:
            return "both"
        return next(iter(self.source_origins)) if self.source_origins else "unknown"


class ConceptExtractor:
    """Extract concepts using NER, noun-phrase chunking, and TF-IDF keywords."""

    def __init__(
        self,
        use_ner: bool = True,
        use_noun_phrases: bool = True,
        use_tfidf: bool = True,
        ner_labels: set[str] | None = None,
        min_freq: int = MIN_CONCEPT_FREQ,
        tfidf_top_n: int = TFIDF_TOP_N,
    ):
        self.use_ner = use_ner
        self.use_noun_phrases = use_noun_phrases
        self.use_tfidf = use_tfidf
        self.ner_labels = ner_labels or NER_LABELS
        self.min_freq = min_freq
        self.tfidf_top_n = tfidf_top_n

    def extract(self, documents: list[ProcessedDocument]) -> list[Concept]:
        """Extract and merge concepts from one or more processed documents."""
        raw_concepts: Counter[str] = Counter()
        concept_types: dict[str, str] = {}
        concept_docs: dict[str, set[str]] = {}
        concept_sents: dict[str, list[int]] = {}
        concept_origins: dict[str, set[str]] = {}

        for doc in documents:
            for sent_idx, sent in enumerate(doc.sentences):
                candidates: list[tuple[str, str]] = []

                if self.use_ner:
                    candidates.extend(self._extract_entities(sent))

                if self.use_noun_phrases:
                    candidates.extend(self._extract_noun_phrases(sent))

                for label, ctype in candidates:
                    label = self._normalise_label(label)
                    if len(label) < 2:
                        continue
                    raw_concepts[label] += 1
                    concept_types.setdefault(label, ctype)
                    concept_docs.setdefault(label, set()).add(doc.doc_id)
                    concept_sents.setdefault(label, []).append(sent_idx)
                    concept_origins.setdefault(label, set()).add(doc.source)

        if self.use_tfidf:
            tfidf_keywords = self._extract_tfidf_keywords(documents)
            for kw in tfidf_keywords:
                kw = self._normalise_label(kw)
                if kw not in concept_types:
                    concept_types[kw] = "keyword"
                    concept_docs.setdefault(kw, set())
                    concept_sents.setdefault(kw, [])
                    concept_origins.setdefault(kw, set())
                raw_concepts[kw] = max(raw_concepts.get(kw, 0), 1)

        concepts = []
        for label, freq in raw_concepts.items():
            if freq < self.min_freq and concept_types.get(label) != "keyword":
                continue
            concepts.append(Concept(
                id=self._make_id(label),
                label=label,
                type=concept_types[label],
                frequency=freq,
                source_docs=concept_docs.get(label, set()),
                source_sentences=concept_sents.get(label, []),
                source_origins=concept_origins.get(label, set()),
            ))

        concepts.sort(key=lambda c: c.frequency, reverse=True)
        return concepts

    def _extract_entities(self, sent) -> list[tuple[str, str]]:
        return [
            (ent_text, "entity")
            for ent_text, ent_label in sent.entities
            if ent_label in self.ner_labels
        ]

    def _extract_noun_phrases(self, sent) -> list[tuple[str, str]]:
        results = []
        for chunk in sent.noun_chunks:
            text = chunk.strip()
            words = text.split()
            if len(words) == 1 and words[0].lower() in CUSTOM_STOP_WORDS:
                continue
            if len(words) >= 2 or (len(words) == 1 and len(words[0]) > 3):
                results.append((text, "noun_phrase"))
        return results

    def _extract_tfidf_keywords(self, documents: list[ProcessedDocument]) -> list[str]:
        corpus = [" ".join(doc.all_lemmas) for doc in documents]
        if not corpus or all(len(c.strip()) == 0 for c in corpus):
            return []
        max_df = 1.0 if len(corpus) <= 2 else 0.95
        vectorizer = TfidfVectorizer(
            max_features=self.tfidf_top_n,
            ngram_range=(1, 2),
            min_df=1,
            max_df=max_df,
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return []
        feature_names = vectorizer.get_feature_names_out()

        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[::-1][:self.tfidf_top_n]
        return [feature_names[i] for i in top_indices]

    @staticmethod
    def _normalise_label(label: str) -> str:
        label = label.strip().lower()
        label = " ".join(label.split())
        return label

    @staticmethod
    def _make_id(label: str) -> str:
        return label.replace(" ", "_").replace("-", "_")
