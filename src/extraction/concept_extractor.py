"""Extract key concepts from processed documents using multiple strategies.

The extractor combines three signals:

* **Named-entity recognition** (NER) — keeps entities whose label is in
  :data:`src.config.NER_LABELS`. ``PERSON`` entities additionally need
  to clear :data:`src.config.MIN_PERSON_FREQ` to avoid review-style
  proper-noun fragments (e.g. "Eli's salad" leaking ``"eli"`` into the
  graph).
* **Noun-phrase chunking** — consumes the *cleaned* chunks produced by
  :class:`SpacyTokeniser` (determiners stripped, lemmatised), so
  ``"the great food"`` arrives here as ``"great food"``.
* **TF-IDF keyword extraction** — tightened with ``max_df`` / ``min_df``
  thresholds and the same domain stop-word set as direct extraction so
  high-frequency boilerplate cannot sneak back in via the keyword path.

All candidates are filtered against the source-specific stop-word set
returned by :func:`src.config.stopwords_for`, so corpus-wide noise like
``"eat"``, ``"place"``, ``"food"`` (Yelp) or ``"section"``, ``"clause"``
(policy) never enters the graph.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    CUSTOM_STOP_WORDS,
    DOMAIN_STOP_WORDS_POLICY,
    DOMAIN_STOP_WORDS_YELP,
    MIN_CONCEPT_FREQ,
    MIN_ENTITY_FREQ,
    MIN_PERSON_FREQ,
    NER_LABELS,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_TOP_N,
    stopwords_for,
)
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
        min_person_freq: int = MIN_PERSON_FREQ,
        tfidf_max_df: float = TFIDF_MAX_DF,
        tfidf_min_df: int = TFIDF_MIN_DF,
        min_entity_freq: dict[str, int] | None = None,
    ):
        self.use_ner = use_ner
        self.use_noun_phrases = use_noun_phrases
        self.use_tfidf = use_tfidf
        self.ner_labels = ner_labels or NER_LABELS
        self.min_freq = min_freq
        self.tfidf_top_n = tfidf_top_n
        self.min_person_freq = min_person_freq
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_min_df = tfidf_min_df

        # Per-NER-type frequency gate. Defaults to ``MIN_ENTITY_FREQ`` from
        # config but layered with backward-compat for the standalone
        # ``min_person_freq`` argument: callers that explicitly pass
        # ``min_person_freq=N`` keep the old behaviour for PERSON
        # entities even when a wider ``min_entity_freq`` dict is also
        # provided. Any NER label missing from the dict falls back to
        # the global ``min_freq`` gate (i.e. no extra restriction).
        merged: dict[str, int] = dict(MIN_ENTITY_FREQ)
        if min_entity_freq:
            merged.update(min_entity_freq)
        # ``min_person_freq`` is the *explicit* override path that older
        # call sites (and the existing CLI flag) use. Honour it if
        # supplied by the caller.
        merged["PERSON"] = min_person_freq
        self.min_entity_freq: dict[str, int] = merged

    # ── Public API ─────────────────────────────────────────────────
    def extract(self, documents: list[ProcessedDocument]) -> list[Concept]:
        """Extract and merge concepts across all documents.

        Filtering happens in three layers, in this order:

        1. Per-candidate: drop candidates whose normalised label is in
           the source-specific stop-word set, is too short, or is a
           pure number.
        2. Per-PERSON: drop NER ``PERSON`` candidates whose total
           frequency across all documents stays below
           ``self.min_person_freq``.
        3. Per-concept: drop any concept whose total frequency stays
           below ``self.min_freq``. TF-IDF keywords also have to clear
           this gate now (previously they bypassed it).
        """
        raw_concepts: Counter[str] = Counter()
        concept_types: dict[str, str] = {}
        concept_docs: dict[str, set[str]] = {}
        concept_sents: dict[str, list[int]] = {}
        concept_origins: dict[str, set[str]] = {}
        ner_label_seen: dict[str, str] = {}   # label → strongest NER label

        for doc in documents:
            src_stop = stopwords_for(doc.source)
            for sent_idx, sent in enumerate(doc.sentences):
                candidates: list[tuple[str, str, str | None]] = []

                if self.use_ner:
                    candidates.extend(self._extract_entities(sent))

                if self.use_noun_phrases:
                    candidates.extend(self._extract_noun_phrases(sent))

                for label, ctype, ent_label in candidates:
                    label = self._normalise_label(label)
                    if not self._label_passes(label, src_stop):
                        continue
                    raw_concepts[label] += 1
                    concept_types.setdefault(label, ctype)
                    if ent_label and ctype == "entity":
                        ner_label_seen.setdefault(label, ent_label)
                    concept_docs.setdefault(label, set()).add(doc.doc_id)
                    concept_sents.setdefault(label, []).append(sent_idx)
                    concept_origins.setdefault(label, set()).add(doc.source)

        if self.use_tfidf:
            tfidf_keywords = self._extract_tfidf_keywords(documents)
            for kw in tfidf_keywords:
                kw = self._normalise_label(kw)
                if not self._label_passes(kw, _global_stopwords()):
                    continue
                if kw not in concept_types:
                    concept_types[kw] = "keyword"
                    concept_docs.setdefault(kw, set())
                    concept_sents.setdefault(kw, [])
                    concept_origins.setdefault(kw, set())
                    # TF-IDF candidates without observed text occurrences
                    # are seeded with frequency 1 so the global ``min_freq``
                    # gate below decides whether to keep them — they no
                    # longer get a free pass.
                    raw_concepts[kw] = max(raw_concepts.get(kw, 0), 1)

        concepts: list[Concept] = []
        for label, freq in raw_concepts.items():
            if freq < self.min_freq:
                continue
            ent_label = ner_label_seen.get(label)
            if ent_label is not None:
                # NER candidates have to clear *both* the universal
                # ``min_freq`` gate and a per-type minimum (PERSON,
                # ORG, PRODUCT have stricter defaults to keep brand
                # / proper-noun fragments out of the network).
                ent_min = self.min_entity_freq.get(ent_label)
                if ent_min is not None and freq < ent_min:
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

    # ── Candidate sources ──────────────────────────────────────────
    def _extract_entities(self, sent) -> list[tuple[str, str, str | None]]:
        return [
            (ent_text, "entity", ent_label)
            for ent_text, ent_label in sent.entities
            if ent_label in self.ner_labels
        ]

    def _extract_noun_phrases(self, sent) -> list[tuple[str, str, str | None]]:
        results: list[tuple[str, str, str | None]] = []
        for chunk in sent.noun_chunks:
            text = chunk.strip()
            if not text:
                continue
            words = text.split()
            # Tokeniser already cleaned chunks; this is a defensive last
            # filter to drop residual single-token stop-words and very
            # short tails like "it".
            if len(words) == 1:
                w = words[0].lower()
                if w in CUSTOM_STOP_WORDS or len(w) < 3:
                    continue
            results.append((text, "noun_phrase", None))
        return results

    def _extract_tfidf_keywords(self, documents: list[ProcessedDocument]) -> list[str]:
        corpus = [" ".join(doc.all_lemmas) for doc in documents]
        if not corpus or all(len(c.strip()) == 0 for c in corpus):
            return []

        n_docs = len(corpus)
        # Relax constraints automatically for tiny corpora so unit tests
        # and demo notebooks (n ≤ a handful of docs) still produce output.
        if n_docs <= 2:
            max_df, min_df = 1.0, 1
        else:
            max_df = min(self.tfidf_max_df, 1.0)
            min_df = max(1, min(self.tfidf_min_df, n_docs - 1))

        vectorizer = TfidfVectorizer(
            max_features=self.tfidf_top_n,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            stop_words=sorted(_global_stopwords()),
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return []
        feature_names = vectorizer.get_feature_names_out()

        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[::-1][:self.tfidf_top_n]
        return [feature_names[i] for i in top_indices]

    # ── Helpers ────────────────────────────────────────────────────
    @staticmethod
    def _label_passes(label: str, stop_set: set[str]) -> bool:
        if not label or len(label) < 2:
            return False
        if label.isdigit():
            return False
        if label in stop_set:
            return False
        if " " not in label and label.replace("-", "").isalpha() is False:
            # single token containing digits / punctuation noise
            return False
        return True

    @staticmethod
    def _normalise_label(label: str) -> str:
        label = label.strip().lower()
        label = " ".join(label.split())
        return label

    @staticmethod
    def _make_id(label: str) -> str:
        return label.replace(" ", "_").replace("-", "_")


def _global_stopwords() -> set[str]:
    """Union of universal + every domain stop-word set (used for TF-IDF)."""
    return CUSTOM_STOP_WORDS | DOMAIN_STOP_WORDS_YELP | DOMAIN_STOP_WORDS_POLICY
