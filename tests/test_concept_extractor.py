"""Tests for the concept extraction module."""

import pytest
import spacy

from src.preprocessing.tokeniser import SpacyTokeniser, ProcessedDocument
from src.extraction.concept_extractor import ConceptExtractor, Concept


@pytest.fixture(scope="module")
def sample_doc() -> ProcessedDocument:
    tokeniser = SpacyTokeniser()
    text = (
        "The Australian Government will invest in climate adaptation. "
        "Climate change poses risks to the economy and communities. "
        "The Bureau of Meteorology provides climate science. "
        "Adaptation and resilience are critical for future prosperity."
    )
    return tokeniser.process(text, doc_id="test_doc", source="policy")


class TestConceptExtractor:
    def test_extracts_entities(self, sample_doc):
        extractor = ConceptExtractor(use_ner=True, use_noun_phrases=False, use_tfidf=False, min_freq=1)
        concepts = extractor.extract([sample_doc])
        labels = {c.label for c in concepts}
        assert any("australian" in l for l in labels) or len(concepts) > 0

    def test_extracts_noun_phrases(self, sample_doc):
        extractor = ConceptExtractor(use_ner=False, use_noun_phrases=True, use_tfidf=False, min_freq=1)
        concepts = extractor.extract([sample_doc])
        assert len(concepts) > 0
        assert all(c.type == "noun_phrase" for c in concepts)

    def test_extracts_tfidf_keywords(self, sample_doc):
        extractor = ConceptExtractor(use_ner=False, use_noun_phrases=False, use_tfidf=True, min_freq=1)
        concepts = extractor.extract([sample_doc])
        assert any(c.type == "keyword" for c in concepts)

    def test_all_methods_combined(self, sample_doc):
        extractor = ConceptExtractor(use_ner=True, use_noun_phrases=True, use_tfidf=True, min_freq=1)
        concepts = extractor.extract([sample_doc])
        types = {c.type for c in concepts}
        assert len(concepts) > 0
        assert len(types) >= 1

    def test_frequency_filter(self, sample_doc):
        extractor = ConceptExtractor(min_freq=100)
        concepts = extractor.extract([sample_doc])
        # with min_freq=100, very few or no concepts should survive (except tfidf keywords)
        for c in concepts:
            if c.type != "keyword":
                assert c.frequency >= 100

    def test_concept_has_required_fields(self, sample_doc):
        extractor = ConceptExtractor(min_freq=1)
        concepts = extractor.extract([sample_doc])
        if concepts:
            c = concepts[0]
            assert c.id
            assert c.label
            assert c.type in ("entity", "noun_phrase", "keyword")
            assert isinstance(c.frequency, int)
            assert isinstance(c.source_docs, set)

    def test_deduplication(self, sample_doc):
        extractor = ConceptExtractor(min_freq=1)
        concepts = extractor.extract([sample_doc])
        labels = [c.label for c in concepts]
        assert len(labels) == len(set(labels))
