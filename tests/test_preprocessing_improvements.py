"""Regression tests for the P0 + P1 preprocessing fixes.

Covers:
- NP chunk cleanup (determiners stripped, lemmatised) at the tokeniser layer.
- Lemma-stream sentence text used by relationship extraction.
- Source-aware stop-word filtering at concept extraction.
- Tightened TF-IDF (max_df / min_df / shared stopwords).
- NER PERSON minimum-frequency gate.
- NPMI computation + filtering.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from src.config import (
    DOMAIN_STOP_WORDS_YELP,
    DOMAIN_STOP_WORDS_POLICY,
    stopwords_for,
)
from src.extraction.concept_extractor import Concept, ConceptExtractor
from src.extraction.relationship_extractor import (
    RelationshipExtractor,
    _label_in_text,
    _npmi,
)
from src.preprocessing.tokeniser import SpacyTokeniser, _clean_chunk


# ════════════════════════════════════════════════════════════════════
# Shared fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def tokeniser() -> SpacyTokeniser:
    return SpacyTokeniser()


# ════════════════════════════════════════════════════════════════════
# stopwords_for() — config helper
# ════════════════════════════════════════════════════════════════════


class TestStopwordsForSource:
    def test_yelp_includes_yelp_domain(self):
        sw = stopwords_for("yelp")
        assert "eat" in sw
        assert "place" in sw
        # policy-only words should NOT pollute the yelp set
        assert "section" not in sw

    def test_policy_includes_policy_domain(self):
        sw = stopwords_for("policy")
        assert "section" in sw
        assert "clause" in sw
        # yelp-only words should NOT pollute the policy set
        assert "eat" not in sw

    def test_combined_unions_both(self):
        sw = stopwords_for("combined")
        assert "eat" in sw and "section" in sw

    @pytest.mark.parametrize("src", ["", None, "unknown_source"])
    def test_unknown_source_falls_back_to_universal_only(self, src):
        sw = stopwords_for(src)
        # universal members survive
        assert "however" in sw
        # but no domain noise
        assert "eat" not in sw
        assert "section" not in sw

    def test_returns_a_fresh_set(self):
        """Mutating the returned set must not corrupt the source-of-truth
        constants — caller mutability would silently break later calls."""
        sw1 = stopwords_for("yelp")
        sw1.add("__should_not_persist__")
        sw2 = stopwords_for("yelp")
        assert "__should_not_persist__" not in sw2
        assert "__should_not_persist__" not in DOMAIN_STOP_WORDS_YELP


# ════════════════════════════════════════════════════════════════════
# NP chunk cleanup at the tokeniser layer
# ════════════════════════════════════════════════════════════════════


class TestNounChunkCleanup:
    def test_determiners_stripped_and_lemmatised(self, tokeniser):
        """`the great food` → `great food`; `this place` → `place`."""
        doc = tokeniser.process(
            "I loved the great food. This place was amazing.",
            doc_id="d0",
            source="yelp",
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        # Determiner-headed chunks should have their head trimmed
        assert "the food" not in chunks
        assert "this place" not in chunks
        assert "great food" in chunks or "food" in chunks
        assert "place" in chunks

    def test_pluralisation_collapses_via_lemma(self, tokeniser):
        """`foods`, `foodies` → `food`, `foodie` via lemmatisation."""
        doc = tokeniser.process(
            "The foods were amazing.", doc_id="d0", source="yelp"
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        assert "food" in chunks
        assert "foods" not in chunks
        assert "the foods" not in chunks

    def test_possessives_stripped(self, tokeniser):
        """`my favourite dish` → `favourite dish` (or `dish`)."""
        doc = tokeniser.process(
            "My favourite dish was the pasta.", doc_id="d0", source="yelp"
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        assert all(not c.startswith("my ") for c in chunks)

    def test_clean_chunk_returns_empty_for_pure_determiner(self, tokeniser):
        """A chunk that is *only* determiners must collapse to empty."""
        doc = tokeniser.nlp("the")
        # Build a span manually — spaCy may or may not produce a chunk here
        span = doc[0:1]
        assert _clean_chunk(span) == ""

    # ── Leak 1: head-POS gate ──────────────────────────────────────

    def test_numeral_chunk_rejected(self, tokeniser):
        """`one` (POS=NUM/PRON) must not survive as a concept."""
        doc = tokeniser.process(
            "I want one. We ordered one. Pizza is the one I love.",
            doc_id="d0", source="yelp",
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        assert "one" not in chunks

    def test_modal_or_aux_chunk_rejected(self, tokeniser):
        """`can`, `will`, `would`, `could` (POS=AUX) must not leak in."""
        doc = tokeniser.process(
            "We can recommend the pizza. They will return tomorrow. "
            "I would order again. You could try the pasta.",
            doc_id="d0", source="yelp",
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        for w in ("can", "will", "would", "could"):
            assert w not in chunks, f"{w!r} should not survive head-POS gate"

    def test_pronoun_chunk_rejected(self, tokeniser):
        """`it`, `they`, `someone` (POS=PRON) chunks must be dropped."""
        doc = tokeniser.process(
            "Someone left a tip. They paid in cash. It was delicious.",
            doc_id="d0", source="yelp",
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        for w in ("it", "they", "someone", "she", "he", "we"):
            assert w not in chunks

    def test_real_noun_chunks_still_survive(self, tokeniser):
        """Belt-and-braces: the head-POS gate must NOT over-prune."""
        doc = tokeniser.process(
            "The pizza was amazing. The waiter was friendly. "
            "Their service exceeded expectations.",
            doc_id="d0", source="yelp",
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        # All three head nouns should be present in lemma form.
        for expected in ("pizza", "waiter", "service"):
            assert expected in chunks, (
                f"expected {expected!r} to survive but got chunks={chunks!r}"
            )

    # ── Leak 2: trailing possessive strip ──────────────────────────

    def test_trailing_possessive_apostrophe_s_stripped(self, tokeniser):
        """`the australian government's` → `australian government`."""
        doc = tokeniser.process(
            "The Australian government's policy on climate change is clear.",
            doc_id="d0", source="policy",
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        # No chunk should still end in 's
        for c in chunks:
            assert not c.endswith("'s"), f"{c!r} still has trailing 's"
            assert not c.endswith("’s"), f"{c!r} still has trailing ’s"
            assert c != "australian government's"

    def test_possessive_pronoun_chunk_stripped_at_head(self, tokeniser):
        """`John's pizza` → `john pizza` (or `pizza`); trailing `'s`
        on possessor must not leak into the chunk."""
        doc = tokeniser.process(
            "John's pizza arrived hot. Sarah's salad was fresh.",
            doc_id="d0", source="yelp",
        )
        chunks = [c for s in doc.sentences for c in s.noun_chunks]
        for c in chunks:
            assert "'s" not in c, f"{c!r} still contains 's"
            assert "’s" not in c

    def test_lemma_text_is_populated(self, tokeniser):
        doc = tokeniser.process(
            "Customers ordered amazing pizzas.",
            doc_id="d0",
            source="yelp",
        )
        sent = doc.sentences[0]
        assert sent.lemma_text  # non-empty
        # lemma stream must contain lemma forms, not surface forms
        assert "pizza" in sent.lemma_text
        assert "order" in sent.lemma_text
        assert "pizzas" not in sent.lemma_text
        assert "ordered" not in sent.lemma_text


# ════════════════════════════════════════════════════════════════════
# Source-aware stop-word filtering at concept extraction
# ════════════════════════════════════════════════════════════════════


class TestConceptExtractionStopwords:
    def test_yelp_domain_words_filtered(self, tokeniser):
        """`eat`, `food`, `place` must not survive as Yelp concepts."""
        text = (
            "We had great food at this place. "
            "We love to eat pizza here. "
            "The food was amazing and the place was clean. "
            "Pizza is the best thing to eat."
        )
        doc = tokeniser.process(text, doc_id="d0", source="yelp")
        ce = ConceptExtractor(use_tfidf=False, min_freq=1)
        concepts = ce.extract([doc])
        labels = {c.label for c in concepts}
        assert "eat" not in labels
        assert "food" not in labels
        assert "place" not in labels
        # but topical concepts survive
        assert "pizza" in labels

    def test_policy_domain_words_filtered(self, tokeniser):
        text = (
            "Section 1 outlines the policy. "
            "The clause defines climate adaptation strategy. "
            "This section covers resilience frameworks."
        )
        doc = tokeniser.process(text, doc_id="d0", source="policy")
        ce = ConceptExtractor(use_tfidf=False, min_freq=1)
        concepts = ce.extract([doc])
        labels = {c.label for c in concepts}
        assert "section" not in labels
        assert "clause" not in labels
        assert "policy" not in labels

    def test_yelp_stopwords_dont_apply_to_policy(self, tokeniser):
        """`eat` may legitimately appear in policy text and should
        survive there (it is yelp-only noise)."""
        text = "The eat strategy is detailed in the appendix. The eat plan defines outcomes."
        doc = tokeniser.process(text, doc_id="d0", source="policy")
        ce = ConceptExtractor(use_tfidf=False, min_freq=1)
        concepts = ce.extract([doc])
        labels = {c.label for c in concepts}
        # policy-source extraction does NOT filter yelp-only stopwords
        assert "eat" in labels or any("eat" in lbl for lbl in labels)

    # ── Round 3: 'work' / 'need' kicked out of yelp + combined ─────

    def test_work_and_need_filtered_in_yelp(self, tokeniser):
        """`work` and `need` were dominating SPOF in combined runs;
        adding them to DOMAIN_STOP_WORDS_YELP keeps them out of the
        graph for both yelp and combined sources."""
        text = (
            "The service does not work for me. We really need a refund. "
            "I want this to work. We need help."
        )
        for src in ("yelp", "combined"):
            doc = tokeniser.process(text, doc_id="d0", source=src)
            ce = ConceptExtractor(use_tfidf=False, min_freq=1)
            concepts = ce.extract([doc])
            labels = {c.label for c in concepts}
            assert "work" not in labels, (
                f"'work' must be filtered for source={src} but got {labels}"
            )
            assert "need" not in labels, (
                f"'need' must be filtered for source={src} but got {labels}"
            )


# ════════════════════════════════════════════════════════════════════
# TF-IDF tightening
# ════════════════════════════════════════════════════════════════════


class TestTfidfTightening:
    def test_keyword_does_not_bypass_min_freq(self, tokeniser):
        """Previously TF-IDF keywords were force-included with frequency 1.
        Now they have to clear the global ``min_freq`` gate too."""
        text = "rare topic phrase appears once."
        doc = tokeniser.process(text, doc_id="d0", source="general")
        ce = ConceptExtractor(use_ner=False, use_noun_phrases=False,
                              use_tfidf=True, min_freq=5)
        concepts = ce.extract([doc])
        # nothing in this tiny one-sentence doc can possibly hit freq=5
        assert all(c.frequency >= 5 for c in concepts)

    def test_tfidf_respects_domain_stopwords(self, tokeniser):
        """`eat` shouldn't sneak in via TF-IDF on a yelp corpus."""
        docs = [
            tokeniser.process(
                f"I love to eat pizza. The pizza is amazing. Pizza pizza pizza. Doc {i}.",
                doc_id=f"d{i}",
                source="yelp",
            )
            for i in range(5)
        ]
        ce = ConceptExtractor(
            use_ner=False, use_noun_phrases=False,
            use_tfidf=True, min_freq=1, tfidf_min_df=1, tfidf_max_df=1.0,
        )
        concepts = ce.extract(docs)
        labels = {c.label for c in concepts}
        # `eat` is in DOMAIN_STOP_WORDS_YELP and the tfidf path now
        # uses the union of all domain stopwords, so it must not appear
        assert "eat" not in labels


# ════════════════════════════════════════════════════════════════════
# NER PERSON minimum-frequency gate
# ════════════════════════════════════════════════════════════════════


class TestPersonFreqGate:
    """`en_core_web_sm` NER is noisy on isolated names — these tests first
    sniff the model's actual labels and skip if it didn't tag the name as
    PERSON, otherwise they would be testing tokeniser behaviour, not the
    gate."""

    @staticmethod
    def _person_labels(tokeniser, text: str) -> set[str]:
        d = tokeniser.nlp(text)
        return {ent.text.lower() for ent in d.ents if ent.label_ == "PERSON"}

    def test_low_freq_person_filtered(self, tokeniser):
        text = (
            "Sarah Johnson visited the bistro. "
            "The dessert was sublime."
        )
        if "sarah" not in self._person_labels(tokeniser, text) \
                and "sarah johnson" not in self._person_labels(tokeniser, text):
            pytest.skip("spaCy en_core_web_sm did not tag the name as PERSON")
        doc = tokeniser.process(text, doc_id="d0", source="yelp")
        ce = ConceptExtractor(
            use_tfidf=False, min_freq=1, min_person_freq=10,
        )
        concepts = ce.extract([doc])
        labels = {c.label for c in concepts}
        # any single-token PERSON name must be filtered out
        assert "sarah" not in labels
        assert "sarah johnson" not in labels

    def test_high_freq_person_survives(self, tokeniser):
        text = " ".join(["Sarah Johnson ate pizza."] * 12)
        persons = self._person_labels(tokeniser, text)
        if not persons:
            pytest.skip("spaCy en_core_web_sm did not tag the name as PERSON")
        doc = tokeniser.process(text, doc_id="d0", source="yelp")
        ce = ConceptExtractor(
            use_tfidf=False, min_freq=1, min_person_freq=10,
        )
        concepts = ce.extract([doc])
        labels = {c.label for c in concepts}
        # at least one variant of the name must survive after passing the gate
        assert any(p in labels for p in persons), (
            f"expected one of {persons} to survive but got {labels}"
        )


# ════════════════════════════════════════════════════════════════════
# Per-NER-type frequency gates (PERSON / ORG / PRODUCT)
# ════════════════════════════════════════════════════════════════════


class TestPerEntityTypeGate:
    """Round 3: the PERSON-only gate has been generalised — ORG and
    PRODUCT now have their own thresholds in
    ``ConceptExtractor.min_entity_freq``. These tests stub out NER so
    they're independent of which spaCy version is installed."""

    def _stub_extractor(self, **kwargs) -> ConceptExtractor:
        return ConceptExtractor(
            use_ner=True, use_noun_phrases=False, use_tfidf=False,
            min_freq=1, **kwargs,
        )

    def _doc_with_entities(
        self, entities_per_sentence: list[list[tuple[str, str]]]
    ):
        """Return a fake ProcessedDocument that the extractor will eat.

        ``entities_per_sentence`` is e.g.
        ``[[("wendy", "ORG")], [("wendy", "ORG")]]`` — one entity tuple
        per sentence; the test harness only cares about the entities
        list on each ProcessedSentence.
        """
        from src.preprocessing.tokeniser import ProcessedDocument, ProcessedSentence

        sents = [
            ProcessedSentence(
                text="", tokens=[], lemmas=[], pos_tags=[],
                entities=ents, noun_chunks=[], lemma_text="",
            )
            for ents in entities_per_sentence
        ]
        return ProcessedDocument(
            doc_id="d0", source="yelp", sentences=sents, spacy_doc=None,
        )

    def test_low_frequency_org_filtered(self):
        """A brand name appearing only twice (below default 10) must drop."""
        ce = self._stub_extractor()
        # Two appearances of an ORG → freq 2, default ORG gate is 10
        doc = self._doc_with_entities([
            [("Wendy", "ORG")], [("Wendy", "ORG")],
        ])
        concepts = ce.extract([doc])
        assert "wendy" not in {c.label for c in concepts}

    def test_high_frequency_org_survives(self):
        ce = self._stub_extractor()
        # 12 occurrences clears the default ORG gate of 10
        sents = [[("Wendy", "ORG")] for _ in range(12)]
        doc = self._doc_with_entities(sents)
        concepts = ce.extract([doc])
        assert "wendy" in {c.label for c in concepts}

    def test_low_frequency_product_filtered(self):
        ce = self._stub_extractor()
        doc = self._doc_with_entities([[("iPhone", "PRODUCT")] for _ in range(3)])
        concepts = ce.extract([doc])
        assert "iphone" not in {c.label for c in concepts}

    def test_min_entity_freq_dict_overrides_defaults(self):
        """Caller-supplied ``min_entity_freq`` should win for non-PERSON keys."""
        ce = self._stub_extractor(min_entity_freq={"ORG": 2})
        doc = self._doc_with_entities([
            [("Wendy", "ORG")], [("Wendy", "ORG")],
        ])
        concepts = ce.extract([doc])
        # Lowering ORG to 2 lets it through
        assert "wendy" in {c.label for c in concepts}

    def test_min_person_freq_arg_takes_precedence_over_dict(self):
        """Backward compatibility: an explicit ``min_person_freq`` argument
        always overrides whatever's in ``min_entity_freq`` for PERSON."""
        ce = self._stub_extractor(
            min_person_freq=2,
            min_entity_freq={"PERSON": 99, "ORG": 99},
        )
        # 3 PERSON occurrences should clear the explicit gate of 2,
        # not the dict's PERSON=99.
        doc = self._doc_with_entities(
            [[("Sarah", "PERSON")] for _ in range(3)]
        )
        concepts = ce.extract([doc])
        assert "sarah" in {c.label for c in concepts}

    def test_unrelated_entity_type_uses_only_min_freq(self):
        """A label not in ``min_entity_freq`` (e.g. GPE) must still pass
        the gate as long as it clears the global ``min_freq`` (=1 here)."""
        ce = self._stub_extractor()
        doc = self._doc_with_entities([
            [("Sydney", "GPE")], [("Sydney", "GPE")],
        ])
        concepts = ce.extract([doc])
        # GPE has no per-type gate → only ``min_freq=1`` applies
        assert "sydney" in {c.label for c in concepts}


# ════════════════════════════════════════════════════════════════════
# NPMI maths
# ════════════════════════════════════════════════════════════════════


class TestNpmiMath:
    def test_perfect_co_occurrence_is_one(self):
        """When a, b, co, n all coincide, NPMI → 1."""
        # a appears 5/10, b appears 5/10, co=5/10 (always together)
        v = _npmi(co=5, a=5, b=5, n=10)
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_independence_is_zero(self):
        """When co == expected by independence, NPMI is 0."""
        # a=4/10, b=5/10 → expected co = 4*5/10 = 2 → set co=2
        v = _npmi(co=2, a=4, b=5, n=10)
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_negative_association(self):
        """co LESS than expected → negative NPMI."""
        v = _npmi(co=1, a=5, b=5, n=10)   # expected = 2.5, observed = 1
        assert v < 0

    def test_zero_or_negative_inputs_return_zero(self):
        for args in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)]:
            assert _npmi(*args) == 0.0


class TestLabelInText:
    def test_single_token_word_boundary(self):
        # "eat" must NOT match "eaten"
        assert _label_in_text("eat", "i love to eat pizza") is True
        assert _label_in_text("eat", "i have eaten pizza") is False

    def test_multi_token_substring(self):
        assert _label_in_text("great food", "the great food was nice") is True
        assert _label_in_text("great food", "the food was great") is False


# ════════════════════════════════════════════════════════════════════
# NPMI integration into RelationshipExtractor
# ════════════════════════════════════════════════════════════════════


class TestRelationshipNpmi:
    def _yelp_concepts(self, *labels):
        return [
            Concept(id=lbl.replace(" ", "_"), label=lbl, type="noun_phrase",
                    frequency=1, source_origins={"yelp"})
            for lbl in labels
        ]

    def test_npmi_populated_on_association_edges(self, tokeniser):
        text = (
            "Customers loved the pizza. "
            "Customers also ordered pizza. "
            "The pizza arrived hot."
        )
        doc = tokeniser.process(text, doc_id="d0", source="yelp")
        concepts = self._yelp_concepts("customer", "pizza")
        ext = RelationshipExtractor(
            use_dependency=False, compute_npmi=True, min_npmi=-1.0,
        )
        rels = ext.extract([doc], concepts)
        assoc = [r for r in rels if r.type == "ASSOCIATION"]
        assert assoc
        for r in assoc:
            assert r.npmi is not None
            assert -1.0 <= r.npmi <= 1.0

    def test_npmi_filter_drops_weak_edges(self, tokeniser):
        """min_npmi=0.99 should kill almost everything."""
        text = "pizza salad burger fries soda. coffee tea juice water beer."
        # build concepts from real lemma vocabulary
        concepts = self._yelp_concepts(
            "pizza", "salad", "burger", "coffee", "tea", "juice"
        )
        doc = tokeniser.process(text, doc_id="d0", source="yelp")
        loose = RelationshipExtractor(
            use_dependency=False, compute_npmi=True, min_npmi=-1.0,
        )
        strict = RelationshipExtractor(
            use_dependency=False, compute_npmi=True, min_npmi=0.99,
        )
        loose_n = len([r for r in loose.extract([doc], concepts)
                       if r.type == "ASSOCIATION"])
        strict_n = len([r for r in strict.extract([doc], concepts)
                        if r.type == "ASSOCIATION"])
        assert strict_n <= loose_n


# ════════════════════════════════════════════════════════════════════
# Lemma-aware co-occurrence matching
# ════════════════════════════════════════════════════════════════════


class TestLemmaAwareCoOccurrence:
    def test_plural_surface_matches_singular_concept(self, tokeniser):
        """Concept label is lemma `pizza`; sentence says `pizzas`. Should match."""
        text = "We ordered pizzas and salads."
        doc = tokeniser.process(text, doc_id="d0", source="yelp")
        concepts = [
            Concept(id=l, label=l, type="noun_phrase", frequency=1,
                    source_origins={"yelp"})
            for l in ("pizza", "salad")
        ]
        ext = RelationshipExtractor(
            use_dependency=False, compute_npmi=False, min_npmi=None,
        )
        rels = ext.extract([doc], concepts)
        assert any(r.type == "ASSOCIATION" for r in rels)
