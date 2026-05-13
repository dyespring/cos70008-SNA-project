"""spaCy-based tokenisation, lemmatisation, and linguistic processing.

Two upstream cleanups happen here that the rest of the pipeline depends on:

1. **Noun chunks are cleaned**: leading determiners / pronouns are stripped
   and the remaining tokens are lemmatised before being stored as a single
   string. So spaCy's raw chunk ``"the great food"`` becomes ``"great food"``,
   and ``"this place"`` becomes ``"place"``. This stops the network from
   treating ``"the food"`` and ``"food"`` as separate concepts.

2. **A lemma-stream version of each sentence is precomputed** and stored as
   :attr:`ProcessedSentence.lemma_text`. Downstream relationship extraction
   uses it for substring matching against (lemmatised) concept labels, so
   tense and pluralisation no longer hide co-occurrences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import spacy
from spacy.tokens import Doc, Span

from src.config import SPACY_MODEL, CUSTOM_STOP_WORDS, PRESERVE_STOP_WORDS


# Token POS / lower-form prefixes to peel off the front of a noun chunk
# before lemmatising. spaCy's ``noun_chunks`` keeps these as part of the
# span, which is exactly what we don't want at the concept layer.
_LEADING_STRIP_POS = {"DET", "PRON"}
_LEADING_STRIP_WORDS = {
    "the", "a", "an",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no", "every", "each",
    "another", "such",
}

# Coarse-grained POS labels that are acceptable as the *syntactic head*
# of a real concept. Anything else is rejected. spaCy's ``noun_chunks``
# is permissive — it returns spans whose root is PRON ("I", "they"),
# AUX/VERB ("can" mis-tagged), or NUM, none of which carry topical
# meaning for a network analysis.
_HEAD_OK_POS = {"NOUN", "PROPN"}

# Possessive / apostrophe markers that survive ``is_punct`` filtering.
# spaCy tokenises Saxon genitive "'s" as a separate ``PART`` token
# embedded in the middle of the chunk (``[John, 's, pizza]``), so we
# strip these from *anywhere* in the chunk — not just the tail.
_POSSESSIVE_FORMS = {"'s", "s'", "'", "’s", "’", "`s", "`"}

# Single-token chunks whose lemma is in this set are vacuous noise even
# when spaCy tags them as NOUN. These come from substantive uses of
# numerals ("the one I love" → ``one`` / NOUN) and generic anaphora
# ("the thing is" → ``thing`` / NOUN). They would otherwise dominate
# centrality metrics as artificial hubs.
_SINGLE_TOKEN_NOISE = {
    # cardinal numerals as words
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "dozen", "hundred", "thousand", "million",
    "billion",
    # ordinals
    "first", "second", "third", "fourth", "fifth", "sixth",
    # generic anaphora not already in CUSTOM_STOP_WORDS
    "ones", "kind", "sort",
}


def _clean_chunk(chunk: Span) -> str:
    """Return a normalised, lemmatised string for a noun chunk.

    Pipeline:

    1. **Pre-cleanup root gate** — fast reject when spaCy's chunk root
       is itself the wrong POS (PRON ``"I"``, NUM, AUX, VERB).
    2. **Leading strip** — peel determiners / demonstratives /
       possessives ("the", "this", "my", …) from the front.
    3. **Mid-chunk filter** — drop punctuation, whitespace, and Saxon
       genitive ``'s`` markers (as PART) wherever they appear, so
       ``"John's pizza"`` collapses to ``[John, pizza]`` instead of
       leaking ``"john 's pizza"``.
    4. **Post-cleanup head gate** — verify the right-most surviving
       token is itself NOUN/PROPN (the cleanup above can change which
       token is the effective head, e.g. after stripping ``"the"``
       from ``"the one"`` the leftover ``"one"`` is the new head).
    5. **Single-token noise gate** — reject chunks that collapse to
       exactly one token in :data:`_SINGLE_TOKEN_NOISE` (numerals,
       generic anaphora) even when POS-tagged NOUN.
    6. **Lemma join** — return the surviving tokens as a single
       lowercase, lemmatised string.

    Returns an empty string if any layer empties the chunk.
    """
    # 1. Fast pre-cleanup rejection on chunk root POS.
    if chunk.root is None or chunk.root.pos_ not in _HEAD_OK_POS:
        return ""

    tokens = list(chunk)

    # 2. Peel off leading DET/PRON or known fillers ("the", "a", "this", …).
    while tokens and (
        tokens[0].pos_ in _LEADING_STRIP_POS
        or tokens[0].lower_ in _LEADING_STRIP_WORDS
    ):
        tokens.pop(0)

    # 3. Drop punctuation, whitespace, and possessive markers anywhere
    # in the chunk (Leak 2 fix — Saxon genitive 's appears mid-chunk
    # as PART, not at the tail).
    cleaned = [
        t for t in tokens
        if not t.is_punct
        and not t.is_space
        and t.lemma_.strip()
        and t.lower_ not in _POSSESSIVE_FORMS
        and t.lemma_.lower() not in _POSSESSIVE_FORMS
    ]

    if not cleaned:
        return ""

    # 4. Post-cleanup head gate. After stripping "the" from "the one"
    # the new head is "one" (NOUN-tagged but vacuous); after stripping
    # "my" from "my favourite dish" the head is "dish" (NOUN, kept).
    if cleaned[-1].pos_ not in _HEAD_OK_POS:
        return ""

    # 5. Single-token noise blocklist (Leak 1 fix — numerals as words).
    if len(cleaned) == 1 and cleaned[0].lemma_.lower() in _SINGLE_TOKEN_NOISE:
        return ""

    return " ".join(t.lemma_.lower() for t in cleaned).strip()


def _sentence_lemma_text(sent: Span) -> str:
    """Build a whitespace-joined lemma stream for substring matching."""
    parts: list[str] = []
    for tok in sent:
        if tok.is_punct or tok.is_space:
            continue
        parts.append(tok.lemma_.lower())
    return " ".join(parts)


@dataclass
class ProcessedSentence:
    """One sentence with its token-level annotations.

    ``noun_chunks`` already contains *cleaned, lemmatised* chunk strings
    (see :func:`_clean_chunk`). ``lemma_text`` is the full sentence as a
    space-joined lemma sequence and is what relationship extraction
    matches concept labels against.
    """
    text: str
    tokens: list[str]
    lemmas: list[str]
    pos_tags: list[str]
    entities: list[tuple[str, str]]   # (text, label)
    noun_chunks: list[str]
    lemma_text: str = ""
    spacy_span: object = field(default=None, repr=False)


@dataclass
class ProcessedDocument:
    """A fully processed document ready for concept/relationship extraction."""
    doc_id: str
    source: str
    sentences: list[ProcessedSentence]
    spacy_doc: object = field(default=None, repr=False)

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def all_lemmas(self) -> list[str]:
        return [lemma for s in self.sentences for lemma in s.lemmas]


class SpacyTokeniser:
    """Wraps a spaCy model to produce :class:`ProcessedDocument` objects."""

    def __init__(self, model_name: str | None = None):
        self.nlp = spacy.load(model_name or SPACY_MODEL)
        self._configure_stop_words()

    def _configure_stop_words(self) -> None:
        for word in CUSTOM_STOP_WORDS:
            self.nlp.vocab[word].is_stop = True
        for word in PRESERVE_STOP_WORDS:
            self.nlp.vocab[word].is_stop = False

    def process(
        self, text: str, doc_id: str = "doc_0", source: str = "general"
    ) -> ProcessedDocument:
        """Process a single document string into a :class:`ProcessedDocument`."""
        doc: Doc = self.nlp(text)
        return self._spacy_doc_to_processed(doc, doc_id, source)

    def process_batch(
        self,
        texts: list[tuple[str, str]],
        source: str = "general",
        batch_size: int = 50,
    ) -> list[ProcessedDocument]:
        """Process multiple ``(doc_id, text)`` pairs efficiently."""
        ids = [t[0] for t in texts]
        raw = [t[1] for t in texts]
        out: list[ProcessedDocument] = []
        for i, spacy_doc in enumerate(self.nlp.pipe(raw, batch_size=batch_size)):
            out.append(self._spacy_doc_to_processed(spacy_doc, ids[i], source))
        return out

    # ── Internals ──────────────────────────────────────────────────
    def _spacy_doc_to_processed(
        self, doc: Doc, doc_id: str, source: str
    ) -> ProcessedDocument:
        sentences: list[ProcessedSentence] = []
        for sent in doc.sents:
            tokens: list[str] = []
            lemmas: list[str] = []
            pos_tags: list[str] = []
            for tok in sent:
                if tok.is_space:
                    continue
                tokens.append(tok.text)
                pos_tags.append(tok.pos_)
                if not tok.is_stop and not tok.is_punct and len(tok.text) > 1:
                    lemmas.append(tok.lemma_.lower())

            entities = [(ent.text, ent.label_) for ent in sent.ents]

            noun_chunks: list[str] = []
            seen_chunks: set[str] = set()
            for chunk in sent.noun_chunks:
                cleaned = _clean_chunk(chunk)
                if not cleaned or cleaned in seen_chunks:
                    continue
                # Drop chunks that collapse to a single short token like "it"
                if len(cleaned) < 2:
                    continue
                noun_chunks.append(cleaned)
                seen_chunks.add(cleaned)

            sentences.append(ProcessedSentence(
                text=sent.text.strip(),
                tokens=tokens,
                lemmas=lemmas,
                pos_tags=pos_tags,
                entities=entities,
                noun_chunks=noun_chunks,
                lemma_text=_sentence_lemma_text(sent),
                spacy_span=sent,
            ))
        return ProcessedDocument(
            doc_id=doc_id,
            source=source,
            sentences=sentences,
            spacy_doc=doc,
        )
