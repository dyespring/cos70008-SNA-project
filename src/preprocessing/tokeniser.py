"""spaCy-based tokenisation, lemmatisation, and linguistic processing."""

from __future__ import annotations

from dataclasses import dataclass, field

import spacy
from spacy.tokens import Doc

from src.config import SPACY_MODEL, CUSTOM_STOP_WORDS, PRESERVE_STOP_WORDS


@dataclass
class ProcessedSentence:
    """One sentence with its token-level annotations."""
    text: str
    tokens: list[str]
    lemmas: list[str]
    pos_tags: list[str]
    entities: list[tuple[str, str]]   # (text, label)
    noun_chunks: list[str]
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
    """Wraps a spaCy model to produce ProcessedDocument objects."""

    def __init__(self, model_name: str | None = None):
        self.nlp = spacy.load(model_name or SPACY_MODEL)
        self._configure_stop_words()

    def _configure_stop_words(self) -> None:
        for word in CUSTOM_STOP_WORDS:
            self.nlp.vocab[word].is_stop = True
        for word in PRESERVE_STOP_WORDS:
            self.nlp.vocab[word].is_stop = False

    def process(self, text: str, doc_id: str = "doc_0", source: str = "general") -> ProcessedDocument:
        """Process a single document string into a ProcessedDocument."""
        doc: Doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            tokens = []
            lemmas = []
            pos_tags = []
            for tok in sent:
                if tok.is_space:
                    continue
                tokens.append(tok.text)
                pos_tags.append(tok.pos_)
                if not tok.is_stop and not tok.is_punct and len(tok.text) > 1:
                    lemmas.append(tok.lemma_.lower())

            entities = [(ent.text, ent.label_) for ent in sent.ents]
            noun_chunks = [
                chunk.text for chunk in sent.noun_chunks
                if len(chunk.text.split()) >= 1
            ]

            sentences.append(ProcessedSentence(
                text=sent.text.strip(),
                tokens=tokens,
                lemmas=lemmas,
                pos_tags=pos_tags,
                entities=entities,
                noun_chunks=noun_chunks,
                spacy_span=sent,
            ))
        return ProcessedDocument(
            doc_id=doc_id,
            source=source,
            sentences=sentences,
            spacy_doc=doc,
        )

    def process_batch(
        self,
        texts: list[tuple[str, str]],
        source: str = "general",
        batch_size: int = 50,
    ) -> list[ProcessedDocument]:
        """Process multiple (doc_id, text) pairs efficiently using nlp.pipe."""
        docs = []
        ids = [t[0] for t in texts]
        raw = [t[1] for t in texts]
        for i, spacy_doc in enumerate(self.nlp.pipe(raw, batch_size=batch_size)):
            pd = self._spacy_doc_to_processed(spacy_doc, ids[i], source)
            docs.append(pd)
        return docs

    def _spacy_doc_to_processed(self, doc: Doc, doc_id: str, source: str) -> ProcessedDocument:
        sentences = []
        for sent in doc.sents:
            tokens = []
            lemmas = []
            pos_tags = []
            for tok in sent:
                if tok.is_space:
                    continue
                tokens.append(tok.text)
                pos_tags.append(tok.pos_)
                if not tok.is_stop and not tok.is_punct and len(tok.text) > 1:
                    lemmas.append(tok.lemma_.lower())
            entities = [(ent.text, ent.label_) for ent in sent.ents]
            noun_chunks = [chunk.text for chunk in sent.noun_chunks if len(chunk.text.split()) >= 1]
            sentences.append(ProcessedSentence(
                text=sent.text.strip(),
                tokens=tokens,
                lemmas=lemmas,
                pos_tags=pos_tags,
                entities=entities,
                noun_chunks=noun_chunks,
                spacy_span=sent,
            ))
        return ProcessedDocument(doc_id=doc_id, source=source, sentences=sentences, spacy_doc=doc)
