"""Extract and classify relationships between concepts.

Two extraction signals are combined:

* **Co-occurrence** within a sliding sentence window — produces
  ``ASSOCIATION`` edges. Matching now operates over the *lemma-stream*
  text on each sentence (:attr:`ProcessedSentence.lemma_text`) so
  surface variants like ``"orders" / "ordered" / "ordering"`` no longer
  hide a co-occurrence with the lemma-canonical concept ``"order"``.
* **Dependency parsing** — produces ``ACTION`` and ``CAUSATION`` edges
  by following ``nsubj/nsubjpass`` and ``dobj/pobj/attr/oprd`` arcs.
  Subject and object matching also use lemmas, for the same reason.

After extraction, edges can optionally be re-weighted with **NPMI**
(normalised pointwise mutual information). NPMI ∈ [-1, 1] measures
*association strength* relative to chance: it is 0 when two concepts
co-occur exactly as often as random, > 0 when they associate more than
chance, < 0 when they avoid each other. Filtering by NPMI removes the
"every concept connects to the global hub" pattern that pure raw
co-occurrence counts produce.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

from src.config import CAUSAL_VERBS, CO_OCCURRENCE_WINDOW, MIN_NPMI
from src.extraction.concept_extractor import Concept
from src.preprocessing.tokeniser import ProcessedDocument


@dataclass
class Relationship:
    """A typed, weighted relationship between two concepts.

    ``verbs`` is a Counter of verb lemmas observed when this relationship was
    extracted via dependency parsing. Co-occurrence-only relationships (type
    = ``ASSOCIATION``) leave this empty. Downstream code uses it to pick a
    ``top_verb`` for display and edge embeddings.

    ``npmi`` is the normalised pointwise mutual information for the pair,
    populated when :class:`RelationshipExtractor` is run with
    ``compute_npmi=True``. It is ``None`` for edges that were not scored
    (e.g. dependency edges or runs with NPMI disabled).
    """
    source_id: str
    target_id: str
    type: str          # ASSOCIATION, ACTION, CAUSATION
    weight: float = 1.0
    directed: bool = False
    source_docs: set[str] = field(default_factory=set)
    verbs: Counter = field(default_factory=Counter)
    npmi: float | None = None


class RelationshipExtractor:
    """Extract relationships using co-occurrence and dependency parsing."""

    def __init__(
        self,
        use_cooccurrence: bool = True,
        use_dependency: bool = True,
        window_size: int = CO_OCCURRENCE_WINDOW,
        causal_verbs: set[str] | None = None,
        compute_npmi: bool = True,
        min_npmi: float | None = MIN_NPMI,
    ):
        self.use_cooccurrence = use_cooccurrence
        self.use_dependency = use_dependency
        self.window_size = window_size
        self.causal_verbs = causal_verbs or CAUSAL_VERBS
        self.compute_npmi = compute_npmi
        # ``min_npmi`` of None or <= -1 disables NPMI filtering.
        self.min_npmi = min_npmi

    # ── Public API ─────────────────────────────────────────────────
    def extract(
        self,
        documents: list[ProcessedDocument],
        concepts: list[Concept],
    ) -> list[Relationship]:
        """Extract relationships between the given concepts across all documents.

        Pipeline per call:

        1. Iterate documents, populating an edge bucket and tracking,
           for each concept, the number of windows it appeared in
           (used as the marginal probability for NPMI).
        2. Build :class:`Relationship` objects, attaching ``npmi`` to
           association edges when ``compute_npmi`` is on.
        3. Drop association edges whose NPMI falls below ``min_npmi``.
        """
        concept_labels = {c.label for c in concepts}
        concept_id_map = {c.label: c.id for c in concepts}

        edge_counter: Counter[tuple[str, str, str]] = Counter()
        edge_docs: dict[tuple[str, str, str], set[str]] = {}
        edge_verbs: dict[tuple[str, str, str], Counter] = {}

        # Marginal counts for NPMI: how many windows each concept appeared
        # in, plus the total number of windows considered. Only the
        # co-occurrence pass populates these; dependency edges don't use
        # NPMI (subject/object pair already encodes a strong signal).
        concept_windows: Counter[str] = Counter()
        total_windows = 0

        for doc in documents:
            if self.use_cooccurrence:
                added = self._extract_cooccurrence(
                    doc, concept_labels, concept_id_map,
                    edge_counter, edge_docs, concept_windows,
                )
                total_windows += added
            if self.use_dependency:
                self._extract_dependency(
                    doc, concept_labels, concept_id_map,
                    edge_counter, edge_docs, edge_verbs,
                )

        relationships: list[Relationship] = []
        for (src, tgt, rtype), weight in edge_counter.items():
            npmi: float | None = None
            if (
                self.compute_npmi
                and rtype == "ASSOCIATION"
                and total_windows > 0
            ):
                npmi = _npmi(
                    co=weight,
                    a=concept_windows[self._reverse_id(concepts, src)],
                    b=concept_windows[self._reverse_id(concepts, tgt)],
                    n=total_windows,
                )
            relationships.append(Relationship(
                source_id=src,
                target_id=tgt,
                type=rtype,
                weight=weight,
                directed=(rtype != "ASSOCIATION"),
                source_docs=edge_docs.get((src, tgt, rtype), set()),
                verbs=edge_verbs.get((src, tgt, rtype), Counter()),
                npmi=npmi,
            ))

        # NPMI filter (only meaningful on ASSOCIATION edges; dependency
        # edges are kept regardless because they carry verb-typed signal).
        if self.min_npmi is not None and self.min_npmi > -1.0:
            relationships = [
                r for r in relationships
                if r.type != "ASSOCIATION"
                or r.npmi is None
                or r.npmi >= self.min_npmi
            ]

        relationships.sort(key=lambda r: r.weight, reverse=True)
        return relationships

    # ── Co-occurrence ──────────────────────────────────────────────
    def _extract_cooccurrence(
        self,
        doc: ProcessedDocument,
        concept_labels: set[str],
        id_map: dict[str, str],
        counter: Counter,
        docs_map: dict,
        concept_windows: Counter,
    ) -> int:
        """Link concepts co-occurring within a sliding sentence window.

        Returns the number of windows considered for this document, so
        the caller can sum a corpus-wide ``total_windows`` for NPMI.
        """
        n_sents = len(doc.sentences)
        windows_seen = 0
        for i in range(n_sents):
            window_sents = doc.sentences[i: i + self.window_size]
            # Lemma-aware match: each sentence already exposes a
            # whitespace-joined lemma stream, so "ordered" / "ordering"
            # both match concept "order".
            window_text = " ".join(s.lemma_text for s in window_sents)
            if not window_text:
                continue
            windows_seen += 1

            present: list[str] = [
                c for c in concept_labels if _label_in_text(c, window_text)
            ]
            if not present:
                continue
            for c in present:
                concept_windows[c] += 1
            for a, b in combinations(sorted(present), 2):
                key = (id_map[a], id_map[b], "ASSOCIATION")
                counter[key] += 1
                docs_map.setdefault(key, set()).add(doc.doc_id)
        return windows_seen

    # ── Dependency parsing ─────────────────────────────────────────
    def _extract_dependency(
        self,
        doc: ProcessedDocument,
        concept_labels: set[str],
        id_map: dict[str, str],
        counter: Counter,
        docs_map: dict,
        verbs_map: dict,
    ) -> None:
        """Extract SVO triples from dependency parses and classify by verb type."""
        spacy_doc = doc.spacy_doc
        if spacy_doc is None:
            return

        for sent in spacy_doc.sents:
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                subjects = [
                    child for child in token.children
                    if child.dep_ in ("nsubj", "nsubjpass")
                ]
                objects = [
                    child for child in token.children
                    if child.dep_ in ("dobj", "pobj", "attr", "oprd")
                ]

                for subj in subjects:
                    subj_text = self._match_concept(subj, concept_labels)
                    if not subj_text:
                        continue
                    for obj in objects:
                        obj_text = self._match_concept(obj, concept_labels)
                        if not obj_text or subj_text == obj_text:
                            continue

                        verb_lemma = token.lemma_.lower()
                        if verb_lemma in self.causal_verbs:
                            rtype = "CAUSATION"
                        else:
                            rtype = "ACTION"

                        key = (id_map[subj_text], id_map[obj_text], rtype)
                        counter[key] += 1
                        docs_map.setdefault(key, set()).add(doc.doc_id)
                        verbs_map.setdefault(key, Counter())[verb_lemma] += 1

    @staticmethod
    def _match_concept(token, concept_labels: set[str]) -> str | None:
        """Match a token (or its subtree) to a known lemma-canonical concept.

        Tries, in order:
        1. The lemma of the entire subtree (multi-word concept like
           ``"great food"``).
        2. The token lemma alone.
        3. Falls back to surface forms in case the corpus contains
           concepts that were not lemma-normalised (legacy data).
        """
        subtree_lemma = " ".join(
            t.lemma_.lower()
            for t in token.subtree
            if not t.is_punct and not t.is_space
        ).strip()
        if subtree_lemma in concept_labels:
            return subtree_lemma

        token_lemma = token.lemma_.lower().strip()
        if token_lemma in concept_labels:
            return token_lemma

        # Surface-form fallback (kept for compatibility with concept lists
        # produced before lemma normalisation, e.g. older test fixtures).
        chunk_text = " ".join(t.text for t in token.subtree).lower().strip()
        if chunk_text in concept_labels:
            return chunk_text
        token_text = token.text.lower().strip()
        if token_text in concept_labels:
            return token_text
        return None

    # ── Internals ──────────────────────────────────────────────────
    @staticmethod
    def _reverse_id(concepts: list[Concept], cid: str) -> str:
        """Map a concept id back to its label (for NPMI lookups)."""
        for c in concepts:
            if c.id == cid:
                return c.label
        return ""


# ── Helpers ───────────────────────────────────────────────────────────


def _label_in_text(label: str, lemma_text: str) -> bool:
    """Word-boundary aware ``in`` check.

    For a single-token concept we require a whole-word match so that
    ``"eat"`` doesn't match ``"eaten"`` (lemma already collapses tense,
    so any leftover suffix is genuinely a different word).

    For a multi-token concept we use plain substring containment, which
    is fine because both sides are space-separated lemma streams.
    """
    if " " in label:
        return label in lemma_text
    return f" {label} " in f" {lemma_text} "


def _npmi(co: int, a: int, b: int, n: int) -> float:
    """Normalised pointwise mutual information for a co-occurrence pair.

    Returns 0.0 if any input is non-positive (numerically degenerate) or
    if the result would be undefined (perfect co-occurrence with both
    marginals == co-occurrence count, giving log(1) / -log(p) issues).
    """
    if co <= 0 or a <= 0 or b <= 0 or n <= 0:
        return 0.0
    p_a = a / n
    p_b = b / n
    p_ab = co / n
    if p_ab <= 0 or p_ab >= 1:
        return 0.0
    pmi = math.log(p_ab / (p_a * p_b))
    denom = -math.log(p_ab)
    if denom == 0:
        return 0.0
    return pmi / denom
