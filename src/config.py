"""Global configuration for the Text-to-Network Engine."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# Auto-load .env from project root so any entry point (pipeline.py, chat.py,
# streamlit, pytest) sees the same environment without needing `source .env`.
#
# Default behaviour: existing process env vars take precedence; .env only
# fills in keys that are not already set.
#
# For credentials and provider selectors that we want to be able to
# *swap* by editing .env (without unsetting shell exports first), we
# unconditionally override. This stops a stale `export DASHSCOPE_API_KEY=…`
# in ~/.zshrc from silently masking a fresher value the user just put
# in .env. To opt out, set ``DOTENV_RESPECT_SHELL=1``.
_DOTENV_OVERRIDE_KEYS: set[str] = {
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "HUGGINGFACE_API_TOKEN",
    "HF_TOKEN",
    "EMBEDDING_MODEL",
}


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    respect_shell = os.environ.get("DOTENV_RESPECT_SHELL", "").lower() in {
        "1", "true", "yes",
    }
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            should_override = (
                key not in os.environ
                or (key in _DOTENV_OVERRIDE_KEYS and not respect_shell)
            )
            if should_override:
                os.environ[key] = value
    except Exception:
        pass


_load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
CONCEPT_DICTIONARY_PATH = CONFIG_DIR / "concept_dictionary.yaml"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Data source paths ──────────────────────────────────────────────
POLICY_PDF_PATH = DATA_DIR / "national-climate-resilience-and-adaptation-strategy.pdf"
YELP_EXTRACTED_DIR = DATA_DIR / "Yelp JSON" / "yelp_dataset"

# ── spaCy model ────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── Preprocessing ──────────────────────────────────────────────────
# Universal stopwords — discourse glue and abstract pronouns that carry no
# topical meaning regardless of corpus. Kept tight so domain stopwords
# below can stay opt-in per source.
CUSTOM_STOP_WORDS: set[str] = {
    "also", "however", "including", "well", "within", "across",
    "example", "e.g.", "i.e.", "etc", "page", "figure", "table",
    "this", "that", "these", "those", "they", "them", "which",
    "what", "some", "both", "each", "every", "other", "another",
    "everyone", "everything", "something", "nothing", "anything",
    "someone", "anyone", "nobody", "many", "much", "more", "most",
    "such", "very", "quite", "really", "actually", "just", "even",
    "still", "already", "often", "always", "never",
}
PRESERVE_STOP_WORDS: set[str] = {
    "shall", "must", "will", "should", "may",
}

# Domain-specific noise. These words are individually meaningful, but
# they appear in nearly every document of the matching corpus and
# therefore drown out genuine signal in centrality / community metrics
# (every restaurant review mentions "eat", "food", "place").
#
# Filtered at concept-extraction time, after lemmatisation. Add only
# words you're sure carry no analytical value for the corpus in question.
DOMAIN_STOP_WORDS_YELP: set[str] = {
    # generic verbs
    "eat", "go", "come", "get", "make", "take", "give", "try", "want",
    "know", "think", "say", "tell", "look", "see", "feel", "find",
    "ask", "leave", "wait", "put", "let", "use", "love", "like",
    # broad action verbs that dominate cross-source SPOF / Hub cards
    # because they are equally generic in policy ("work plan",
    # "needs assessment") and yelp ("doesn't work", "really need").
    "work", "need",
    # generic nouns
    "place", "time", "day", "night", "thing", "way", "lot", "people",
    "guy", "person", "kind", "sort", "type", "bit", "minute", "hour",
    "week", "month", "year", "moment", "side", "back", "front", "end",
    "part", "stuff",
    # generic restaurant nouns (kept liberal — flip individually if business
    # cares about "service" vs. "food" as standalone topics)
    "food", "meal", "dish", "menu", "table", "order",
    # generic adjectives that survive POS filtering as standalone keywords
    "good", "great", "nice", "okay", "ok", "fine", "decent", "bad",
    "amazing", "awesome", "delicious", "tasty", "fresh", "hot", "cold",
    "small", "big", "little", "long", "short",
}

DOMAIN_STOP_WORDS_POLICY: set[str] = {
    # document structure
    "section", "clause", "paragraph", "subsection", "schedule", "appendix",
    "annex", "preamble", "chapter", "article",
    # generic policy verbs / nouns
    "policy", "policies", "regulation", "regulations", "framework",
    "approach", "objective", "outcome", "principle",
}


def stopwords_for(source: str | None) -> set[str]:
    """Return the union of universal + domain-specific stopwords for a source.

    ``source`` is the per-document tag set by the ingest layer:
    ``"yelp"``, ``"policy"``, ``"combined"`` or ``"general"``. Unknown
    sources fall back to the universal set only.
    """
    base = set(CUSTOM_STOP_WORDS)
    s = (source or "").lower()
    if s == "yelp":
        return base | DOMAIN_STOP_WORDS_YELP
    if s == "policy":
        return base | DOMAIN_STOP_WORDS_POLICY
    if s in ("combined", "general", "both"):
        return base | DOMAIN_STOP_WORDS_YELP | DOMAIN_STOP_WORDS_POLICY
    return base


# ── Concept extraction ─────────────────────────────────────────────
NER_LABELS = {"PERSON", "ORG", "GPE", "EVENT", "NORP", "FAC", "LOC", "PRODUCT", "LAW"}
MIN_CONCEPT_FREQ = 3                  # bumped from 2 — drops singleton noise
TFIDF_TOP_N = 100

# Extra protection against NER PERSON noise (review datasets are full of
# first names mentioned a handful of times each, e.g. "Eli's salad").
# A PERSON entity is only kept as a concept if it appears at least this
# many times across the entire corpus.
MIN_PERSON_FREQ = 10

# Per-NER-type minimum frequency gates. PERSON / ORG / PRODUCT all leak
# brand-name and proper-noun fragments into the network in review data
# ("wendy", "uber", "apple"); the gate keeps the noise out without
# losing genuinely high-volume entities. PERSON's entry mirrors
# ``MIN_PERSON_FREQ`` for backward compatibility — the standalone
# ``MIN_PERSON_FREQ`` constant is still exposed.
#
# Any NER label not listed here falls back to the global
# ``MIN_CONCEPT_FREQ`` gate, so adding new labels here is purely
# subtractive: it tightens the threshold, never relaxes it.
MIN_ENTITY_FREQ: dict[str, int] = {
    "PERSON":  MIN_PERSON_FREQ,
    "ORG":     10,
    "PRODUCT": 10,
}

# TF-IDF tuning. ``TFIDF_MAX_DF`` drops ngrams that appear in more than
# this fraction of documents (high-DF tokens carry little discriminative
# signal). ``TFIDF_MIN_DF`` drops ngrams that appear in fewer than this
# many documents (one-off noise). 0.5 / 3 is conservative for corpora
# of >= 100 documents.
TFIDF_MAX_DF = 0.5
TFIDF_MIN_DF = 3

# ── Relationship extraction ────────────────────────────────────────
CO_OCCURRENCE_WINDOW = 2          # sentences
CAUSAL_VERBS = {
    "cause", "lead", "result", "drive", "enable", "support",
    "contribute", "influence", "affect", "impact", "reduce",
    "increase", "improve", "strengthen", "weaken", "threaten",
    "promote", "facilitate", "hinder", "prevent",
}

# ── Network construction ───────────────────────────────────────────
# Minimum raw co-occurrence count required for an edge to enter the
# graph. Bumping from 1 → 2 cuts coincidental pairs without losing any
# repeated association. Pipeline CLI default lives in ``pipeline.py``
# (``--min-weight``) and overrides this constant when set.
MIN_EDGE_WEIGHT = 2

# Minimum normalised pointwise mutual information for an edge to survive.
# NPMI ∈ [-1, 1]: 1 = perfectly co-occurring, 0 = independent, < 0 =
# negatively associated. A threshold of 0.0 keeps only positively
# associated pairs and aggressively prunes "everything pairs with the
# global hubs" noise. Set to None / 0.0 to disable.
MIN_NPMI = 0.0

# ── Visualisation ──────────────────────────────────────────────────
TOP_N_DISPLAY = 30
GRAPH_LAYOUT = "spring"

# ── LLM / RAG chatbot (Stage 8) ────────────────────────────────────
# Provider is one of:
#   "openai" | "dashscope" | "huggingface_api" | "huggingface_local" | "echo"
# "echo" is a zero-dependency fallback that returns retrieved context verbatim —
# useful for local testing when no LLM keys are configured.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "echo")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# API keys are read from the environment by each provider; centralised here
# only so callers can check whether a provider is configured.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "") or os.getenv(
    "HF_TOKEN", ""
)
# DashScope (Aliyun ModelScope) — uses the OpenAI-compatible endpoint so we
# can reuse the `openai` Python client. Default model is qwen-turbo, override
# via LLM_MODEL (e.g. qwen-plus, qwen-max, qwen2.5-72b-instruct).
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ── Neo4j backend (required) ───────────────────────────────────────
# Neo4j is the single source of truth for the conceptual network. The
# pipeline, dashboard and chatbot all read from / write to it directly,
# and there is no NetworkX fallback. NEO4J_PASSWORD must be set in your
# environment (or .env) before any entry point will run.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_VECTOR_INDEX = os.getenv("NEO4J_VECTOR_INDEX", "concept_embedding")
NEO4J_EDGE_VECTOR_INDEX = os.getenv("NEO4J_EDGE_VECTOR_INDEX", "related_embedding")
NEO4J_EMBEDDING_DIM = int(os.getenv("NEO4J_EMBEDDING_DIM", "384"))


def require_neo4j_config() -> None:
    """Hard-fail if Neo4j env vars aren't configured.

    Called by every CLI entry point (``pipeline.py``, ``chat.py``,
    ``dashboard.py``) so misconfiguration surfaces immediately rather
    than silently degrading.
    """
    missing: list[str] = []
    if not NEO4J_URI:
        missing.append("NEO4J_URI")
    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")
    if missing:
        raise RuntimeError(
            "Neo4j is required but not configured. Missing environment "
            f"variables: {', '.join(missing)}. "
            "Copy `.env.example` to `.env`, set NEO4J_PASSWORD, and run "
            "`docker compose up -d`."
        )
