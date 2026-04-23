"""Global configuration for the Text-to-Network Engine."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
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

# ── Concept extraction ─────────────────────────────────────────────
NER_LABELS = {"PERSON", "ORG", "GPE", "EVENT", "NORP", "FAC", "LOC", "PRODUCT", "LAW"}
MIN_CONCEPT_FREQ = 2
TFIDF_TOP_N = 100

# ── Relationship extraction ────────────────────────────────────────
CO_OCCURRENCE_WINDOW = 2          # sentences
CAUSAL_VERBS = {
    "cause", "lead", "result", "drive", "enable", "support",
    "contribute", "influence", "affect", "impact", "reduce",
    "increase", "improve", "strengthen", "weaken", "threaten",
    "promote", "facilitate", "hinder", "prevent",
}

# ── Network construction ───────────────────────────────────────────
MIN_EDGE_WEIGHT = 1

# ── Visualisation ──────────────────────────────────────────────────
TOP_N_DISPLAY = 30
GRAPH_LAYOUT = "spring"

# ── LLM / RAG chatbot (Stage 8) ────────────────────────────────────
# Provider is one of: "openai", "huggingface_api", "huggingface_local", "echo".
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
