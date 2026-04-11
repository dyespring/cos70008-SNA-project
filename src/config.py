"""Global configuration for the Text-to-Network Engine."""

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
