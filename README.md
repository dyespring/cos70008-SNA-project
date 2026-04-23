# Text-to-Network Engine

**From Text to Networks: Building a Conceptual Network Engine for Data Science and Social Network Analysis**

A Python pipeline that transforms unstructured text into conceptual networks for social network analysis (SNA). Built for SNA Toolbox as part of COS70008 Technology Innovation Project.

## Overview

The engine ingests raw text (policy documents, Yelp reviews, or narrative text), extracts key concepts and their relationships using NLP techniques, constructs a network graph, and applies SNA methods to produce actionable insights.

## Pipeline Stages

1. **Data Ingestion** — Read PDF policy documents or Yelp JSON reviews
2. **Preprocessing** — Clean, tokenise, lemmatise, and filter text
3. **Concept & Relationship Extraction** — NER, noun-phrase chunking, TF-IDF; co-occurrence windows, dependency parsing, and causal pattern matching
4. **Network Construction** — Build NetworkX graphs with typed, weighted edges
5. **Network Analysis** — Centrality, community detection, brokerage, path analysis
6. **Visualisation** — Static matplotlib plots and interactive pyvis HTML graphs
7. **Temporal Comparison** *(optional)* — Build separate networks per time slice / document section and compare overlap and drift
8. **Conversational Exploration** *(optional)* — LLM/RAG chatbot grounded in the conceptual network

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the full pipeline on the policy document
python pipeline.py --source policy --output results/

# Run on Yelp reviews (sampled)
python pipeline.py --source yelp --yelp-category Restaurants --yelp-sample 500 --output results/

# Add temporal comparison (3 slices) and sentiment-weighted edges
python pipeline.py --source combined --temporal 3 --sentiment --output results/

# Launch the interactive Streamlit dashboard (5 analysis tabs + Temporal + Chat)
streamlit run src/extensions/dashboard.py

# Stage 8: conversational Q&A over the knowledge graph
python chat.py --source policy              # standalone
python pipeline.py --source policy --chat   # appended to the main pipeline
```

### LLM / chatbot configuration

The chatbot (`chat.py`, `--chat`, and the dashboard **Chat** tab) is
provider-agnostic. Select a backend via `LLM_PROVIDER`:

| Provider             | Env vars                        | Install                       |
|----------------------|----------------------------------|-------------------------------|
| `echo` *(default)*   | none                             | none — no-LLM fallback        |
| `openai`             | `OPENAI_API_KEY`, `LLM_MODEL`    | `pip install openai`          |
| `huggingface_api`    | `HUGGINGFACE_API_TOKEN`, `LLM_MODEL` | `pip install huggingface_hub` |
| `huggingface_local`  | `LLM_MODEL` (local model id)    | `pip install transformers`    |

Optional semantic retrieval (improves routing for paraphrased questions)
requires `sentence-transformers` + `faiss-cpu` (already in `requirements.txt`).

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python chat.py --source combined --vector-store
```

## Project Structure

```
src/
  config.py               # Global settings and parameters (including LLM/embedding)
  ingest/                 # Data readers (PDF, Yelp JSON)
  preprocessing/          # Text cleaning and tokenisation
  extraction/             # Concept and relationship extraction
  network/                # Graph construction and SNA analysis
  visualisation/          # Static and interactive visualisations
  extensions/             # Sentiment, concept dictionary, temporal,
                          # Streamlit dashboard, chatbot (Stage 8)
notebooks/                # Jupyter notebooks for exploration and demos
config/                   # User-editable concept_dictionary.yaml
tests/                    # Unit tests
pipeline.py               # CLI entry point (stages 1-8)
chat.py                   # Standalone Stage-8 chatbot entry point
```

## Optional Extensions

- **Sentiment-weighted edges** — Attach VADER/TextBlob sentiment scores to relationships (`--sentiment`)
- **User-defined concept dictionary** — Analyst-controlled vocabularies (`--concept-dictionary default`, see `config/concept_dictionary.yaml`)
- **Temporal comparison** — Compare networks across document sections or time periods (`--temporal N`, or the Temporal tab in the dashboard)
- **Interactive dashboard** — Streamlit app for exploring networks in the browser, including a Chat tab wired to the Stage-8 chatbot
- **LLM/RAG chatbot (Stage 8)** — Conversational Q&A grounded in the graph, with pluggable OpenAI / Hugging Face / local providers (`python chat.py`, `--chat`, or the dashboard Chat tab)

## Data Sources

- Australian National Climate Resilience and Adaptation Strategy (2021–2025)
- Yelp Open Dataset (academic review subset)

**Local setup:** Large files are gitignored. After clone, place the policy PDF and/or extracted Yelp JSON under `data/` as described in [`data/README.md`](data/README.md). Configured paths are defined in `src/config.py` (`POLICY_PDF_PATH`, `YELP_EXTRACTED_DIR`).

**GitHub hygiene:** Do not commit `venv/` (reinstall from `requirements.txt`). The `papers/`, `project_brief/`, and `Research_Ethics/` folders are excluded via `.gitignore` for this upload.
