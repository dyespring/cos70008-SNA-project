# Text-to-Network Engine

**From Text to Networks: Building a Conceptual Network Engine for Data Science and Social Network Analysis**

A Python pipeline that transforms unstructured text into conceptual networks for social network analysis (SNA). Built for SNA Toolbox as part of COS70008 Technology Innovation Project.

## Overview

The engine ingests raw text (policy documents, Yelp reviews, or narrative text), extracts key concepts and their relationships using NLP techniques, constructs a network graph, and applies SNA methods to produce actionable insights.

## Pipeline Stages

1. **Data Ingestion** — Read PDF policy documents or Yelp JSON reviews
2. **Preprocessing** — Clean, tokenise, lemmatise, and filter text
3. **Concept Extraction** — NER, noun-phrase chunking, and TF-IDF keyword extraction
4. **Relationship Extraction** — Co-occurrence windows, dependency parsing, and causal pattern matching
5. **Network Construction** — Build NetworkX graphs with typed, weighted edges
6. **Network Analysis** — Centrality, community detection, brokerage, path analysis
7. **Visualisation** — Static matplotlib plots and interactive pyvis HTML graphs

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the full pipeline on the policy document
python pipeline.py --source policy --output results/

# Run on Yelp reviews (sampled)
python pipeline.py --source yelp --yelp-category Restaurants --yelp-sample 500 --output results/
```

## Project Structure

```
src/
  config.py               # Global settings and parameters
  ingest/                 # Data readers (PDF, Yelp JSON)
  preprocessing/          # Text cleaning and tokenisation
  extraction/             # Concept and relationship extraction
  network/                # Graph construction and SNA analysis
  visualisation/          # Static and interactive visualisations
  extensions/             # Sentiment, temporal, Streamlit dashboard
notebooks/                # Jupyter notebooks for exploration and demos
tests/                    # Unit tests
pipeline.py               # CLI entry point
```

## Optional Extensions

- **Sentiment-weighted edges** — Attach VADER/TextBlob sentiment scores to relationships
- **Temporal comparison** — Compare networks across document sections or time periods
- **Interactive dashboard** — Streamlit app for exploring networks in the browser

## Data Sources

- Australian National Climate Resilience and Adaptation Strategy (2021–2025)
- Yelp Open Dataset (academic review subset)

**Local setup:** Large files are gitignored. After clone, place the policy PDF and/or extracted Yelp JSON under `data/` as described in [`data/README.md`](data/README.md). Configured paths are defined in `src/config.py` (`POLICY_PDF_PATH`, `YELP_EXTRACTED_DIR`).

**GitHub hygiene:** Do not commit `venv/` (reinstall from `requirements.txt`). The `papers/`, `project_brief/`, and `Research_Ethics/` folders are excluded via `.gitignore` for this upload.
