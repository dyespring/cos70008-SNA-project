# Text-to-Network Engine

**From Text to Networks: Building a Conceptual Network Engine for Data Science and Social Network Analysis**

A Python pipeline that transforms unstructured text into conceptual networks for social network analysis (SNA). Built for SNA Toolbox as part of COS70008 Technology Innovation Project.

The engine is **Neo4j-only**: every concept, relationship, embedding and SNA metric lives in a Neo4j 5 database. The pipeline writes into it (Stage 1-4), Neo4j Graph Data Science (GDS) computes centralities and communities server-side (Stage 5), and downstream consumers (dashboard, chatbot, visualisations) read straight back via Cypher.

## Overview

The engine ingests raw text (policy documents, Yelp reviews, or narrative text), extracts key concepts and their relationships using NLP techniques, writes them into Neo4j as `(:Concept)-[:RELATED]->(:Concept)`, then asks the GDS plugin to compute centrality, communities, brokerage and paths. A Streamlit dashboard and an LLM/RAG chatbot let analysts explore the resulting graph.

## Architecture

```
Stage 1-2 (Python)      Stage 3 (Python)        Stage 4 (Python)
ingest + spaCy   ───►   Concept[] + Rel[]  ───► Neo4jGraphWriter (Cypher UNWIND)
                                                          │
                                                          ▼
                                                      Neo4j 5
                                                  (:Concept)-[:RELATED]->
                                                          ▲
                                  Stage 5 (server-side)  │
                                  GdsAnalysisRunner ─────┘
                                  · pageRank · betweenness
                                  · louvain  · wcc · degree
                                                          │
            Stage 6 (Python, transient)                   │
            cypher_to_nx.fetch_subgraph(top_n=200) ◄──────┤
            → matplotlib + pyvis HTML                     │
                                                          │
            Stage 7 (Python -> Neo4j)                     │
            TemporalAnalyser writes per-slice subgraphs   │
            tagged with `slice_id`; comparison via Cypher │
                                                          │
            Stage 8 (Python -> Neo4j)                     │
            Neo4jGraphContext + Neo4jVectorStore  ◄───────┘
            → GraphChatbot (echo / OpenAI / DashScope / HF)
```

NetworkX is used in only one place: the visualisation adapter pulls a top-N subgraph back into an `nx.DiGraph` for `matplotlib` / `pyvis` rendering. There is no in-memory NetworkX state in the long-running services.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Bring up Neo4j (with APOC + GDS already enabled in docker-compose.yml)
cp .env.example .env                       # set NEO4J_PASSWORD here
docker compose up -d
#    Browser UI: http://localhost:7474

# 3. Run the pipeline (each subcommand is independent)
python pipeline.py etl     --source policy --reset
python pipeline.py analyse --source policy --embed
python pipeline.py viz     --source policy --top-n 200 --export-csv

# Or one-shot:
python pipeline.py all --source combined --reset --sentiment --embed

# 4. Explore
python -m streamlit run src/extensions/dashboard.py
python pipeline.py chat --source policy --vector-store
```

## CLI subcommands

| Command | Stage(s) | What it does |
|---|---|---|
| `etl` | 1-4 | Ingest PDF / Yelp, extract concepts + relationships, **write to Neo4j**. Optional: `--sentiment`, `--embed`. |
| `analyse` | 5 | Run GDS algorithms server-side (PageRank, Louvain, betweenness, closeness, eigenvector, WCC, local clustering); write metrics back as node properties. |
| `viz` | 6 | Pull a top-N subgraph from Neo4j and render PNG + interactive HTML. `--export-csv` also dumps centralities/communities/brokers. |
| `temporal` | 7 | Build N per-slice subgraphs (each tagged with a unique `slice_id`); compute jaccard / overlap via Cypher. |
| `chat` | 8 | Drop into the chatbot grounded in the Neo4j graph (`--vector-store` enables semantic search). |
| `all` | 1-6 (+7/8 opt) | Run `etl` -> `analyse` -> `viz` in one shot. |

Every subcommand requires `NEO4J_PASSWORD` in the environment and a reachable Neo4j instance — there is no fallback.

## LLM / chatbot configuration

The chatbot (`chat.py`, `pipeline.py chat`, and the dashboard **Chat** tab) is provider-agnostic. Select a backend via `LLM_PROVIDER`:

| Provider             | Env vars                         | Install                       |
|----------------------|----------------------------------|-------------------------------|
| `echo` *(default)*   | none                             | none — no-LLM fallback        |
| `openai`             | `OPENAI_API_KEY`, `LLM_MODEL`    | `pip install openai`          |
| `dashscope` *(Qwen)* | `DASHSCOPE_API_KEY`, `LLM_MODEL` | `pip install openai`          |
| `huggingface_api`    | `HUGGINGFACE_API_TOKEN`, `LLM_MODEL` | `pip install huggingface_hub` |
| `huggingface_local`  | `LLM_MODEL` (local model id)     | `pip install transformers`    |

Semantic retrieval (Neo4j native vector index) requires `sentence-transformers`; embeddings are populated by `pipeline.py etl --embed` or `pipeline.py analyse --embed`.

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python pipeline.py chat --source combined --vector-store

# Aliyun DashScope (Qwen) — OpenAI-compatible
export LLM_PROVIDER=dashscope
export LLM_MODEL=qwen-turbo
export DASHSCOPE_API_KEY=sk-...
python pipeline.py chat --source combined --vector-store
```

## Project Structure

```
src/
  config.py                         # Global settings; `require_neo4j_config()` hard-fails on misconfig
  ingest/                           # PDF / Yelp readers
  preprocessing/                    # Cleaning + spaCy tokenisation
  extraction/                       # Concept and relationship extraction
  network/
    neo4j_writer.py                 # Concept[] + Relationship[] -> Cypher UNWIND
    gds_analyser.py                 # GDS PageRank / Louvain / betweenness etc.
  visualisation/
    cypher_to_nx.py                 # Top-N subgraph fetcher (Neo4j -> transient nx.DiGraph)
    static_viz.py                   # matplotlib renderers
    interactive_viz.py              # pyvis HTML renderer
  extensions/
    sentiment.py                    # VADER/TextBlob -> SET r.sentiment
    concept_dictionary.py           # User-defined concept allow-list / aliases
    temporal.py                     # Per-slice subgraphs (slice_id) + Cypher comparison
    temporal_slicing.py             # Build (label, [ProcessedDocument]) lists
    neo4j_store.py                  # Driver lifetime, schema, embeddings, vector indexes
    neo4j_graph_context.py          # Cypher-backed read API for the chatbot
    neo4j_vectorstore.py            # Node vector search (db.index.vector.queryNodes)
    neo4j_edge_vectorstore.py       # Edge vector search (db.index.vector.queryRelationships)
    chatbot.py                      # LLM providers + QueryRouter + GraphChatbot
    dashboard.py                    # Streamlit app (reads everything from Neo4j)
config/                             # User-editable concept_dictionary.yaml
tests/                              # Unit + Neo4j integration tests
pipeline.py                         # CLI: etl / analyse / viz / temporal / chat / all
chat.py                             # Standalone chatbot entry point
```

## Neo4j schema

```
(:Concept {
  id, label, concept_type, source_type, frequency,
  source_label, slice_id,
  pagerank, betweenness, closeness, eigenvector,
  degree, community, wcc_component, local_clustering,
  embedding                            # 384-d cosine, sentence-transformers
})

-[:RELATED {
  weight, types, directed,
  source_label, slice_id,
  top_verb, verb_count, verb_list,
  sentiment, sentiment_label,          # if --sentiment
  embedding                            # 384-d cosine, edge-level
}]->

Schema objects (all created idempotently by Neo4jStore.ensure_schema):
  CONSTRAINT concept_id          UNIQUE (c.id)
  INDEX concept_source_slice     ON (c.source_label, c.slice_id)
  INDEX concept_label            ON (c.label)
  INDEX concept_community        ON (c.community)
  INDEX related_source_slice     ON [r:RELATED] (r.source_label, r.slice_id)
  VECTOR INDEX concept_embedding ON (c.embedding)
  VECTOR INDEX related_embedding ON [r:RELATED] (r.embedding)
```

`source_label` (e.g. `policy`, `yelp`, `combined`) and `slice_id` (set only by the temporal stage) form a composite scope so multiple datasets / snapshots coexist in the same database without colliding.

### Edge-level semantic search

Verb lemmas from spaCy dependency parsing are preserved on the edge (`r.top_verb`, `r.verb_count`, `r.verb_list`) so the chatbot can answer relational questions like *"who recommended the food?"* rather than only "which concepts are important?".

To keep storage and runtime reasonable we embed a curated subset of edges:
- **all** edges produced by dependency parsing (ACTION + CAUSATION — i.e. every edge with `top_verb` set), and
- the **top-N** ASSOCIATION edges by weight (default N = 2000, configurable via `--edge-embed-top-n`).

```bash
# Embed default top-2000 association edges
python pipeline.py etl --source combined --embed

# Richer edge recall (slower, more storage)
python pipeline.py etl --source combined --embed --edge-embed-top-n 5000

# Verb-only embedding (skip ASSOCIATION co-occurrence edges entirely)
python pipeline.py etl --source combined --embed --edge-embed-top-n 0
```

The chatbot's `QueryRouter` automatically switches on edge-level search when the question contains a relational verb (recommend, hate, order, cause, prefer, ...) or a `who/what ... verb ...` pattern.

## Optional Extensions

- **Sentiment-weighted edges** — VADER / TextBlob scoring, written back as `r.sentiment` (`pipeline.py etl --sentiment`).
- **User-defined concept dictionary** — Analyst-controlled vocabularies via `pipeline.py etl --concept-dictionary default` (see `config/concept_dictionary.yaml`).
- **Temporal comparison + insights** — `pipeline.py temporal --source combined --temporal 3` builds three sliced subgraphs (each tagged with its own `slice_id`), runs GDS write-back per slice, and the dashboard's **Temporal** tab surfaces trend / drift / comparison insight cards. Add `--skip-gds` to skip the per-slice GDS pass when iterating quickly.
- **Interactive dashboard** — Streamlit app rendering the live Neo4j subgraph through vis-network (no NetworkX rebuild on the hot path), with per-tab LLM blurbs on Centrality / Communities / Brokers / Summary.
- **LLM / RAG chatbot (Stage 8)** — Conversational Q&A grounded in the graph, with pluggable OpenAI / DashScope / Hugging Face / local providers.

## Tuning the extraction pipeline

All extraction thresholds live in [`src/config.py`](src/config.py) and have CLI overrides on every subcommand that runs ETL. Reach for them in this order if the default network is too noisy or too sparse:

| Knob | CLI flag | Purpose |
|---|---|---|
| `MIN_CONCEPT_FREQ` | `--min-concept-freq` | Minimum corpus-wide frequency to keep a concept (default 3). |
| `MIN_PERSON_FREQ` | `--min-person-freq` | Stricter gate for NER `PERSON` entities (default 10). |
| `TFIDF_MIN_DF` / `TFIDF_MAX_DF` | `--tfidf-min-df`, `--tfidf-max-df` | Drop ngrams that appear in too few / too many documents. |
| `MIN_EDGE_WEIGHT` | `--min-weight` | Drop edges whose raw co-occurrence count is below this. |
| `MIN_NPMI` | `--min-npmi` | Drop association edges with NPMI below this (`-1` disables). |
| `CO_OCCURRENCE_WINDOW` | `--cooccurrence-window` | Sliding sentence window for co-occurrence edges. |
| Source-aware stopwords | `src.config.stopwords_for(source)` | Universal + per-source stopword union (`yelp` / `policy` / `combined`). |

For an opt-in second pass that folds substring duplicates (e.g. `climate` into `climate change`) and consolidates evidence, add:

```bash
python pipeline.py etl --source combined --advanced-extraction \
    --advanced-substring-ratio 0.6 --reset
```

The merger is dependency-free, deterministic, and lives in [`src/extraction/advanced.py`](src/extraction/advanced.py). Use [`candidate_substring_pairs`](src/extraction/advanced.py) to preview which pairs would merge before running the flag.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Use `neo4j+s://...` for AuraDB |
| `NEO4J_USER` | `neo4j` | |
| `NEO4J_PASSWORD` | *(required)* | Also consumed by `docker compose` |
| `NEO4J_DATABASE` | `neo4j` | |
| `NEO4J_VECTOR_INDEX` | `concept_embedding` | |
| `NEO4J_EDGE_VECTOR_INDEX` | `related_embedding` | |
| `NEO4J_EMBEDDING_DIM` | `384` | Must match `EMBEDDING_MODEL` output |
| `LLM_PROVIDER` | `echo` | `openai` / `dashscope` / `huggingface_api` / `huggingface_local` / `echo` |
| `LLM_MODEL` | `gpt-4o-mini` | Provider-specific model id |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer used for both node and edge embeddings |

## Data Sources

- Australian National Climate Resilience and Adaptation Strategy (2021–2025)
- Yelp Open Dataset (academic review subset)

**Local setup:** Large files are gitignored. After clone, place the policy PDF and/or extracted Yelp JSON under `data/` as described in [`data/README.md`](data/README.md). Configured paths are defined in `src/config.py` (`POLICY_PDF_PATH`, `YELP_EXTRACTED_DIR`).

**GitHub hygiene:** Do not commit `venv/` (reinstall from `requirements.txt`). The `papers/`, `project_brief/`, and `Research_Ethics/` folders are excluded via `.gitignore` for this upload.
