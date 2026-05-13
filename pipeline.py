"""CLI entry point for the Neo4j-only Text-to-Network Engine.

Subcommands
-----------

``etl``       Stage 1-4: ingest + preprocess + extract + write to Neo4j.
``analyse``   Stage 5: run GDS algorithms (pagerank / louvain / betweenness …),
              writing centrality + community properties back onto :Concept nodes.
``viz``       Stage 6: pull a top-N subgraph back out and render PNG + HTML.
``temporal``  Stage 7: build per-slice subgraphs (slice_id property) + Cypher
              comparison tables.
``chat``      Stage 8: interactive Q&A over the populated Neo4j graph.
``all``       Run etl → analyse → viz in one shot (the typical demo path).

Neo4j is always required. There is no NetworkX-only fallback.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import (
    CONCEPT_DICTIONARY_PATH,
    POLICY_PDF_PATH,
    RESULTS_DIR,
    require_neo4j_config,
)


# ════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════


def _resolve_dictionary(arg: str | None) -> Path | None:
    raw = (arg or "").strip()
    if raw == "default":
        return CONCEPT_DICTIONARY_PATH
    if raw:
        return Path(raw)
    return None


def _load_documents(args: argparse.Namespace):
    """Run Stages 1-2: ingest + preprocess.

    Returns ``(documents, pages, df)`` where ``pages`` and ``df`` are the
    raw inputs the temporal stage needs to re-slice the corpus.
    """
    from src.preprocessing.cleaner import clean_text
    from src.preprocessing.tokeniser import SpacyTokeniser

    tokeniser = SpacyTokeniser()
    documents = []
    pages: list = []
    df = None

    if args.source in ("policy", "combined"):
        from src.ingest.pdf_reader import read_policy_pdf

        pages = read_policy_pdf(args.pdf_path)
        raw_policy = [(f"policy_page_{p}", text) for p, text in pages]
        cleaned = [(did, clean_text(t, source="policy")) for did, t in raw_policy]
        cleaned = [(did, t) for did, t in cleaned if t.strip()]
        full = " ".join(t for _, t in cleaned)
        documents.append(
            tokeniser.process(full, doc_id="policy_doc", source="policy")
        )
        print(f"       Loaded {len(pages)} pages from policy PDF")

    if args.source in ("yelp", "combined"):
        from src.ingest.yelp_reader import load_yelp_reviews

        yelp_n = args.yelp_sample if args.yelp_sample > 0 else None
        df = load_yelp_reviews(
            category_filter=args.yelp_category,
            sample_n=yelp_n,
        )
        raw_yelp = [(row["review_id"], row["text"]) for _, row in df.iterrows()]
        cleaned = [(did, clean_text(t, source="yelp")) for did, t in raw_yelp]
        cleaned = [(did, t) for did, t in cleaned if t.strip()]
        documents.extend(tokeniser.process_batch(cleaned, source="yelp"))
        print(f"       Loaded {len(df)} Yelp reviews")

    if not documents:
        sys.exit("No documents were loaded. Check --source and paths.")

    return documents, pages, df, tokeniser


def _open_store():
    """Open a connected :class:`Neo4jStore`. Raises on misconfiguration."""
    require_neo4j_config()
    from src.extensions.neo4j_store import Neo4jStore, Neo4jUnavailableError

    try:
        store = Neo4jStore.from_config()
        store.connect()
        return store
    except Neo4jUnavailableError as e:
        sys.exit(f"Neo4j is required but unreachable: {e}")


# ════════════════════════════════════════════════════════════════════
# Commands
# ════════════════════════════════════════════════════════════════════


def cmd_etl(args: argparse.Namespace) -> None:
    """Stage 1-4: ingest -> preprocess -> extract -> write to Neo4j."""
    print(f"[ETL] source={args.source}")
    t0 = time.time()

    documents, _pages, _df, _tok = _load_documents(args)

    # Stage 3 — extract
    print("[3/4] Extracting concepts and relationships...")
    from src.extraction.concept_extractor import ConceptExtractor
    from src.extraction.relationship_extractor import RelationshipExtractor

    methods = set(args.methods)
    extractor = ConceptExtractor(
        use_ner=("ner" in methods or "all" in methods),
        use_noun_phrases=("np" in methods or "all" in methods),
        use_tfidf=("tfidf" in methods or "all" in methods),
        min_freq=args.min_concept_freq,
        min_person_freq=args.min_person_freq,
        tfidf_top_n=args.tfidf_top_n,
        tfidf_max_df=args.tfidf_max_df,
        tfidf_min_df=args.tfidf_min_df,
        min_entity_freq={
            "PERSON":  args.min_person_freq,
            "ORG":     args.min_org_freq,
            "PRODUCT": args.min_product_freq,
        },
    )
    concepts = extractor.extract(documents)
    print(
        f"       {len(concepts)} concepts extracted "
        f"(min_freq={args.min_concept_freq}, "
        f"min_person_freq={args.min_person_freq}, "
        f"tfidf_min_df={args.tfidf_min_df}, tfidf_max_df={args.tfidf_max_df})"
    )

    if args.advanced_extraction:
        from src.extraction.advanced import refine_concepts

        before = len(concepts)
        concepts = refine_concepts(
            concepts,
            merge_substring_pairs=True,
            min_substring_ratio=args.advanced_substring_ratio,
        )
        print(
            f"       Advanced refine: {before} → {len(concepts)} "
            f"(merged {before - len(concepts)} substring duplicates)"
        )

    cd = _resolve_dictionary(args.concept_dictionary)
    if cd is not None:
        from src.extensions.concept_dictionary import apply_concept_dictionary

        concepts = apply_concept_dictionary(concepts, cd)
        print(f"       {len(concepts)} concepts after concept dictionary")

    rel_extractor = RelationshipExtractor(
        use_cooccurrence=True,
        use_dependency=True,
        window_size=args.cooccurrence_window,
        compute_npmi=True,
        min_npmi=args.min_npmi,
    )
    relationships = rel_extractor.extract(documents, concepts)
    print(
        f"       {len(relationships)} relationships extracted "
        f"(window={args.cooccurrence_window}, NPMI ≥ {args.min_npmi})"
    )

    # Stage 4 — write
    print("[4/4] Writing graph to Neo4j...")
    from src.network.neo4j_writer import Neo4jGraphWriter

    store = _open_store()
    try:
        writer = Neo4jGraphWriter(
            store,
            source_label=args.source,
            min_edge_weight=args.min_weight,
        )
        counts = writer.write(concepts, relationships, reset=args.reset)
        print(
            f"       Wrote {counts['nodes']} nodes / {counts['edges']} edges "
            f"under source_label='{args.source}'"
        )

        if args.sentiment:
            print("       Annotating edges with sentiment...")
            from src.extensions.sentiment import SentimentAnnotator

            try:
                ann = SentimentAnnotator(method="vader")
            except ImportError:
                ann = SentimentAnnotator(method="textblob")
            n = ann.annotate_edges_in_neo4j(
                store, documents, source_label=args.source
            )
            print(f"       Sentiment annotated on {n} edges")

        if args.embed:
            print("       Embedding nodes + edges...")
            n_nodes = store.embed_and_store(source_label=args.source)
            n_edges = store.embed_edges(
                source_label=args.source,
                top_n_association=args.edge_embed_top_n,
            )
            print(f"       {n_nodes} node embeddings, {n_edges} edge embeddings")
    finally:
        store.close()

    print(f"[ETL] done in {time.time() - t0:.1f}s")


def cmd_analyse(args: argparse.Namespace) -> None:
    """Stage 5: run GDS algorithms, write metrics back onto nodes."""
    print(f"[ANALYSE] source={args.source}")
    t0 = time.time()

    store = _open_store()
    try:
        from src.network.gds_analyser import GdsAnalysisRunner

        gds = GdsAnalysisRunner(store, source_label=args.source)
        results = gds.run_all()
        for k, v in results.items():
            print(f"       {k}: wrote {v} node properties")

        stats = gds.summary_stats()
        print(
            f"       Graph: {stats['nodes']} nodes, {stats['edges']} edges, "
            f"density={stats['density']:.4f}, "
            f"avg_clustering={stats['avg_clustering']:.4f}"
        )

        comp = gds.source_comparison_summary()
        if args.source == "combined":
            print(
                f"       Source split: policy={comp['policy_only_concepts']}, "
                f"yelp={comp['yelp_only_concepts']}, "
                f"both={comp['shared_concepts']} "
                f"({comp['overlap_pct']:.1f}% overlap)"
            )

        if args.embed:
            print("       Embedding nodes + edges...")
            n_nodes = store.embed_and_store(source_label=args.source)
            n_edges = store.embed_edges(
                source_label=args.source,
                top_n_association=args.edge_embed_top_n,
            )
            print(f"       {n_nodes} node embeddings, {n_edges} edge embeddings")
    finally:
        store.close()

    print(f"[ANALYSE] done in {time.time() - t0:.1f}s")


def cmd_viz(args: argparse.Namespace) -> None:
    """Stage 6: render top-N PNG + interactive HTML from Neo4j."""
    print(f"[VIZ] source={args.source}")
    t0 = time.time()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = _open_store()
    try:
        from src.network.gds_analyser import GdsAnalysisRunner
        from src.visualisation.cypher_to_nx import fetch_partition, fetch_subgraph
        from src.visualisation.interactive_viz import InteractiveVisualiser
        from src.visualisation.static_viz import StaticVisualiser

        G = fetch_subgraph(store, args.source, top_n=args.top_n)
        if G.number_of_nodes() == 0:
            sys.exit(
                f"No nodes found for source_label='{args.source}'. "
                f"Run `python pipeline.py etl --source {args.source}` first."
            )
        partition = fetch_partition(store, args.source)
        gds = GdsAnalysisRunner(store, source_label=args.source)
        centrality_df = gds.all_centralities()

        static_viz = StaticVisualiser(output_dir)
        if args.source == "combined":
            static_viz.plot_network_by_source(
                G, title="Combined Network — Coloured by Source"
            )
            static_viz.plot_source_overlap(G)
        static_viz.plot_network(
            G, partition=partition,
            title=f"Conceptual Network ({args.source.title()} Data)",
        )
        static_viz.plot_top_concepts(centrality_df, top_n=20)
        static_viz.plot_community_distribution(partition)
        static_viz.plot_cooccurrence_heatmap(G, top_n=20)

        interactive_viz = InteractiveVisualiser(output_dir)
        html_path = interactive_viz.create_interactive_network(
            G,
            partition=partition,
            title=f"Conceptual Network Explorer - {args.source.title()} Data",
            colour_by_source=(args.source == "combined"),
            colour_edges_by_sentiment=any(
                "sentiment" in d for _, _, d in G.edges(data=True)
            ),
        )
        print(f"       Static plots saved under {output_dir}/")
        print(f"       Interactive HTML: {html_path}")

        if args.export_csv:
            print("       Exporting CSVs...")
            centrality_df.to_csv(output_dir / "centralities.csv", index=False)
            gds.community_summary().to_csv(
                output_dir / "communities.csv", index=False
            )
            gds.find_brokers(top_n=20).to_csv(
                output_dir / "brokers.csv", index=False
            )
            if args.source == "combined":
                gds.source_distribution().to_csv(
                    output_dir / "source_distribution.csv", index=False
                )
                gds.cross_source_edges().to_csv(
                    output_dir / "cross_source_edges.csv", index=False
                )
                gds.bridging_concepts(top_n=20).to_csv(
                    output_dir / "bridging_concepts.csv", index=False
                )
    finally:
        store.close()

    print(f"[VIZ] done in {time.time() - t0:.1f}s")


def cmd_temporal(args: argparse.Namespace) -> None:
    """Stage 7: build per-slice subgraphs and compare them."""
    print(f"[TEMPORAL] source={args.source}, slices={args.temporal}")
    t0 = time.time()

    if args.temporal < 2:
        sys.exit("--temporal must be >= 2.")

    documents, pages, df, tokeniser = _load_documents(args)

    from src.extensions.temporal import TemporalAnalyser
    from src.extensions.temporal_slicing import (
        policy_pages_to_temporal_slices,
        yelp_reviews_to_year_slices,
    )

    slices: list = []
    if args.source in ("policy", "combined") and pages:
        slices.extend(
            policy_pages_to_temporal_slices(
                pages, tokeniser, n_chunks=args.temporal
            )
        )
    if args.source in ("yelp", "combined") and df is not None:
        slices.extend(
            yelp_reviews_to_year_slices(
                df, tokeniser, sample_per_year=args.yelp_sample or 150
            )
        )
    if len(slices) < 2:
        sys.exit("Not enough slices (need >= 2).")

    store = _open_store()
    try:
        ta = TemporalAnalyser(
            store, source_label=args.source, min_edge_weight=args.min_weight,
        )
        tslices = ta.build_slices(slices)

        # Default behaviour: also run GDS write-back per slice so the
        # downstream Insight Engine has c.pagerank / c.community to play
        # with. ``--skip-gds`` opts out for fast smoke runs.
        if not getattr(args, "skip_gds", False):
            print("       Running GDS write-back per slice…")
            gds_results = ta.run_gds_per_slice(tslices)
            for sid, res in gds_results.items():
                if "_error" in res:
                    print(f"         {sid}: GDS failed ({res['_error']})")
                else:
                    print(
                        f"         {sid}: wrote "
                        f"{sum(int(v) for v in res.values())} node properties"
                    )

        summary_df = ta.enriched_summary(tslices)
        pairwise_df = ta.comparison_table(tslices)

        if args.export_csv:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_dir / "temporal_slice_summary.csv", index=False)
            pairwise_df.to_csv(output_dir / "temporal_pairwise.csv", index=False)
            print(f"       CSVs saved under {output_dir}/")

        print(f"       Built {len(tslices)} slices.")
        for _, row in summary_df.iterrows():
            sent = row.get("avg_sentiment")
            sent_str = f", sent={sent:+.2f}" if sent is not None else ""
            print(
                f"         {row['slice']}: {row['nodes']} nodes, "
                f"{row['edges']} edges{sent_str} "
                f"(top: {row['top_concept']})"
            )
    finally:
        store.close()

    print(f"[TEMPORAL] done in {time.time() - t0:.1f}s")


def cmd_chat(args: argparse.Namespace) -> None:
    """Delegate to chat.py — kept here so `pipeline.py chat` still works."""
    import runpy

    sys.argv = [
        "chat.py",
        "--source", args.source,
        *(("--vector-store",) if args.vector_store else ()),
        *(("--provider", args.provider) if args.provider else ()),
    ]
    runpy.run_path("chat.py", run_name="__main__")


def cmd_all(args: argparse.Namespace) -> None:
    """Run etl -> analyse -> viz in one shot."""
    cmd_etl(args)
    cmd_analyse(args)
    cmd_viz(args)
    if args.temporal and args.temporal >= 2:
        cmd_temporal(args)
    if args.chat:
        cmd_chat(args)


# ════════════════════════════════════════════════════════════════════
# Argparse plumbing
# ════════════════════════════════════════════════════════════════════


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    from src.config import (
        CO_OCCURRENCE_WINDOW,
        MIN_CONCEPT_FREQ,
        MIN_NPMI,
        MIN_PERSON_FREQ,
        TFIDF_MAX_DF,
        TFIDF_MIN_DF,
        TFIDF_TOP_N,
    )

    p.add_argument(
        "--source",
        choices=["policy", "yelp", "combined"],
        default="combined",
        help="Data source / Neo4j source_label scope.",
    )
    p.add_argument("--pdf-path", type=str, default=str(POLICY_PDF_PATH))
    p.add_argument("--yelp-category", type=str, default="Restaurants")
    p.add_argument("--yelp-sample", type=int, default=500)

    # ── Concept-extraction tuning ──────────────────────────────────
    p.add_argument(
        "--min-concept-freq", type=int, default=MIN_CONCEPT_FREQ,
        help=(
            "Drop concepts whose total frequency is below this threshold. "
            f"Defaults to {MIN_CONCEPT_FREQ} (config.MIN_CONCEPT_FREQ)."
        ),
    )
    p.add_argument(
        "--min-person-freq", type=int, default=MIN_PERSON_FREQ,
        help=(
            "Drop NER PERSON concepts below this corpus-wide frequency. "
            f"Defaults to {MIN_PERSON_FREQ}."
        ),
    )
    p.add_argument(
        "--min-org-freq", type=int, default=10,
        help=(
            "Drop NER ORG concepts below this frequency. Filters brand-name "
            "fragments like 'wendy' / 'uber' that pollute review corpora. "
            "Defaults to 10."
        ),
    )
    p.add_argument(
        "--min-product-freq", type=int, default=10,
        help=(
            "Drop NER PRODUCT concepts below this frequency. Defaults to 10."
        ),
    )
    p.add_argument(
        "--tfidf-top-n", type=int, default=TFIDF_TOP_N,
        help=f"Keep top-N TF-IDF keywords (default {TFIDF_TOP_N}).",
    )
    p.add_argument(
        "--tfidf-min-df", type=int, default=TFIDF_MIN_DF,
        help=(
            "Drop TF-IDF ngrams that appear in fewer than N documents. "
            f"Default {TFIDF_MIN_DF}."
        ),
    )
    p.add_argument(
        "--tfidf-max-df", type=float, default=TFIDF_MAX_DF,
        help=(
            "Drop TF-IDF ngrams that appear in more than this *fraction* "
            f"of documents (0–1). Default {TFIDF_MAX_DF}."
        ),
    )

    # ── Relationship-extraction tuning ─────────────────────────────
    p.add_argument(
        "--min-weight", type=int, default=2,
        help="Minimum raw co-occurrence count required for an edge.",
    )
    p.add_argument(
        "--min-npmi", type=float, default=MIN_NPMI,
        help=(
            "Minimum NPMI for ASSOCIATION edges (range -1..1). "
            "0.0 keeps only positively-associated pairs and prunes "
            "the 'everything pairs with hub words' pattern. "
            "Set to -1 to disable NPMI filtering."
        ),
    )
    p.add_argument(
        "--cooccurrence-window", type=int, default=CO_OCCURRENCE_WINDOW,
        help=(
            "Sliding sentence window for co-occurrence edges. "
            f"Default {CO_OCCURRENCE_WINDOW}."
        ),
    )

    # ── Optional advanced extraction (Phase 2 — opt-in) ────────────
    p.add_argument(
        "--advanced-extraction",
        action="store_true",
        help=(
            "Enable optional post-extraction refinements "
            "(substring-pair concept merging) — see "
            "src/extraction/advanced.py."
        ),
    )
    p.add_argument(
        "--advanced-substring-ratio", type=float, default=0.6,
        help=(
            "When --advanced-extraction is on, only merge a shorter concept "
            "into a longer one if their token-overlap ratio is at least "
            "this value (0–1). Default 0.6."
        ),
    )

    p.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Concept extraction methods: ner / np / tfidf / all (default: all)",
    )
    p.add_argument(
        "--concept-dictionary",
        default=None,
        help="Path to a YAML dictionary or 'default' for config/concept_dictionary.yaml",
    )
    p.add_argument("--output", type=str, default=str(RESULTS_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Text-to-Network Engine — Neo4j-only pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_etl = sub.add_parser("etl", help="Ingest + preprocess + extract + write to Neo4j")
    _add_shared_args(p_etl)
    p_etl.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing Concept nodes for this source_label first.",
    )
    p_etl.add_argument(
        "--sentiment",
        action="store_true",
        help="Annotate edges with VADER/TextBlob sentiment.",
    )
    p_etl.add_argument(
        "--embed",
        action="store_true",
        help="Compute and store sentence-transformer embeddings.",
    )
    p_etl.add_argument(
        "--edge-embed-top-n",
        type=int,
        default=2000,
        help="Top-N association edges to embed (in addition to all verb edges).",
    )
    p_etl.set_defaults(func=cmd_etl)

    p_an = sub.add_parser("analyse", help="Run GDS algorithms server-side")
    _add_shared_args(p_an)
    p_an.add_argument(
        "--embed",
        action="store_true",
        help="Also (re)compute embeddings after analysis.",
    )
    p_an.add_argument("--edge-embed-top-n", type=int, default=2000)
    p_an.set_defaults(func=cmd_analyse)

    p_viz = sub.add_parser(
        "viz", help="Pull top-N subgraph + render PNG / HTML"
    )
    _add_shared_args(p_viz)
    p_viz.add_argument("--top-n", type=int, default=200)
    p_viz.add_argument(
        "--export-csv",
        action="store_true",
        help="Also dump centralities/communities/brokers as CSV.",
    )
    p_viz.set_defaults(func=cmd_viz)

    p_t = sub.add_parser(
        "temporal", help="Build per-slice subgraphs + Cypher comparison"
    )
    _add_shared_args(p_t)
    p_t.add_argument("--temporal", type=int, default=3)
    p_t.add_argument("--export-csv", action="store_true")
    p_t.add_argument(
        "--skip-gds",
        action="store_true",
        help=(
            "Skip the per-slice GDS write-back. Faster, but the "
            "TemporalInsightEngine will fall back to frequency-only "
            "rankings for slices."
        ),
    )
    p_t.set_defaults(func=cmd_temporal)

    p_chat = sub.add_parser("chat", help="Interactive chatbot over the Neo4j graph")
    _add_shared_args(p_chat)
    p_chat.add_argument(
        "--vector-store",
        action="store_true",
        help="Enable Neo4j-backed semantic search.",
    )
    p_chat.add_argument("--provider", default=None)
    p_chat.set_defaults(func=cmd_chat)

    p_all = sub.add_parser(
        "all", help="Run etl -> analyse -> viz (and optionally temporal/chat)"
    )
    _add_shared_args(p_all)
    p_all.add_argument("--reset", action="store_true")
    p_all.add_argument("--sentiment", action="store_true")
    p_all.add_argument("--embed", action="store_true")
    p_all.add_argument("--edge-embed-top-n", type=int, default=2000)
    p_all.add_argument("--top-n", type=int, default=200)
    p_all.add_argument("--export-csv", action="store_true")
    p_all.add_argument("--temporal", type=int, default=0)
    p_all.add_argument("--chat", action="store_true")
    p_all.add_argument("--vector-store", action="store_true")
    p_all.add_argument("--provider", default=None)
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
