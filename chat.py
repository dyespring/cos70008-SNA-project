"""Standalone chat entry point — Stage 8 of the Neo4j-only pipeline.

Connects to a populated Neo4j instance and drops the user into an
interactive chatbot grounded in the conceptual network.

Usage::

    python chat.py --source policy
    python chat.py --source combined --vector-store

Prerequisites:
    1. ``docker compose up -d``                            # Neo4j running
    2. ``python pipeline.py etl --source <src>``           # graph populated
    3. ``python pipeline.py analyse --source <src>``       # GDS metrics
"""

from __future__ import annotations

import argparse

from src.config import require_neo4j_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 8: conversational exploration of the conceptual network "
        "(reads from Neo4j)."
    )
    parser.add_argument(
        "--source",
        choices=["policy", "yelp", "combined"],
        default="combined",
        help="source_label scope to query in Neo4j (must match what etl was run with).",
    )
    parser.add_argument(
        "--slice-id",
        default=None,
        help="Optional slice_id scope (set when querying a temporal subgraph).",
    )
    parser.add_argument(
        "--vector-store",
        action="store_true",
        help="Enable Neo4j-backed semantic vector search (requires "
        "sentence-transformers and embeddings already populated).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override LLM_PROVIDER for this session "
        "(openai | dashscope | huggingface_api | huggingface_local | echo).",
    )
    args = parser.parse_args()

    require_neo4j_config()

    if args.provider:
        from src import config

        config.LLM_PROVIDER = args.provider

    from src.extensions.chatbot import GraphChatbot, build_default_provider
    from src.extensions.neo4j_edge_vectorstore import Neo4jEdgeVectorStore
    from src.extensions.neo4j_graph_context import Neo4jGraphContext
    from src.extensions.neo4j_store import Neo4jStore
    from src.extensions.neo4j_vectorstore import Neo4jVectorStore

    store = Neo4jStore.from_config()
    store.connect()
    try:
        gc = Neo4jGraphContext(
            store, source_label=args.source, slice_id=args.slice_id
        )
        if not gc.all_labels():
            raise SystemExit(
                f"No concepts found in Neo4j for source_label='{args.source}'"
                + (f", slice_id='{args.slice_id}'" if args.slice_id else "")
                + ". Run `python pipeline.py etl --source "
                f"{args.source}` first."
            )
        print(
            f"  Backend: Neo4j (source_label='{args.source}', "
            f"{len(gc.all_labels())} concepts loaded)"
        )

        vs = None
        evs = None
        if args.vector_store:
            vs = Neo4jVectorStore(store, source_label=args.source)
            if not vs.available:
                print(
                    "  (node vector store unavailable; run `python pipeline.py "
                    "analyse --source <src> --embed`)"
                )
                vs = None
            evs = Neo4jEdgeVectorStore(store, source_label=args.source)
            if not evs.available:
                evs = None

        bot = GraphChatbot(
            graph_context=gc,
            llm_provider=build_default_provider(),
            vector_store=vs,
            edge_vector_store=evs,
        )

        print("\n" + "=" * 60)
        print("  Graph Chatbot — ask questions about the conceptual network")
        print("  Type 'exit' or press Ctrl+D to quit.")
        print("=" * 60)
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit", ":q"}:
                print("Goodbye.")
                break
            try:
                answer = bot.ask(q)
            except Exception as e:
                answer = f"(error: {e})"
            print(f"\nBot: {answer}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
