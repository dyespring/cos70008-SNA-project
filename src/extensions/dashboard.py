"""Streamlit interactive dashboard for exploring conceptual networks.

Run with:
    streamlit run src/extensions/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Text-to-Network Explorer", layout="wide")


@st.cache_resource
def load_pipeline(
    source_key: str,
    methods_tuple: tuple[str, ...],
    min_weight: int,
    yelp_sample: int,
    apply_sentiment: bool,
):
    """Run ingest → extract → graph → optional sentiment; match CLI pipeline behaviour."""
    from src.extraction.concept_extractor import ConceptExtractor
    from src.extraction.relationship_extractor import RelationshipExtractor
    from src.network.graph_analysis import GraphAnalyser
    from src.network.graph_builder import GraphBuilder
    from src.preprocessing.cleaner import clean_text
    from src.preprocessing.tokeniser import SpacyTokeniser

    tokeniser = SpacyTokeniser()
    documents = []

    if source_key in ("policy", "combined"):
        from src.ingest.pdf_reader import read_policy_pdf
        pages = read_policy_pdf()
        raw_policy = [(f"policy_page_{p}", text) for p, text in pages]
        cleaned_policy = [(did, clean_text(t, source="policy")) for did, t in raw_policy]
        cleaned_policy = [(did, t) for did, t in cleaned_policy if t.strip()]
        full_clean = " ".join(t for _, t in cleaned_policy)
        documents.append(
            tokeniser.process(full_clean, doc_id="policy_doc", source="policy")
        )

    if source_key in ("yelp", "combined"):
        from src.ingest.yelp_reader import load_yelp_reviews
        yelp_n = yelp_sample if yelp_sample > 0 else None
        df = load_yelp_reviews(
            category_filter="Restaurants",
            sample_n=yelp_n,
        )
        raw_yelp = [(row["review_id"], row["text"]) for _, row in df.iterrows()]
        cleaned_yelp = [(did, clean_text(t, source="yelp")) for did, t in raw_yelp]
        cleaned_yelp = [(did, t) for did, t in cleaned_yelp if t.strip()]
        documents.extend(tokeniser.process_batch(cleaned_yelp, source="yelp"))

    use_ner = "NER" in methods_tuple
    use_np = "Noun Phrases" in methods_tuple
    use_tfidf = "TF-IDF" in methods_tuple
    if not methods_tuple:
        use_ner = use_np = use_tfidf = True

    extractor = ConceptExtractor(
        use_ner=use_ner,
        use_noun_phrases=use_np,
        use_tfidf=use_tfidf,
    )
    concepts = extractor.extract(documents)
    rels = RelationshipExtractor().extract(documents, concepts)
    G = GraphBuilder(min_edge_weight=min_weight).build(concepts, rels)

    if apply_sentiment and documents:
        from src.extensions.sentiment import SentimentAnnotator
        try:
            ann = SentimentAnnotator(method="vader")
        except ImportError:
            ann = SentimentAnnotator(method="textblob")
        concept_labels = {G.nodes[n].get("label", n) for n in G.nodes()}
        G = ann.annotate_graph(G, documents, concept_labels)

    analyser = GraphAnalyser(G)
    centrality_df = analyser.all_centralities()

    try:
        partition = analyser.detect_communities_louvain()
    except ImportError:
        partition = analyser.detect_communities_label_propagation()

    comm_df = analyser.community_summary(partition)
    brokers_df = analyser.find_brokers(partition)
    G = analyser.annotate_graph(partition)

    return G, centrality_df, partition, comm_df, brokers_df, documents


def main():
    st.title("Text-to-Network Explorer")
    st.markdown("Transform unstructured text into conceptual networks for SNA analysis.")

    with st.sidebar:
        st.header("Configuration")
        source = st.selectbox(
            "Data Source",
            [
                "Policy Document",
                "Yelp Reviews",
                "Combined (Policy + Yelp)",
            ],
        )
        source_key = {
            "Policy Document": "policy",
            "Yelp Reviews": "yelp",
            "Combined (Policy + Yelp)": "combined",
        }[source]

        methods = st.multiselect(
            "Extraction Methods",
            ["NER", "Noun Phrases", "TF-IDF"],
            default=["NER", "Noun Phrases", "TF-IDF"],
        )
        min_weight = st.slider("Min Edge Weight", 1, 10, 1)
        top_n = st.slider("Top N Concepts", 10, 100, 30)
        yelp_sample = 500
        if source_key in ("yelp", "combined"):
            yelp_sample = st.slider("Yelp review sample size", 100, 2000, 500, step=50)
        apply_sentiment = st.checkbox(
            "Sentiment-weighted edges (VADER / TextBlob)",
            value=False,
            help="Annotate edges using sentence sentiment where endpoint concepts co-occur.",
        )

    methods_tuple = tuple(sorted(methods))

    try:
        G, centrality_df, partition, comm_df, brokers_df, documents = load_pipeline(
            source_key,
            methods_tuple,
            min_weight,
            yelp_sample,
            apply_sentiment,
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("For Yelp sources, extract the Yelp academic dataset JSONL files into `data/Yelp JSON/yelp_dataset/`.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Network", "Centrality", "Communities", "Brokers", "Extensions"]
    )

    with tab1:
        st.subheader("Interactive Network")
        from src.visualisation.interactive_viz import InteractiveVisualiser
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            iv = InteractiveVisualiser(tmpdir)
            path = iv.create_interactive_network(
                G,
                partition=partition,
                top_n=top_n,
                colour_by_source=(source_key == "combined"),
                colour_edges_by_sentiment=apply_sentiment,
            )
            html_content = path.read_text()
            st.components.v1.html(html_content, height=700, scrolling=True)

    with tab2:
        st.subheader(f"Top {top_n} Concepts by Centrality")
        st.dataframe(centrality_df.head(top_n), use_container_width=True)

        metric = st.selectbox(
            "Plot metric", ["pagerank", "betweenness", "degree", "closeness"]
        )
        chart_data = centrality_df.nlargest(top_n, metric).set_index("label")[metric]
        st.bar_chart(chart_data)

    with tab3:
        st.subheader("Community Structure")
        st.dataframe(comm_df, use_container_width=True)

        from collections import Counter

        sizes = Counter(partition.values())
        st.bar_chart(pd.Series(sizes).sort_index().rename("Size"))

    with tab4:
        st.subheader("Broker Concepts")
        st.markdown("Nodes that bridge different thematic communities.")
        st.dataframe(brokers_df, use_container_width=True)

    with tab5:
        st.subheader("Optional extensions")
        st.markdown(
            """
**Temporal comparison** — Run `notebooks/06_optional_extensions_temporal.ipynb` to build networks
per policy page chunk or Yelp year and export `temporal_*` CSV summaries under `results/`.

**User-defined concepts** — Edit `config/concept_dictionary.yaml` and run the CLI with
`--concept-dictionary default` (or a path to your YAML).

**CLI** — `python pipeline.py --source combined --sentiment` produces sentiment-coloured edges
in `interactive_network.html` (same logic as the checkbox in this app).
            """
        )


if __name__ == "__main__":
    main()
