"""Streamlit dashboard for the Neo4j-only Text-to-Network Engine.

Run with::

    python -m streamlit run src/extensions/dashboard.py

Prerequisites: a populated Neo4j instance.

    docker compose up -d
    python pipeline.py etl     --source <src>
    python pipeline.py analyse --source <src>

The dashboard reads everything from Neo4j; there is no in-process
``nx.DiGraph`` rebuild path. Visualisations are rendered from a
top-N subgraph fetched via :func:`src.visualisation.cypher_to_nx.fetch_subgraph`.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Ensure project root is on path when invoked from anywhere.
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st

from src import config
from src.config import require_neo4j_config

st.set_page_config(page_title="Text-to-Network Explorer", layout="wide")


# ── Resource loaders (cached) ──────────────────────────────────────


@st.cache_resource
def _open_store():
    """Open one persistent Neo4j connection per Streamlit session."""
    require_neo4j_config()
    from src.extensions.neo4j_store import Neo4jStore

    store = Neo4jStore.from_config()
    store.connect()
    return store


@st.cache_data(ttl=60)
def load_metadata(source_key: str):
    """Pull node metadata + centralities + community partition from Neo4j."""
    store = _open_store()
    from src.network.gds_analyser import GdsAnalysisRunner
    from src.visualisation.cypher_to_nx import fetch_partition

    gds = GdsAnalysisRunner(store, source_label=source_key)
    centrality_df = gds.all_centralities()
    partition = fetch_partition(store, source_key)
    comm_df = gds.community_summary()
    link_df = gds.community_linkage_stats()
    if not comm_df.empty and not link_df.empty:
        comm_df = comm_df.merge(link_df, on="community", how="left")
    brokers_df = gds.find_brokers(top_n=20)
    stats = gds.summary_stats()
    src_dist = gds.source_distribution()
    return centrality_df, partition, comm_df, brokers_df, stats, src_dist


@st.cache_resource
def load_subgraph(source_key: str, top_n: int):
    """Fetch a top-N display subgraph + partition from Neo4j."""
    store = _open_store()
    from src.visualisation.cypher_to_nx import fetch_partition, fetch_subgraph

    G = fetch_subgraph(store, source_key, top_n=top_n)
    partition = fetch_partition(store, source_key)
    return G, partition


def _community_extreme_cohesion_metric(
    *,
    label: str,
    community_id: int,
    cohesion: float,
    porous: bool,
    help_text: str,
) -> None:
    """Porous / insular summary with Streamlit-like red/green tint and a simple icon.

    Avoids ``st.metric(..., delta=...)`` so Streamlit does not draw misleading
    delta arrows; colours match the usual inverse / normal delta palette.
    """
    if porous:
        icon, fg, bg = "◇", "#ff4b4b", "rgba(255, 75, 75, 0.14)"
    else:
        icon, fg, bg = "◆", "#09ab3b", "rgba(9, 171, 59, 0.14)"
    co = f"{float(cohesion):.2f}"
    st.markdown(
        f'<div title="{html.escape(help_text)}">'
        '<div style="color:rgba(49,51,63,0.6);font-size:0.875rem;'
        'line-height:1.25;margin-bottom:0.2rem;">'
        f"{html.escape(label)}</div>"
        '<div style="font-size:1.5rem;font-weight:600;color:rgb(49,51,63);'
        'line-height:1.2;">'
        f"comm {int(community_id)}</div>"
        '<div style="margin-top:0.35rem;">'
        '<span style="display:inline-block;padding:0.15rem 0.5rem;'
        "border-radius:0.35rem;"
        f'background:{bg};color:{fg};font-size:0.9375rem;font-weight:500;">'
        f"{html.escape(icon)} cohesion {html.escape(co)}</span></div></div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _load_vis_payload(
    source_key: str,
    top_n: int,
    rank_property: str,
    min_edge_weight: int,
    slice_id: str | None,
    colour_by: str,
) -> dict:
    """Cypher → vis-network JSON payload, cached on the user-visible knobs."""
    store = _open_store()
    from src.visualisation.cypher_to_vis_json import fetch_vis_payload

    return fetch_vis_payload(
        store,
        source_key,
        top_n=top_n,
        rank_property=rank_property,
        min_edge_weight=min_edge_weight,
        slice_id=slice_id,
        colour_by=colour_by,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _load_ego_payload(
    source_key: str,
    seed_id: str,
    depth: int,
    neighbour_limit: int,
    min_edge_weight: int,
    slice_id: str | None,
    colour_by: str,
) -> dict:
    """Cypher → ego-graph vis-network payload, cached on the user-visible knobs."""
    store = _open_store()
    from src.visualisation.cypher_to_vis_json import fetch_ego_payload

    return fetch_ego_payload(
        store,
        source_key,
        seed_id=seed_id,
        depth=depth,
        neighbour_limit=neighbour_limit,
        min_edge_weight=min_edge_weight,
        slice_id=slice_id,
        colour_by=colour_by,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _search_concepts(
    source_key: str,
    query: str,
    slice_id: str | None,
    limit: int = 20,
) -> list[dict]:
    """Return concepts whose label matches the keyword (case-insensitive).

    Ordered by exact match first, then ``frequency`` desc, then
    ``pagerank`` desc — so the best candidate sits at the top of the
    picker even when many labels contain the query substring.
    """
    if not query or not query.strip():
        return []
    store = _open_store()
    needle = query.strip().lower()
    slice_filter = " AND c.slice_id = $sid" if slice_id is not None else ""
    cypher = (
        "MATCH (c:Concept) "
        "WHERE c.source_label = $sl "
        "  AND toLower(c.label) CONTAINS $q"
        + slice_filter
        + " RETURN c.id AS id, c.label AS label, "
        "        coalesce(c.frequency, 0)    AS frequency, "
        "        coalesce(c.pagerank, 0.0)   AS pagerank, "
        "        coalesce(c.betweenness, 0.0) AS betweenness, "
        "        coalesce(c.degree, 0.0)     AS degree, "
        "        coalesce(c.community, -1)   AS community, "
        "        coalesce(c.source_type, 'unknown') AS source_type, "
        "        (toLower(c.label) = $q) AS exact "
        " ORDER BY exact DESC, frequency DESC, pagerank DESC "
        " LIMIT $lim"
    )
    params: dict = {"sl": source_key, "q": needle, "lim": int(limit)}
    if slice_id is not None:
        params["sid"] = slice_id
    with store.session() as s:
        return [dict(r) for r in s.run(cypher, **params)]


@st.cache_data(ttl=120, show_spinner=False)
def _load_concept_rank_stats(
    source_key: str,
    seed_id: str,
    slice_id: str | None,
) -> dict:
    """Return PageRank / betweenness / degree **ranks** for a concept.

    Ranks are 1-based across the whole scoped source (and optional slice),
    so a small rank number means "near the top". Cheap single Cypher
    call; cached for 120 s.
    """
    store = _open_store()
    me_slice = " AND me.slice_id = $sid" if slice_id is not None else ""
    n_slice = " AND n.slice_id = $sid" if slice_id is not None else ""
    cypher = (
        "MATCH (me:Concept {id: $id, source_label: $sl}) "
        "WHERE 1=1" + me_slice + " "
        "WITH me, coalesce(me.pagerank, 0.0) AS pr, "
        "        coalesce(me.betweenness, 0.0) AS bt, "
        "        coalesce(me.degree, 0.0) AS dg "
        "MATCH (n:Concept) "
        "WHERE n.source_label = $sl" + n_slice + " "
        "WITH me, pr, bt, dg, "
        "     sum(CASE WHEN coalesce(n.pagerank, 0.0) > pr THEN 1 ELSE 0 END) AS pr_above, "
        "     sum(CASE WHEN coalesce(n.betweenness, 0.0) > bt THEN 1 ELSE 0 END) AS bt_above, "
        "     sum(CASE WHEN coalesce(n.degree, 0.0) > dg THEN 1 ELSE 0 END) AS dg_above, "
        "     count(n) AS total "
        "RETURN pr_above + 1 AS pr_rank, "
        "       bt_above + 1 AS bt_rank, "
        "       dg_above + 1 AS dg_rank, "
        "       total, pr, bt, dg"
    )
    params: dict = {"id": seed_id, "sl": source_key}
    if slice_id is not None:
        params["sid"] = slice_id
    with store.session() as s:
        rec = s.run(cypher, **params).single()
    return dict(rec) if rec else {}


@st.cache_data(ttl=120, show_spinner=False)
def _list_slice_ids(source_key: str) -> list[str]:
    """Return existing ``slice_id`` values for a source (for the slice picker)."""
    store = _open_store()
    with store.session() as s:
        rows = s.run(
            "MATCH (c:Concept {source_label: $sl}) "
            "WHERE c.slice_id IS NOT NULL "
            "RETURN DISTINCT c.slice_id AS slice_id ORDER BY slice_id",
            sl=source_key,
        ).data()
    return [r["slice_id"] for r in rows]


@st.cache_resource
def _build_chatbot(
    provider: str,
    enable_vector: bool,
    source_key: str,
):
    """Build a GraphChatbot wired to the live Neo4j store."""
    from src.extensions.chatbot import GraphChatbot, build_default_provider
    from src.extensions.neo4j_edge_vectorstore import Neo4jEdgeVectorStore
    from src.extensions.neo4j_graph_context import Neo4jGraphContext
    from src.extensions.neo4j_vectorstore import Neo4jVectorStore

    store = _open_store()

    prior_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = provider
    llm = build_default_provider()
    config.LLM_PROVIDER = prior_provider

    gc = Neo4jGraphContext(store, source_label=source_key)

    vs = None
    evs = None
    if enable_vector:
        vs = Neo4jVectorStore(store, source_label=source_key)
        if not vs.available:
            vs = None
        evs = Neo4jEdgeVectorStore(store, source_label=source_key)
        if not evs.available:
            evs = None

    return GraphChatbot(
        graph_context=gc,
        llm_provider=llm,
        vector_store=vs,
        edge_vector_store=evs,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _load_raw_insights(source_key: str) -> dict:
    """Compute the deterministic (un-polished) insight bundle.

    Cached per source — re-runs only when the user picks a different
    source or hits Streamlit's "Rerun" with a hard cache clear.
    The bundle is returned as a plain dict so it survives Streamlit's
    pickle round-trip without dragging in the live driver session.
    """
    store = _open_store()
    from src.extensions.insight_engine import InsightEngine

    engine = InsightEngine(store, source_label=source_key)
    bundle = engine.all_insights(polish=False)
    return {
        cat: [_insight_to_dict(ins) for ins in items]
        for cat, items in bundle.items()
    }


def _insight_to_dict(ins) -> dict:
    return {
        "id": ins.id,
        "category": ins.category,
        "severity": ins.severity,
        "title": ins.title,
        "body": ins.body,
        "concepts": list(ins.concepts),
        "metric": ins.metric,
        "available": ins.available,
        "unavailable_reason": ins.unavailable_reason,
    }


_SEVERITY_ORDER_EXEC = {"high": 3, "medium": 2, "low": 1, "info": 0}


def _first_executive_insight(items: list[dict]) -> dict | None:
    """Pick the strongest single card for the executive strip."""
    if not items:
        return None
    return max(
        items,
        key=lambda x: (
            1 if x.get("available") else 0,
            _SEVERITY_ORDER_EXEC.get(str(x.get("severity") or "info"), 0),
            len(x.get("body") or ""),
        ),
    )


def _executive_stats_bullets(stats: dict) -> list[str]:
    """Two to three short, stakeholder-friendly lines from GDS summary stats."""
    n = int(stats.get("nodes") or 0)
    e = int(stats.get("edges") or 0)
    d = float(stats.get("density") or 0.0)
    acc = stats.get("avg_clustering")
    acc_f = float(acc) if acc is not None else None
    wcc = int(stats.get("weakly_connected_components") or 0)
    out: list[str] = []
    if n > 1:
        tone = (
            "fairly sparse"
            if d < 0.001
            else "moderately connected"
            if d < 0.01
            else "relatively dense"
        )
        out.append(
            f"The active scope has **{n:,}** concepts and **{e:,}** directed "
            f"relationships (overall density **{d:.4f}**) — **{tone}** for a "
            "text-derived conceptual network."
        )
    else:
        out.append("Too few nodes to characterise density meaningfully.")
    if wcc > 1:
        out.append(
            f"**{wcc}** weakly-connected components mean the story is not one "
            "single giant component; look for bridge concepts if you need "
            "cross-theme narratives."
        )
    else:
        out.append(
            "The graph is **one** weakly-connected piece, so paths and "
            "community structure are globally interpretable."
        )
    if acc_f is not None and acc_f > 0:
        tri = "noticeable triadic closure" if acc_f > 0.25 else "looser local triangles"
        out.append(
            f"Typical local clustering averages **{acc_f:.3f}** — neighbourhoods "
            f"show **{tri}**."
        )
    else:
        out.append(
            "Clustering is **not populated** yet — run `pipeline.py analyse` "
            "so GDS can write `local_clustering`."
        )
    return out[:3]


@st.cache_data(ttl=120, show_spinner=False)
def _load_source_comparison_summary(source_key: str) -> dict:
    store = _open_store()
    from src.network.gds_analyser import GdsAnalysisRunner

    return GdsAnalysisRunner(
        store, source_label=source_key,
    ).source_comparison_summary()


def _executive_summary_tab(
    stats: dict,
    src_dist,
    source_key: str,
    top_n: int,
) -> None:
    """Full body of the Executive Summary tab (used as the first dashboard tab)."""
    st.subheader("Executive summary")
    st.caption(
        "One-screen snapshot for stakeholders: scale, structure, corpus mix "
        "(when combined), and the single strongest insight per category. "
        "Open **Insights** for the full card deck."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Concepts", f"{int(stats.get('nodes', 0) or 0):,}")
    with m2:
        st.metric("Relationships", f"{int(stats.get('edges', 0) or 0):,}")
    with m3:
        st.metric(
            "Density",
            f"{float(stats.get('density', 0) or 0):.4f}",
        )
    with m4:
        acc_v = stats.get("avg_clustering")
        st.metric(
            "Avg clustering",
            f"{float(acc_v):.3f}" if acc_v is not None else "—",
        )
    with m5:
        st.metric(
            "WCC components",
            f"{int(stats.get('weakly_connected_components', 0) or 0)}",
        )

    st.markdown("**What this means**")
    for line in _executive_stats_bullets(stats):
        st.markdown(f"- {line}")

    if source_key == "combined":
        st.divider()
        st.markdown("#### Corpus mix (policy vs reviews)")
        try:
            sc = _load_source_comparison_summary(source_key)
        except Exception as e:
            sc = {}
            st.warning(f"Could not load corpus comparison: {e}")
        if sc:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Policy-leaning concepts",
                    f"{int(sc.get('policy_only_concepts', 0) or 0):,}",
                )
            with c2:
                st.metric(
                    "Yelp-leaning concepts",
                    f"{int(sc.get('yelp_only_concepts', 0) or 0):,}",
                )
            with c3:
                st.metric(
                    "Shared (`both`)",
                    f"{int(sc.get('shared_concepts', 0) or 0):,}",
                )
            with c4:
                st.metric(
                    "Overlap",
                    f"{float(sc.get('overlap_pct', 0) or 0):.1f}%",
                    help="Share of concepts tagged as appearing in both corpora.",
                )
            st.caption(
                "High **Shared** counts usually mean the two sources reuse "
                "the same vocabulary (e.g. climate, business). Use the "
                "**Brokers** tab for vocabulary that explicitly bridges communities."
            )

    st.divider()
    st.markdown("#### TL;DR — strongest SNA card per category")
    try:
        ex_bundle = _load_raw_insights(source_key)
    except Exception as e:
        ex_bundle = {"key": [], "risk": [], "action": []}
        st.warning(f"Insights unavailable: {e}")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**Key**")
        k_ins = _first_executive_insight(ex_bundle.get("key", []))
        if k_ins:
            _render_insight_card(k_ins, None)
        else:
            st.caption("No key insight computed.")
    with e2:
        st.markdown("**Risk**")
        r_ins = _first_executive_insight(ex_bundle.get("risk", []))
        if r_ins:
            _render_insight_card(r_ins, None)
        else:
            st.caption("No risk insight computed.")
    with e3:
        st.markdown("**Action**")
        a_ins = _first_executive_insight(ex_bundle.get("action", []))
        if a_ins:
            _render_insight_card(a_ins, None)
        else:
            st.caption("No action insight computed.")
    st.info(
        "For up to **three** cards per category and optional LLM polish, "
        "open the **Insights** tab."
    )

    st.divider()
    st.markdown("**Source distribution (node tags)**")
    st.dataframe(src_dist, use_container_width=True)

    _llm_blurb_panel(
        title="One-paragraph executive read on the numbers",
        metric_name="Graph summary",
        description=(
            "Nodes, edges, density, average clustering, weakly-connected "
            "components, and (if combined) policy/yelp/both concept mix. "
            "Write a tight executive summary; do not invent metrics."
        ),
        stats=stats,
        state_key="blurb_summary",
        helper_caption=(
            "Optional — **Generate** sends only this aggregate snapshot to the LLM."
        ),
    )

    with st.expander("Reproduce from the CLI", expanded=False):
        st.markdown(
            """
```bash
python pipeline.py etl     --source {src}
python pipeline.py analyse --source {src} --embed
python pipeline.py viz     --source {src} --top-n {top}
python pipeline.py chat    --source {src}
```
            """.format(src=source_key, top=top_n)
        )


@st.cache_data(ttl=600, show_spinner=False)
def _polish_insight_body(
    insight_id: str, body: str, provider: str, model: str
) -> str:
    """LLM-polish one insight body. Cached on (id, raw body, provider, model)
    so the same observation is only sent to the LLM once per session."""
    from src.extensions.chatbot import build_default_provider
    from src.extensions.insight_engine import (
        Insight,
        _polish_text,
    )

    prior_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = provider
    llm = build_default_provider()
    config.LLM_PROVIDER = prior_provider

    fake = Insight(id=insight_id, category="", title="", body=body)
    return _polish_text(llm, fake)


@st.cache_data(ttl=600, show_spinner=False)
def _metric_df_blurb(
    cache_key: str,           # noqa: ARG001 — cache discriminator only
    metric_name: str,
    description: str,
    sample_records: list,
    provider: str,
    model: str,               # noqa: ARG001 — cache discriminator only
) -> str:
    """Streamlit-cached wrapper around :func:`metric_llm.df_blurb`.

    The DataFrame is passed pre-serialised (``list[dict]``) so the cache
    key stays deterministic — Streamlit can't hash a DataFrame
    reliably across reruns.
    """
    import pandas as pd

    from src.extensions.chatbot import build_default_provider
    from src.extensions.metric_llm import df_blurb

    prior_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = provider
    llm = build_default_provider()
    config.LLM_PROVIDER = prior_provider

    if not sample_records:
        return f"_No data for {metric_name}._"

    return df_blurb(
        llm,
        pd.DataFrame(sample_records),
        metric_name=metric_name,
        description=description,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _metric_stats_blurb(
    cache_key: str,           # noqa: ARG001 — cache discriminator only
    metric_name: str,
    description: str,
    stats: dict,
    provider: str,
    model: str,               # noqa: ARG001 — cache discriminator only
) -> str:
    from src.extensions.chatbot import build_default_provider
    from src.extensions.metric_llm import stats_blurb

    prior_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = provider
    llm = build_default_provider()
    config.LLM_PROVIDER = prior_provider

    return stats_blurb(
        llm,
        stats,
        metric_name=metric_name,
        description=description,
    )


def _llm_blurb_panel(
    *,
    title: str,
    metric_name: str,
    description: str,
    df=None,
    stats: dict | None = None,
    rank_column: str | None = None,
    top_rows: int = 5,
    state_key: str,
    helper_caption: str | None = None,
):
    """Render an opt-in LLM blurb section under a metric.

    Behaves like a tiny three-state widget (button / spinner / blurb +
    refresh). Off by default so we don't burn LLM tokens on every
    dashboard rerun.
    """
    from src.extensions.metric_llm import df_signature, stats_signature

    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(f"#### ✨ {title}")
        st.caption(
            helper_caption
            or (
                "Short narrative summary of the metric above, generated on "
                "demand. Uses your configured LLM provider (see Chat tab → "
                "Chatbot settings). Cached so re-runs are free."
            )
        )
    with cols[1]:
        provider_pick = st.selectbox(
            "Provider",
            ["echo", "openai", "dashscope", "huggingface_api",
             "huggingface_local"],
            index=_default_provider_index(),
            key=f"{state_key}_provider",
            label_visibility="collapsed",
        )

    btn_a, btn_b = st.columns([1, 1])
    with btn_a:
        run = st.button("Generate", key=f"{state_key}_run", type="primary")
    with btn_b:
        clear = st.button("Refresh", key=f"{state_key}_clear")

    if clear:
        # Reset cached LLM output by changing the discriminator.
        st.session_state[f"{state_key}_nonce"] = (
            st.session_state.get(f"{state_key}_nonce", 0) + 1
        )

    nonce = st.session_state.get(f"{state_key}_nonce", 0)

    if not run:
        return

    if df is not None:
        sig = f"{df_signature(df, top_rows=top_rows)}::{nonce}"
        sample = df.sort_values(rank_column, ascending=False).head(top_rows) \
            if rank_column and rank_column in df.columns else df.head(top_rows)
        with st.spinner("Asking the LLM…"):
            text = _metric_df_blurb(
                sig, metric_name, description,
                sample.to_dict(orient="records"),
                provider_pick, config.LLM_MODEL,
            )
        st.info(text)
    elif stats is not None:
        sig = f"{stats_signature(stats)}::{nonce}"
        with st.spinner("Asking the LLM…"):
            text = _metric_stats_blurb(
                sig, metric_name, description, stats,
                provider_pick, config.LLM_MODEL,
            )
        st.info(text)


def _default_provider_index() -> int:
    options = [
        "echo", "openai", "dashscope",
        "huggingface_api", "huggingface_local",
    ]
    default = (config.LLM_PROVIDER or "echo").lower()
    alias = {
        "qwen": "dashscope", "aliyun": "dashscope", "modelscope": "dashscope",
    }
    default = alias.get(default, default)
    if default not in options:
        default = "echo"
    return options.index(default)


def _severity_meta(severity: str) -> tuple[str, str]:
    """Map severity → (emoji, streamlit container border colour hint)."""
    return {
        "high":   ("🔴", "high"),
        "medium": ("🟠", "medium"),
        "low":    ("🟡", "low"),
        "info":   ("🔵", "info"),
    }.get(severity, ("⚪", "info"))


def _render_insight_card(ins: dict, polished_body: str | None = None) -> None:
    """Render one insight as a Streamlit bordered card."""
    emoji, _ = _severity_meta(ins["severity"])
    body = polished_body or ins["body"]
    with st.container(border=True):
        st.markdown(f"#### {emoji} {ins['title']}")
        if not ins["available"]:
            st.caption(f"_Unavailable — {ins['unavailable_reason']}_")
        else:
            st.write(body)
            chips: list[str] = []
            if ins["concepts"]:
                shown = ", ".join(f"`{c}`" for c in ins["concepts"][:5])
                chips.append(f"**Concepts**: {shown}")
            if ins["metric"]:
                chips.append(f"**Metric**: `{ins['metric']}`")
            if chips:
                st.caption(" · ".join(chips))


def _temporal_slice_summary(source_key: str):
    """Read existing slice subgraphs out of Neo4j and summarise them."""
    from src.extensions.temporal import (
        is_yelp_calendar_slice_id,
        yelp_calendar_year_from_slice_id,
    )

    store = _open_store()
    with store.session() as s:
        rows = s.run(
            "MATCH (c:Concept {source_label: $sl}) "
            "WHERE c.slice_id IS NOT NULL "
            "WITH c.slice_id AS slice_id, count(c) AS nodes "
            "OPTIONAL MATCH (a:Concept {source_label: $sl, slice_id: slice_id})"
            "-[r:RELATED {source_label: $sl, slice_id: slice_id}]->"
            "(b:Concept {source_label: $sl, slice_id: slice_id}) "
            "RETURN slice_id, nodes, count(r) AS edges "
            "ORDER BY slice_id",
            sl=source_key,
        ).data()
    df = pd.DataFrame(rows)
    if source_key == "combined" and not df.empty:
        df = df.loc[df["slice_id"].map(is_yelp_calendar_slice_id)].copy()
        df["_y"] = df["slice_id"].map(
            lambda sid: yelp_calendar_year_from_slice_id(str(sid)) or 0,
        )
        df = df.sort_values("_y").drop(columns=["_y"])
    return df


@st.cache_data(ttl=120, show_spinner=False)
def _load_community_members(
    source_key: str, community_id: int, top_n: int = 25,
):
    store = _open_store()
    from src.network.gds_analyser import GdsAnalysisRunner

    return GdsAnalysisRunner(
        store, source_label=source_key,
    ).community_members(community_id, top_n=top_n)


@st.cache_data(ttl=120, show_spinner=False)
def _load_community_edges(
    source_key: str, community_id: int, top_n: int = 15,
):
    store = _open_store()
    from src.network.gds_analyser import GdsAnalysisRunner

    return GdsAnalysisRunner(
        store, source_label=source_key,
    ).community_edges(community_id, top_n=top_n)


@st.cache_data(ttl=120, show_spinner=False)
def _load_community_neighbours(
    source_key: str, community_id: int, top_n: int = 10,
):
    store = _open_store()
    from src.network.gds_analyser import GdsAnalysisRunner

    return GdsAnalysisRunner(
        store, source_label=source_key,
    ).community_neighbours(community_id, top_n=top_n)


@st.cache_data(ttl=120, show_spinner=False)
def _load_broker_drilldown(source_key: str, label: str) -> dict:
    """Pull the broker drill-down payload via GdsAnalysisRunner."""
    store = _open_store()
    from src.network.gds_analyser import GdsAnalysisRunner

    return GdsAnalysisRunner(
        store, source_label=source_key,
    ).broker_drilldown(label, top_n_edges=12)


@st.cache_data(ttl=120, show_spinner=False)
def _load_temporal_slices(source_key: str) -> list:
    """Reconstruct the in-memory ``TemporalSlice`` list from Neo4j."""
    store = _open_store()
    from src.extensions.temporal import TemporalAnalyser

    return TemporalAnalyser(
        store, source_label=source_key
    ).existing_slices()


@st.cache_data(ttl=120, show_spinner=False)
def _load_enriched_temporal_summary(source_key: str):
    from src.extensions.temporal import TemporalAnalyser, dashboard_temporal_slices

    store = _open_store()
    ta = TemporalAnalyser(store, source_label=source_key)
    slices = dashboard_temporal_slices(source_key, ta.existing_slices())
    if not slices:
        return pd.DataFrame()
    return ta.enriched_summary(slices)


@st.cache_data(ttl=120, show_spinner=False)
def _load_pairwise_compare(source_key: str, *, _overlap_algo: int = 4):
    from src.extensions.temporal import TemporalAnalyser, dashboard_temporal_slices

    _ = _overlap_algo  # Streamlit cache fingerprint when overlap logic changes

    store = _open_store()
    ta = TemporalAnalyser(store, source_label=source_key)
    slices = dashboard_temporal_slices(source_key, ta.existing_slices())
    if len(slices) < 2:
        return pd.DataFrame()
    return ta.comparison_table(slices)


@st.cache_data(ttl=120, show_spinner=False)
def _load_detailed_slice_compare(
    source_key: str,
    slice_id_a: str,
    slice_id_b: str,
    *,
    _overlap_algo: int = 4,
) -> dict | None:
    """Pairwise drill-down (labels, Jaccard, sentiment delta) for any two slices."""
    _ = _overlap_algo  # Streamlit cache fingerprint when overlap logic changes
    store = _open_store()
    from src.extensions.temporal import TemporalAnalyser

    ta = TemporalAnalyser(store, source_label=source_key)
    by_id = {s.slice_id: s for s in ta.existing_slices()}
    if slice_id_a not in by_id or slice_id_b not in by_id or slice_id_a == slice_id_b:
        return None
    return ta.detailed_compare(by_id[slice_id_a], by_id[slice_id_b], top_k=8)


@st.cache_data(ttl=300, show_spinner=False)
def _load_temporal_insights(source_key: str) -> dict:
    """Compute the temporal insight bundle as plain dicts (cache-safe)."""
    from src.extensions.temporal import TemporalAnalyser, dashboard_temporal_slices
    from src.extensions.temporal_insights import TemporalInsightEngine

    store = _open_store()
    ta = TemporalAnalyser(store, source_label=source_key)
    slices = dashboard_temporal_slices(source_key, ta.existing_slices())
    engine = TemporalInsightEngine(store, source_label=source_key, analyser=ta)
    bundle = engine.all_insights(slices=slices, polish=False)
    out = {"trend": [], "drift": [], "comparison": []}
    for cat, items in bundle.as_dict().items():
        out[cat] = [_insight_to_dict(ins) for ins in items]
    return out


@st.cache_data(ttl=60)
def _load_bridging_concepts(source_key: str):
    """Policy↔Yelp vocabulary bridges (meaningful only for ``combined``)."""
    store = _open_store()
    from src.network.gds_analyser import GdsAnalysisRunner

    return GdsAnalysisRunner(store, source_label=source_key).bridging_concepts(
        top_n=18,
    )


# Short label + full question for the Chat tab onboarding row.
_CHAT_ONBOARD_QUESTIONS: list[tuple[str, str]] = [
    (
        "Top influencers",
        "Which concepts have the highest PageRank in this source scope, "
        "and why might they matter for understanding this corpus?",
    ),
    (
        "Policy vs reviews",
        "How does vocabulary differ between policy text and Yelp reviews? "
        "Name any bridging concepts that connect the two sides.",
    ),
    (
        "Community themes",
        "What are the main themes in the largest graph communities "
        "(Louvain clusters), and which terms anchor each cluster?",
    ),
    (
        "Sentiment hotspots",
        "Where do we see the strongest positive or negative sentiment on "
        "relationships, and what topics or concepts are involved?",
    ),
    (
        "Bridges & brokers",
        "Which concepts act as bridges or brokers between parts of the "
        "network, and what relationships support that view?",
    ),
]


# ── Main app ───────────────────────────────────────────────────────


def main():
    st.title("Text-to-Network Explorer")
    st.markdown(
        "Browse the conceptual network stored in Neo4j. Run "
        "`python pipeline.py etl --source <src>` and "
        "`python pipeline.py analyse --source <src>` to populate the data."
    )

    try:
        require_neo4j_config()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        st.header("Configuration")
        _source_labels = [
            "Combined (Policy + Yelp)",
            "Policy Document",
            "Yelp Reviews",
        ]
        source = st.selectbox(
            "Data Source",
            _source_labels,
            index=0,
            help="Default is combined graph scope (matches `pipeline.py` default).",
        )
        source_key = {
            "Combined (Policy + Yelp)": "combined",
            "Policy Document": "policy",
            "Yelp Reviews": "yelp",
        }[source]
        top_n = st.slider("Top N concepts (visualisation)", 10, 500, 100)

    try:
        centrality_df, partition, comm_df, brokers_df, stats, src_dist = (
            load_metadata(source_key)
        )
    except Exception as e:
        st.error(
            f"Failed to load data from Neo4j: {e}. "
            f"Have you run `python pipeline.py etl --source {source_key}`?"
        )
        st.stop()

    if centrality_df.empty:
        st.warning(
            f"No concepts found in Neo4j for source_label='{source_key}'. "
            f"Run `python pipeline.py etl --source {source_key}` and "
            f"`python pipeline.py analyse --source {source_key}` first."
        )
        st.stop()

    G, _partition_full = load_subgraph(source_key, top_n)

    tab_sum, tab_net, tab_ins, tab2, tab3, tab4, tab5, tab_chat = st.tabs(
        [
            "Summary",
            "Network",
            "Insights",
            "Centrality",
            "Communities",
            "Brokers",
            "Temporal",
            "Chat",
        ]
    )

    with tab_sum:
        _executive_summary_tab(stats, src_dist, source_key, top_n)

    with tab_net:
        st.subheader("Interactive Network")
        st.caption(
            "Rendered straight from Neo4j as a vis-network payload — no "
            "NetworkX rebuild on the hot path. Use the controls below to "
            "change ranking, prune weak edges, or focus on a temporal slice."
        )
        with st.container(border=True):
            st.markdown("### 👋 How to use this view")
            st.markdown(
                """
- **Top N + rank** — loads a **high-impact subgraph** (not every node in Neo4j at once).
- **Colour** — **community** shows Louvain themes; **source** (on *combined*) shows policy- vs review-anchored vocabulary.
- **Tooltips** — hover nodes/edges for weight, optional **sentiment**, and top verb when stored.
- The canvas is **interactive** (pan/zoom/drag); change filters **above** to refresh what is pulled from the database.
                """
            )

        # ── Network controls ────────────────────────────────────────
        avail_slices = _list_slice_ids(source_key)
        ctrl_a, ctrl_b, ctrl_c, ctrl_d, ctrl_e = st.columns([1, 1, 1, 1, 1])
        with ctrl_a:
            net_top_n = st.slider(
                "Top N nodes", 20, 500, value=min(top_n, 200), step=10,
                key="net_top_n",
            )
        with ctrl_b:
            rank_property = st.selectbox(
                "Rank by",
                ["pagerank", "betweenness", "degree", "closeness",
                 "eigenvector", "frequency"],
                index=0,
                key="net_rank_property",
            )
        with ctrl_c:
            min_edge_weight = st.slider(
                "Min edge weight", 1, 10, value=1, key="net_min_edge_weight"
            )
        with ctrl_d:
            colour_options = ["community", "source"]
            default_colour_idx = (
                1 if source_key == "combined" else 0
            )
            colour_by = st.selectbox(
                "Colour by",
                colour_options,
                index=default_colour_idx,
                key="net_colour_by",
            )
        with ctrl_e:
            slice_choices = ["(all)"] + avail_slices
            slice_pick = st.selectbox(
                "Slice", slice_choices, index=0, key="net_slice_pick",
                help=(
                    "Filter to a single temporal slice (must have run "
                    "`pipeline.py temporal --source <src>` first)."
                    if avail_slices
                    else "No temporal slices found for this source."
                ),
            )
        slice_id = None if slice_pick == "(all)" else slice_pick

        physics_choice = st.radio(
            "Layout solver",
            ["barnesHut", "forceAtlas2Based", "repulsion"],
            index=0,
            horizontal=True,
            key="net_physics",
        )

        # ── Keyword / concept search ─────────────────────────────────
        with st.container(border=True):
            st.markdown("#### 🔎 Search the network by keyword or concept")
            st.caption(
                "Type a concept label (or any keyword that appears in one). "
                "We fetch the **ego graph** around the match plus a side "
                "panel of stats and neighbours. Leave empty to keep the "
                "Top-N view above."
            )
            s1, s2, s3 = st.columns([3, 1, 1])
            with s1:
                search_q = st.text_input(
                    "Keyword or concept",
                    placeholder="e.g. resilience, food, climate change",
                    key="net_search_q",
                    label_visibility="collapsed",
                )
            with s2:
                ego_depth = st.selectbox(
                    "Hops",
                    [1, 2],
                    index=0,
                    key="net_ego_depth",
                    help=(
                        "1 = direct neighbours of the match. 2 = also "
                        "include neighbours-of-neighbours."
                    ),
                )
            with s3:
                ego_per_hop = st.slider(
                    "Per hop",
                    5, 60, value=20, step=5,
                    key="net_ego_per_hop",
                    help=(
                        "Cap on **new** nodes added per hop, picked by "
                        "edge weight. Total ego size ≤ 1 + hops × per-hop."
                    ),
                )

        search_q_clean = (search_q or "").strip()
        search_mode = bool(search_q_clean)

        if search_mode:
            try:
                hits = _search_concepts(
                    source_key, search_q_clean, slice_id, limit=20,
                )
            except Exception as e:
                hits = []
                st.error(f"Search failed: {e}")

            if not hits:
                st.warning(
                    f"No concept matches '{search_q_clean}'. Try a "
                    "different spelling, broaden the slice, or clear the "
                    "search box to see the Top-N view."
                )
                payload = {"nodes": [], "edges": [], "stats": {}}
            else:
                if len(hits) > 1:
                    pick_idx = st.selectbox(
                        f"Pick a match ({len(hits)} candidates):",
                        options=list(range(len(hits))),
                        index=0,
                        format_func=lambda i: (
                            f"{hits[i]['label']}  "
                            f"(freq={int(hits[i]['frequency'])}, "
                            f"PR={float(hits[i]['pagerank']):.3f}, "
                            f"community={int(hits[i]['community'])})"
                        ),
                        key="net_search_pick",
                    )
                    seed = hits[pick_idx]
                else:
                    seed = hits[0]
                    st.caption(
                        f"Matched concept: **{seed['label']}** "
                        f"(freq={int(seed['frequency'])}, "
                        f"community={int(seed['community'])})"
                    )

                try:
                    payload = _load_ego_payload(
                        source_key,
                        seed["id"],
                        int(ego_depth),
                        int(ego_per_hop),
                        min_edge_weight,
                        slice_id,
                        colour_by,
                    )
                except Exception as e:
                    st.error(f"Failed to build ego graph: {e}")
                    payload = {"nodes": [], "edges": [], "stats": {}}

                # ── Insight side panel ──────────────────────────────
                with st.container(border=True):
                    st.markdown(f"### 💡 Insights for **{seed['label']}**")

                    try:
                        rank_stats = _load_concept_rank_stats(
                            source_key, seed["id"], slice_id,
                        )
                    except Exception:
                        rank_stats = {}

                    total = int(rank_stats.get("total") or 0)
                    cols_m = st.columns(4)
                    with cols_m[0]:
                        st.metric(
                            "Community",
                            (
                                f"#{int(seed['community'])}"
                                if int(seed["community"]) >= 0
                                else "—"
                            ),
                        )
                    with cols_m[1]:
                        pr_rank = rank_stats.get("pr_rank")
                        st.metric(
                            "PageRank rank",
                            (
                                f"#{int(pr_rank)} / {total}"
                                if pr_rank and total
                                else "—"
                            ),
                            help=f"PageRank value: {float(seed['pagerank']):.4f}",
                        )
                    with cols_m[2]:
                        bt_rank = rank_stats.get("bt_rank")
                        st.metric(
                            "Betweenness rank",
                            (
                                f"#{int(bt_rank)} / {total}"
                                if bt_rank and total
                                else "—"
                            ),
                            help=f"Betweenness value: {float(seed['betweenness']):.4f}",
                        )
                    with cols_m[3]:
                        dg_rank = rank_stats.get("dg_rank")
                        st.metric(
                            "Degree rank",
                            (
                                f"#{int(dg_rank)} / {total}"
                                if dg_rank and total
                                else "—"
                            ),
                            help=f"Degree (weighted): {float(seed['degree']):.2f}",
                        )

                    nodes_in_ego = payload.get("nodes") or []
                    edges_in_ego = payload.get("edges") or []
                    communities_reached = sorted({
                        int(n["raw"]["community"])
                        for n in nodes_in_ego
                        if int(n["raw"].get("community", -1)) >= 0
                        and n["id"] != seed["id"]
                    })

                    summary_lines = [
                        f"- **Source type:** {seed['source_type']}",
                        f"- **Mentions in corpus:** {int(seed['frequency'])}",
                        f"- **Ego graph size:** {len(nodes_in_ego)} concepts, "
                        f"{len(edges_in_ego)} relationships "
                        f"(depth={int(ego_depth)}, per-hop={int(ego_per_hop)}).",
                        f"- **Communities reached by ego:** "
                        + (
                            ", ".join(f"#{c}" for c in communities_reached)
                            if communities_reached
                            else "none"
                        ),
                    ]
                    if int(seed["community"]) >= 0:
                        summary_lines.append(
                            "- Click the **Communities** tab and search for "
                            f"community **#{int(seed['community'])}** to see "
                            "the rest of this concept's home cluster."
                        )
                    st.markdown("\n".join(summary_lines))

                    # ── Direct neighbours table ─────────────────────
                    if edges_in_ego:
                        nbr_rows: list[dict] = []
                        label_by_id = {
                            n["id"]: n["raw"]["label"] for n in nodes_in_ego
                        }
                        community_by_id = {
                            n["id"]: int(n["raw"].get("community", -1))
                            for n in nodes_in_ego
                        }
                        seed_id_key = seed["id"]
                        for e in edges_in_ego:
                            if e["from"] != seed_id_key and e["to"] != seed_id_key:
                                continue
                            if e["from"] == seed_id_key:
                                other_id = e["to"]
                                direction = "→"
                            else:
                                other_id = e["from"]
                                direction = "←"
                            nbr_rows.append({
                                "direction": direction,
                                "neighbour": label_by_id.get(other_id, other_id),
                                "community": community_by_id.get(other_id, -1),
                                "weight": float(e["raw"].get("weight") or 0.0),
                                "top_verb": e["raw"].get("top_verb") or "",
                                "types": e["raw"].get("types") or "",
                                "sentiment": e["raw"].get("sentiment"),
                            })
                        if nbr_rows:
                            nbr_df = pd.DataFrame(nbr_rows).sort_values(
                                "weight", ascending=False,
                            )
                            st.markdown("**Direct neighbours**")
                            st.dataframe(
                                nbr_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "weight": st.column_config.NumberColumn(
                                        "Weight", format="%.1f",
                                    ),
                                    "sentiment": st.column_config.NumberColumn(
                                        "Sentiment", format="%+.2f",
                                    ),
                                    "community": st.column_config.NumberColumn(
                                        "Comm", format="%d",
                                    ),
                                },
                            )
        else:
            try:
                payload = _load_vis_payload(
                    source_key,
                    net_top_n,
                    rank_property,
                    min_edge_weight,
                    slice_id,
                    colour_by,
                )
            except Exception as e:
                st.error(f"Failed to fetch vis payload: {e}")
                payload = {"nodes": [], "edges": [], "stats": {}}

        n_nodes = len(payload.get("nodes", []))
        n_edges = len(payload.get("edges", []))
        if search_mode and n_nodes > 0:
            seed_label = locals().get("seed", {}).get("label", search_q_clean)
            st.caption(
                f"Showing the **ego graph** around **{seed_label}** — "
                f"{n_nodes} nodes, {n_edges} edges "
                f"(depth={int(ego_depth)}, per-hop={int(ego_per_hop)}, "
                f"min weight={min_edge_weight}, colour=`{colour_by}`"
                + (f", slice=`{slice_id}`" if slice_id else "")
                + "). Clear the search box above to return to the Top-N view."
            )
        else:
            st.caption(
                f"Showing **{n_nodes}** nodes and **{n_edges}** edges "
                f"(rank=`{rank_property}`, min weight={min_edge_weight}, "
                f"colour=`{colour_by}`"
                + (f", slice=`{slice_id}`" if slice_id else "")
                + ")."
            )
        vs_stats = payload.get("stats") or {}
        if n_nodes > 0 and vs_stats:
            z1, z2, z3, z4 = st.columns(4)
            with z1:
                st.metric(
                    "Top rank in view",
                    f"{float(vs_stats.get('max_rank') or 0):.4f}",
                    help=f"Maximum {vs_stats.get('rank_property', rank_property)} among displayed nodes.",
                )
            with z2:
                st.metric(
                    "Strongest edge (weight)",
                    f"{float(vs_stats.get('max_weight') or 0):.1f}",
                )
            with z3:
                st.metric(
                    "Sentiment on edges",
                    "yes" if vs_stats.get("has_sentiment") else "no",
                )
            with z4:
                st.metric(
                    "Colour mode",
                    str(vs_stats.get("colour_by") or colour_by),
                )

        if n_nodes > 0:
            with st.expander("Why can this differ from the Insights tab?", expanded=False):
                st.markdown(
                    "This tab renders a **Top-N subgraph** for clarity. "
                    "**Insights** runs on the **full** scoped graph in Neo4j "
                    "(centralities, communities, SPOF logic). If a concept does "
                    "not appear here, raise **Top N** or switch the **rank** metric."
                )

        if n_nodes == 0:
            st.info(
                "No nodes match the current filters. Try lowering "
                "the minimum edge weight or picking a different slice."
            )
        else:
            from src.visualisation.vis_network_html import render_vis_html

            html = render_vis_html(
                payload,
                height="720px",
                physics=physics_choice,
                show_legend=True,
            )
            st.components.v1.html(html, height=740, scrolling=False)

    with tab_ins:
        st.subheader("📊 SNA Insights")
        st.caption(
            "Cypher-driven observations turned into Key findings, Risk "
            "warnings and Action recommendations. Optional LLM polishing "
            "rewrites each card in executive-friendly language."
        )

        # ── Controls ────────────────────────────────────────────────
        ctrl_a, ctrl_b = st.columns([1, 2])
        with ctrl_a:
            polish = st.toggle(
                "✨ Polish with LLM",
                value=False,
                help=(
                    "Rewrite each insight using the configured LLM provider "
                    "(see the Chat tab → Chatbot settings to pick a "
                    "provider). Off = deterministic templates, free, fast."
                ),
            )
        with ctrl_b:
            _provider_options = [
                "echo", "openai", "dashscope",
                "huggingface_api", "huggingface_local",
            ]
            _default = (config.LLM_PROVIDER or "echo").lower()
            _alias = {"qwen": "dashscope", "aliyun": "dashscope",
                      "modelscope": "dashscope"}
            _default = _alias.get(_default, _default)
            if _default not in _provider_options:
                _default = "echo"
            polish_provider = st.selectbox(
                "LLM provider for polishing",
                _provider_options,
                index=_provider_options.index(_default),
                disabled=not polish,
                key="ins_polish_provider",
            )

        # ── Compute raw insights ───────────────────────────────────
        try:
            bundle = _load_raw_insights(source_key)
        except Exception as e:
            st.error(f"Failed to compute insights: {e}")
            st.stop()

        # ── Polish on demand ───────────────────────────────────────
        polished_lookup: dict[str, str] = {}
        if polish:
            with st.spinner("Polishing insights with LLM…"):
                for items in bundle.values():
                    for ins in items:
                        if not ins["available"]:
                            continue
                        try:
                            polished_lookup[ins["id"]] = _polish_insight_body(
                                ins["id"],
                                ins["body"],
                                polish_provider,
                                config.LLM_MODEL,
                            )
                        except Exception as e:
                            polished_lookup[ins["id"]] = ins["body"]
                            st.toast(
                                f"Polish failed for {ins['title']}: {e}",
                                icon="⚠️",
                            )

        st.divider()

        # ── Three-column layout ────────────────────────────────────
        col_key, col_risk, col_act = st.columns(3)

        with col_key:
            st.markdown("### 🔑 Key Insights")
            st.caption("What the network looks like.")
            for ins in bundle["key"]:
                _render_insight_card(ins, polished_lookup.get(ins["id"]))

        with col_risk:
            st.markdown("### ⚠️ Risk Insights")
            st.caption("Where the network is fragile.")
            for ins in bundle["risk"]:
                _render_insight_card(ins, polished_lookup.get(ins["id"]))

        with col_act:
            st.markdown("### ✅ Action Recommendations")
            st.caption("What to do about it.")
            for ins in bundle["action"]:
                _render_insight_card(ins, polished_lookup.get(ins["id"]))

        # ── Footer ─────────────────────────────────────────────────
        with st.expander("How are these computed?"):
            st.markdown(
                """
Every card is produced by one Cypher query against the Neo4j graph,
using metrics already written by `pipeline.py analyse` (PageRank,
betweenness, Louvain communities, weakly-connected components,
optional sentiment).

* **Key Insights** — descriptive: hubs, communities, bridges,
  isolated clusters, cross-source overlap.
* **Risk Insights** — diagnostic: single points of failure (high
  PageRank × high betweenness), echo chambers (>85% intra-community
  edges), negative-sentiment clusters, sparse coverage,
  network fragmentation.
* **Action Recommendations** — prescriptive: diversify away from
  hubs, leverage bridge concepts, investigate isolates, anchor
  cross-source communication.

Toggle **Polish with LLM** to re-phrase each observation using the
selected provider. The deterministic template is always available
as a fallback.
                """
            )

    with tab2:
        st.subheader("Centrality — who matters, and how?")
        with st.container(border=True):
            st.markdown("### 👋 Start here")
            st.markdown(
                """
**This tab is about influence.** You will see the same network through
five different lenses — from “famous in the graph” to “sits on the
shortest paths between other ideas.”

- **Tables below** — the headline concepts under each lens (great for
  scanning before a meeting).
- **Scatter plot** — mixes two lenses at once so you can spot ideas that
  are both *popular* and *load-bearing* (often worth a closer look).
- **Generate** — when you are ready, ask the LLM for a short, plain-English
  story of what stands out (no tokens are used until you click).

_Open **Optional: metric cheat sheet** only when you want the definitions._
                """
            )
        _llm_blurb_panel(
            title="Turn the leaderboards into a short brief",
            metric_name="Centrality leaderboard",
            description=(
                "Top concepts by PageRank, betweenness, degree, closeness, "
                "and eigenvector for this source; highlight unusual leaders "
                "and anything that looks like a bottleneck."
            ),
            df=centrality_df[
                [c for c in [
                    "label", "pagerank", "betweenness", "degree",
                    "closeness", "eigenvector", "community",
                ] if c in centrality_df.columns]
            ],
            rank_column="pagerank" if "pagerank" in centrality_df.columns else None,
            top_rows=10,
            state_key="blurb_centrality_overview",
            helper_caption=(
                "Optional one-click summary — pick a provider, then **Generate**. "
                "Perfect when you want wording you can paste into notes or slides."
            ),
        )

        with st.expander("Optional: metric cheat sheet", expanded=False):
            st.markdown(
                """
- **PageRank** — *importance by association*. A concept with a high
  PageRank is referenced by many other influential concepts. Useful for
  finding **anchors / hubs** in the discourse.
- **Betweenness** — *control over information flow*. Concepts that lie
  on many shortest paths between pairs of other concepts. High
  betweenness = **bridge / broker**.
- **Degree** — *raw connectivity*. How many distinct neighbours.
- **Closeness** — *reach*. Inverse of average distance to every other
  concept; high = **fast spreader**.
- **Eigenvector** — *prestige*. Like PageRank but recursive: high if
  connected to other high-eigenvector concepts.

**Reading the scatter chart:** a concept that sits **high on both**
PageRank and betweenness behaves like a **single point of failure** —
very visible *and* structurally load-bearing. High PageRank alone tends
to mean “popular anchor”; high betweenness alone often means “quiet
bridge” between themes.
                """
            )

        # ── Side-by-side leaderboards ──────────────────────────────
        st.markdown("#### Top 10 by each metric")
        leaderboard_cols = st.columns(5)
        metrics = ["pagerank", "betweenness", "degree", "closeness", "eigenvector"]
        for col, metric_name in zip(leaderboard_cols, metrics):
            with col:
                st.markdown(f"**{metric_name.title()}**")
                if metric_name in centrality_df.columns:
                    top10 = (
                        centrality_df[["label", metric_name, "community"]]
                        .nlargest(10, metric_name)
                        .reset_index(drop=True)
                    )
                    top10.index = top10.index + 1
                    st.dataframe(
                        top10,
                        use_container_width=True,
                        hide_index=False,
                        column_config={
                            metric_name: st.column_config.NumberColumn(
                                metric_name, format="%.4f",
                            ),
                            "community": st.column_config.NumberColumn(
                                "comm", format="%d",
                            ),
                        },
                    )
                else:
                    st.caption("(metric not populated)")

        # ── PageRank vs Betweenness scatter ────────────────────────
        if (
            "pagerank" in centrality_df.columns
            and "betweenness" in centrality_df.columns
        ):
            st.markdown("#### PageRank × Betweenness — spot the “can’t lose” ideas")
            st.caption(
                "**Upper-right** — loud *and* load-bearing (often your “if this "
                "disappears, the story wobbles” concepts). **Upper-left** — "
                "connects themes without being the biggest star. **Lower-right** "
                "— big names that mostly stay inside their neighbourhood. "
                "Hover any dot for the label."
            )
            try:
                import altair as alt

                scatter_top = (
                    centrality_df.nlargest(80, "pagerank")
                    .copy()
                )
                # Rank product highlights SPOFs (high on both metrics).
                pr_rank = scatter_top["pagerank"].rank(ascending=False)
                bc_rank = scatter_top["betweenness"].rank(ascending=False)
                scatter_top["spof_rank"] = pr_rank + bc_rank
                scatter_top["is_spof"] = scatter_top["spof_rank"] <= 10

                chart = (
                    alt.Chart(scatter_top)
                    .mark_circle(size=120, opacity=0.7)
                    .encode(
                        x=alt.X(
                            "pagerank:Q",
                            scale=alt.Scale(zero=False),
                            title="PageRank →",
                        ),
                        y=alt.Y(
                            "betweenness:Q",
                            scale=alt.Scale(zero=False),
                            title="Betweenness ↑",
                        ),
                        color=alt.Color(
                            "is_spof:N",
                            scale=alt.Scale(
                                domain=[True, False],
                                range=["#d62728", "#7f8fa6"],
                            ),
                            legend=alt.Legend(title="Pinch-point pick?"),
                        ),
                        size=alt.Size(
                            "frequency:Q" if "frequency" in scatter_top.columns
                            else "pagerank:Q",
                            legend=None,
                        ),
                        tooltip=[
                            "label",
                            alt.Tooltip("pagerank:Q", format=".4f"),
                            alt.Tooltip("betweenness:Q", format=".4f"),
                            "community:Q",
                        ],
                    )
                    .properties(height=380)
                    .interactive()
                )
                # Label the SPOFs explicitly so the dot's identity is
                # readable without hovering.
                labels = (
                    alt.Chart(scatter_top[scatter_top["is_spof"]])
                    .mark_text(
                        align="left", dx=8, dy=-4, fontSize=11,
                        fontWeight="bold", color="#d62728",
                    )
                    .encode(x="pagerank:Q", y="betweenness:Q", text="label:N")
                )
                st.altair_chart(chart + labels, use_container_width=True)
            except Exception as e:
                st.warning(f"Scatter plot unavailable: {e}")

        # ── Detailed per-metric drill-down ─────────────────────────
        with st.expander("Detailed table + bar chart for one metric", expanded=False):
            metric = st.selectbox(
                "Plot metric",
                metrics,
                key="centrality_drill_metric",
            )
            if metric in centrality_df.columns:
                st.dataframe(
                    centrality_df.head(top_n),
                    use_container_width=True,
                )
                chart_data = (
                    centrality_df.nlargest(top_n, metric)
                    .set_index("label")[metric]
                )
                st.bar_chart(chart_data)
            else:
                st.info(
                    f"Metric '{metric}' is not present. "
                    f"Run `python pipeline.py analyse --source {source_key}` first."
                )

    with tab3:
        st.subheader("Communities — the hidden “theme teams”")
        with st.container(border=True):
            st.markdown("### 👋 Start here")
            st.markdown(
                """
**Think of each row as a conversation theme** — ideas that tend to show
up together more than with the rest of the network.

- **Two anchor columns** help you read each cluster quickly: what shows
  up *often* versus what acts like a *hub* inside the theme.
- **Cohesion** (when shown) tells you how “self-contained” a theme is —
  low scores usually mean the cluster talks a lot *outward* to other
  themes (interface topics).
- **Inspect one community** — zoom into members, sample links, and which
  other clusters it reaches.

_Use **Generate** when you want a friendly narrative of the table below._
                """
            )
        _llm_blurb_panel(
            title="Summarise the theme landscape in plain English",
            metric_name="Louvain communities",
            description=(
                "Community table: sizes, top concepts by frequency and "
                "PageRank, cohesion and sentiment when present; call out "
                "the largest, most porous, and most self-contained clusters."
            ),
            df=comm_df,
            rank_column="size",
            top_rows=8,
            state_key="blurb_communities",
            helper_caption=(
                "Optional LLM brief — same data as the table below, "
                "rephrased for stakeholders. No API call until you click **Generate**."
            ),
        )

        with st.expander("Optional: how clusters are built", expanded=False):
            st.markdown(
                """
Communities are groups of concepts that mention each other more than
they mention the rest of the network — Louvain modularity finds them
automatically. Each community is shown by **two anchor lists**:

- **By frequency** — what people mention most inside this cluster
  (surface-level vocabulary).
- **By PageRank** — what is most structurally important inside this
  cluster (the cluster's "hub words"). The Hub / SPOF cards on the
  Insights tab use the PageRank ranking, so this column is the one
  to cross-reference.

Pick a community below to see its members, sample edges, and which
neighbouring communities it talks to.

The **All communities** table adds **cohesion** when linkage stats are
available: internal vs outward cross-community edge weight from nodes in
that cluster (low cohesion ≈ interface / porous theme).
                """
            )

        st.markdown("#### All communities")
        if not comm_df.empty:
            if "cohesion" in comm_df.columns:
                st.caption(
                    "**Cohesion** is a simple readability score: how much of a "
                    "theme’s “link budget” stays *inside* the cluster versus "
                    "flows outward. **Lower** usually means the theme talks a lot "
                    "to others (a bridge topic). **Higher** means it is more of its "
                    "own world."
                )
                sized = comm_df[comm_df["size"] >= 5].copy()
                sized_co = sized.dropna(subset=["cohesion"])
                if not sized_co.empty:
                    most_porous = sized_co.nsmallest(1, "cohesion").iloc[0]
                    most_tight = sized_co.nlargest(1, "cohesion").iloc[0]
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Communities", f"{len(comm_df)}")
                    with m2:
                        st.metric(
                            "Largest cluster",
                            f"n={int(comm_df['size'].max())}",
                        )
                    with m3:
                        _community_extreme_cohesion_metric(
                            label="Most porous (n≥5)",
                            community_id=int(most_porous["community"]),
                            cohesion=float(most_porous["cohesion"]),
                            porous=True,
                            help_text=(
                                "Lowest cohesion among clusters with n≥5 "
                                "(more outward links = more porous)."
                            ),
                        )
                    with m4:
                        _community_extreme_cohesion_metric(
                            label="Most insular (n≥5)",
                            community_id=int(most_tight["community"]),
                            cohesion=float(most_tight["cohesion"]),
                            porous=False,
                            help_text=(
                                "Highest cohesion among clusters with n≥5 "
                                "(more inward links = more self-contained)."
                            ),
                        )
                else:
                    st.metric("Communities", f"{len(comm_df)}")
            else:
                st.metric("Communities", f"{len(comm_df)}")

        # Reorder columns so the PageRank-anchor list is right next to
        # the frequency-anchor list (when both columns exist).
        display_cols = [
            c for c in [
                "community", "size", "top_concepts",
                "top_pagerank", "avg_pagerank",
                "cohesion", "internal_weight", "bridge_out_weight",
                "avg_internal_sentiment", "bridge_out_edges",
            ] if c in comm_df.columns
        ]
        st.dataframe(
            comm_df[display_cols] if display_cols else comm_df,
            use_container_width=True,
            column_config={
                "top_concepts":  st.column_config.TextColumn("Top by frequency"),
                "top_pagerank":  st.column_config.TextColumn("Top by PageRank"),
                "avg_pagerank":  st.column_config.NumberColumn(
                    "Avg PageRank", format="%.4f",
                ),
                "cohesion": st.column_config.NumberColumn(
                    "Cohesion", format="%.2f",
                    help="Internal weight share of directed mass leaving nodes in this community.",
                ),
                "internal_weight": st.column_config.NumberColumn(
                    "Int. weight", format="%.0f",
                ),
                "bridge_out_weight": st.column_config.NumberColumn(
                    "Bridge-out Δw", format="%.0f",
                ),
                "avg_internal_sentiment": st.column_config.NumberColumn(
                    "Avg int. sent.", format="%+.2f",
                ),
                "bridge_out_edges": st.column_config.NumberColumn(
                    "Bridge-out edges", format="%d",
                ),
            },
        )

        from collections import Counter

        sizes = Counter(partition.values())
        st.markdown("#### Community size distribution")
        st.bar_chart(pd.Series(sizes).sort_index().rename("Size"))

        # ── Per-community drill-down ───────────────────────────────
        if not comm_df.empty:
            st.divider()
            st.markdown("#### Pick a theme to zoom in")
            st.caption("Choose a cluster — we will show its stars, its bridges, and a few strong internal links.")

            ranked_communities = comm_df.sort_values(
                "size", ascending=False,
            )
            picker_options = []
            for _, row in ranked_communities.iterrows():
                cid = int(row["community"])
                anchors = (
                    row.get("top_pagerank")
                    or row.get("top_concepts")
                    or ""
                )
                short = ", ".join(anchors.split(",")[:3]) if anchors else "n/a"
                picker_options.append(
                    (cid, f"community {cid} — {short} (n={int(row['size'])})")
                )
            chosen = st.selectbox(
                "Which cluster?",
                options=[opt[0] for opt in picker_options],
                format_func=lambda cid: dict(picker_options)[cid],
                key="comm_drill_pick",
            )

            try:
                members_df = _load_community_members(source_key, chosen, top_n=25)
                edges_df = _load_community_edges(source_key, chosen, top_n=15)
                neigh_df = _load_community_neighbours(
                    source_key, chosen, top_n=10,
                )
            except Exception as e:
                st.error(f"Drill-down query failed: {e}")
                members_df = edges_df = neigh_df = pd.DataFrame()

            d1, d2 = st.columns([2, 1])
            with d1:
                st.markdown(f"**Top concepts in community {chosen}**")
                if members_df.empty:
                    st.info("No members found for this community.")
                else:
                    st.dataframe(
                        members_df,
                        use_container_width=True,
                        column_config={
                            "pagerank":     st.column_config.NumberColumn(format="%.4f"),
                            "betweenness":  st.column_config.NumberColumn(format="%.4f"),
                        },
                    )
            with d2:
                st.markdown("**Reaches into…**")
                if neigh_df.empty:
                    st.caption("No cross-community edges from this cluster.")
                else:
                    neigh_df = neigh_df.copy()
                    neigh_df["neighbour_community"] = neigh_df[
                        "neighbour_community"
                    ].astype(int)
                    st.dataframe(
                        neigh_df,
                        use_container_width=True,
                        column_config={
                            "total_weight": st.column_config.NumberColumn(
                                "Edge weight", format="%.1f",
                            ),
                            "neighbour_community": st.column_config.NumberColumn(
                                "→ comm", format="%d",
                            ),
                        },
                    )

            st.markdown("**Strongest internal edges**")
            if edges_df.empty:
                st.caption("No edges found for this community.")
            else:
                ec = {
                    "weight": st.column_config.NumberColumn(format="%.1f"),
                    "sentiment": st.column_config.NumberColumn(format="%+.2f"),
                }
                if "top_verb" in edges_df.columns:
                    ec["top_verb"] = st.column_config.TextColumn("verb")
                st.dataframe(
                    edges_df,
                    use_container_width=True,
                    column_config=ec,
                )

    with tab4:
        st.subheader("Brokers — who translates one theme into another?")
        with st.container(border=True):
            st.markdown("### 👋 Start here")
            st.markdown(
                """
**Brokers are the “bilingual” ideas** — they sit between Louvain
communities, so they often explain how one storyline links to another.

- **Scores below** rank who does the most *cross-community* heavy lifting.
- **Inspect a broker** to see its home cluster, where it reaches, and the
  strongest bridges it builds.
- **Generate** turns the table into a short narrative you can share.

_On **Combined** data you also get a second lens — vocabulary that mixes
**policy** and **Yelp** neighbours (not the same as Louvain brokers, but
often fascinating)._
                """
            )
        if not brokers_df.empty:
            _llm_blurb_panel(
                title="Explain who is stitching the network together",
                metric_name="Broker score",
                description=(
                    "Top broker concepts: broker_score, betweenness, home "
                    "community, cross-community edge counts; describe who "
                    "bridges themes and why that matters for risk or opportunity."
                ),
                df=brokers_df,
                rank_column=(
                    "broker_score" if "broker_score" in brokers_df.columns else None
                ),
                top_rows=8,
                state_key="blurb_brokers",
                helper_caption=(
                    "Optional LLM story — uses the broker table below. "
                    "Pick a provider, then **Generate** (nothing is sent until you do)."
                ),
            )

        with st.expander("Optional: what “broker score” actually measures", expanded=False):
            st.markdown(
                """
A **hub** is a popular concept *inside* its own community. A **broker**
is different: its neighbours span **multiple** communities — it is the
kind of vocabulary that helps one cluster “talk to” another.

The score is **`betweenness × cross_community_edges`**:

- **High betweenness alone** can still happen mostly *inside* one dense
  theme — not necessarily a bridge.
- **A high broker score** rewards concepts whose neighbours really do sit
  in different communities.

Use the drill-down to see *where* each broker lives and *which* clusters
it reaches with the strongest edges.
                """
            )

        if source_key == "combined":
            st.markdown("#### When policy language meets review language")
            st.caption(
                "A softer lens than Louvain brokers: these concepts have "
                "neighbours from **both** corpora (`cross_source_neighbors`). "
                "Great for spotting shared vocabulary between formal text and "
                "everyday reviews."
            )
            try:
                bx = _load_bridging_concepts(source_key)
                if bx.empty:
                    st.caption("No cross-source bridge rows returned yet.")
                else:
                    bcols = [
                        c for c in [
                            "label", "source_type", "betweenness",
                            "cross_source_neighbors", "total_neighbors",
                            "cross_ratio", "bridge_score",
                        ] if c in bx.columns
                    ]
                    st.dataframe(
                        bx[bcols],
                        use_container_width=True,
                        column_config={
                            "betweenness": st.column_config.NumberColumn(
                                format="%.4f",
                            ),
                            "cross_ratio": st.column_config.NumberColumn(
                                "cross nb %", format="%.2f",
                            ),
                            "bridge_score": st.column_config.NumberColumn(
                                format="%.3f",
                            ),
                        },
                    )
            except Exception as e:
                st.warning(f"Cross-source bridge table unavailable: {e}")

        if brokers_df.empty:
            st.info(
                "No broker scores yet — run "
                f"`python pipeline.py analyse --source {source_key}` once your "
                "graph is loaded."
            )
        else:
            display_brokers = brokers_df.head(20).copy()
            b0, b1, b2 = st.columns(3)
            with b0:
                st.metric("Brokers listed", len(display_brokers))
            with b1:
                st.metric(
                    "Top broker score",
                    f"{float(display_brokers.iloc[0]['broker_score']):.1f}",
                    help="betweenness × distinct cross-community neighbours",
                )
            with b2:
                mx = display_brokers["cross_community_edges"].max()
                st.metric("Max cross-community degree", int(mx))
            display_brokers.index = display_brokers.index + 1
            st.dataframe(
                display_brokers,
                use_container_width=True,
                column_config={
                    "broker_score": st.column_config.NumberColumn(format="%.2f"),
                    "betweenness":  st.column_config.NumberColumn(format="%.4f"),
                    "community":    st.column_config.NumberColumn("home", format="%d"),
                    "cross_community_edges": st.column_config.NumberColumn("cross"),
                    "total_edges": st.column_config.NumberColumn("total"),
                },
            )

            st.divider()
            st.markdown("#### Pick a broker to follow the bridges")
            st.caption("See where they “live”, which clusters they touch, and their strongest cross-cluster links.")
            broker_pick = st.selectbox(
                "Which broker?",
                options=brokers_df["label"].head(20).tolist(),
                key="broker_drill_pick",
            )

            try:
                drill = _load_broker_drilldown(source_key, broker_pick)
                home = int(drill.get("home_community", -1))
                reach = drill.get("reached_communities", pd.DataFrame())
                cross = drill.get("cross_edges", pd.DataFrame())
            except Exception as e:
                st.error(f"Broker drill-down failed: {e}")
                home, reach, cross = -1, pd.DataFrame(), pd.DataFrame()

            d1, d2 = st.columns([1, 2])
            with d1:
                st.markdown("**Home community**")
                if home < 0:
                    st.caption("(no community assigned)")
                else:
                    st.metric("Community ID", home)
                st.markdown("**Reaches into…**")
                if reach.empty:
                    st.caption("No cross-community edges from this broker.")
                else:
                    reach_disp = reach.copy()
                    reach_disp["neighbour_community"] = (
                        reach_disp["neighbour_community"].astype(int)
                    )
                    st.dataframe(
                        reach_disp,
                        use_container_width=True,
                        column_config={
                            "total_weight": st.column_config.NumberColumn(
                                "weight", format="%.1f",
                            ),
                            "neighbour_community": st.column_config.NumberColumn(
                                "comm", format="%d",
                            ),
                        },
                    )

            with d2:
                st.markdown("**Strongest cross-community edges**")
                if cross.empty:
                    st.caption("No cross-community edges to show.")
                else:
                    cross_disp = cross.copy()
                    cross_disp["neighbour_community"] = (
                        cross_disp["neighbour_community"].astype(int)
                    )
                    xc = {
                        "weight": st.column_config.NumberColumn(format="%.1f"),
                        "neighbour_community": st.column_config.NumberColumn(
                            "→ comm", format="%d",
                        ),
                    }
                    if "top_verb" in cross_disp.columns:
                        xc["top_verb"] = st.column_config.TextColumn("verb")
                    if "sentiment" in cross_disp.columns:
                        xc["sentiment"] = st.column_config.NumberColumn(
                            format="%+.2f",
                        )
                    st.dataframe(
                        cross_disp,
                        use_container_width=True,
                        column_config=xc,
                    )

    with tab5:
        st.subheader("Temporal — how the story changes slice by slice")
        st.caption(
            "Each **slice** is a tagged subgraph in Neo4j (`slice_id`). "
            "Run `python pipeline.py temporal --source <src> --temporal N` "
            "to build slices; omit `--skip-gds` if you want centralities per slice."
        )
        if source_key == "combined":
            st.caption(
                "**Combined** scope: this tab lists **Yelp review calendar years** "
                "only (slice ids like `slice_00_2019`). Policy PDF chunks have no "
                "real dates, so page-range slices stay in Neo4j but are hidden here."
            )
        try:
            temp_df = _temporal_slice_summary(source_key)
        except Exception as e:
            temp_df = pd.DataFrame()
            st.error(f"Failed to read temporal slices: {e}")
        if temp_df.empty:
            if source_key == "combined":
                st.info(
                    "No **Yelp calendar-year** temporal slices found for "
                    f"**{source_key}** (ids matching `slice_XX_YYYY`). "
                    "Run `python pipeline.py temporal --source combined "
                    "--temporal 3` after ETL. "
                    "If you only see policy page-chunk slices in Neo4j, they are "
                    "hidden here because they are not calendar-dated."
                )
            else:
                st.info(
                    f"No temporal slices found for **{source_key}**. "
                    f"Run `python pipeline.py temporal --source {source_key} "
                    "--temporal 3` (or `4`) after ETL."
                )
        else:
            enriched_df = pd.DataFrame()
            pairwise_df = pd.DataFrame()
            try:
                enriched_df = _load_enriched_temporal_summary(source_key)
            except Exception as e:
                st.warning(f"Enriched temporal summary unavailable: {e}")
            try:
                pairwise_df = _load_pairwise_compare(source_key)
            except Exception as e:
                st.warning(f"Pairwise comparison unavailable: {e}")

            # ── 1) Stakeholder insights (top) ─────────────────────────
            with st.container(border=True):
                st.markdown("### 👋 What you are looking at")
                st.markdown(
                    """
**Three angles** on the same slice sequence:

| Angle | Plain language |
|-------|------------------|
| **Trend** | Is the network **growing or shrinking**? Getting **denser**? **Mood** shifting? |
| **Drift** | Between **neighbouring** slices, what vocabulary **shows up**, **drops out**, and how much **overlap** remains? |
| **Comparison** | **First vs last** snapshot — how different did the discourse become overall? |

Use **Polish with LLM** when you want board-ready wording. Nothing is sent
to the model until you enable polish.
                """
                )

            st.markdown("### 🕒 Temporal insights")
            tcol_a, tcol_b = st.columns([1, 2])
            with tcol_a:
                t_polish = st.toggle(
                    "✨ Polish with LLM",
                    value=False,
                    key="temporal_polish",
                )
            with tcol_b:
                provider_options = [
                    "echo", "openai", "dashscope",
                    "huggingface_api", "huggingface_local",
                ]
                t_provider = st.selectbox(
                    "LLM provider",
                    provider_options,
                    index=_default_provider_index(),
                    disabled=not t_polish,
                    key="temporal_polish_provider",
                )

            try:
                bundle_dict = _load_temporal_insights(source_key)
            except Exception as e:
                bundle_dict = {"trend": [], "drift": [], "comparison": []}
                st.error(f"Failed to compute temporal insights: {e}")

            polished_lookup: dict[str, str] = {}
            if t_polish:
                with st.spinner("Polishing temporal insights with LLM…"):
                    for items in bundle_dict.values():
                        for ins in items:
                            if not ins["available"]:
                                continue
                            try:
                                polished_lookup[ins["id"]] = (
                                    _polish_insight_body(
                                        ins["id"],
                                        ins["body"],
                                        t_provider,
                                        config.LLM_MODEL,
                                    )
                                )
                            except Exception as e:
                                polished_lookup[ins["id"]] = ins["body"]
                                st.toast(
                                    f"Polish failed for {ins['title']}: {e}",
                                    icon="⚠️",
                                )

            col_t, col_d, col_c = st.columns(3)
            with col_t:
                st.markdown("#### 📈 Trend")
                st.caption("Big-picture shape of the network over time.")
                for ins in bundle_dict["trend"]:
                    _render_insight_card(ins, polished_lookup.get(ins["id"]))
            with col_d:
                st.markdown("#### 🔀 Drift")
                st.caption("Step-by-step vocabulary change.")
                if bundle_dict["drift"]:
                    for ins in bundle_dict["drift"]:
                        _render_insight_card(ins, polished_lookup.get(ins["id"]))
                else:
                    st.info("Need at least two slices for drift cards.")
            with col_c:
                st.markdown("#### 🆚 Comparison")
                st.caption("Earliest vs latest snapshot.")
                for ins in bundle_dict["comparison"]:
                    _render_insight_card(ins, polished_lookup.get(ins["id"]))

            st.divider()

            # ── 2) Trend visuals + filters ─────────────────────────────
            st.markdown("### 📊 Trend explorer")
            st.caption(
                "Filter which slices and metrics appear in the charts — "
                "useful when presenting only part of the timeline."
            )
            if not enriched_df.empty and "slice_id" in enriched_df.columns:
                all_ids = sorted(enriched_df["slice_id"].dropna().unique().tolist())
                _ef = enriched_df.drop_duplicates(subset=["slice_id"], keep="first")
                label_map = dict(zip(_ef["slice_id"], _ef["slice"]))
                sel_ids = st.multiselect(
                    "Slices to include",
                    options=all_ids,
                    default=all_ids,
                    format_func=lambda sid: f"{label_map.get(sid, sid)}",
                    key="temporal_plot_slice_ids",
                )
                metric_map: dict[str, str] = {}
                if "nodes" in enriched_df.columns:
                    metric_map["Nodes"] = "nodes"
                if "edges" in enriched_df.columns:
                    metric_map["Edges"] = "edges"
                if "density" in enriched_df.columns:
                    metric_map["Density"] = "density"
                if "avg_sentiment" in enriched_df.columns:
                    if enriched_df["avg_sentiment"].notna().any():
                        metric_map["Avg sentiment"] = "avg_sentiment"
                default_metrics = [k for k in ["Nodes", "Edges"] if k in metric_map]
                if not default_metrics:
                    default_metrics = list(metric_map.keys())[:2]
                sel_metrics = st.multiselect(
                    "Metrics to plot",
                    options=list(metric_map.keys()),
                    default=default_metrics,
                    key="temporal_plot_metrics",
                )
                norm_overlay = st.checkbox(
                    "Overlay metrics on one chart (0–1 scale per metric)",
                    value=False,
                    key="temporal_plot_normalize",
                )
                plot_base = enriched_df[
                    enriched_df["slice_id"].isin(sel_ids or all_ids)
                ].copy()
                plot_base["_ord"] = plot_base["slice_id"].map(
                    {sid: i for i, sid in enumerate(all_ids)},
                )
                plot_base = plot_base.sort_values("_ord")
                inv_label = {v: k for k, v in metric_map.items()}
                if sel_metrics and not plot_base.empty:
                    cols = [metric_map[m] for m in sel_metrics if m in metric_map]
                    cols = [c for c in cols if c in plot_base.columns]
                    if cols:
                        idx = plot_base["slice"].fillna(plot_base["slice_id"])
                        if norm_overlay and len(cols) > 1:
                            ndf = plot_base[["slice_id", "slice"] + cols].copy()
                            for c in cols:
                                lo, hi = ndf[c].min(), ndf[c].max()
                                ndf[c + "_n"] = (
                                    (ndf[c] - lo) / (hi - lo)
                                    if hi is not None and hi > lo
                                    else 0.5
                                )
                            idx_n = ndf["slice"].fillna(ndf["slice_id"])
                            chart_df = ndf.set_index(idx_n)[[c + "_n" for c in cols]]
                            chart_df.columns = [inv_label[c] for c in cols]
                            st.line_chart(chart_df)
                            st.caption(
                                "Each line is min–max normalised within the "
                                "selected slices so you can compare shape, not units."
                            )
                        else:
                            chart_df = plot_base.set_index(idx)[cols]
                            chart_df.columns = [inv_label[c] for c in cols]
                            st.line_chart(chart_df)
                else:
                    st.caption("Pick at least one metric to plot.")
            else:
                st.info(
                    "Enriched per-slice stats are not available — charts need "
                    "`enriched_summary` data (check temporal run completed)."
                )

            if (
                not pairwise_df.empty
                and "jaccard_nodes" in pairwise_df.columns
            ):
                st.markdown("#### Concept overlap between consecutive slices")
                st.caption(
                    "**Jaccard (nodes)** compares **canonical lemmas** (lowercase; "
                    "spaces and hyphens become underscores so labels line up with "
                    "``Concept.id``). Near 1 means the two slices reuse most of the "
                    "same vocabulary."
                )
                pw = pairwise_df.copy()
                if "slice_a" in pw.columns and "slice_b" in pw.columns:
                    pw["step"] = (
                        pw["slice_a"].astype(str) + " → " + pw["slice_b"].astype(str)
                    )
                    step_opts = pw["step"].tolist()
                    sel_steps = st.multiselect(
                        "Transitions to show",
                        options=step_opts,
                        default=step_opts,
                        key="temporal_jaccard_steps",
                    )
                    if sel_steps:
                        pw = pw[pw["step"].isin(sel_steps)]
                if pw.empty:
                    st.caption("No transitions selected for the chart.")
                else:
                    try:
                        import altair as alt

                        if "step" not in pw.columns and "slice_a" in pw.columns:
                            pw["step"] = (
                                pw["slice_a"].astype(str)
                                + " → "
                                + pw["slice_b"].astype(str)
                            )
                        jcol = "jaccard_nodes"
                        chart_pw = (
                            alt.Chart(pw)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "step:N",
                                    sort=None,
                                    title="Transition",
                                    axis=alt.Axis(labelAngle=-35, labelLimit=200),
                                ),
                                y=alt.Y(
                                    f"{jcol}:Q",
                                    title="Jaccard (nodes)",
                                    scale=alt.Scale(domain=[0, 1]),
                                ),
                                tooltip=[
                                    "step",
                                    alt.Tooltip(f"{jcol}:Q", format=".3f"),
                                ],
                            )
                            .properties(height=260)
                        )
                        st.altair_chart(chart_pw, use_container_width=True)
                    except Exception:
                        st.dataframe(pw, use_container_width=True)
            elif not pairwise_df.empty:
                st.dataframe(pairwise_df, use_container_width=True)

            with st.expander("Compare any two slices (pick A vs B)", expanded=False):
                ids_for_pick = sorted(temp_df["slice_id"].dropna().unique().tolist())
                if len(ids_for_pick) >= 2:
                    c1, c2, c3 = st.columns([2, 2, 1])
                    with c1:
                        id_a = st.selectbox(
                            "Slice A",
                            ids_for_pick,
                            index=0,
                            key="temporal_cmp_a",
                        )
                    with c2:
                        id_b = st.selectbox(
                            "Slice B",
                            ids_for_pick,
                            index=min(1, len(ids_for_pick) - 1),
                            key="temporal_cmp_b",
                        )
                    with c3:
                        run_cmp = st.button("Compare", key="temporal_cmp_run")
                    if run_cmp and id_a and id_b and id_a != id_b:
                        detail = _load_detailed_slice_compare(
                            source_key, id_a, id_b,
                        )
                        if detail:
                            jn = detail.get("jaccard_nodes")
                            je = detail.get("jaccard_edges")
                            st.metric(
                                "Shared vocabulary (Jaccard, nodes)",
                                f"{float(jn):.2f}" if jn is not None else "n/a",
                            )
                            st.caption(
                                "Lemma overlap: same concept after **normalising** "
                                "spaces/hyphens to underscores (matches stored ``id``), "
                                "not raw display text."
                            )
                            if je is not None:
                                st.caption(
                                    f"Jaccard (edges): **{float(je):.2f}** — "
                                    "overlap of **undirected concept pairs** "
                                    "(same two labels linked in each slice)."
                                )
                            sd = detail.get("sentiment_delta")
                            if sd is not None:
                                st.caption(
                                    f"Average sentiment shift (B − A): **{sd:+.3f}**"
                                )
                            ap = detail.get("appeared_top") or []
                            dp = detail.get("disappeared_top") or []
                            if ap:
                                st.markdown("**Notable new terms in B**")
                                st.write(", ".join(f"`{t}`" for t in ap[:12]))
                            if dp:
                                st.markdown("**Notable terms that faded from A**")
                                st.write(", ".join(f"`{t}`" for t in dp[:12]))
                        else:
                            st.warning("Could not load that pair — check slice ids.")
                else:
                    st.caption("Need at least two slices for a custom comparison.")

            st.divider()
            st.markdown("### 📋 Data tables")
            st.markdown("**Per-slice summary (Neo4j)**")
            st.dataframe(temp_df, use_container_width=True)
            if not enriched_df.empty:
                st.markdown("**Enriched view (density + sentiment + top concept)**")
                st.dataframe(enriched_df, use_container_width=True)
                if "avg_sentiment" in enriched_df.columns:
                    sent = (
                        enriched_df.set_index("slice")[["avg_sentiment"]]
                        .dropna()
                    )
                    if not sent.empty:
                        st.markdown("**Sentiment trajectory**")
                        st.line_chart(sent)
                if "nodes" in enriched_df.columns:
                    st.markdown("**Slice size (nodes & edges)**")
                    st.bar_chart(
                        enriched_df.set_index("slice")[["nodes", "edges"]]
                    )
            if not pairwise_df.empty:
                st.markdown("**Pairwise drift (consecutive slices)**")
                st.dataframe(pairwise_df, use_container_width=True)

    with tab_chat:
        st.subheader("Chat with the knowledge graph")
        st.caption(
            "Conversational Q&A grounded in the Neo4j graph. Default "
            "provider is loaded from `LLM_PROVIDER` in `.env` (or "
            "`src/config.py`); `echo` is a no-LLM fallback that returns "
            "retrieved graph context verbatim."
        )

        with st.expander("Chatbot settings", expanded=False):
            _provider_options = [
                "echo",
                "openai",
                "dashscope",
                "huggingface_api",
                "huggingface_local",
            ]
            _default_provider = (config.LLM_PROVIDER or "echo").lower()
            _alias = {
                "qwen": "dashscope",
                "aliyun": "dashscope",
                "modelscope": "dashscope",
            }
            _default_provider = _alias.get(_default_provider, _default_provider)
            if _default_provider not in _provider_options:
                _default_provider = "echo"
            provider_choice = st.selectbox(
                "LLM provider",
                _provider_options,
                index=_provider_options.index(_default_provider),
            )
            use_vec = st.checkbox(
                "Enable semantic vector search (Neo4j)",
                value=False,
                help=(
                    "Requires embeddings to have been populated, e.g. "
                    "`python pipeline.py analyse --source <src> --embed`."
                ),
            )

        try:
            bot = _build_chatbot(provider_choice, use_vec, source_key)
        except Exception as e:
            st.error(f"Failed to build chatbot: {e}")
            st.stop()

        st.caption("Backend in use: **Neo4j** (Cypher + native vector index)")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        pending_q = st.session_state.pop("_chat_pending_example", None)

        with st.container(border=True):
            st.markdown("##### Example questions")
            st.caption(
                "Tap a prompt to send it, or type your own in the box below."
            )
            onboard_cols = st.columns(5)
            for idx, ((short, full), col) in enumerate(
                zip(_CHAT_ONBOARD_QUESTIONS, onboard_cols),
            ):
                with col:
                    if st.button(
                        short,
                        key=f"chat_onboard_{idx}",
                        help=full,
                        use_container_width=True,
                    ):
                        st.session_state["_chat_pending_example"] = full
                        st.rerun()

        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        user_q = pending_q or st.chat_input(
            "Ask about concepts, communities, or overall structure…"
        )
        if user_q:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_q}
            )
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer, routed = bot.ask_with_context(user_q)
                    except Exception as e:
                        answer = f"(error: {e})"
                        routed = None
                st.markdown(answer)
                if routed and routed.snippets:
                    with st.expander("Graph context used"):
                        for i, s in enumerate(routed.snippets, 1):
                            st.code(s, language="text")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

        if st.button("Clear conversation"):
            st.session_state.chat_history = []

if __name__ == "__main__":
    main()
