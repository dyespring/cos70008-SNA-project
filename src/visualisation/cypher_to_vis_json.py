"""Pull a top-N subgraph out of Neo4j as a vis-network JSON payload.

This is the modern counterpart to :func:`cypher_to_nx.fetch_subgraph` —
instead of materialising an ``nx.DiGraph`` (which then has to be re-walked
by pyvis), we hand the same Neo4j subgraph straight to the browser as
``{"nodes": [...], "edges": [...]}`` shaped for `vis-network
<https://visjs.github.io/vis-network/docs/network/>`_.

Why it exists:

* Streamlit can embed the JSON in an HTML template that loads the local
  vis-network bundle (``src/extensions/lib/vis-9.1.2/``) — no Python-side
  graph rebuild, no NetworkX traversal on the hot path.
* The same payload powers the ``Network`` dashboard tab and any future
  consumer (e.g. an HTTP endpoint).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


# Distinct, colour-blind friendly palette for community colouring.
_COMMUNITY_COLOURS: list[str] = [
    "#4363d8", "#e6194b", "#3cb44b", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#ffe119", "#000000",
]
_SOURCE_COLOURS = {
    "policy":  "#4363d8",
    "yelp":    "#e6194b",
    "both":    "#3cb44b",
    "unknown": "#808080",
}


def fetch_vis_payload(
    store: "Neo4jStore",
    source_label: str,
    *,
    slice_id: str | None = None,
    top_n: int = 200,
    rank_property: str = "pagerank",
    min_edge_weight: int = 1,
    colour_by: str = "community",
) -> dict[str, Any]:
    """Return a vis-network-ready ``{"nodes", "edges", "stats"}`` dict.

    Parameters
    ----------
    store:
        Connected :class:`Neo4jStore`.
    source_label:
        Same scope filter used by the writer / GDS runner.
    slice_id:
        Optional ``slice_id`` filter — when set, only nodes/edges tagged
        for that temporal slice are returned.
    top_n:
        Maximum number of nodes to include. Default 200.
    rank_property:
        Node property used to rank nodes when truncating
        (``"pagerank"`` / ``"betweenness"`` / ``"degree"`` /
        ``"frequency"``). Falls back to ``frequency`` if the chosen
        property hasn't been written yet.
    min_edge_weight:
        Drop edges whose ``r.weight`` is strictly below this threshold.
        ``1`` keeps everything; bumping to 2-3 declutters the canvas
        considerably without hiding the structurally important edges.
    colour_by:
        ``"community"`` (default) colours nodes by Louvain community,
        ``"source"`` colours by ``source_type`` (policy / yelp / both).
    """
    rank_property = _safe_ident(rank_property)
    params: dict[str, Any] = {
        "sl": source_label,
        "n": int(top_n),
        "min_w": float(min_edge_weight),
    }
    slice_filter = ""
    edge_slice_filter = ""
    if slice_id is not None:
        params["sid"] = slice_id
        slice_filter = " AND c.slice_id = $sid"
        edge_slice_filter = " AND r.slice_id = $sid"

    rank_expr = (
        f"coalesce(c.{rank_property}, "
        "         toFloat(coalesce(c.frequency, 0)))"
    )

    node_query = (
        "MATCH (c:Concept) "
        "WHERE c.source_label = $sl"
        + slice_filter
        + f" WITH c, {rank_expr} AS rank "
        "ORDER BY rank DESC LIMIT $n "
        "RETURN c.id              AS id, "
        "       c.label           AS label, "
        "       coalesce(c.concept_type, 'concept') AS concept_type, "
        "       coalesce(c.source_type, 'unknown')  AS source_type, "
        "       coalesce(c.frequency, 0)            AS frequency, "
        "       coalesce(c.pagerank, 0.0)           AS pagerank, "
        "       coalesce(c.betweenness, 0.0)        AS betweenness, "
        "       coalesce(c.degree, 0.0)             AS degree, "
        "       coalesce(c.community, -1)           AS community"
    )
    with store.session() as s:
        node_rows = list(s.run(node_query, **params))

    if not node_rows:
        return {"nodes": [], "edges": [], "stats": _empty_stats()}

    node_ids: list[str] = []
    nodes: list[dict[str, Any]] = []
    max_rank = 0.0
    for r in node_rows:
        rank_val = float(r.get(rank_property, 0.0) or 0.0)
        if rank_val > max_rank:
            max_rank = rank_val

    palette_size = len(_COMMUNITY_COLOURS)
    for r in node_rows:
        nid = str(r["id"])
        node_ids.append(nid)
        community = int(r["community"])
        source_type = r["source_type"]
        rank_val = float(r.get(rank_property, 0.0) or 0.0)

        if colour_by == "source":
            colour = _SOURCE_COLOURS.get(source_type, _SOURCE_COLOURS["unknown"])
        else:
            colour = (
                _COMMUNITY_COLOURS[community % palette_size]
                if community >= 0
                else "#cccccc"
            )

        # vis-network ``value`` drives the node radius; scale to a
        # readable 8 → 60 px range based on the chosen rank metric.
        scaled = (rank_val / max_rank) if max_rank > 0 else 0.0
        size_value = 8 + 52 * scaled

        title_text = _node_tooltip_text(
            label=r["label"] or nid,
            concept_type=r["concept_type"],
            source_type=source_type,
            frequency=int(r["frequency"]),
            pagerank=float(r["pagerank"]),
            betweenness=float(r["betweenness"]),
            degree=float(r["degree"]),
            community=community,
            rank_property=rank_property,
            rank_val=rank_val,
            max_rank=max_rank,
            colour_by=colour_by,
        )

        nodes.append({
            "id": nid,
            "label": _short_label(r["label"] or nid, 28),
            "title": title_text,
            "value": size_value,
            "color": colour,
            "group": community if community >= 0 else None,
            "shape": "dot",
            "font": {"size": 12, "face": "Inter, system-ui, sans-serif"},
            "raw": {
                "label": r["label"] or nid,
                "source_type": source_type,
                "concept_type": r["concept_type"],
                "frequency": int(r["frequency"]),
                "pagerank": float(r["pagerank"]),
                "betweenness": float(r["betweenness"]),
                "community": community,
            },
        })

    edge_query = (
        "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
        "WHERE a.id IN $ids AND b.id IN $ids "
        "  AND r.source_label = $sl "
        "  AND coalesce(r.weight, 1.0) >= $min_w"
        + edge_slice_filter
        + " RETURN a.id AS src, b.id AS tgt, "
        "        coalesce(a.label, toString(a.id)) AS src_label, "
        "        coalesce(b.label, toString(b.id)) AS tgt_label, "
        "        coalesce(r.weight, 1.0) AS weight, "
        "        coalesce(r.types, '')   AS types, "
        "        r.sentiment             AS sentiment, "
        "        r.sentiment_label       AS sentiment_label, "
        "        r.top_verb              AS top_verb"
    )
    edge_params = dict(params, ids=node_ids)
    with store.session() as s:
        edge_rows = list(s.run(edge_query, **edge_params))

    edges: list[dict[str, Any]] = []
    max_weight = max((float(r["weight"] or 1.0) for r in edge_rows), default=1.0)
    has_sentiment = False
    for r in edge_rows:
        weight = float(r["weight"] or 1.0)
        types = r["types"] or ""
        sentiment = r["sentiment"]
        if sentiment is not None:
            has_sentiment = True
        sentiment_label = r["sentiment_label"]
        verb = r["top_verb"]

        directed = bool(types) and "ASSOCIATION" not in types.split(",")

        if sentiment is not None:
            colour = _sentiment_colour(float(sentiment))
        else:
            colour = "#9aa6b2"

        src_label = r.get("src_label")
        tgt_label = r.get("tgt_label")
        title_text = _edge_tooltip_text(
            src_id=str(r["src"]),
            tgt_id=str(r["tgt"]),
            src_label=str(src_label) if src_label is not None else None,
            tgt_label=str(tgt_label) if tgt_label is not None else None,
            weight=weight,
            types=types or "ASSOCIATION",
            directed=directed,
            verb=verb,
            sentiment=float(sentiment) if sentiment is not None else None,
            sentiment_label=sentiment_label,
        )

        edges.append({
            "from": str(r["src"]),
            "to": str(r["tgt"]),
            "title": title_text,
            "value": (weight / max_weight) if max_weight > 0 else 1.0,
            "width": 1 + 5.0 * ((weight / max_weight) if max_weight > 0 else 0.0),
            "color": {"color": colour, "opacity": 0.55},
            "arrows": {"to": {"enabled": directed, "scaleFactor": 0.6}},
            "smooth": {"type": "continuous"},
            "raw": {
                "weight": weight,
                "types": types or "ASSOCIATION",
                "sentiment": float(sentiment) if sentiment is not None else None,
                "sentiment_label": sentiment_label,
                "top_verb": verb,
            },
        })

    stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "max_weight": max_weight,
        "max_rank": max_rank,
        "rank_property": rank_property,
        "min_edge_weight": float(min_edge_weight),
        "has_sentiment": has_sentiment,
        "colour_by": colour_by,
        "slice_id": slice_id,
    }
    logger.info(
        "fetch_vis_payload: %d nodes / %d edges (source_label=%s, top_n=%d, "
        "rank=%s, min_edge_weight=%s, slice=%s)",
        len(nodes), len(edges),
        source_label, top_n, rank_property, min_edge_weight, slice_id,
    )
    return {"nodes": nodes, "edges": edges, "stats": stats}


def _empty_stats() -> dict[str, Any]:
    return {
        "nodes": 0, "edges": 0,
        "max_weight": 0.0, "max_rank": 0.0,
        "rank_property": "", "min_edge_weight": 0.0,
        "has_sentiment": False, "colour_by": "",
        "slice_id": None,
    }


# Whitelist of allowed rank properties — prevents Cypher injection via
# the ``rank_property`` argument since it's interpolated directly.
_ALLOWED_RANK_PROPS = {
    "pagerank", "betweenness", "degree", "closeness",
    "eigenvector", "frequency",
}


def _safe_ident(prop: str) -> str:
    if prop not in _ALLOWED_RANK_PROPS:
        raise ValueError(
            f"rank_property must be one of {sorted(_ALLOWED_RANK_PROPS)}, "
            f"got {prop!r}"
        )
    return prop


def _short_label(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _source_type_plain(source_type: str) -> str:
    """Short stakeholder-friendly label for ``source_type``."""
    return {
        "policy": "Policy documents",
        "yelp": "Public reviews (Yelp)",
        "both": "Policy and reviews",
        "unknown": "Source not tagged",
    }.get(source_type, str(source_type))


def _rank_metric_blurb(rank_property: str) -> str:
    """One-line hint for what the chosen ranking measures."""
    return {
        "pagerank": 'How much "attention" would flow here if someone wandered the graph at random.',
        "betweenness": "How often this idea sits on shortest paths between others - a broker / bridge.",
        "degree": "How many weighted ties this concept has in the subgraph.",
        "closeness": "How few hops on average to reach the rest of the network from here.",
        "eigenvector": "Connected to other well-connected ideas, not only raw degree.",
        "frequency": "How often the phrase was seen in the scoped corpus.",
    }.get(rank_property, "Ranking metric for this view.")


def _concept_type_plain(concept_type: str) -> str:
    ct = (concept_type or "concept").strip().lower().replace("_", " ")
    if ct == "noun phrase":
        return "Noun phrase"
    return ct[:1].upper() + ct[1:] if ct else "Concept"


def _tooltip_plain_segment(value: Any, *, max_len: int = 240) -> str:
    """Single-line user-derived fragment safe for vis-network ``title`` (innerText)."""
    s = "" if value is None else str(value).replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _node_tooltip_text(
    *,
    label: str,
    concept_type: str,
    source_type: str,
    frequency: int,
    pagerank: float,
    betweenness: float,
    degree: float,
    community: int,
    rank_property: str,
    rank_val: float,
    max_rank: float,
    colour_by: str,
) -> str:
    """Plain ``title`` for vis-network (popup uses innerText, not HTML)."""
    name = _tooltip_plain_segment(label or nid, max_len=120)
    rel_size = ""
    if max_rank > 0 and rank_val >= 0:
        pct = 100.0 * (rank_val / max_rank)
        rel_size = (
            f"\nRelative strength: about {pct:.0f}% of the strongest node on this "
            f"canvas (by {rank_property})."
        )
    comm_line = (
        "Not clustered (no community id)."
        if community < 0
        else (
            f"Cluster #{community} - same colour means the same theme group "
            "(Louvain). See the Communities tab for a readable cluster name."
        )
    )
    colour_line = (
        "Node colour follows data source (policy / Yelp / both)."
        if colour_by == "source"
        else "Node colour follows theme cluster (see line above)."
    )
    return (
        f"{name}\n"
        f"{_concept_type_plain(concept_type)} - {_source_type_plain(source_type)}\n"
        f"\n"
        f"Mentions in corpus: {int(frequency)}\n"
        f"Size on chart: driven by {rank_property} ({_rank_metric_blurb(rank_property)})"
        f"{rel_size}\n"
        f"\n"
        f"PageRank: {float(pagerank):.4f} (overall influence)\n"
        f"Betweenness: {float(betweenness):.4f} (broker / bottleneck signal)\n"
        f"Degree (weighted): {float(degree):.2f} (how connected it is here)\n"
        f"\n"
        f"Community: {comm_line}\n"
        f"{colour_line}"
    )


def _edge_tooltip_text(
    *,
    src_id: str,
    tgt_id: str,
    src_label: str | None,
    tgt_label: str | None,
    weight: float,
    types: str,
    directed: bool,
    verb: str | None,
    sentiment: float | None,
    sentiment_label: str | None,
) -> str:
    """Plain edge ``title`` for vis-network (innerText)."""
    sl = _tooltip_plain_segment(src_label or src_id)
    tl = _tooltip_plain_segment(tgt_label or tgt_id)
    type_str = (types or "ASSOCIATION").strip() or "ASSOCIATION"
    orient = (
        'Directed - read the arrow as "influences / relates toward".'
        if directed
        else "Undirected - co-occurrence / association (no arrow bias)."
    )
    lines = [
        f"{sl} -> {tl}",
        f"IDs: {src_id} -> {tgt_id}",
        "",
        f"Link strength: {weight:.1f} (thicker line = more co-occurrence evidence)",
        f"Edge type: {type_str}. {orient}",
    ]
    if verb:
        lines.append(
            f'Typical wording: "{_tooltip_plain_segment(verb, max_len=80)}" '
            "(most common verb in snippets)"
        )
    if sentiment is not None:
        lbl = f" ({_tooltip_plain_segment(sentiment_label)})" if sentiment_label else ""
        lines.append(
            f"Stance (aggregated): {float(sentiment):+.2f}{lbl} "
            "- green / grey / red edge tint reflects this."
        )
    return "\n".join(lines)


_HTML_ESCAPES = {
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}


def _html_escape(value: Any) -> str:
    s = "" if value is None else str(value)
    return "".join(_HTML_ESCAPES.get(ch, ch) for ch in s)


def _sentiment_colour(score: float) -> str:
    """Map sentiment ∈ [-1, 1] to a green / grey / red ramp."""
    if score >= 0.05:
        return "#2ca02c"   # positive → green
    if score <= -0.05:
        return "#d62728"   # negative → red
    return "#9aa6b2"       # neutral → grey
