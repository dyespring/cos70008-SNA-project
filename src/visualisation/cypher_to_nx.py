"""Pull a top-N subgraph out of Neo4j as a transient ``nx.DiGraph``.

This is the only place in the Neo4j-only pipeline where NetworkX is still
used: ``StaticVisualiser`` and ``InteractiveVisualiser`` need a graph to
render, and rebuilding one from Cypher results is the simplest way to keep
those visualisers untouched.

The adapter intentionally fetches a *bounded* subgraph (default top-N by
``pagerank``) — drawing 10k nodes is meaningless and slow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


def fetch_subgraph(
    store: "Neo4jStore",
    source_label: str,
    slice_id: str | None = None,
    top_n: int = 200,
    rank_property: str = "pagerank",
) -> nx.DiGraph:
    """Return the top-N subgraph by ``rank_property`` as an ``nx.DiGraph``.

    Parameters
    ----------
    store:
        Connected :class:`Neo4jStore`.
    source_label:
        Scope filter — same value used by the writer / GDS runner.
    slice_id:
        Optional slice scope.
    top_n:
        Maximum number of nodes to include. Default 200 (interactive HTML
        becomes unreadable past ~500 anyway).
    rank_property:
        Node property used to rank nodes when truncating. Defaults to
        ``pagerank``; falls back to ``frequency`` if pagerank hasn't been
        written.
    """
    params: dict = {"sl": source_label, "n": int(top_n)}
    slice_filter = ""
    if slice_id is not None:
        params["sid"] = slice_id
        slice_filter = " AND c.slice_id = $sid"

    rank_expr = (
        f"coalesce(c.{rank_property}, "
        "         toFloat(coalesce(c.frequency, 0)))"
    )

    G = nx.DiGraph()

    with store.session() as s:
        node_rows = list(
            s.run(
                "MATCH (c:Concept) "
                "WHERE c.source_label = $sl"
                + slice_filter
                + f" WITH c, {rank_expr} AS rank "
                "ORDER BY rank DESC LIMIT $n "
                "RETURN c.id AS id, c.label AS label, "
                "       coalesce(c.concept_type, 'concept') AS type, "
                "       coalesce(c.source_type, 'unknown') AS source_type, "
                "       coalesce(c.frequency, 0)            AS frequency, "
                "       coalesce(c.pagerank, 0.0)           AS pagerank, "
                "       coalesce(c.betweenness, 0.0)        AS betweenness, "
                "       coalesce(c.community, -1)           AS community",
                **params,
            )
        )
    if not node_rows:
        return G

    node_ids: list[str] = []
    for r in node_rows:
        nid = str(r["id"])
        node_ids.append(nid)
        G.add_node(
            nid,
            label=r["label"] or nid,
            type=r["type"],
            source_type=r["source_type"],
            frequency=int(r["frequency"]),
            pagerank=float(r["pagerank"]),
            betweenness=float(r["betweenness"]),
            community=int(r["community"]),
        )

    edge_params = dict(params)
    edge_params["ids"] = node_ids
    edge_slice_filter = ""
    if slice_id is not None:
        edge_slice_filter = " AND r.slice_id = $sid"

    with store.session() as s:
        edge_rows = s.run(
            "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            "WHERE a.id IN $ids AND b.id IN $ids "
            "  AND r.source_label = $sl"
            + edge_slice_filter
            + " RETURN a.id AS src, b.id AS tgt, "
            "        coalesce(r.weight, 1.0) AS weight, "
            "        coalesce(r.types, '')   AS types, "
            "        r.sentiment             AS sentiment, "
            "        r.sentiment_label       AS sentiment_label, "
            "        r.top_verb              AS top_verb",
            **edge_params,
        )
        for r in edge_rows:
            types = r["types"] or ""
            type_set = set(t for t in types.split(",") if t)
            attrs = {
                "weight": float(r["weight"]),
                "types": type_set or {"ASSOCIATION"},
                "top_verb": r["top_verb"] or "",
            }
            if r["sentiment"] is not None:
                attrs["sentiment"] = float(r["sentiment"])
            if r["sentiment_label"]:
                attrs["sentiment_label"] = r["sentiment_label"]
            G.add_edge(str(r["src"]), str(r["tgt"]), **attrs)

    logger.info(
        "fetch_subgraph: returned %d nodes / %d edges (source_label=%s, top_n=%d)",
        G.number_of_nodes(),
        G.number_of_edges(),
        source_label,
        top_n,
    )
    return G


def fetch_partition(
    store: "Neo4jStore",
    source_label: str,
    slice_id: str | None = None,
) -> dict[str, int]:
    """Return ``{node_id -> community}`` for the full source / slice."""
    params: dict = {"sl": source_label}
    slice_filter = ""
    if slice_id is not None:
        params["sid"] = slice_id
        slice_filter = " AND c.slice_id = $sid"
    with store.session() as s:
        rows = s.run(
            "MATCH (c:Concept) "
            "WHERE c.source_label = $sl"
            + slice_filter
            + " RETURN c.id AS id, coalesce(c.community, -1) AS community",
            **params,
        ).data()
    return {r["id"]: int(r["community"]) for r in rows}
