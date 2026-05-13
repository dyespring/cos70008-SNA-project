"""SNA InsightEngine: turn Neo4j metrics into Key / Risk / Action insights.

The engine is the bridge between the raw GDS metrics on
``(:Concept)`` nodes (``pagerank``, ``betweenness``, ``community``,
``wcc_component`` etc.) and the dashboard's narrative cards.

Three insight categories
------------------------
* **Key Insights**       — descriptive: what the network looks like
* **Risk Insights**      — diagnostic:  where the network is fragile
* **Action Recommendations** — prescriptive: what to do about it

Every insight is computed deterministically from one Cypher query, then
optionally rewritten in professional prose by an :class:`LLMProvider`
(reuses ``src.extensions.chatbot.build_default_provider``).

Typical use
-----------
::

    from src.extensions.insight_engine import InsightEngine

    engine = InsightEngine(store, source_label="combined")
    bundle = engine.all_insights()                         # raw templates
    bundle = engine.all_insights(polish=True)              # LLM-polished
    for ins in bundle["risk"]:
        print(ins.severity, ins.title, ins.body)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from src.extensions.chatbot import LLMProvider
    from src.extensions.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════


@dataclass
class Insight:
    """One narrative card produced by :class:`InsightEngine`."""

    id: str                                # stable id for caching / referencing
    category: str                          # "key" | "risk" | "action"
    severity: str = "info"                 # "info" | "low" | "medium" | "high"
    title: str = ""
    body: str = ""                         # deterministic template text
    body_polished: str | None = None       # LLM-rephrased version (optional)
    concepts: list[str] = field(default_factory=list)
    metric: str | None = None              # source GDS metric, if any
    data: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    unavailable_reason: str | None = None

    def render(self) -> str:
        """Return the best available text (LLM polished if present)."""
        if self.body_polished:
            return self.body_polished
        return self.body


# ════════════════════════════════════════════════════════════════════
# LLM polishing
# ════════════════════════════════════════════════════════════════════


_POLISH_SYSTEM_PROMPT = """You are a senior data analyst writing executive
summaries of social network analysis (SNA) findings. Your job is to rewrite
short analytical observations in clear, professional, neutral language.

Strict rules:
- Keep every concrete detail from the source: concept names (in single quotes),
  numerical values, community ids, percentages.
- Be concise: 2 to 3 sentences maximum.
- Use an analytical tone (not marketing, not casual).
- Never invent details or insights not present in the source observation.
- Do not start with "This insight..." or "The data shows..."; lead with the
  finding itself."""


def _polish_text(
    llm: "LLMProvider",
    insight: Insight,
    cache: Optional[dict[str, str]] = None,
) -> str:
    """Run an Insight body through the LLM and return polished prose."""
    if cache is not None and insight.id in cache:
        return cache[insight.id]

    user_msg = (
        f"Category: {insight.category}\n"
        f"Title: {insight.title}\n"
        f"Severity: {insight.severity}\n"
        f"Observation:\n{insight.body}\n\n"
        "Rewrite the observation following the rules in the system prompt."
    )
    try:
        polished = llm.complete(_POLISH_SYSTEM_PROMPT, user_msg).strip()
    except Exception as e:  # pragma: no cover — LLM call is environmental
        logger.warning("LLM polish failed for %s: %s", insight.id, e)
        polished = insight.body

    if cache is not None:
        cache[insight.id] = polished
    return polished


# ════════════════════════════════════════════════════════════════════
# Engine
# ════════════════════════════════════════════════════════════════════


class InsightEngine:
    """Compute SNA-driven Key/Risk/Action insights from a Neo4j subgraph."""

    def __init__(
        self,
        store: "Neo4jStore",
        source_label: str = "combined",
        slice_id: str | None = None,
        llm: Optional["LLMProvider"] = None,
    ):
        self.store = store
        self.source_label = source_label
        self.slice_id = slice_id
        self.llm = llm
        self._polish_cache: dict[str, str] = {}
        # Cache of community id → human-readable anchor list, populated
        # lazily by :meth:`_describe_community`.
        self._community_anchors: dict[int, str] | None = None

    # Each insight category surfaces at most this many cards in the
    # dashboard. Available insights are sorted by ``severity_rank`` and
    # by their internal ``data`` size before truncation; unavailable
    # cards are appended after available ones (so missing-data hints
    # don't crowd out real findings).
    TOP_PER_CATEGORY: int = 3

    _SEVERITY_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1, "info": 0}

    # ── Public API ─────────────────────────────────────────────────
    def all_insights(self, polish: bool = False) -> dict[str, list[Insight]]:
        """Return the full bundle of insights, grouped by category.

        Each category is capped at :attr:`TOP_PER_CATEGORY` (default 3)
        so the dashboard always shows a uniform, scannable layout.
        Cards are ranked by ``severity`` first, then by how rich their
        backing data is (a card with five flagged communities outranks
        one with a single fallback observation at the same severity).
        """
        bundle = {
            "key":    self._top(self.key_insights()),
            "risk":   self._top(self.risk_insights()),
            "action": self._top(self.action_recommendations()),
        }
        if polish and self.llm is not None:
            for cat in bundle.values():
                for ins in cat:
                    if ins.available:
                        ins.body_polished = _polish_text(
                            self.llm, ins, self._polish_cache
                        )
        return bundle

    def _top(self, insights: list[Insight]) -> list[Insight]:
        """Sort by (available, severity, data richness) and truncate."""
        def _key(ins: Insight) -> tuple[int, int, int]:
            return (
                1 if ins.available else 0,
                self._SEVERITY_RANK.get(ins.severity, 0),
                self._data_size(ins.data),
            )

        ranked = sorted(insights, key=_key, reverse=True)
        return ranked[: self.TOP_PER_CATEGORY]

    @staticmethod
    def _data_size(data: dict[str, Any]) -> int:
        """Tie-breaker score: how much evidence backs this insight?"""
        if not data:
            return 0
        score = 0
        for v in data.values():
            if isinstance(v, list):
                score += len(v)
            elif isinstance(v, dict):
                score += len(v)
            elif isinstance(v, (int, float)):
                score += 1
        return score

    def key_insights(self) -> list[Insight]:
        return self._collect(
            self._hub_concepts,
            self._community_landscape,
            self._bridge_concepts,
            self._isolated_clusters,
            self._cross_source_overlap,
        )

    def risk_insights(self) -> list[Insight]:
        return self._collect(
            self._single_point_of_failure,
            self._echo_chambers,
            self._negative_sentiment_clusters,
            self._sparse_coverage,
            self._weakly_connected_fragmentation,
        )

    def action_recommendations(self) -> list[Insight]:
        return self._collect(
            self._action_diversify_hubs,
            self._action_leverage_bridges,
            self._action_investigate_isolates,
            self._action_cross_source_translation,
        )

    # ── Helpers ────────────────────────────────────────────────────
    def _collect(self, *fns: Callable[[], Insight | None]) -> list[Insight]:
        out: list[Insight] = []
        for fn in fns:
            try:
                ins = fn()
            except Exception as e:
                logger.warning("Insight %s failed: %s", fn.__name__, e)
                continue
            if ins is not None:
                out.append(ins)
        return out

    def _params(self) -> dict[str, Any]:
        if self.slice_id is None:
            return {"sl": self.source_label}
        return {"sl": self.source_label, "sid": self.slice_id}

    def _node_filter(self, alias: str = "c") -> str:
        if self.slice_id is None:
            return f"{alias}.source_label = $sl"
        return f"{alias}.source_label = $sl AND {alias}.slice_id = $sid"

    def _query_all(self, cypher: str, **params) -> list[dict[str, Any]]:
        merged = {**self._params(), **params}
        with self.store.session() as s:
            return s.run(cypher, **merged).data()

    def _query_one(self, cypher: str, **params) -> dict[str, Any] | None:
        rows = self._query_all(cypher, **params)
        return rows[0] if rows else None

    @staticmethod
    def _unavailable(
        id_: str, category: str, title: str, reason: str
    ) -> Insight:
        return Insight(
            id=id_,
            category=category,
            title=title,
            body=f"Not available — {reason}",
            available=False,
            unavailable_reason=reason,
            severity="info",
        )

    @staticmethod
    def _quote(label: str) -> str:
        return f"'{label}'"

    @staticmethod
    def _join_quoted(labels: list[str]) -> str:
        return ", ".join(f"'{lbl}'" for lbl in labels)

    # ── Human-readable community labels ─────────────────────────────
    #
    # Every card that mentions a community id — Hub, SPOF, Echo,
    # Bridge, Sentiment, Diversify, Leverage Bridge — should make the
    # community immediately recognisable instead of asking the reader
    # to swivel back to the Communities card.
    #
    # The first call hydrates a per-engine cache of
    # ``community_id → "'climate change', 'restaurant', 'business'"``
    # (top 3 by PageRank, falling back to frequency); subsequent calls
    # are O(1).

    _COMMUNITY_ANCHOR_LIMIT: int = 3

    def _community_anchor_index(self) -> dict[int, str]:
        """Map ``community_id → joined-quoted anchor labels`` (cached)."""
        if self._community_anchors is not None:
            return self._community_anchors

        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.community IS NOT NULL "
            "RETURN c.community AS cid, c.label AS label, "
            "       coalesce(c.frequency, 0) AS freq, "
            "       coalesce(c.pagerank, 0.0) AS pr"
        )
        per_cid: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            cid = int(r["cid"])
            per_cid.setdefault(cid, []).append(r)

        index: dict[int, str] = {}
        limit = self._COMMUNITY_ANCHOR_LIMIT
        for cid, members in per_cid.items():
            # PageRank-first ordering — these labels need to line up
            # with what the Hub / SPOF cards consider central.
            ranked = sorted(
                members,
                key=lambda m: (
                    float(m.get("pr") or 0.0),
                    int(m.get("freq") or 0),
                ),
                reverse=True,
            )[:limit]
            labels = [str(m["label"]) for m in ranked if m.get("label")]
            index[cid] = self._join_quoted(labels) if labels else ""

        self._community_anchors = index
        return index

    def _describe_community(self, cid: int | None) -> str:
        """Return ``"community 38 (X, Y, Z)"`` — fallback to bare id on miss."""
        if cid is None:
            return "an unnamed community"
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            return f"community {cid}"
        anchors = self._community_anchor_index().get(cid_int, "")
        if not anchors:
            return f"community {cid_int}"
        return f"community {cid_int} ({anchors})"

    # ════════════════════════════════════════════════════════════
    # KEY INSIGHTS
    # ════════════════════════════════════════════════════════════

    def _hub_concepts(self) -> Insight | None:
        """Top concepts by PageRank — the most central / influential nodes."""
        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.pagerank IS NOT NULL "
            "RETURN c.label AS label, c.pagerank AS pagerank, "
            "       c.community AS community "
            "ORDER BY c.pagerank DESC LIMIT 5"
        )
        if not rows:
            return self._unavailable(
                "key.hub", "key", "Hub Concepts",
                "PageRank not yet computed (run `pipeline.py analyse`).",
            )
        labels = [r["label"] for r in rows]
        top = rows[0]
        body = (
            f"The most influential concepts are {self._join_quoted(labels[:3])}. "
            f"{self._quote(top['label'])} dominates the network with a PageRank "
            f"of {float(top['pagerank']):.4f}, anchoring discussion across "
            f"{self._describe_community(top['community'])}."
        )
        return Insight(
            id="key.hub",
            category="key",
            severity="info",
            title="Hub Concepts",
            body=body,
            concepts=labels,
            metric="pagerank",
            data={"top": rows},
        )

    def _community_landscape(self) -> Insight | None:
        """Number of detected communities + dual-anchor view of the largest.

        Each community is described by **two** anchor lists so the
        reader can reconcile this card with Hub / SPOF cards:

        * ``top_freq``     — the highest-frequency concepts (most-mentioned
          surface vocabulary).
        * ``top_pagerank`` — the most-central concepts (structural
          influence). Hub Concepts ranks by PageRank, so this list is
          what makes a community recognisable as "the one containing
          'climate change'".

        The two lists often differ: the largest community frequently
        contains both yelp's high-volume words ("restaurant", "drink")
        and policy's high-influence words ("climate change",
        "business"); collapsing to a single anchor hides that.
        """
        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.community IS NOT NULL "
            "WITH c.community AS cid, "
            "     collect({label: c.label, "
            "              freq: coalesce(c.frequency, 0), "
            "              pr:   coalesce(c.pagerank, 0.0)}) AS members "
            "WITH cid, members, size(members) AS sz, "
            "     reduce(acc = [], m IN "
            "         apoc.coll.sortMaps(members, '^freq')[..-4..-1] | acc + m.label) AS _ignore_apoc "
            "RETURN cid, sz, members "
            "ORDER BY sz DESC LIMIT 5"
        ) if False else self._query_all(  # APOC-free Cypher fallback below.
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.community IS NOT NULL "
            "WITH c.community AS cid, "
            "     collect({label: c.label, "
            "              freq: coalesce(c.frequency, 0), "
            "              pr:   coalesce(c.pagerank, 0.0)}) AS members "
            "RETURN cid, size(members) AS sz, members "
            "ORDER BY sz DESC LIMIT 5"
        )
        if not rows:
            return self._unavailable(
                "key.community", "key", "Thematic Communities",
                "Communities not yet detected (run `pipeline.py analyse`).",
            )
        # Re-shape rows in Python to keep the Cypher portable across
        # Neo4j versions (no APOC dependency).
        for r in rows:
            members = r["members"] or []
            top_freq = sorted(
                members, key=lambda m: (m["freq"] or 0), reverse=True,
            )[:3]
            top_pr = sorted(
                members, key=lambda m: (m["pr"] or 0.0), reverse=True,
            )[:3]
            r["top"] = [m["label"] for m in top_freq]
            r["top_pagerank"] = [m["label"] for m in top_pr]

        total = sum(int(r["sz"]) for r in rows)
        biggest = rows[0]
        # Keep the body concise but include both anchors when they
        # actually differ — otherwise the second list adds noise.
        anchor_freq = self._join_quoted(biggest["top"])
        anchor_pr = self._join_quoted(biggest["top_pagerank"])
        if biggest["top"] == biggest["top_pagerank"]:
            anchor_clause = f"and is anchored by {anchor_freq}"
        else:
            anchor_clause = (
                f"with high-frequency anchors {anchor_freq} and "
                f"high-influence anchors {anchor_pr}"
            )
        body = (
            f"The corpus splits into {len(rows)} thematic communities. The "
            f"largest covers {biggest['sz']} concepts "
            f"({biggest['sz'] / total:.0%} of the visible network) "
            f"{anchor_clause}."
        )
        return Insight(
            id="key.community",
            category="key",
            severity="info",
            title="Thematic Communities",
            body=body,
            # Hub-style concepts (PageRank-ranked) are the better
            # surface for the chip strip — they're what shows up in the
            # SPOF / Hub cards too.
            concepts=biggest["top_pagerank"] or biggest["top"],
            metric="community",
            data={
                "communities": [
                    {
                        "cid": r["cid"],
                        "sz": r["sz"],
                        "top_freq": r["top"],
                        "top_pagerank": r["top_pagerank"],
                    }
                    for r in rows
                ]
            },
        )

    def _bridge_concepts(self) -> Insight | None:
        """Brokers — high betweenness nodes connecting communities."""
        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.betweenness IS NOT NULL AND c.community IS NOT NULL "
            "OPTIONAL MATCH (c)-[:RELATED]-(nb:Concept) "
            f"WHERE {self._node_filter('nb')} "
            "WITH c, count(DISTINCT CASE WHEN nb.community <> c.community "
            "                            THEN nb END) AS cross_n "
            "WHERE cross_n > 0 "
            "RETURN c.label AS label, c.betweenness AS bc, "
            "       c.community AS community, cross_n "
            "ORDER BY c.betweenness * cross_n DESC LIMIT 3"
        )
        if not rows:
            return self._unavailable(
                "key.bridge", "key", "Bridge Concepts",
                "Betweenness or communities not yet computed.",
            )
        top = rows[0]
        labels = [r["label"] for r in rows]
        body = (
            f"{self._join_quoted(labels)} act as bridges between thematic "
            f"clusters. {self._quote(top['label'])} connects "
            f"{top['cross_n']} concepts outside its own community "
            f"({self._describe_community(top['community'])}), "
            f"giving it disproportionate influence over information flow."
        )
        return Insight(
            id="key.bridge",
            category="key",
            severity="info",
            title="Bridge Concepts",
            body=body,
            concepts=labels,
            metric="betweenness",
            data={"bridges": rows},
        )

    def _isolated_clusters(self) -> Insight | None:
        """Small weakly-connected components — fringe topics."""
        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.wcc_component IS NOT NULL "
            "WITH c.wcc_component AS w, collect(c.label) AS labels "
            "WHERE size(labels) <= 4 AND size(labels) >= 1 "
            "RETURN w, labels, size(labels) AS sz "
            "ORDER BY sz ASC LIMIT 3"
        )
        if not rows:
            return Insight(
                id="key.isolated",
                category="key",
                severity="info",
                title="Isolated Concepts",
                body="No fringe clusters detected — the network is well "
                     "connected with no orphan topic islands.",
                metric="wcc_component",
                available=True,
            )
        examples = ", ".join(
            f"{r['sz']} concepts ({self._join_quoted(r['labels'][:3])})"
            for r in rows
        )
        body = (
            f"{len(rows)} small clusters sit disconnected from the main "
            f"discourse: {examples}. These represent fringe or under-developed "
            f"topics that don't yet integrate with the broader network."
        )
        return Insight(
            id="key.isolated",
            category="key",
            severity="low",
            title="Isolated Concepts",
            body=body,
            concepts=[lbl for r in rows for lbl in r["labels"]],
            metric="wcc_component",
            data={"clusters": rows},
        )

    def _cross_source_overlap(self) -> Insight | None:
        """Combined-only: shared concepts across policy + yelp."""
        if self.source_label != "combined":
            return None
        rec = self._query_one(
            "MATCH (c:Concept {source_label: $sl}) "
            "WITH count(c) AS total, "
            "     sum(CASE c.source_type WHEN 'policy' THEN 1 ELSE 0 END) AS p, "
            "     sum(CASE c.source_type WHEN 'yelp'   THEN 1 ELSE 0 END) AS y, "
            "     sum(CASE c.source_type WHEN 'both'   THEN 1 ELSE 0 END) AS b "
            "RETURN total, p, y, b"
        )
        if not rec or not rec.get("total"):
            return None
        total = int(rec["total"])
        b = int(rec["b"] or 0)
        p = int(rec["p"] or 0)
        y = int(rec["y"] or 0)
        overlap_pct = 100 * b / total if total else 0.0
        body = (
            f"Of {total} concepts, {b} ({overlap_pct:.1f}%) appear in both the "
            f"policy and Yelp corpora; {p} are policy-only and {y} are "
            f"Yelp-only. The overlap defines the shared vocabulary between "
            f"institutional language and customer language."
        )
        return Insight(
            id="key.cross_source",
            category="key",
            severity="info",
            title="Cross-Source Overlap",
            body=body,
            metric="source_type",
            data={"total": total, "policy": p, "yelp": y, "both": b,
                  "overlap_pct": overlap_pct},
        )

    # ════════════════════════════════════════════════════════════
    # RISK INSIGHTS
    # ════════════════════════════════════════════════════════════

    def _single_point_of_failure(self) -> Insight | None:
        """Concepts simultaneously high pagerank AND high betweenness."""
        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.pagerank IS NOT NULL AND c.betweenness IS NOT NULL "
            "WITH c, c.pagerank * c.betweenness AS score "
            "RETURN c.label AS label, c.pagerank AS pr, "
            "       c.betweenness AS bc, score "
            "ORDER BY score DESC LIMIT 3"
        )
        if not rows or float(rows[0]["score"] or 0) <= 0:
            return self._unavailable(
                "risk.spof", "risk", "Single Point of Failure",
                "PageRank and betweenness must be computed first.",
            )
        top = rows[0]
        labels = [r["label"] for r in rows]
        body = (
            f"{self._quote(top['label'])} is a structural single point of "
            f"failure: it ranks both top in influence (PageRank "
            f"{float(top['pr']):.4f}) and top in path-control (betweenness "
            f"{float(top['bc']):.4f}). Removing it would fragment information "
            f"flow across the network. Other vulnerable hubs include "
            f"{self._join_quoted(labels[1:3])}."
        )
        return Insight(
            id="risk.spof",
            category="risk",
            severity="high",
            title="Single Point of Failure",
            body=body,
            concepts=labels,
            metric="pagerank*betweenness",
            data={"candidates": rows},
        )

    # Echo-chamber heuristic — a community is "insular" only when it is
    # *both* highly self-referential AND a small minority of the network.
    # Without the size cap, the network's *main backbone* (typically the
    # largest community, dominating cross-source vocabulary) would
    # always trigger this card, which is the opposite of an echo
    # chamber.
    _ECHO_INTERNAL_RATIO: float = 0.85
    _ECHO_MAX_SIZE_RATIO: float = 0.40
    _ECHO_MIN_EDGES: int = 5

    def _echo_chambers(self) -> Insight | None:
        """Communities that are simultaneously insular AND a minority slice.

        Two filters guard against false positives:

        * ``internal_ratio >= ECHO_INTERNAL_RATIO`` — most edges stay inside
          the community.
        * ``size_ratio < ECHO_MAX_SIZE_RATIO`` — the community is not the
          backbone of the whole network. A 60% community whose 90% of
          edges stay internal is just "the network has a dominant
          theme", not an echo chamber.
        """
        # Total node count for the active scope — used to compute
        # community size ratios in the same query.
        total_rec = self._query_one(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.community IS NOT NULL "
            "RETURN count(c) AS n"
        )
        total_nodes = int(total_rec["n"]) if total_rec else 0
        if total_nodes <= 0:
            return self._unavailable(
                "risk.echo", "risk", "Echo Chambers",
                "No communities populated (run `pipeline.py analyse`).",
            )

        rows = self._query_all(
            "MATCH (a:Concept)-[r:RELATED]-(b:Concept) "
            f"WHERE {self._node_filter('a')} AND {self._node_filter('b')} "
            "  AND a.community IS NOT NULL AND b.community IS NOT NULL "
            "  AND id(a) < id(b) "
            "WITH a.community AS cid, "
            "     sum(CASE WHEN a.community = b.community THEN 1 ELSE 0 END) AS internal, "
            "     sum(CASE WHEN a.community <> b.community THEN 1 ELSE 0 END) AS external "
            "WHERE internal + external >= $min_edges "
            "WITH cid, internal, external, "
            "     toFloat(internal) / (internal + external) AS ratio "
            "WHERE ratio >= $min_ratio "
            "RETURN cid, internal, external, ratio "
            "ORDER BY ratio DESC LIMIT 8",
            min_edges=self._ECHO_MIN_EDGES,
            min_ratio=self._ECHO_INTERNAL_RATIO,
        )

        # Attach community size + size_ratio so we can filter out the
        # backbone, and rank by (ratio, smallness) — small *and* tight
        # is the more interesting echo signal.
        sizes = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.community IS NOT NULL "
            "RETURN c.community AS cid, count(c) AS sz"
        )
        size_lookup = {int(r["cid"]): int(r["sz"]) for r in sizes}

        enriched: list[dict[str, Any]] = []
        for r in rows:
            cid = int(r["cid"])
            sz = size_lookup.get(cid, 0)
            size_ratio = sz / total_nodes if total_nodes else 0.0
            if size_ratio >= self._ECHO_MAX_SIZE_RATIO:
                # Backbone — too big to be an echo chamber.
                continue
            enriched.append({
                **r,
                "size": sz,
                "size_ratio": size_ratio,
            })

        if not enriched:
            return Insight(
                id="risk.echo",
                category="risk",
                severity="info",
                title="Echo Chambers",
                body=(
                    f"No echo-chamber communities detected (looking for "
                    f">= {self._ECHO_INTERNAL_RATIO:.0%} internal edges in "
                    f"a community covering < {self._ECHO_MAX_SIZE_RATIO:.0%} "
                    "of the network). High-ratio backbone communities are "
                    "deliberately excluded — they reflect a dominant theme, "
                    "not insularity."
                ),
                metric="community_internal_ratio",
                available=True,
            )

        # Top echo: smallest among the qualifying high-ratio communities,
        # so we surface the most "siloed" cluster first.
        enriched.sort(key=lambda e: (e["size_ratio"], -e["ratio"]))
        top = enriched[0]
        body = (
            f"{self._describe_community(top['cid']).capitalize()} is a "
            f"small insular pocket: {top['internal']} of "
            f"{top['internal'] + top['external']} edges "
            f"({float(top['ratio']):.0%}) stay inside the cluster, "
            f"yet the cluster only accounts for {top['size']} concepts "
            f"({top['size_ratio']:.1%} of the network). This signals an "
            f"echo-chamber pattern where the topic rarely connects with "
            f"the broader discourse."
        )
        return Insight(
            id="risk.echo",
            category="risk",
            severity="medium",
            title="Echo Chambers",
            body=body,
            metric="community_internal_ratio",
            data={
                "communities": enriched[:3],
                "total_nodes": total_nodes,
                "ratio_threshold": self._ECHO_INTERNAL_RATIO,
                "size_threshold": self._ECHO_MAX_SIZE_RATIO,
            },
        )

    def _negative_sentiment_clusters(self) -> Insight | None:
        """Communities whose intra-edges average strongly negative sentiment.

        Three outcomes:

        1. **No ``r.sentiment`` at all** — tell the user to run ETL with
           ``--sentiment``.
        2. **Sentiment exists, at least one community below −0.05** —
           classic risk insight (complaints / criticism cluster).
        3. **Sentiment exists but every community ≥ −0.05** — this is *not*
           missing data (Step A/B can show 20k+ intra-community edges with
           positive averages). Return an informational card citing the
           lowest-averaging communities instead of a misleading
           "not populated" message.
        """
        nf_a = self._node_filter("a")
        nf_b = self._node_filter("b")

        cnt_rec = self._query_one(
            f"MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            f"WHERE {nf_a} AND {nf_b} AND r.sentiment IS NOT NULL "
            "RETURN count(r) AS n"
        )
        edges_with_sentiment = int(cnt_rec["n"]) if cnt_rec else 0
        if edges_with_sentiment == 0:
            return self._unavailable(
                "risk.sentiment",
                "risk",
                "Negative Sentiment Clusters",
                "Edge sentiment not populated (run `pipeline.py etl --sentiment`).",
            )

        rows_neg = self._query_all(
            "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            f"WHERE {nf_a} AND {nf_b} "
            "  AND r.sentiment IS NOT NULL "
            "  AND a.community IS NOT NULL AND a.community = b.community "
            "WITH a.community AS cid, avg(r.sentiment) AS avg_s, count(r) AS n "
            "WHERE n >= 3 AND avg_s < -0.05 "
            "RETURN cid, avg_s, n ORDER BY avg_s ASC LIMIT 3"
        )
        if rows_neg:
            top = rows_neg[0]
            body = (
                f"{self._describe_community(top['cid']).capitalize()} "
                f"carries a negative sentiment signal: average polarity "
                f"{float(top['avg_s']):+.2f} across {top['n']} relationships. "
                f"This concentration warrants attention — it likely reflects "
                f"a cluster of complaints, criticism or adverse outcomes."
            )
            return Insight(
                id="risk.sentiment",
                category="risk",
                severity="high",
                title="Negative Sentiment Clusters",
                body=body,
                metric="sentiment",
                data={
                    "communities": rows_neg,
                    "edges_with_sentiment": edges_with_sentiment,
                },
            )

        # Sentiment populated but no community crosses the negative threshold.
        rows_low = self._query_all(
            "MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            f"WHERE {nf_a} AND {nf_b} "
            "  AND r.sentiment IS NOT NULL "
            "  AND a.community IS NOT NULL AND a.community = b.community "
            "WITH a.community AS cid, avg(r.sentiment) AS avg_s, count(r) AS n "
            "WHERE n >= 3 "
            "RETURN cid, avg_s, n ORDER BY avg_s ASC LIMIT 3"
        )
        if not rows_low:
            body = (
                f"{edges_with_sentiment:,} edges carry sentiment scores, but "
                "fewer than three intra-community edges exist per community — "
                "community-level sentiment cannot be aggregated. Run "
                "`pipeline.py analyse` after ETL so Louvain writes "
                "`c.community`, or lower the minimum-edge threshold in code."
            )
            return Insight(
                id="risk.sentiment",
                category="risk",
                severity="info",
                title="Negative Sentiment Clusters",
                body=body,
                metric="sentiment",
                data={
                    "edges_with_sentiment": edges_with_sentiment,
                    "communities": [],
                },
            )

        lowest_bits = ", ".join(
            f"{self._describe_community(r['cid'])} "
            f"(avg {float(r['avg_s']):+.2f}, n={r['n']})"
            for r in rows_low
        )
        body = (
            "No community crosses the negative-risk threshold "
            "(average intra-community edge sentiment below −0.05). "
            f"The lowest averages observed are: {lowest_bits}. "
            f"Across {edges_with_sentiment:,} scored edges, discourse in "
            "these clusters skews neutral-to-positive rather than "
            "complaint-heavy."
        )
        return Insight(
            id="risk.sentiment",
            category="risk",
            severity="info",
            title="Negative Sentiment Clusters",
            body=body,
            metric="sentiment",
            data={
                "communities": rows_low,
                "threshold": -0.05,
                "edges_with_sentiment": edges_with_sentiment,
            },
        )

    def _sparse_coverage(self) -> Insight | None:
        """Network density too low to support reliable structural analysis."""
        rec = self._query_one(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "WITH count(c) AS n "
            "OPTIONAL MATCH (a:Concept)-[r:RELATED]->(b:Concept) "
            f"WHERE {self._node_filter('a')} AND {self._node_filter('b')} "
            "RETURN n, count(r) AS e"
        )
        if not rec:
            return None
        n = int(rec["n"] or 0)
        e = int(rec["e"] or 0)
        if n < 2:
            return None
        density = e / (n * (n - 1))
        if density >= 0.005:
            return None  # not sparse enough to flag
        body = (
            f"The network is very sparse: density {density:.4f} across "
            f"{n} concepts and {e} edges. Centrality-based insights become "
            f"unstable below ~0.005 — consider lowering --min-weight, "
            f"sampling more documents, or aggregating concepts via the "
            f"concept dictionary."
        )
        return Insight(
            id="risk.sparse",
            category="risk",
            severity="medium",
            title="Sparse Network Structure",
            body=body,
            metric="density",
            data={"nodes": n, "edges": e, "density": density},
        )

    def _weakly_connected_fragmentation(self) -> Insight | None:
        """Many disconnected components → conversation is fragmented."""
        rec = self._query_one(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.wcc_component IS NOT NULL "
            "WITH count(c) AS total, count(DISTINCT c.wcc_component) AS k "
            "RETURN total, k"
        )
        if not rec or not rec.get("total"):
            return None
        total = int(rec["total"])
        k = int(rec["k"])
        if k <= max(2, total * 0.05):
            return None  # well connected
        ratio = k / total
        body = (
            f"The network fragments into {k} weakly-connected components "
            f"across {total} concepts (1 component per "
            f"{1/ratio:.1f} concepts). This level of fragmentation indicates "
            f"the corpus mixes multiple unrelated discussions — consider "
            f"separating sources or filtering by topic before analysis."
        )
        return Insight(
            id="risk.fragmentation",
            category="risk",
            severity="medium",
            title="Network Fragmentation",
            body=body,
            metric="wcc_component",
            data={"total": total, "components": k},
        )

    # ════════════════════════════════════════════════════════════
    # ACTION RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════

    def _action_diversify_hubs(self) -> Insight | None:
        """If a SPOF exists, recommend diversifying away from it."""
        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.pagerank IS NOT NULL AND c.betweenness IS NOT NULL "
            "RETURN c.label AS label, c.pagerank AS pr, "
            "       c.betweenness AS bc "
            "ORDER BY c.pagerank * c.betweenness DESC LIMIT 1"
        )
        if not rows or float(rows[0]["pr"] or 0) <= 0:
            return None
        top = rows[0]
        body = (
            f"Reduce the network's dependency on {self._quote(top['label'])}. "
            f"It currently absorbs both attention (PageRank "
            f"{float(top['pr']):.4f}) and structural control (betweenness "
            f"{float(top['bc']):.4f}). Investigate what makes it indispensable "
            f"and surface alternative concepts that can carry similar "
            f"connections."
        )
        return Insight(
            id="action.diversify",
            category="action",
            severity="high",
            title="Diversify Away From Hub",
            body=body,
            concepts=[top["label"]],
            metric="pagerank*betweenness",
            data=top,
        )

    def _action_leverage_bridges(self) -> Insight | None:
        """Encourage using bridge concepts as translators between groups."""
        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.betweenness IS NOT NULL AND c.community IS NOT NULL "
            "OPTIONAL MATCH (c)-[:RELATED]-(nb:Concept) "
            f"WHERE {self._node_filter('nb')} AND nb.community <> c.community "
            "WITH c, count(DISTINCT nb) AS cross_n, "
            "     collect(DISTINCT nb.community) AS reached "
            "WHERE cross_n >= 2 AND size(reached) >= 2 "
            "RETURN c.label AS label, c.community AS community, "
            "       reached, cross_n "
            "ORDER BY size(reached) DESC, cross_n DESC LIMIT 1"
        )
        if not rows:
            return None
        top = rows[0]
        reached = top["reached"] or []
        reached_descs = ", ".join(
            self._describe_community(c) for c in reached[:3]
        )
        if len(reached) > 3:
            reached_descs += f" and {len(reached) - 3} more"
        body = (
            f"Use {self._quote(top['label'])} as a translator concept when "
            f"communicating across audiences. From its home in "
            f"{self._describe_community(top['community'])}, it already "
            f"reaches {len(reached)} other communities — {reached_descs} — "
            f"through {top['cross_n']} cross-cluster ties, making it the "
            f"natural anchor for multi-audience messaging."
        )
        return Insight(
            id="action.bridge",
            category="action",
            severity="medium",
            title="Leverage Bridge Concept",
            body=body,
            concepts=[top["label"]],
            metric="cross_community_neighbors",
            data=top,
        )

    def _action_investigate_isolates(self) -> Insight | None:
        """If small WCC components exist, suggest investigating them."""
        rows = self._query_all(
            f"MATCH (c:Concept) WHERE {self._node_filter()} "
            "  AND c.wcc_component IS NOT NULL "
            "WITH c.wcc_component AS w, collect(c.label) AS labels "
            "WHERE size(labels) BETWEEN 2 AND 5 "
            "RETURN w, labels, size(labels) AS sz "
            "ORDER BY sz ASC LIMIT 1"
        )
        if not rows:
            return None
        top = rows[0]
        body = (
            f"Investigate the isolated cluster around "
            f"{self._join_quoted(top['labels'][:3])} "
            f"({top['sz']} concepts, disconnected from the main graph). "
            f"Either it represents a genuinely siloed topic worth surfacing, "
            f"or extraction noise that should be filtered out via the concept "
            f"dictionary."
        )
        return Insight(
            id="action.isolates",
            category="action",
            severity="low",
            title="Investigate Isolated Cluster",
            body=body,
            concepts=top["labels"][:5],
            metric="wcc_component",
            data=top,
        )

    def _action_cross_source_translation(self) -> Insight | None:
        """Combined-only: highlight the strongest cross-source bridge."""
        if self.source_label != "combined":
            return None
        rows = self._query_all(
            "MATCH (c:Concept {source_label: 'combined'}) "
            "WHERE coalesce(c.source_type, 'unknown') = 'both' "
            "  AND c.pagerank IS NOT NULL "
            "RETURN c.label AS label, c.pagerank AS pr "
            "ORDER BY c.pagerank DESC LIMIT 3"
        )
        if not rows:
            return None
        labels = [r["label"] for r in rows]
        body = (
            f"Anchor cross-source communication around "
            f"{self._join_quoted(labels)}. These concepts already appear in "
            f"both the policy and Yelp corpora and rank highest in influence — "
            f"making them the most efficient vocabulary for translating "
            f"between institutional and customer perspectives."
        )
        return Insight(
            id="action.cross_source",
            category="action",
            severity="medium",
            title="Anchor Cross-Source Communication",
            body=body,
            concepts=labels,
            metric="pagerank+source_type=both",
            data={"top": rows},
        )
