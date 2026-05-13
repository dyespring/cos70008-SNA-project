"""Offline tests for InsightEngine helpers and the LLM polish layer.

Live Neo4j integration is exercised in `tests/test_neo4j_pipeline.py`.
This module isolates and tests the pure-Python plumbing.
"""

from __future__ import annotations

import pytest

from src.extensions.insight_engine import (
    Insight,
    InsightEngine,
    _polish_text,
)


class _FakeLLM:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.output


class _BoomLLM:
    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("LLM down")


def test_insight_render_prefers_polished_body():
    raw = "Concept 'foo' has high pagerank."
    pol = "'foo' dominates the network."
    ins = Insight(id="x", category="key", title="t", body=raw, body_polished=pol)
    assert ins.render() == pol

    ins.body_polished = None
    assert ins.render() == raw


def test_polish_text_uses_llm_and_caches():
    llm = _FakeLLM("polished prose")
    ins = Insight(
        id="key.hub",
        category="key",
        title="Hub Concepts",
        body="The most influential concept is 'spam'.",
    )
    cache: dict[str, str] = {}

    out1 = _polish_text(llm, ins, cache)
    out2 = _polish_text(llm, ins, cache)

    assert out1 == "polished prose"
    assert out2 == "polished prose"
    assert len(llm.calls) == 1, "second call must hit the cache"
    assert "Hub Concepts" in llm.calls[0][1]
    assert "'spam'" in llm.calls[0][1]


def test_polish_text_falls_back_on_llm_failure():
    ins = Insight(id="x", category="key", title="t", body="raw observation")
    out = _polish_text(_BoomLLM(), ins, cache={})
    assert out == "raw observation"


def test_unavailable_helper_marks_insight():
    ins = InsightEngine._unavailable(
        "key.hub", "key", "Hub Concepts", "no pagerank"
    )
    assert ins.available is False
    assert ins.unavailable_reason == "no pagerank"
    assert "no pagerank" in ins.body
    assert ins.severity == "info"


def test_quote_helpers():
    assert InsightEngine._quote("alpha") == "'alpha'"
    assert (
        InsightEngine._join_quoted(["alpha", "beta"]) == "'alpha', 'beta'"
    )


def test_node_filter_with_and_without_slice():
    class _DummyStore:
        def session(self):  # pragma: no cover
            raise NotImplementedError

    eng = InsightEngine(_DummyStore(), source_label="combined")
    assert eng._node_filter() == "c.source_label = $sl"
    assert eng._params() == {"sl": "combined"}

    eng2 = InsightEngine(
        _DummyStore(), source_label="combined", slice_id="2024-Q1"
    )
    assert eng2._node_filter() == (
        "c.source_label = $sl AND c.slice_id = $sid"
    )
    assert eng2._node_filter("nb") == (
        "nb.source_label = $sl AND nb.slice_id = $sid"
    )
    assert eng2._params() == {"sl": "combined", "sid": "2024-Q1"}


def test_collect_swallows_failing_insight_fns():
    class _DummyStore: ...

    eng = InsightEngine(_DummyStore(), source_label="combined")

    def good() -> Insight:
        return Insight(id="g", category="key", title="g", body="g")

    def bad() -> Insight:
        raise RuntimeError("kaboom")

    out = eng._collect(good, bad, good)
    assert [i.id for i in out] == ["g", "g"]


def test_collect_skips_none_returning_fns():
    class _DummyStore: ...

    eng = InsightEngine(_DummyStore(), source_label="policy")

    def empty() -> Insight | None:
        return None

    def real() -> Insight:
        return Insight(id="r", category="key", title="r", body="r")

    out = eng._collect(empty, real, empty)
    assert [i.id for i in out] == ["r"]


def test_cross_source_insight_is_skipped_for_non_combined():
    class _DummyStore: ...

    eng = InsightEngine(_DummyStore(), source_label="policy")
    assert eng._cross_source_overlap() is None
    assert eng._action_cross_source_translation() is None


# ── Top-K cap & ranking (Todo 2) ──────────────────────────────────


def _make_insight(id_, severity="info", available=True, data=None):
    return Insight(
        id=id_, category="key", title=id_,
        body=id_, severity=severity,
        available=available, data=data or {},
    )


def test_top_caps_to_three_per_category():
    class _DummyStore: ...

    eng = InsightEngine(_DummyStore(), source_label="combined")
    items = [_make_insight(f"i{i}", severity="info") for i in range(5)]
    capped = eng._top(items)
    assert len(capped) == 3


def test_top_prefers_high_severity_over_low():
    class _DummyStore: ...

    eng = InsightEngine(_DummyStore(), source_label="combined")
    items = [
        _make_insight("low",    severity="low"),
        _make_insight("info",   severity="info"),
        _make_insight("high",   severity="high"),
        _make_insight("medium", severity="medium"),
    ]
    capped = eng._top(items)
    assert [i.id for i in capped] == ["high", "medium", "low"]


def test_top_prefers_available_over_unavailable():
    class _DummyStore: ...

    eng = InsightEngine(_DummyStore(), source_label="combined")
    items = [
        _make_insight("avail_info", severity="info", available=True),
        _make_insight("missing_high", severity="high", available=False),
        _make_insight("avail_low",  severity="low",  available=True),
    ]
    capped = eng._top(items)
    # Available insights should come first regardless of severity rank.
    assert capped[0].id in {"avail_info", "avail_low"}
    assert capped[-1].id == "missing_high"


def test_top_uses_data_size_as_tie_breaker():
    class _DummyStore: ...

    eng = InsightEngine(_DummyStore(), source_label="combined")
    items = [
        _make_insight("rich",  severity="medium", data={"top": [1, 2, 3, 4, 5]}),
        _make_insight("poor",  severity="medium", data={"top": [1]}),
    ]
    capped = eng._top(items)
    assert capped[0].id == "rich"


def test_top_per_category_is_constant_across_categories():
    """Default cap should be the same for key / risk / action."""
    assert InsightEngine.TOP_PER_CATEGORY == 3


# ── Echo chamber backbone exclusion ───────────────────────────────


class _FakeStoreScripted:
    """Stand-in store whose ``session().run()`` returns scripted Cypher results.

    The mapping is keyed by *substrings* of the query so we don't have
    to write the exact Cypher in test fixtures.
    """

    def __init__(self, scripted: list[tuple[str, list[dict]]]):
        self.scripted = scripted
        self.calls: list[str] = []

    def session(self):
        return _FakeSession(self)


class _FakeSession:
    def __init__(self, store: _FakeStoreScripted):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher: str, **params):
        self.store.calls.append(cypher)
        for needle, rows in self.store.scripted:
            if needle in cypher:
                return _FakeResult(rows)
        return _FakeResult([])


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self) -> list[dict]:
        return list(self._rows)

    def single(self) -> dict | None:
        return self._rows[0] if self._rows else None


def _engine_with_store(store, source_label: str = "combined") -> InsightEngine:
    eng = InsightEngine.__new__(InsightEngine)
    eng.store = store
    eng.source_label = source_label
    eng.slice_id = None
    eng.llm = None
    eng._polish_cache = {}
    # Mirror what __init__ does — _describe_community needs this attribute
    # even when nothing has been hydrated yet.
    eng._community_anchors = None
    return eng


def test_echo_chamber_excludes_backbone_community():
    """A high-internal-ratio community that covers >40% of the network
    must NOT be flagged as an echo chamber — it's the main backbone."""
    store = _FakeStoreScripted([
        # Total node count
        ("RETURN count(c) AS n", [{"n": 100}]),
        # High-ratio rows: community 38 has 90% internal but is 60% of the network.
        ("OR id(a) <", []),  # never matches
        ("id(a) < id(b)", [
            {"cid": 38, "internal": 90, "external": 10, "ratio": 0.9},
        ]),
        # Per-community sizes: 38 has 60 nodes (60% of 100).
        ("RETURN c.community AS cid, count(c) AS sz", [
            {"cid": 38, "sz": 60},
        ]),
    ])
    eng = _engine_with_store(store)
    ins = eng._echo_chambers()
    assert ins is not None
    # Backbone is excluded → "no echo chamber" outcome with explanatory body
    assert "No echo-chamber communities detected" in ins.body
    assert ins.severity == "info"


def test_echo_chamber_keeps_small_insular_pocket():
    """A high-internal-ratio community covering <40% should be flagged."""
    store = _FakeStoreScripted([
        ("RETURN count(c) AS n", [{"n": 100}]),
        ("id(a) < id(b)", [
            {"cid": 7, "internal": 50, "external": 5, "ratio": 0.91},
        ]),
        ("RETURN c.community AS cid, count(c) AS sz", [
            {"cid": 7, "sz": 12},  # 12% of network — pocket
        ]),
        # Anchor index for _describe_community
        ("AS pr", [
            {"cid": 7, "label": "vegan", "freq": 12, "pr": 1.4},
        ]),
    ])
    eng = _engine_with_store(store)
    ins = eng._echo_chambers()
    assert ins is not None
    assert ins.available is True
    assert ins.severity == "medium"
    assert "community 7" in ins.body.lower()
    assert "12 concepts" in ins.body


def test_echo_chamber_picks_smallest_among_qualifying():
    """When multiple small communities qualify, the *smallest* leads."""
    store = _FakeStoreScripted([
        ("RETURN count(c) AS n", [{"n": 200}]),
        ("id(a) < id(b)", [
            {"cid": 5, "internal": 80, "external": 5, "ratio": 0.94},
            {"cid": 9, "internal": 30, "external": 2, "ratio": 0.94},
        ]),
        ("RETURN c.community AS cid, count(c) AS sz", [
            {"cid": 5, "sz": 30},   # 15%
            {"cid": 9, "sz": 8},    # 4% — much smaller
        ]),
        ("AS pr", [
            {"cid": 5, "label": "alpha", "freq": 30, "pr": 0.5},
            {"cid": 9, "label": "beta",  "freq": 8,  "pr": 0.4},
        ]),
    ])
    eng = _engine_with_store(store)
    ins = eng._echo_chambers()
    assert ins is not None
    # Smallest insular pocket should be the headline
    assert "community 9" in ins.body.lower()


def test_echo_chamber_thresholds_are_constants():
    """Constants are class-level so they can be tweaked without touching
    method internals — lock the contract."""
    assert InsightEngine._ECHO_INTERNAL_RATIO == 0.85
    assert InsightEngine._ECHO_MAX_SIZE_RATIO == 0.40
    assert InsightEngine._ECHO_MIN_EDGES == 5


# ── Community landscape: dual anchors ─────────────────────────────


def test_community_landscape_lists_both_anchor_kinds():
    """Output should expose ``top_freq`` AND ``top_pagerank`` per community,
    and the body should mention both when they differ."""
    members = [
        {"label": "restaurant",     "freq": 500, "pr": 0.8},
        {"label": "drink",          "freq": 400, "pr": 0.3},
        {"label": "server",         "freq": 380, "pr": 0.2},
        {"label": "climate change", "freq": 50,  "pr": 4.9},
        {"label": "business",       "freq": 80,  "pr": 3.1},
        {"label": "investment",     "freq": 40,  "pr": 2.5},
    ]
    store = _FakeStoreScripted([
        ("collect({label: c.label", [
            {"cid": 38, "sz": 6, "members": members},
        ]),
    ])
    eng = _engine_with_store(store)
    ins = eng._community_landscape()
    assert ins is not None
    payload = ins.data["communities"][0]
    assert payload["top_freq"]      == ["restaurant", "drink", "server"]
    assert payload["top_pagerank"]  == ["climate change", "business", "investment"]
    # Body should mention both anchor lists since they differ.
    assert "high-frequency anchors" in ins.body
    assert "high-influence anchors" in ins.body
    # Concepts chip should track PageRank (so it lines up with Hub card).
    assert ins.concepts == ["climate change", "business", "investment"]


def test_community_landscape_collapses_when_anchors_coincide():
    """When freq-top and pagerank-top agree, body should not be redundant."""
    members = [
        {"label": "alpha", "freq": 50, "pr": 0.9},
        {"label": "beta",  "freq": 30, "pr": 0.5},
        {"label": "gamma", "freq": 20, "pr": 0.3},
    ]
    store = _FakeStoreScripted([
        ("collect({label: c.label", [
            {"cid": 1, "sz": 3, "members": members},
        ]),
    ])
    eng = _engine_with_store(store)
    ins = eng._community_landscape()
    assert ins is not None
    assert "anchored by" in ins.body
    assert "high-frequency anchors" not in ins.body


def test_community_landscape_returns_unavailable_without_data():
    store = _FakeStoreScripted([("collect({label: c.label", [])])
    eng = _engine_with_store(store)
    ins = eng._community_landscape()
    assert ins is not None
    assert ins.available is False
    assert "Communities not yet detected" in ins.unavailable_reason


# ── Community label helper ───────────────────────────────────────


def test_describe_community_uses_pagerank_anchors():
    """``_describe_community`` should pick top-3 by PageRank, falling back
    to frequency only when PageRank ties."""
    store = _FakeStoreScripted([
        # Anchor index query returns one row per concept.
        ("AS pr", [
            {"cid": 38, "label": "climate change", "freq": 50,  "pr": 4.9},
            {"cid": 38, "label": "restaurant",     "freq": 500, "pr": 4.6},
            {"cid": 38, "label": "business",       "freq": 80,  "pr": 3.1},
            {"cid": 38, "label": "drink",          "freq": 400, "pr": 0.3},
            {"cid": 1,  "label": "policy",         "freq": 100, "pr": 2.0},
        ]),
    ])
    eng = _engine_with_store(store)
    desc = eng._describe_community(38)
    # Should contain id + the three highest-PageRank labels in order.
    assert desc == "community 38 ('climate change', 'restaurant', 'business')"

    # Different community → different anchors.
    assert eng._describe_community(1) == "community 1 ('policy')"


def test_describe_community_caches_index():
    """The anchor index should be hydrated once per engine instance."""
    store = _FakeStoreScripted([
        ("AS pr", [
            {"cid": 1, "label": "alpha", "freq": 5, "pr": 1.0},
        ]),
    ])
    eng = _engine_with_store(store)
    eng._describe_community(1)
    eng._describe_community(1)
    eng._describe_community(1)
    # Only one anchor-index Cypher call should have been issued.
    anchor_calls = [c for c in store.calls if "AS pr" in c]
    assert len(anchor_calls) == 1


def test_describe_community_falls_back_to_bare_id_on_miss():
    store = _FakeStoreScripted([
        ("AS pr", [
            {"cid": 1, "label": "alpha", "freq": 5, "pr": 1.0},
        ]),
    ])
    eng = _engine_with_store(store)
    # 99 is not in the index → bare "community 99"
    assert eng._describe_community(99) == "community 99"


def test_describe_community_handles_none_or_garbage():
    store = _FakeStoreScripted([("AS pr", [])])
    eng = _engine_with_store(store)
    assert eng._describe_community(None) == "an unnamed community"
    assert eng._describe_community("not-a-number") == "community not-a-number"
