"""Offline tests for the per-metric LLM blurb helpers.

We don't call any real LLM — a fake provider records prompts so we can
verify (a) the data is actually included and (b) cache signatures are
deterministic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.extensions.metric_llm import (
    df_blurb,
    df_signature,
    stats_blurb,
    stats_signature,
)


class _FakeLLM:
    def __init__(self, output: str = "fake summary"):
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.output


class _BoomLLM:
    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("LLM down")


def test_df_blurb_handles_empty_without_calling_llm():
    llm = _FakeLLM()
    out = df_blurb(
        llm, pd.DataFrame(),
        metric_name="centrality", description="…",
    )
    assert "No data" in out
    assert llm.calls == []


def test_df_blurb_includes_top_rows_in_prompt():
    llm = _FakeLLM("two-sentence interpretation.")
    df = pd.DataFrame([
        {"label": "climate", "pagerank": 0.5},
        {"label": "policy",  "pagerank": 0.3},
        {"label": "service", "pagerank": 0.2},
        {"label": "menu",    "pagerank": 0.1},
        {"label": "tip",     "pagerank": 0.05},
    ])
    out = df_blurb(
        llm, df, metric_name="PageRank centrality",
        description="influence ranking",
        rank_column="pagerank", top_rows=3,
    )
    assert out == "two-sentence interpretation."
    assert len(llm.calls) == 1
    user_msg = llm.calls[0][1]
    # Top three labels should appear; bottom two should not.
    assert "climate" in user_msg
    assert "policy" in user_msg
    assert "service" in user_msg
    assert "menu" not in user_msg
    assert "tip" not in user_msg


def test_df_blurb_caps_top_rows_correctly():
    """Only the requested top_rows appear in the prompt — no spillover."""
    llm = _FakeLLM("ok")
    df = pd.DataFrame({"label": [f"c{i}" for i in range(20)],
                       "pagerank": list(range(20, 0, -1))})
    df_blurb(
        llm, df, metric_name="x", description="y",
        rank_column="pagerank", top_rows=5,
    )
    # 5 records → roughly 5 occurrences of "label"
    user_msg = llm.calls[0][1]
    assert user_msg.count('"label"') == 5


def test_df_blurb_falls_back_on_llm_error():
    out = df_blurb(
        _BoomLLM(), pd.DataFrame({"a": [1]}),
        metric_name="x", description="y",
    )
    assert "LLM blurb unavailable" in out


def test_stats_blurb_basic():
    llm = _FakeLLM("nice")
    out = stats_blurb(
        llm, {"nodes": 100, "edges": 300, "density": 0.001},
        metric_name="Graph summary", description="overall stats",
    )
    assert out == "nice"
    user_msg = llm.calls[0][1]
    assert "100" in user_msg and "300" in user_msg


def test_stats_blurb_handles_empty():
    llm = _FakeLLM()
    out = stats_blurb(llm, {}, metric_name="x", description="y")
    assert "No data" in out
    assert llm.calls == []


def test_signatures_are_stable_and_data_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    sig1 = df_signature(df)
    sig2 = df_signature(df)
    assert sig1 == sig2
    # Same data, different order → different signature (we don't sort the
    # underlying table, signature should reflect actual content shown).
    df2 = df.iloc[::-1].reset_index(drop=True)
    assert df_signature(df2) != sig1


def test_signatures_handle_empty_inputs():
    assert df_signature(pd.DataFrame()) == "empty"
    assert stats_signature({}) == "empty"


def test_stats_signature_changes_with_values():
    s1 = stats_signature({"nodes": 100})
    s2 = stats_signature({"nodes": 101})
    assert s1 != s2


def test_df_signature_handles_pandas_types():
    """numpy / pandas types should round-trip without raising."""
    df = pd.DataFrame({
        "a": pd.Series([1, 2, 3], dtype="int64"),
        "b": pd.Series([0.1, 0.2, 0.3], dtype="float64"),
        "c": pd.Series(["x", "y", "z"]),
    })
    sig = df_signature(df)
    assert isinstance(sig, str) and len(sig) == 16
