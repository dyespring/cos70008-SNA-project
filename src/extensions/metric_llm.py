"""LLM-driven, per-metric narrative blurbs for the dashboard tabs.

These are *cheap, structured* prompts (a couple of dozen rows shoved
into one user message) — fundamentally different from the
``insight_engine`` polishing pass which rewrites a single observation.
Each blurb summarises a *whole metric* (centrality table, community
table, broker table, summary stats) in one short paragraph.

Wraps :func:`src.extensions.chatbot.build_default_provider` for provider
lookup so the dashboard can switch backends (echo / openai / dashscope)
without re-instantiating the chatbot.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from src.extensions.chatbot import LLMProvider


logger = logging.getLogger(__name__)


# ── Prompt prelude shared by every metric blurb ────────────────────


_SYSTEM_PROMPT = """You are a senior data analyst writing a short
analytical paragraph about one slice of a social network analysis
result. Your audience is a non-technical decision maker.

Strict rules:
- 2 to 3 sentences, no more.
- Quote concept names in single quotes.
- Cite at least one specific number from the data (top value, count,
  ratio, percentage).
- Neutral analytical tone — never marketing copy, never speculation.
- Never invent details that are not in the data."""


_DEFAULT_TOP_ROWS = 5


def df_blurb(
    llm: "LLMProvider",
    df: "pd.DataFrame",
    *,
    metric_name: str,
    description: str,
    rank_column: str | None = None,
    top_rows: int = _DEFAULT_TOP_ROWS,
    columns: list[str] | None = None,
) -> str:
    """Summarise the top rows of a DataFrame in 2-3 sentences via the LLM.

    Parameters
    ----------
    llm:
        Any :class:`LLMProvider` (echo, openai, dashscope, …).
    df:
        The data the dashboard tab is showing. Empty DataFrames return
        an explanatory placeholder string instead of calling the LLM.
    metric_name:
        Short human label, e.g. ``"PageRank centrality"``.
    description:
        Sentence telling the LLM what the metric represents (so it can
        speak about it correctly without inventing).
    rank_column:
        Sort the table descending by this column before truncating to
        ``top_rows``. ``None`` keeps the existing order.
    top_rows:
        Number of rows to include in the prompt. 5 is enough for a
        useful narrative without bloating the token count.
    columns:
        Subset of columns to include (defaults to every column).
    """
    if df is None or df.empty:
        return f"_No data for {metric_name} — run the relevant pipeline step first._"

    work = df
    if rank_column is not None and rank_column in df.columns:
        work = df.sort_values(rank_column, ascending=False)
    if columns:
        work = work[[c for c in columns if c in work.columns]]
    work = work.head(top_rows)

    sample = work.to_dict(orient="records")
    user_msg = (
        f"Metric: {metric_name}\n"
        f"What it measures: {description}\n"
        f"Sample (top {len(sample)} rows):\n"
        f"{json.dumps(sample, default=_json_default, ensure_ascii=False)}\n\n"
        "Write 2 to 3 sentences interpreting what the data shows for a "
        "non-technical reader. Quote concept labels in single quotes; "
        "cite at least one specific number; do not invent."
    )
    try:
        return llm.complete(_SYSTEM_PROMPT, user_msg).strip()
    except Exception as e:  # pragma: no cover — network / provider issue
        logger.warning("LLM blurb failed for %s: %s", metric_name, e)
        return f"_(LLM blurb unavailable: {e})_"


def stats_blurb(
    llm: "LLMProvider",
    stats: dict[str, Any],
    *,
    metric_name: str,
    description: str,
) -> str:
    """Summarise a flat ``{name: value}`` stats dict in 2-3 sentences."""
    if not stats:
        return f"_No data for {metric_name}._"
    user_msg = (
        f"Metric: {metric_name}\n"
        f"What it measures: {description}\n"
        f"Values:\n"
        f"{json.dumps(stats, default=_json_default, ensure_ascii=False)}\n\n"
        "Write 2 to 3 sentences interpreting the values for a "
        "non-technical reader. Cite at least one specific number; "
        "do not invent."
    )
    try:
        return llm.complete(_SYSTEM_PROMPT, user_msg).strip()
    except Exception as e:  # pragma: no cover
        logger.warning("LLM blurb failed for %s: %s", metric_name, e)
        return f"_(LLM blurb unavailable: {e})_"


def df_signature(df: "pd.DataFrame", *, top_rows: int = _DEFAULT_TOP_ROWS) -> str:
    """Return a deterministic short hash of the top rows of a DataFrame.

    Used as a Streamlit cache key so a blurb is recomputed only when the
    underlying numbers change — rerunning the dashboard with the same
    data hits the cache.
    """
    if df is None or df.empty:
        return "empty"
    head = df.head(top_rows).to_dict(orient="records")
    payload = json.dumps(head, default=_json_default, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stats_signature(stats: dict[str, Any]) -> str:
    if not stats:
        return "empty"
    payload = json.dumps(stats, default=_json_default, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _json_default(value: Any) -> Any:
    """Make pandas / numpy types JSON-serialisable inside prompts."""
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
