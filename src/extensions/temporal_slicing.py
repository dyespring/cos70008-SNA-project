"""Build ProcessedDocument lists for temporal / sectional network comparison."""

from __future__ import annotations

import math
import pandas as pd

from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokeniser import ProcessedDocument, SpacyTokeniser


def policy_pages_to_temporal_slices(
    pages: list[tuple[int, str]],
    tokeniser: SpacyTokeniser,
    n_chunks: int = 2,
    source: str = "policy",
) -> list[tuple[str, list[ProcessedDocument]]]:
    """Split policy PDF pages into sequential chunks; one ProcessedDocument per chunk.

    Parameters
    ----------
    pages : (page_num, text) from read_policy_pdf
    n_chunks : number of contiguous slices (e.g. 2 = first half / second half)
    """
    if not pages:
        return []
    n_chunks = max(1, n_chunks)
    chunk_size = max(1, math.ceil(len(pages) / n_chunks))
    slices: list[tuple[str, list[ProcessedDocument]]] = []

    for i in range(0, len(pages), chunk_size):
        chunk = pages[i : i + chunk_size]
        p0, p1 = chunk[0][0], chunk[-1][0]
        label = f"Pages {p0}–{p1}"
        full = "\n\n".join(t for _, t in chunk)
        cleaned = clean_text(full, source=source)
        if not cleaned.strip():
            continue
        doc = tokeniser.process(
            cleaned,
            doc_id=f"policy_slice_{p0}_{p1}",
            source=source,
        )
        slices.append((label, [doc]))
    return slices


def yelp_reviews_to_year_slices(
    df: pd.DataFrame,
    tokeniser: SpacyTokeniser,
    sample_per_year: int | None = 150,
    min_reviews: int = 5,
    random_state: int = 42,
) -> list[tuple[str, list[ProcessedDocument]]]:
    """Group Yelp reviews by calendar year; tokenise each year as a document batch.

    Expects columns review_id, text, date (ISO-like string).
    """
    if df.empty or "date" not in df.columns:
        return []

    d = df.copy()
    d["year"] = pd.to_datetime(d["date"], errors="coerce").dt.year
    d = d.dropna(subset=["year"])
    d["year"] = d["year"].astype(int)

    slices: list[tuple[str, list[ProcessedDocument]]] = []
    for year in sorted(d["year"].unique()):
        sub = d[d["year"] == year]
        if len(sub) < min_reviews:
            continue
        if sample_per_year and len(sub) > sample_per_year:
            sub = sub.sample(n=sample_per_year, random_state=random_state)
        pairs = [
            (str(row["review_id"]), clean_text(str(row["text"]), source="yelp"))
            for _, row in sub.iterrows()
        ]
        pairs = [(rid, t) for rid, t in pairs if t.strip()]
        if not pairs:
            continue
        docs = tokeniser.process_batch(pairs, source="yelp")
        slices.append((str(year), docs))
    return slices
