"""Load and sample Yelp review data from the extracted academic dataset."""

from __future__ import annotations

import json

import pandas as pd

from src.config import YELP_EXTRACTED_DIR

_BIZ_PATH = YELP_EXTRACTED_DIR / "yelp_academic_dataset_business.json"
_REV_PATH = YELP_EXTRACTED_DIR / "yelp_academic_dataset_review.json"


def load_yelp_reviews(
    category_filter: str | None = "Restaurants",
    star_filter: int | None = None,
    sample_n: int | None = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load Yelp reviews, optionally filtering by category and star rating.

    Set sample_n=None to load ALL matching reviews (no sampling).

    Returns a DataFrame with columns:
        review_id, business_id, text, stars, date, business_name, categories
    """
    if not _BIZ_PATH.exists() or not _REV_PATH.exists():
        raise FileNotFoundError(
            f"Extracted Yelp JSONL files not found in {YELP_EXTRACTED_DIR}. "
            "Please extract yelp_dataset.tar into that directory first."
        )
    print("       Using extracted Yelp JSONL files")

    print("       Reading businesses...")
    biz_df = pd.read_json(_BIZ_PATH, lines=True, dtype_backend="numpy_nullable")[
        ["business_id", "name", "categories"]
    ]
    biz_df = biz_df.rename(columns={"name": "business_name"})

    if category_filter:
        biz_df = biz_df[
            biz_df["categories"]
            .fillna("")
            .str.contains(category_filter, case=False)
        ]
        biz_ids = set(biz_df["business_id"])
        print(f"       {len(biz_ids)} businesses match '{category_filter}'")
    else:
        biz_ids = None

    max_rows = (sample_n * 5) if sample_n else None
    print(f"       Reading reviews (streaming, limit={max_rows or 'ALL'})...")
    reviews: list[dict] = []
    with open(_REV_PATH, "r") as f:
        for line in f:
            obj = json.loads(line)
            if biz_ids is not None and obj.get("business_id") not in biz_ids:
                continue
            reviews.append(obj)
            if max_rows and len(reviews) >= max_rows:
                break
    print(f"       {len(reviews)} candidate reviews read")

    rev_df = pd.DataFrame(reviews)[["review_id", "business_id", "text", "stars", "date"]]

    if star_filter is not None:
        rev_df = rev_df[rev_df["stars"] == star_filter]

    if sample_n and len(rev_df) > sample_n:
        rev_df = rev_df.sample(n=sample_n, random_state=random_state)

    result = rev_df.merge(biz_df, on="business_id", how="left")
    print(f"       Final dataset: {len(result)} reviews")
    return result.reset_index(drop=True)
