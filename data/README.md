# Data directory

Large or licensed datasets are **not** committed to Git. After cloning the repo, add files locally as described below. Paths match [`src/config.py`](../src/config.py).

## Policy PDF (optional)

**Expected path:** `data/national-climate-resilience-and-adaptation-strategy.pdf`

- Variable: `POLICY_PDF_PATH` → `DATA_DIR / "national-climate-resilience-and-adaptation-strategy.pdf"`
- Obtain the Australian National Climate Resilience and Adaptation Strategy (2021–2025) PDF from the official government source and save it under that filename, **or** pass a custom path to the pipeline: `python pipeline.py --pdf-path /path/to/file.pdf`

## Yelp academic dataset (optional)

**Expected directory:** `data/Yelp JSON/yelp_dataset/`

- Variable: `YELP_EXTRACTED_DIR` → `DATA_DIR / "Yelp JSON" / "yelp_dataset"`
- Download the [Yelp Open Dataset](https://www.yelp.com/dataset) (academic use agreement), extract the archive, and copy at least these JSON files into `yelp_dataset/`:
  - `yelp_academic_dataset_business.json`
  - `yelp_academic_dataset_review.json`

The pipeline and [`src/ingest/yelp_reader.py`](../src/ingest/yelp_reader.py) read those paths directly.

**Do not commit** Yelp JSON (multi‑GB) or `Yelp Photos/`; they are listed in the root `.gitignore`.

## GitHub size limits

Keep individual files under ~50–100 MB. Use local copies or institutional storage for full dumps; document links here or in the main README if needed.
