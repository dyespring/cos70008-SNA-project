"""Extract text from policy PDF documents using pdfplumber."""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from src.config import POLICY_PDF_PATH


def read_policy_pdf(
    pdf_path: str | Path | None = None,
    skip_pages: set[int] | None = None,
) -> list[tuple[int, str]]:
    """Read a PDF and return a list of (page_number, raw_text) tuples.

    Parameters
    ----------
    pdf_path : path to the PDF file (defaults to the configured policy PDF)
    skip_pages : 1-indexed page numbers to skip (e.g. cover, TOC)
    """
    pdf_path = Path(pdf_path) if pdf_path else POLICY_PDF_PATH
    skip_pages = skip_pages or {1, 2, 3, 5}  # cover, copyright, acknowledgement, TOC

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if i in skip_pages:
                continue
            text = page.extract_text() or ""
            text = _strip_page_artefacts(text, i)
            if text.strip():
                pages.append((i, text))
    return pages


_HEADER_RE = re.compile(
    r"^\d+\s+National Climate Resilience and Adaptation Strategy\s*",
    re.MULTILINE,
)
_FOOTER_RE = re.compile(r"(?:Photo|©|Source):.*$", re.MULTILINE)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)


def _strip_page_artefacts(text: str, page_num: int) -> str:
    """Remove headers, footers, photo credits and standalone page numbers."""
    text = _HEADER_RE.sub("", text)
    text = _FOOTER_RE.sub("", text)
    text = _PAGE_NUM_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pages_to_document(pages: list[tuple[int, str]]) -> str:
    """Concatenate page tuples into a single document string."""
    return "\n\n".join(text for _, text in pages)
