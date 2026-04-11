"""Text cleaning and noise removal utilities."""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str, source: str = "general") -> str:
    """Apply a sequence of cleaning steps to raw text.

    Parameters
    ----------
    text : the raw input string
    source : one of 'policy', 'yelp', or 'general' to apply source-specific rules
    """
    text = _normalise_unicode(text)
    text = _remove_urls(text)
    text = _remove_emails(text)

    if source == "policy":
        text = _remove_policy_artefacts(text)

    text = _normalise_whitespace(text)
    return text.strip()


def _normalise_unicode(text: str) -> str:
    """Replace smart quotes, em-dashes, and other Unicode punctuation with ASCII equivalents."""
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2022": " ",   # bullet
        "\u00b7": " ",   # middle dot
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKD", text)
    return text


def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def _remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+\.\S+", "", text)


_POLICY_ARTEFACTS = re.compile(
    r"|".join([
        r"National Climate Resilience and Adaptation Strategy",
        r"^\d{1,3}\s*$",
        r"-- \d+ of \d+ --",
        r"ISBN \S+",
        r"CC BY \S+",
        r"GPO Box.*",
        r"Telephone \d[\d\s]+",
        r"Web awe\.gov\.au",
    ]),
    re.MULTILINE,
)


def _remove_policy_artefacts(text: str) -> str:
    """Remove repeated headers, page markers, and boilerplate from the policy PDF."""
    return _POLICY_ARTEFACTS.sub("", text)


def _normalise_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
