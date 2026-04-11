"""Tests for the text cleaning module."""

import pytest

from src.preprocessing.cleaner import clean_text


class TestCleanText:
    def test_removes_urls(self):
        text = "Visit https://example.com for more info."
        result = clean_text(text)
        assert "https://example.com" not in result
        assert "Visit" in result

    def test_removes_emails(self):
        text = "Contact admin@example.com today."
        result = clean_text(text)
        assert "admin@example.com" not in result
        assert "Contact" in result

    def test_normalises_smart_quotes(self):
        text = "\u201cHello\u201d \u2018world\u2019"
        result = clean_text(text)
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert '"Hello"' in result

    def test_normalises_em_dashes(self):
        text = "climate\u2014resilience"
        result = clean_text(text)
        assert "\u2014" not in result
        assert "climate-resilience" in result

    def test_collapses_whitespace(self):
        text = "word1   word2\n\n\n\nword3"
        result = clean_text(text)
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_policy_artefact_removal(self):
        text = "National Climate Resilience and Adaptation Strategy\nImportant content here."
        result = clean_text(text, source="policy")
        assert "National Climate Resilience and Adaptation Strategy" not in result
        assert "Important content here" in result

    def test_page_marker_removal(self):
        text = "Some text\n-- 5 of 47 --\nMore text"
        result = clean_text(text, source="policy")
        assert "-- 5 of 47 --" not in result

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_general_mode_preserves_policy_headers(self):
        text = "National Climate Resilience and Adaptation Strategy"
        result = clean_text(text, source="general")
        assert "National Climate" in result
