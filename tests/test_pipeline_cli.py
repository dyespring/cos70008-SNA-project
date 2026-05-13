"""Argparse-level tests for the new tuning flags on ``pipeline.py``.

These tests don't run the actual stages — they just parse argv and
verify that the new knobs are wired up with sensible defaults and
override correctly. Keeps the CLI contract from drifting silently.
"""

from __future__ import annotations

import pytest

from pipeline import build_parser


# Common base argv slot for every subcommand we want to test against.
_SOURCE_FOR = {
    "etl":      ["--source", "policy"],
    "analyse":  ["--source", "policy"],
    "viz":      ["--source", "policy"],
    "temporal": ["--source", "policy"],
    "all":      ["--source", "policy"],
}


@pytest.mark.parametrize("subcmd", ["etl", "analyse", "viz", "temporal", "all"])
def test_default_source_is_combined(subcmd):
    parser = build_parser()
    args = parser.parse_args([subcmd])
    assert args.source == "combined"


@pytest.mark.parametrize("subcmd", ["etl", "analyse", "viz", "temporal", "all"])
def test_shared_tuning_flags_present_with_defaults(subcmd):
    parser = build_parser()
    args = parser.parse_args([subcmd, *_SOURCE_FOR[subcmd]])

    # Every shared knob should expose a default — value isn't asserted
    # here (defaults live in ``src.config``), only that the attribute
    # exists.
    for attr in (
        "min_concept_freq", "min_person_freq",
        "min_org_freq", "min_product_freq",
        "tfidf_top_n", "tfidf_min_df", "tfidf_max_df",
        "min_weight", "min_npmi", "cooccurrence_window",
        "advanced_extraction", "advanced_substring_ratio",
    ):
        assert hasattr(args, attr), (
            f"{subcmd}: --{attr.replace('_', '-')} flag missing"
        )


def test_etl_overrides_propagate_to_args():
    parser = build_parser()
    args = parser.parse_args([
        "etl",
        "--source", "yelp",
        "--min-concept-freq", "5",
        "--min-person-freq", "20",
        "--tfidf-min-df", "4",
        "--tfidf-max-df", "0.4",
        "--min-weight", "3",
        "--min-npmi", "0.1",
        "--cooccurrence-window", "1",
        "--advanced-extraction",
        "--advanced-substring-ratio", "0.7",
    ])
    assert args.min_concept_freq == 5
    assert args.min_person_freq == 20
    assert args.tfidf_min_df == 4
    assert args.tfidf_max_df == pytest.approx(0.4)
    assert args.min_weight == 3
    assert args.min_npmi == pytest.approx(0.1)
    assert args.cooccurrence_window == 1
    assert args.advanced_extraction is True
    assert args.advanced_substring_ratio == pytest.approx(0.7)


def test_advanced_extraction_off_by_default():
    parser = build_parser()
    args = parser.parse_args(["etl", "--source", "policy"])
    assert args.advanced_extraction is False


def test_temporal_skip_gds_flag_exists():
    parser = build_parser()
    args = parser.parse_args([
        "temporal", "--source", "policy", "--temporal", "3", "--skip-gds",
    ])
    assert getattr(args, "skip_gds", False) is True


def test_temporal_skip_gds_default_false():
    parser = build_parser()
    args = parser.parse_args([
        "temporal", "--source", "policy", "--temporal", "3",
    ])
    assert getattr(args, "skip_gds", True) is False


def test_min_npmi_negative_one_disables_filter():
    parser = build_parser()
    args = parser.parse_args(["etl", "--source", "policy", "--min-npmi", "-1"])
    assert args.min_npmi == pytest.approx(-1.0)


def test_per_entity_freq_flags_override():
    parser = build_parser()
    args = parser.parse_args([
        "etl", "--source", "yelp",
        "--min-org-freq", "5",
        "--min-product-freq", "8",
    ])
    assert args.min_org_freq == 5
    assert args.min_product_freq == 8


def test_per_entity_freq_flags_have_sane_defaults():
    parser = build_parser()
    args = parser.parse_args(["etl", "--source", "policy"])
    assert args.min_org_freq == 10
    assert args.min_product_freq == 10
