from dataclasses import FrozenInstanceError

import pytest

from Stock.forecast_methodology_audit import (
    audit_candidate_specs,
    build_audit_candidate,
    build_capital_efficiency_normalization,
    build_five_year_shadow,
    build_mature_margin_normalization,
    classify_quarterly_growth,
    methodology_hypotheses,
    spec_for_ticker,
    terminal_economics,
)


EXPECTED_TICKERS = {"NVDA", "GOOGL", "META", "MSFT", "MU", "AMZN", "AVGO", "AAPL"}


def test_audit_universe_contains_exactly_eight_requested_issuers():
    specs = audit_candidate_specs()

    assert len(specs) == 8
    assert {spec.ticker for spec in specs} == EXPECTED_TICKERS
    assert len({spec.issuer for spec in specs}) == 8


def test_alphabet_share_class_alias_resolves_to_one_issuer():
    assert spec_for_ticker("goog") == spec_for_ticker("GOOGL")


def test_out_of_universe_ticker_is_rejected():
    with pytest.raises(ValueError, match="outside the eight-issuer audit universe"):
        spec_for_ticker("TSLA")


def test_candidates_are_explicitly_separate_from_profile_workflow():
    for spec in audit_candidate_specs():
        assert "Candidate" in spec.candidate_status
        assert "accepted" not in spec.candidate_status.lower()
        assert "applied" not in spec.candidate_status.lower()


def test_nvda_three_year_candidate_matches_existing_research_path():
    spec = spec_for_ticker("NVDA")
    candidate = build_audit_candidate(spec, starting_operating_margin=0.64)

    assert candidate.near_term_revenue_growth == (0.55, 0.40, 0.25)
    assert candidate.forecast_years == 11
    assert candidate.revenue_fade_years == 8
    assert candidate.mature_operating_margin == pytest.approx(0.45)


def test_alphabet_candidate_preserves_reassessed_research_inputs():
    spec = spec_for_ticker("GOOGL")
    candidate = build_audit_candidate(spec, starting_operating_margin=0.3311)

    assert candidate.near_term_revenue_growth == (0.23, 0.20, 0.17)
    assert candidate.mature_operating_margin == pytest.approx(0.34)
    assert candidate.starting_sales_to_capital == pytest.approx(0.50)
    assert candidate.mature_sales_to_capital == pytest.approx(0.70)


def test_five_year_shadow_extends_explicit_path_without_shortening_fade():
    spec = spec_for_ticker("MSFT")
    base = build_audit_candidate(spec, starting_operating_margin=0.4678)
    shadow = build_five_year_shadow(spec, starting_operating_margin=0.4678)

    assert shadow.three_year_growth == (0.18, 0.18, 0.15)
    assert shadow.five_year_growth == (0.18, 0.18, 0.15, 0.14, 0.12)
    assert shadow.assumptions.forecast_years == base.forecast_years + 2
    assert shadow.assumptions.revenue_fade_years == base.revenue_fade_years
    assert shadow.confidence[-2:] == ("Low", "Low")
    assert shadow.purpose == "research_only_not_a_production_profile"


def test_shadow_result_is_immutable():
    shadow = build_five_year_shadow(spec_for_ticker("AAPL"), 0.33)

    with pytest.raises(FrozenInstanceError):
        shadow.ticker = "MSFT"  # type: ignore[misc]


def test_one_factor_capital_efficiency_normalization_changes_only_sales_to_capital():
    spec = spec_for_ticker("META")
    base = build_audit_candidate(spec, 0.38)
    normalized = build_capital_efficiency_normalization(base, spec)

    assert normalized is not None
    assert normalized.starting_sales_to_capital == pytest.approx(0.55)
    assert normalized.mature_sales_to_capital == pytest.approx(0.80)
    assert normalized.near_term_revenue_growth == base.near_term_revenue_growth
    assert normalized.mature_operating_margin == base.mature_operating_margin
    assert normalized.wacc == base.wacc


def test_one_factor_mature_margin_normalization_changes_only_mature_margin():
    spec = spec_for_ticker("AMZN")
    base = build_audit_candidate(spec, 0.11155)
    normalized = build_mature_margin_normalization(base, spec)

    assert normalized.mature_operating_margin == pytest.approx(0.14)
    assert normalized.near_term_revenue_growth == base.near_term_revenue_growth
    assert normalized.starting_sales_to_capital == base.starting_sales_to_capital
    assert normalized.wacc == base.wacc


def test_terminal_economics_use_existing_assumption_identities():
    assumptions = build_audit_candidate(spec_for_ticker("NVDA"), 0.64)
    result = terminal_economics(assumptions)

    assert result.terminal_roic == pytest.approx(assumptions.derived_terminal_roic)
    assert result.terminal_reinvestment_rate == pytest.approx(
        assumptions.terminal_growth / assumptions.derived_terminal_roic
    )
    assert result.terminal_fcff_to_nopat == pytest.approx(
        1.0 - result.terminal_reinvestment_rate
    )


@pytest.mark.parametrize(
    ("values", "cyclical", "expected"),
    [
        ((0.10, 0.11, 0.12), False, "insufficient_data"),
        ((0.10, 0.11, 0.12, 0.13), False, "accelerating"),
        ((0.13, 0.12, 0.11, 0.10), False, "decelerating"),
        ((0.10, 0.12, 0.11, 0.13), False, "stable"),
        ((-0.20, 0.80, 0.10, 2.00), True, "cyclical/rebounding"),
    ],
)
def test_quarterly_growth_classification(values, cyclical, expected):
    assert classify_quarterly_growth(values, cyclical=cyclical) == expected


def test_hypothesis_audit_is_balanced_and_complete():
    hypotheses = methodology_hypotheses()

    assert tuple(item.hypothesis_id for item in hypotheses) == tuple(
        f"H{index}" for index in range(1, 9)
    )
    assert all(item.evidence_for for item in hypotheses)
    assert all(item.evidence_against for item in hypotheses)


def test_fit_scorecard_uses_only_allowed_conclusions():
    allowed = {"FIT", "FIT WITH CAUTION", "NEEDS ADAPTATION", "POOR FIT"}
    assert {spec.methodology_fit for spec in audit_candidate_specs()} <= allowed
    assert spec_for_ticker("MU").methodology_fit == "POOR FIT"
    assert spec_for_ticker("NVDA").methodology_fit == "FIT WITH CAUTION"


def test_audit_module_does_not_carry_market_price_or_recommendation_fields():
    field_names = set(spec_for_ticker("NVDA").__dataclass_fields__)

    assert "market_price" not in field_names
    assert "upside" not in field_names
    assert "recommendation" not in field_names
    assert "review_status" not in field_names
