from dataclasses import fields, replace

import pytest

from Stock import stock_valuation_mvp as app
from Stock.company_profile_one_click import build_one_click_review_apply
from Stock.company_profiles import (
    CompanyResearchProfile,
    build_multistage_assumptions_from_profile,
    get_company_profile,
    normalize_profile_issuer,
)
from Stock.hyperscaler_research import (
    build_meta_research_profile,
    build_microsoft_research_profile,
)
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_alphabet_research import (
    current_assumptions,
    history,
    inputs,
)


@pytest.mark.parametrize(
    "builder,ticker,growth,margin,start_sc,mature_sc,tax,wacc",
    [
        (build_microsoft_research_profile, "MSFT", (0.18, 0.19, 0.17), 0.42, 0.48, 0.70, 0.19, 0.0925),
        (build_meta_research_profile, "META", (0.24, 0.20, 0.17), 0.36, 0.47, 0.75, 0.16, 0.0975),
    ],
)
def test_profiles_are_complete_unreviewed_candidates(
    builder, ticker, growth, margin, start_sc, mature_sc, tax, wacc
):
    result = builder(current_assumptions(), history(), retrieved_at="2026-08-23")
    profile = result.lookup.profile
    translated = build_multistage_assumptions_from_profile(profile)
    assert result.lookup.available
    assert profile.issuer_id == ticker
    assert profile.profile_status == "research_in_progress"
    assert profile.last_reviewed_at is None
    assert translated.available
    assert translated.assumptions.near_term_revenue_growth == growth
    assert translated.assumptions.revenue_fade_years == 8
    assert translated.assumptions.forecast_years == 11
    assert translated.assumptions.mature_operating_margin == pytest.approx(margin)
    assert translated.assumptions.starting_sales_to_capital == pytest.approx(start_sc)
    assert translated.assumptions.mature_sales_to_capital == pytest.approx(mature_sc)
    assert translated.assumptions.operating_tax_rate == pytest.approx(tax)
    assert translated.assumptions.wacc == pytest.approx(wacc)
    assert translated.assumptions.terminal_growth == pytest.approx(0.0325)


@pytest.mark.parametrize(
    "builder,ticker,expected_roic",
    [
        (build_microsoft_research_profile, "MSFT", 0.42 * 0.81 * 0.70),
        (build_meta_research_profile, "META", 0.36 * 0.84 * 0.75),
    ],
)
def test_implied_fade_terminal_economics_and_full_dcf(builder, ticker, expected_roic):
    profile = builder(current_assumptions(), history()).lookup.profile
    candidate = build_multistage_assumptions_from_profile(profile).assumptions
    run = run_multistage_dcf(inputs(ticker), candidate)
    assert run.forecast_path.years[3].revenue_growth == pytest.approx(0.1528125)
    assert run.forecast_path.years[4].revenue_growth == pytest.approx(0.135625)
    assert run.forecast_path.years[-1].revenue_growth == pytest.approx(0.0325)
    assert candidate.derived_terminal_roic == pytest.approx(expected_roic)
    assert candidate.terminal_reinvestment_rate == pytest.approx(0.0325 / expected_roic)
    assert run.per_share_value is not None
    assert 0 < run.enterprise_value.terminal_value_share < 1


@pytest.mark.parametrize("builder,ticker", [
    (build_microsoft_research_profile, "MSFT"),
    (build_meta_research_profile, "META"),
])
def test_generic_one_click_is_eligible_but_builder_does_not_review_or_apply(builder, ticker):
    profile = builder(current_assumptions(), history()).lookup.profile
    result = build_one_click_review_apply(
        profile, current_assumptions(), reviewed_at="2026-08-23T10:00:00+00:00",
        applied_at="2026-08-23T10:00:01+00:00", preview_validated=True,
    )
    assert result.reviewed_snapshot.profile.issuer_id == ticker
    assert result.application.issuer == ticker
    assert profile.profile_status == "research_in_progress"
    assert profile.last_reviewed_at is None


def test_issuer_registration_and_no_y4_y5_or_market_price_fields():
    assert normalize_profile_issuer("msft") == "MSFT"
    assert normalize_profile_issuer("meta") == "META"
    assert get_company_profile("MSFT").available
    assert get_company_profile("META").available
    names = {item.name for item in fields(CompanyResearchProfile)}
    assert not any("year4" in name or "year5" in name or "market_price" in name for name in names)


def test_candidate_generation_is_market_price_independent_and_formula_wacc_is_not_mutated():
    current = current_assumptions()
    microsoft = build_microsoft_research_profile(current, history()).lookup.profile
    meta = build_meta_research_profile(current, history()).lookup.profile
    assert microsoft.wacc_framework.wacc_audit is None
    assert meta.wacc_framework.wacc_audit is None
    assert "market price" in microsoft.rationale.lower()
    assert "market price" in meta.rationale.lower()
    assert current == current_assumptions()


def test_existing_candidate_profile_is_immutable_when_a_new_candidate_is_built():
    original = build_microsoft_research_profile(current_assumptions(), history()).lookup.profile
    signature = app.candidate_assumption_signature(original)
    changed_current = replace(current_assumptions(), wacc=0.11)
    newer = build_microsoft_research_profile(changed_current, history()).lookup.profile
    assert app.candidate_assumption_signature(original) == signature
    assert newer.profile_status == "research_in_progress"


def test_no_hybrid_reinvestment_enters_production_profile():
    profile = build_meta_research_profile(current_assumptions(), history()).lookup.profile
    text = repr(profile).lower()
    assert "delta working capital" not in text
    assert "capex - depreciation" not in text
