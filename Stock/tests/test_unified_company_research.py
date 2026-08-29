from dataclasses import replace
import inspect

import pytest

from Stock import valuation
from Stock.company_profile_review import initialize_profile_review
from Stock.company_profiles import (
    build_multistage_assumptions_from_profile,
    get_company_profile,
)
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_amazon_research import amazon_history, amazon_inputs
from Stock.tests.test_alphabet_research import current_assumptions
from Stock.unified_company_research import (
    build_amd_research_profile,
    build_apple_research_profile,
    build_broadcom_research_profile,
    build_micron_research_profile,
)


BUILDERS = {
    "MU": build_micron_research_profile,
    "AAPL": build_apple_research_profile,
    "AVGO": build_broadcom_research_profile,
    "AMD": build_amd_research_profile,
}


def result(ticker):
    return BUILDERS[ticker](current_assumptions(), amazon_history())


@pytest.mark.parametrize("ticker", ("AMZN", "MU", "AAPL", "AVGO", "AMD"))
def test_phase4_profile_definition_exists(ticker):
    lookup = get_company_profile(ticker)
    assert lookup.available
    assert lookup.profile.issuer_id == ticker


@pytest.mark.parametrize("ticker", ("MU", "AAPL", "AVGO", "AMD"))
def test_candidate_is_unreviewed_unapplied_and_uses_common_schema(ticker):
    profile = result(ticker).lookup.profile
    state = initialize_profile_review(profile)
    assert profile.profile_status == "research_in_progress"
    assert profile.last_reviewed_at is None
    assert profile.reinvestment_strategy is None
    assert state.reviewed_snapshot is None
    assert state.incomplete_groups
    assert profile.revenue_framework.year1_growth is not None
    assert profile.revenue_framework.year2_growth is not None
    assert profile.revenue_framework.year3_growth is not None


@pytest.mark.parametrize(
    "ticker,growth,margin,sc,wacc,g,roic,risk",
    (
        ("MU", (1.55, .20, .15), .28, .55, .105, .025, .28*.85*.55, "High"),
        ("AAPL", (.12, .08, .06), .32, 1.80, .085, .03, .32*.84*1.80, "Medium"),
        ("AVGO", (.80, .60, .25), .46, .75, .095, .03, .46*.85*.75, "High"),
        ("AMD", (.46, .60, .35), .28, 1.50, .115, .03, .28*.85*1.50, "High"),
    ),
)
def test_researched_assumptions_and_terminal_economics(
    ticker, growth, margin, sc, wacc, g, roic, risk
):
    profile = result(ticker).lookup.profile
    translated = build_multistage_assumptions_from_profile(profile)
    assert translated.available
    assumptions = translated.assumptions
    assert assumptions.near_term_revenue_growth == growth
    assert assumptions.revenue_fade_years == 8
    assert assumptions.forecast_years == 11
    assert assumptions.mature_operating_margin == pytest.approx(margin)
    assert assumptions.mature_sales_to_capital == pytest.approx(sc)
    assert assumptions.wacc == pytest.approx(wacc)
    assert assumptions.terminal_growth == pytest.approx(g)
    assert assumptions.derived_terminal_roic == pytest.approx(roic)
    assert profile.model_risk == risk


@pytest.mark.parametrize("ticker", ("MU", "AAPL", "AVGO", "AMD"))
def test_candidate_runs_through_standard_sales_to_capital_engine(ticker):
    profile = result(ticker).lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    inputs = replace(
        amazon_inputs(), ticker=ticker,
        starting_revenue=float(profile.revenue_framework.starting_revenue.value),
    )
    run = run_multistage_dcf(inputs, assumptions)
    assert run.per_share_value.intrinsic_value_per_share == pytest.approx(
        run.equity_value.equity_value / inputs.shares_outstanding
    )
    assert len(run.operating_forecast.years) == 11
    for year in run.operating_forecast.years:
        assert year.reinvestment == pytest.approx(
            year.delta_revenue / year.sales_to_capital
        )


def test_micron_cycle_is_normalized_in_assumptions_not_engine():
    profile = result("MU").lookup.profile
    text = " ".join(profile.uncertainty_notes)
    assert "Remaining memory cyclicality is moderated" in text
    assert "not eliminated" in text
    assert profile.margin_framework.mature_operating_margin.value < .804


def test_apple_economic_capital_efficiency_is_explicit():
    profile = result("AAPL").lookup.profile
    text = " ".join(profile.uncertainty_notes)
    assert "economic research assumption" in text
    assert "accounting invested capital" in text


def test_broadcom_debt_equity_bridge_remains_separate_from_operating_profile():
    profile = result("AVGO").lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    inputs = replace(
        amazon_inputs(), ticker="AVGO", starting_revenue=80e9,
        net_debt=50e9,
    )
    run = run_multistage_dcf(inputs, assumptions)
    assert run.equity_value.equity_value == pytest.approx(
        run.enterprise_value.enterprise_value - 50e9
    )
    assert "Acquisition and segment complexity" in " ".join(profile.uncertainty_notes)


def test_amd_profile_preserves_gaap_non_gaap_distinction_and_official_growth_evidence():
    profile = result("AMD").lookup.profile
    evidence = {item.evidence_id: item for item in profile.evidence_items}

    assert evidence["analyst_day_growth"].value == ">35% CAGR"
    assert evidence["analyst_day_margin"].value == ">35%"
    assert profile.margin_framework.mature_operating_margin.value == pytest.approx(.28)
    assert "GAAP-oriented" in profile.margin_framework.mature_operating_margin.rationale
    assert "customer warrants" in " ".join(profile.uncertainty_notes)


@pytest.mark.parametrize("builder", BUILDERS.values())
def test_market_price_is_not_a_candidate_input(builder):
    parameters = inspect.signature(builder).parameters
    assert "market_price" not in parameters
    assert "current_price" not in parameters


def test_valuation_engine_has_no_ticker_specific_dispatch():
    source = inspect.getsource(valuation).lower()
    for ticker in ("amzn", "micron", "aapl", "avgo", "amd"):
        assert ticker not in source
    assert "hybrid_explicit_with_handoff" not in source
