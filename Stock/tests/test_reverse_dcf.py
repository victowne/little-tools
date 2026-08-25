from dataclasses import FrozenInstanceError, replace
from copy import deepcopy

import pandas as pd
import pytest

from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.reverse_dcf import (
    AMBIGUOUS,
    GROWTH_UPLIFT,
    MARKET_PRICE_UNAVAILABLE,
    MATURE_MARGIN,
    MATURE_SALES_TO_CAPITAL,
    OUTSIDE_REASONABLE_RANGE,
    SOLVED,
    WACC,
    ReverseVariableConfig,
    assumptions_for_reverse_value,
    run_reverse_dcf,
    research_ranges_from_profile,
    solve_reverse_variable,
)
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions


def assumptions(**overrides):
    values = {
        "forecast_years": 10,
        "near_term_revenue_growth": (0.20, 0.15, 0.10),
        "revenue_fade_years": 7,
        "terminal_growth": 0.03,
        "starting_operating_margin": 0.30,
        "mature_operating_margin": 0.25,
        "starting_sales_to_capital": 1.5,
        "mature_sales_to_capital": 1.2,
        "operating_tax_rate": 0.20,
        "wacc": 0.09,
    }
    values.update(overrides)
    return MultiStageDCFAssumptions(**values)


def company_inputs(ticker="TEST"):
    shares = 10_000_000_000.0
    normalized = NormalizedShareCount(
        ticker=ticker,
        shares_outstanding=shares,
        source="fixture",
        source_period=pd.Timestamp("2025-12-31"),
        scope="consolidated_common",
        method="fixture",
        components=(),
        warnings=(),
        available=True,
        reason=None,
    )
    return RealCompanyDCFInputs(
        ticker=ticker,
        starting_revenue=100_000_000_000.0,
        starting_revenue_source="ttm",
        starting_revenue_periods=(),
        net_debt=5_000_000_000.0,
        net_debt_source="fixture",
        net_debt_period=None,
        shares_outstanding=shares,
        normalized_share_count=normalized,
        historical_sales_to_capital_3y=1.4,
        current_accounting_roic=0.30,
    )


def dcf_value(inputs, model):
    return run_multistage_dcf(inputs, model).per_share_value.intrinsic_value_per_share


@pytest.mark.parametrize(
    ("variable", "known_value"),
    [
        (GROWTH_UPLIFT, 0.075),
        (MATURE_MARGIN, 0.40),
        (MATURE_SALES_TO_CAPITAL, 1.80),
        (WACC, 0.075),
    ],
)
def test_exact_synthetic_market_target_recovers_one_changed_lever(variable, known_value):
    inputs = company_inputs()
    base = assumptions()
    target_model = assumptions_for_reverse_value(base, variable, known_value)
    target = dcf_value(inputs, target_model)
    analysis = run_reverse_dcf(inputs, base, target)
    result = analysis.result_for(variable)

    assert result.status == SOLVED
    assert result.implied_value == pytest.approx(known_value, abs=2e-5)
    assert result.implied_dcf_value == pytest.approx(target, abs=0.011)
    assert result.enterprise_value is not None
    assert result.equity_value is not None


def test_growth_uplift_changes_all_three_years_by_the_same_delta():
    base = assumptions()
    changed = assumptions_for_reverse_value(base, GROWTH_UPLIFT, 0.06)

    assert changed.near_term_revenue_growth == pytest.approx((0.26, 0.21, 0.16))
    assert base.near_term_revenue_growth == (0.20, 0.15, 0.10)


def test_monotonic_directions_match_economic_expectations():
    inputs = company_inputs()
    base = assumptions()
    value = lambda model: dcf_value(inputs, model)

    assert value(replace(base, mature_operating_margin=0.35)) > value(base)
    assert value(replace(base, mature_sales_to_capital=2.0)) > value(base)
    assert value(replace(base, wacc=0.11)) < value(base)


def test_market_price_outside_reasonable_bound_is_not_forced_to_a_solution():
    result = run_reverse_dcf(company_inputs(), assumptions(), 1_000_000.0)

    assert all(item.status == OUTSIDE_REASONABLE_RANGE for item in result.results)
    assert all(item.implied_value is None for item in result.results)


def test_non_monotonic_multiple_roots_are_reported_as_ambiguous():
    base = assumptions()
    config = ReverseVariableConfig(GROWTH_UPLIFT, -0.30, 0.50, "increasing")

    result = solve_reverse_variable(
        base,
        1.0,
        config,
        lambda model: 1.0 + (
            model.near_term_revenue_growth[0]
            - base.near_term_revenue_growth[0]
        ) ** 2 - 0.01,
    )

    assert result.status == AMBIGUOUS
    assert result.implied_value is None
    assert result.monotonic is False
    assert result.root_interval_count >= 2


def test_missing_market_price_is_explicitly_unavailable():
    result = run_reverse_dcf(company_inputs(), assumptions(), None)

    assert all(item.status == MARKET_PRICE_UNAVAILABLE for item in result.results)
    assert result.market_price is None


def test_results_and_base_assumptions_are_immutable():
    base = assumptions()
    original = base
    result = run_reverse_dcf(company_inputs(), base, dcf_value(company_inputs(), base))

    with pytest.raises(FrozenInstanceError):
        result.base_source = "changed"
    with pytest.raises(FrozenInstanceError):
        result.results[0].status = SOLVED
    assert base == original


def test_candidate_reviewed_snapshot_and_applied_base_are_not_mutated():
    from Stock.amazon_research import build_amazon_research_profile
    from Stock.company_profile_application import ReviewedProfileApplication
    from Stock.company_profile_review import ReviewedCompanyProfileSnapshot
    from Stock.tests.test_amazon_research import amazon_history

    base = assumptions()
    profile = build_amazon_research_profile(base, amazon_history()).lookup.profile
    snapshot = ReviewedCompanyProfileSnapshot(
        profile=profile,
        reviewed_at="2026-08-23T00:00:00Z",
        group_reviews=(),
        overall_review_note="fixture",
        assumption_signature="fixture",
        evidence_signature="fixture",
    )
    application = ReviewedProfileApplication(
        source="reviewed_company_profile",
        issuer="AMZN",
        reviewed_at=snapshot.reviewed_at,
        applied_at="2026-08-23T00:01:00Z",
        snapshot_fingerprint="fixture",
        assumptions=base,
    )
    before = deepcopy((profile, snapshot, application))

    run_reverse_dcf(company_inputs("AMZN"), application.assumptions, 25.0)

    assert (profile, snapshot, application) == before


@pytest.mark.parametrize(
    "ticker",
    ("NVDA", "GOOGL", "META", "MSFT", "AMZN", "MU", "AAPL", "AVGO", "AMD"),
)
def test_all_target_tickers_use_the_same_generic_reverse_workflow(ticker):
    inputs = company_inputs(ticker)
    base = assumptions()
    target = dcf_value(inputs, base)

    result = run_reverse_dcf(inputs, base, target, ticker=ticker)

    assert result.ticker == ticker
    assert len(result.results) == 4
    assert all(item.status == SOLVED for item in result.results)


def test_explicit_profile_ranges_are_read_without_inference():
    from Stock.amazon_research import build_amazon_research_profile
    from Stock.tests.test_amazon_research import amazon_history

    profile = build_amazon_research_profile(
        assumptions(), amazon_history()
    ).lookup.profile
    ranges = research_ranges_from_profile(profile)

    assert ranges[MATURE_MARGIN].lower == pytest.approx(0.1233)
    assert ranges[MATURE_MARGIN].upper == pytest.approx(0.2389)
    assert ranges[MATURE_SALES_TO_CAPITAL].lower == pytest.approx(0.588)
    assert ranges[MATURE_SALES_TO_CAPITAL].upper == pytest.approx(1.095)
