from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from Stock.assumption_diagnostics import build_assumption_diagnostics
from Stock.fundamentals import (
    FCF_MARGIN,
    OPERATING_MARGIN,
    REVENUE,
    ROIC,
    SALES_TO_CAPITAL,
    FundamentalHistory,
    HistoricalDCFAnchors,
    RevenueCAGRResult,
    TTMResult,
)
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions


PERIODS = tuple(pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]))


def assumptions(**overrides):
    values = {
        "forecast_years": 10,
        "near_term_revenue_growth": (0.30, 0.25, 0.20),
        "revenue_fade_years": 7,
        "terminal_growth": 0.035,
        "starting_operating_margin": 0.60,
        "mature_operating_margin": 0.40,
        "starting_sales_to_capital": 1.5,
        "mature_sales_to_capital": 1.2,
        "operating_tax_rate": 0.16,
        "wacc": 0.09,
    }
    values.update(overrides)
    return MultiStageDCFAssumptions(**values)


def history(*, cagr=0.20, ttm_margin=0.58, fcf_margin=0.30, annual_margin=0.55, latest_stc=1.4):
    annual = pd.DataFrame(
        {
            REVENUE: [80.0, 100.0],
            OPERATING_MARGIN: [0.50, annual_margin],
            SALES_TO_CAPITAL: [1.3, latest_stc],
            ROIC: [0.25, 0.30],
        },
        index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
    )
    cagr_result = RevenueCAGRResult(
        cagr,
        cagr is not None,
        pd.Timestamp("2022-12-31"),
        pd.Timestamp("2025-12-31"),
        3,
        None if cagr is not None else "insufficient_history",
    )
    return FundamentalHistory(
        annual=annual,
        ttm={
            REVENUE: TTMResult(100.0, True, PERIODS, None),
            OPERATING_MARGIN: TTMResult(
                ttm_margin, ttm_margin is not None, PERIODS if ttm_margin is not None else (),
                None if ttm_margin is not None else "unavailable",
            ),
            FCF_MARGIN: TTMResult(fcf_margin, True, PERIODS, None),
        },
        annual_reasons=pd.DataFrame(index=annual.index),
        dcf_anchors=HistoricalDCFAnchors(revenue_cagr={3: cagr_result}),
    )


def inputs(*, historical_stc=1.4, current_roic=0.30):
    normalized = NormalizedShareCount(
        "TEST", 10.0, "fixture", pd.Timestamp("2025-12-31"),
        "consolidated_common", "fixture", (), (), True, None,
    )
    return RealCompanyDCFInputs(
        ticker="TEST",
        starting_revenue=100.0,
        starting_revenue_source="ttm",
        starting_revenue_periods=PERIODS,
        net_debt=5.0,
        net_debt_source="fixture",
        net_debt_period=pd.Timestamp("2025-12-31"),
        shares_outstanding=10.0,
        normalized_share_count=normalized,
        historical_sales_to_capital_3y=historical_stc,
        current_accounting_roic=current_roic,
    )


def diagnostic(model=None, evidence=None, company_inputs=None):
    model = model or assumptions()
    evidence = evidence or history()
    company_inputs = company_inputs or inputs()
    run = run_multistage_dcf(company_inputs, model)
    result = build_assumption_diagnostics(
        evidence, company_inputs, model, run.forecast_path,
        run.operating_forecast, run.terminal_value, run.enterprise_value,
    )
    return result, run


def test_revenue_diagnostics_expose_history_path_and_scale():
    result, run = diagnostic()

    assert result.revenue.historical_cagr_3y == 0.20
    assert (result.revenue.year_1_growth, result.revenue.year_2_growth, result.revenue.year_3_growth) == (0.30, 0.25, 0.20)
    assert result.revenue.year_5_revenue == pytest.approx(run.operating_forecast.years[4].revenue)
    assert result.revenue.final_forecast_revenue == pytest.approx(run.operating_forecast.ending_revenue)
    assert result.revenue.final_to_starting_revenue_multiple == pytest.approx(run.operating_forecast.ending_revenue / 100.0)
    assert result.revenue.year_1_growth_minus_historical_cagr_3y == pytest.approx(0.10)
    assert result.revenue.final_explicit_growth == pytest.approx(0.035)


def test_unavailable_historical_cagr_remains_missing():
    result, _ = diagnostic(evidence=history(cagr=None))
    assert result.revenue.historical_cagr_3y is None
    assert result.revenue.year_1_growth_minus_historical_cagr_3y is None


@pytest.mark.parametrize(
    ("start", "mature", "direction"),
    [(0.60, 0.40, "contraction"), (0.30, 0.40, "expansion"), (0.40, 0.40, "unchanged")],
)
def test_margin_direction_is_descriptive(start, mature, direction):
    result, _ = diagnostic(model=assumptions(starting_operating_margin=start, mature_operating_margin=mature))
    assert result.operating_margin.direction == direction
    assert result.operating_margin.total_margin_change == pytest.approx(mature - start)


def test_missing_ttm_margin_preserves_unavailable_distance():
    result, _ = diagnostic(evidence=history(ttm_margin=None))
    assert result.operating_margin.latest_ttm_margin is None
    assert result.operating_margin.start_minus_current_ttm is None


@pytest.mark.parametrize(
    ("forecast_stc", "historical_stc", "flagged"),
    [(1.45, 1.40, False), (2.00, 1.40, True), (0.80, 1.40, True)],
)
def test_sales_to_capital_distance_flag_is_symmetric(forecast_stc, historical_stc, flagged):
    result, _ = diagnostic(
        model=assumptions(starting_sales_to_capital=forecast_stc),
        company_inputs=inputs(historical_stc=historical_stc),
    )
    assert ("forecast_start_sales_to_capital_far_from_historical" in result.flags) is flagged
    assert result.sales_to_capital.start_minus_historical_3y == pytest.approx(forecast_stc - historical_stc)


def test_missing_historical_sales_to_capital_remains_unavailable():
    result, _ = diagnostic(company_inputs=inputs(historical_stc=None))
    assert result.sales_to_capital.historical_normalized_3y is None
    assert result.sales_to_capital.start_to_historical_3y_ratio is None


@pytest.mark.parametrize(
    ("current_roic", "mature_margin", "mature_stc", "above"),
    [(0.50, 0.30, 1.0, False), (0.10, 0.40, 1.2, True), (None, 0.40, 1.2, False)],
)
def test_terminal_roic_comparison(current_roic, mature_margin, mature_stc, above):
    result, _ = diagnostic(
        model=assumptions(mature_operating_margin=mature_margin, mature_sales_to_capital=mature_stc),
        company_inputs=inputs(current_roic=current_roic),
    )
    assert ("terminal_roic_above_current_accounting_roic" in result.flags) is above
    if current_roic is None:
        assert result.roic.terminal_minus_current_accounting_roic is None


def test_implied_operating_roic_uses_after_tax_margin_times_sales_to_capital():
    result, run = diagnostic()
    year_5 = run.operating_forecast.years[4]
    expected = year_5.operating_margin * (1 - 0.16) * year_5.sales_to_capital
    assert result.roic.year_5_implied_operating_roic == pytest.approx(expected)
    assert result.roic.terminal_derived_roic == pytest.approx(0.40 * 0.84 * 1.2)


def test_positive_growth_cash_flow_ratios_are_exposed():
    result, run = diagnostic()
    year_1 = run.operating_forecast.years[0]
    assert result.cash_flow_economics.year_1.reinvestment > 0
    assert result.cash_flow_economics.year_1.fcff_to_nopat == pytest.approx(year_1.fcff / year_1.nopat)
    assert result.cash_flow_economics.year_1.fcff_margin == pytest.approx(year_1.fcff / year_1.revenue)
    assert result.cash_flow_economics.historical_fundamental_ttm_fcf_margin == 0.30


def test_zero_growth_has_zero_reinvestment():
    model = assumptions(near_term_revenue_growth=(0.0, 0.0, 0.0), terminal_growth=0.0)
    result, _ = diagnostic(model=model)
    assert result.cash_flow_economics.year_1.reinvestment == 0.0
    assert result.cash_flow_economics.terminal_reinvestment_rate == 0.0


def test_negative_growth_exposes_capital_release():
    model = assumptions(near_term_revenue_growth=(-0.10, -0.05, 0.0), terminal_growth=0.0)
    result, _ = diagnostic(model=model)
    assert result.cash_flow_economics.year_1.reinvestment < 0


def test_negative_fcff_is_preserved_without_subjective_flag():
    model = assumptions(
        near_term_revenue_growth=(0.80, 0.50, 0.30),
        starting_operating_margin=0.05,
        mature_operating_margin=0.20,
        starting_sales_to_capital=0.5,
        mature_sales_to_capital=1.0,
    )
    result, _ = diagnostic(model=model)
    assert result.cash_flow_economics.year_1.fcff < 0


def test_terminal_dependency_matches_existing_enterprise_result():
    result, run = diagnostic()
    assert result.terminal_dependency.explicit_forecast_pv == run.enterprise_value.explicit_forecast_pv
    assert result.terminal_dependency.terminal_value_pv == run.enterprise_value.terminal_value_pv
    assert result.terminal_dependency.terminal_value_share == run.enterprise_value.terminal_value_share


def test_terminal_dominance_warning_and_flag_are_preserved():
    model = assumptions()
    evidence = history()
    company_inputs = inputs()
    run = run_multistage_dcf(company_inputs, model)
    enterprise = replace(
        run.enterprise_value,
        terminal_value_share=0.90,
        warnings=run.enterprise_value.warnings + ("terminal_value_dominates_enterprise_value",),
    )
    result = build_assumption_diagnostics(
        evidence, company_inputs, model, run.forecast_path,
        run.operating_forecast, run.terminal_value, enterprise,
    )
    assert "terminal_value_dominates_enterprise_value" in result.warnings
    assert "terminal_value_dominates_enterprise_value" in result.flags


def test_margin_distance_threshold_is_transparent_and_objective():
    result, _ = diagnostic(
        model=assumptions(starting_operating_margin=0.64),
        evidence=history(ttm_margin=0.58),
    )
    assert "forecast_start_margin_far_from_current" in result.flags


def test_diagnostic_result_is_immutable():
    result, _ = diagnostic()
    with pytest.raises(FrozenInstanceError):
        result.flags = ()
