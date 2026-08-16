from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from Stock.fundamentals import (
    REVENUE,
    ROIC,
    FundamentalHistory,
    HistoricalDCFAnchors,
    SalesToCapitalResult,
    TTMResult,
)
from Stock.multistage_integration import (
    RealCompanyDCFInputs,
    extract_real_company_dcf_inputs,
    run_multistage_dcf,
    run_real_company_multistage_dcf,
)
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions
from Stock.valuation_support import FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED


PERIODS = tuple(
    pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"])
)


def assumptions(**overrides):
    values = {
        "forecast_years": 3,
        "near_term_revenue_growth": (0.10,),
        "revenue_fade_years": 2,
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


def history(*, ttm_value=400.0, ttm_available=True, annual_latest=350.0):
    annual = pd.DataFrame(
        {
            REVENUE: [300.0, annual_latest],
            ROIC: [0.20, 0.25],
        },
        index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
    )
    anchor = SalesToCapitalResult(
        value=1.4,
        available=True,
        start_period=pd.Timestamp("2022-12-31"),
        end_period=pd.Timestamp("2025-12-31"),
        years=3,
        reason=None,
    )
    return FundamentalHistory(
        annual=annual,
        ttm={
            REVENUE: TTMResult(
                ttm_value if ttm_available else None,
                ttm_available,
                PERIODS if ttm_available else (),
                None if ttm_available else "fewer_than_four_quarters",
            )
        },
        annual_reasons=pd.DataFrame(index=annual.index),
        dcf_anchors=HistoricalDCFAnchors(
            normalized_sales_to_capital={3: anchor}
        ),
    )


def consolidated_share_statement(value=10.0):
    return pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [value]},
        index=["OrdinarySharesNumber"],
    )


def normalized_shares(value=10.0):
    return NormalizedShareCount(
        ticker="TEST",
        shares_outstanding=value,
        source="fixture",
        source_period=pd.Timestamp("2025-12-31"),
        scope="consolidated_common",
        method="fixture",
        components=(),
        warnings=(),
        available=True,
        reason=None,
    )


def test_valid_ttm_revenue_is_selected_with_periods(snapshot_factory):
    snapshot = snapshot_factory(
        net_debt_source="annual_balance_debt_minus_cash",
        net_debt_period=pd.Timestamp("2025-12-31"),
    )

    inputs = extract_real_company_dcf_inputs(snapshot, history())

    assert inputs.starting_revenue == 400.0
    assert inputs.starting_revenue_source == "ttm"
    assert inputs.starting_revenue_periods == PERIODS
    assert inputs.net_debt_source == "annual_balance_debt_minus_cash"
    assert inputs.net_debt_period == pd.Timestamp("2025-12-31")
    assert inputs.historical_sales_to_capital_3y == 1.4
    assert inputs.current_accounting_roic == 0.25


def test_unavailable_ttm_uses_explicit_latest_annual_fallback(snapshot_factory):
    inputs = extract_real_company_dcf_inputs(
        snapshot_factory(), history(ttm_available=False)
    )

    assert inputs.starting_revenue == 350.0
    assert inputs.starting_revenue_source == "annual_fallback"
    assert inputs.starting_revenue_periods == (pd.Timestamp("2025-12-31"),)


def test_missing_latest_annual_revenue_is_not_replaced_by_older_year(
    snapshot_factory,
):
    with pytest.raises(ValueError, match="starting_revenue_unavailable"):
        extract_real_company_dcf_inputs(
            snapshot_factory(),
            history(ttm_available=False, annual_latest=float("nan")),
        )


def test_missing_net_debt_is_rejected(snapshot_factory):
    with pytest.raises(ValueError, match="net_debt_unavailable"):
        extract_real_company_dcf_inputs(
            snapshot_factory(net_debt=None), history()
        )


@pytest.mark.parametrize("shares", [None, 0.0, -1.0, float("nan")])
def test_missing_consolidated_shares_preserves_equity_but_not_per_share(
    snapshot_factory, shares
):
    snapshot = snapshot_factory(
        shares_outstanding=shares,
        ticker_shares_outstanding=shares,
    )

    result = run_real_company_multistage_dcf(snapshot, history(), assumptions())

    assert result.equity_value.equity_value is not None
    assert result.per_share_value is None
    assert result.per_share_unavailable_reason == (
        "consolidated_share_count_unavailable"
    )


def test_net_cash_flows_through_existing_equity_bridge(snapshot_factory):
    snapshot = snapshot_factory(net_debt=-50.0, shares_outstanding=10.0)

    result = run_real_company_multistage_dcf(snapshot, history(), assumptions())

    assert result.equity_value.equity_value == pytest.approx(
        result.enterprise_value.enterprise_value + 50.0
    )
    assert "net_cash_position" in result.warnings


def test_orchestration_results_are_internally_consistent(snapshot_factory):
    snapshot = snapshot_factory(
        net_debt=25.0,
        shares_outstanding=10.0,
        quarterly_balance=consolidated_share_statement(),
    )

    result = run_real_company_multistage_dcf(snapshot, history(), assumptions())

    explicit_pv = sum(
        year.present_value_fcff for year in result.discounted_forecast.years
    )
    assert result.enterprise_value.explicit_forecast_pv == pytest.approx(explicit_pv)
    assert result.enterprise_value.enterprise_value == pytest.approx(
        explicit_pv + result.terminal_value.present_value_terminal_value
    )
    assert result.equity_value.equity_value == pytest.approx(
        result.enterprise_value.enterprise_value - 25.0
    )
    assert result.per_share_value.intrinsic_value_per_share == pytest.approx(
        result.equity_value.equity_value / 10.0
    )
    assert result.terminal_value.final_forecast_revenue == pytest.approx(
        result.operating_forecast.ending_revenue
    )


def test_foreign_listing_preserves_issuer_values_but_fails_per_security_closed(
    snapshot_factory,
):
    snapshot = snapshot_factory(
        ticker="TSM",
        financial_currency="TWD",
        price_currency="USD",
        shares_outstanding=25.0,
        quarterly_balance=consolidated_share_statement(25.0),
    )

    result = run_real_company_multistage_dcf(snapshot, history(), assumptions())

    assert result.enterprise_value.enterprise_value != 0
    assert result.equity_value.equity_value != 0
    assert result.per_share_value is None
    assert not result.per_security_valuation_supported
    assert result.per_share_unavailable_reason == (
        FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED
    )
    assert result.per_share_unavailable_reason != 0


def test_supported_domestic_listing_still_returns_numeric_per_share(
    snapshot_factory,
):
    snapshot = snapshot_factory(
        ticker="NVDA",
        financial_currency="USD",
        price_currency="USD",
        shares_outstanding=10.0,
        quarterly_balance=consolidated_share_statement(10.0),
    )

    result = run_real_company_multistage_dcf(snapshot, history(), assumptions())

    assert result.per_security_valuation_supported
    assert result.per_share_unavailable_reason is None
    assert result.per_share_value.intrinsic_value_per_share == pytest.approx(
        result.equity_value.equity_value / 10.0
    )


def test_explicit_inputs_run_has_no_market_price_or_automatic_anchor_binding():
    inputs = RealCompanyDCFInputs(
        ticker="TEST",
        starting_revenue=400.0,
        starting_revenue_source="ttm",
        starting_revenue_periods=PERIODS,
        net_debt=25.0,
        net_debt_source="fixture",
        net_debt_period=pd.Timestamp("2025-12-31"),
        shares_outstanding=10.0,
        normalized_share_count=normalized_shares(),
        historical_sales_to_capital_3y=99.0,
        current_accounting_roic=9.0,
    )
    model = assumptions(starting_sales_to_capital=1.5)

    result = run_multistage_dcf(inputs, model)

    assert result.forecast_path.years[0].sales_to_capital == 1.5
    assert not hasattr(result, "market_price")
    assert result.assumptions is model
    with pytest.raises(FrozenInstanceError):
        result.inputs.starting_revenue = 0.0
