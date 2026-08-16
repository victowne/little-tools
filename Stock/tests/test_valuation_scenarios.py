from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from Stock.fundamentals import FundamentalHistory, HistoricalDCFAnchors
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions
from Stock.valuation_scenarios import (
    create_scenario_from_base,
    run_multi_scenario_dcf,
)
from Stock.valuation_support import FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED


def assumptions(**overrides):
    values = {
        "forecast_years": 10,
        "near_term_revenue_growth": (0.15, 0.12, 0.10),
        "revenue_fade_years": 5,
        "terminal_growth": 0.03,
        "starting_operating_margin": 0.28,
        "mature_operating_margin": 0.25,
        "starting_sales_to_capital": 1.4,
        "mature_sales_to_capital": 1.0,
        "operating_tax_rate": 0.20,
        "wacc": 0.09,
    }
    values.update(overrides)
    return MultiStageDCFAssumptions(**values)


def history():
    return FundamentalHistory(
        annual=pd.DataFrame(), ttm={}, annual_reasons=pd.DataFrame(),
        dcf_anchors=HistoricalDCFAnchors(),
    )


def inputs(*, net_debt=10.0, shares=10.0, shares_available=True):
    normalized = NormalizedShareCount(
        ticker="TEST",
        shares_outstanding=shares if shares_available else None,
        source="fixture" if shares_available else None,
        source_period=None,
        scope="consolidated_common" if shares_available else None,
        method="fixture" if shares_available else None,
        components=(), warnings=(), available=shares_available,
        reason=None if shares_available else "consolidated_share_count_unavailable",
    )
    return RealCompanyDCFInputs(
        ticker="TEST", starting_revenue=100.0,
        starting_revenue_source="ttm", starting_revenue_periods=(),
        net_debt=net_debt, net_debt_source="fixture", net_debt_period=None,
        shares_outstanding=shares if shares_available else None,
        normalized_share_count=normalized,
        historical_sales_to_capital_3y=None,
        current_accounting_roic=None,
    )


def cases(base=None):
    base = base or assumptions()
    return (
        create_scenario_from_base(
            "bear", base, rationale="fixture bear",
            near_term_revenue_growth=(0.08, 0.06, 0.04),
            revenue_fade_years=4, terminal_growth=0.02,
            mature_operating_margin=0.20, mature_sales_to_capital=0.8,
            research_wacc=0.105,
        ),
        create_scenario_from_base("base", base, rationale="fixture base"),
        create_scenario_from_base(
            "bull", base, rationale="fixture bull",
            near_term_revenue_growth=(0.22, 0.18, 0.14),
            revenue_fade_years=7, terminal_growth=0.035,
            mature_operating_margin=0.30, mature_sales_to_capital=1.3,
            research_wacc=0.08,
        ),
    )


def run(*, company_inputs=None, scenario_cases=None):
    bear, base, bull = scenario_cases or cases()
    return run_multi_scenario_dcf(
        inputs=company_inputs or inputs(), fundamentals=history(),
        bear=bear, base=base, bull=bull,
    )


def test_three_complete_valid_scenarios_run_full_dcf_chain():
    result = run()
    assert all(item.available for item in result.scenarios)
    assert all(item.dcf_result is not None for item in result.scenarios)
    assert all(item.metrics.intrinsic_value_per_share is not None for item in result.scenarios)
    assert result.bear.assumptions.near_term_revenue_growth == (0.08, 0.06, 0.04)
    assert result.base.assumptions == assumptions()
    assert result.bull.assumptions.wacc == pytest.approx(0.08)


def test_base_exactly_reconciles_with_standalone_dcf():
    model = assumptions()
    result = run(scenario_cases=cases(model)).base
    standalone = run_multistage_dcf(inputs(), model)
    assert result.dcf_result.per_share_value.intrinsic_value_per_share == pytest.approx(
        standalone.per_share_value.intrinsic_value_per_share, abs=1e-12
    )
    assert result.metrics.enterprise_value == pytest.approx(
        standalone.enterprise_value.enterprise_value, abs=1e-12
    )
    assert result.metrics.equity_value == pytest.approx(
        standalone.equity_value.equity_value, abs=1e-12
    )
    assert result.metrics.terminal_value_share == pytest.approx(
        standalone.enterprise_value.terminal_value_share, abs=1e-12
    )


def test_invalid_bear_does_not_prevent_base_or_bull():
    bear, base, bull = cases()
    invalid_bear = create_scenario_from_base(
        "bear", assumptions(), wacc=0.02, terminal_growth=0.03
    )
    result = run(scenario_cases=(invalid_bear, base, bull))
    assert not result.bear.available
    assert "wacc must be greater" in result.bear.reason
    assert result.base.available and result.bull.available
    assert "bear_scenario_unavailable" in result.warnings


def test_invalid_bull_does_not_prevent_base_or_bear():
    bear, base, _ = cases()
    invalid_bull = create_scenario_from_base(
        "bull", assumptions(), mature_sales_to_capital=0.0
    )
    result = run(scenario_cases=(bear, base, invalid_bull))
    assert result.bear.available and result.base.available
    assert not result.bull.available
    assert "mature_sales_to_capital must be positive" in result.bull.reason


def test_scenario_assumptions_and_results_are_immutable():
    result = run()
    with pytest.raises(FrozenInstanceError):
        result.base.assumptions.wacc = 0.0
    with pytest.raises(FrozenInstanceError):
        result.base.metrics.research_wacc = 0.0
    with pytest.raises(FrozenInstanceError):
        result.warnings = ()


def test_base_assumptions_are_not_mutated_by_override_helper():
    base = assumptions()
    bear = create_scenario_from_base("bear", base, mature_operating_margin=0.10)
    assert base.mature_operating_margin == pytest.approx(0.25)
    assert bear.assumptions.mature_operating_margin == pytest.approx(0.10)


def test_unexpected_value_order_is_warning_not_error():
    base = assumptions()
    high_bear = create_scenario_from_base(
        "bear", base, near_term_revenue_growth=(0.30, 0.25, 0.20),
        mature_operating_margin=0.35, mature_sales_to_capital=1.5, wacc=0.07,
    )
    low_bull = create_scenario_from_base(
        "bull", base, near_term_revenue_growth=(0.02, 0.01, 0.0),
        mature_operating_margin=0.15, mature_sales_to_capital=0.7, wacc=0.12,
    )
    result = run(scenario_cases=(high_bear, create_scenario_from_base("base", base), low_bull))
    assert all(item.available for item in result.scenarios)
    assert "scenario_value_order_unexpected" in result.warnings


def test_negative_equity_value_is_preserved():
    result = run(company_inputs=inputs(net_debt=10_000.0))
    assert result.base.metrics.equity_value < 0
    assert result.base.metrics.intrinsic_value_per_share < 0


def test_unavailable_normalized_shares_keeps_enterprise_and_equity_results():
    result = run(company_inputs=inputs(shares_available=False))
    assert result.base.available
    assert result.base.metrics.intrinsic_value_per_share is None
    assert result.base.metrics.enterprise_value > 0
    assert result.base.reason == "consolidated_share_count_unavailable"


def test_unsupported_listing_keeps_scenario_economics_but_not_per_security_values():
    unsupported = replace(
        inputs(),
        per_security_valuation_supported=False,
        per_security_valuation_unsupported_reason=(
            FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED
        ),
        statement_currency="TWD",
        security_currency="USD",
    )

    result = run(company_inputs=unsupported)

    assert all(item.available for item in result.scenarios)
    assert all(item.metrics.enterprise_value != 0 for item in result.scenarios)
    assert all(
        item.metrics.intrinsic_value_per_share is None
        for item in result.scenarios
    )
    assert all(
        item.reason == FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED
        for item in result.scenarios
    )
    assert result.base.delta_vs_base.intrinsic_value_difference is None


def test_net_cash_company_adds_cash_in_equity_bridge():
    net_cash = run(company_inputs=inputs(net_debt=-50.0)).base.metrics
    debt = run(company_inputs=inputs(net_debt=10.0)).base.metrics
    assert net_cash.enterprise_value == pytest.approx(debt.enterprise_value)
    assert net_cash.equity_value - debt.equity_value == pytest.approx(60.0)


def test_diagnostics_and_all_required_economic_metrics_are_attached():
    base = run().base
    assert base.diagnostics is not None
    assert base.metrics.year_5_revenue is not None
    assert base.metrics.year_5_operating_margin is not None
    assert base.metrics.year_5_sales_to_capital is not None
    assert base.metrics.terminal_roic == pytest.approx(base.assumptions.derived_terminal_roic)
    assert base.metrics.terminal_reinvestment_rate == pytest.approx(
        base.assumptions.terminal_reinvestment_rate
    )
    assert base.metrics.year_1_fcff_margin is not None
    assert base.metrics.year_5_fcff_margin is not None
    assert base.metrics.final_year_fcff_margin is not None
    assert base.metrics.terminal_fcff_to_nopat is not None


def test_bear_and_bull_deltas_are_numeric_differences_not_probabilities():
    result = run()
    assert result.bear.delta_vs_base.intrinsic_value_difference == pytest.approx(
        result.bear.metrics.intrinsic_value_per_share
        - result.base.metrics.intrinsic_value_per_share
    )
    assert result.bull.delta_vs_base.intrinsic_value_percentage_difference == pytest.approx(
        result.bull.metrics.intrinsic_value_per_share
        / result.base.metrics.intrinsic_value_per_share - 1
    )
    assert not hasattr(result, "probabilities")
    assert not hasattr(result, "expected_value")


def test_changing_bear_does_not_change_base_or_bull():
    original = run()
    bear, base, bull = cases()
    changed_bear = create_scenario_from_base(
        "bear", assumptions(), near_term_revenue_growth=(-0.05, 0.0, 0.02),
        mature_operating_margin=0.12, mature_sales_to_capital=0.6, wacc=0.14,
    )
    changed = run(scenario_cases=(changed_bear, base, bull))
    assert changed.base.dcf_result == original.base.dcf_result
    assert changed.bull.dcf_result == original.bull.dcf_result
    assert changed.bear.dcf_result != original.bear.dcf_result


def test_changing_bull_does_not_change_base_or_bear():
    original = run()
    bear, base, _ = cases()
    changed_bull = create_scenario_from_base(
        "bull", assumptions(), near_term_revenue_growth=(0.35, 0.30, 0.25),
        mature_operating_margin=0.40, mature_sales_to_capital=1.8, wacc=0.07,
    )
    changed = run(scenario_cases=(bear, base, changed_bull))
    assert changed.base.dcf_result == original.base.dcf_result
    assert changed.bear.dcf_result == original.bear.dcf_result
    assert changed.bull.dcf_result != original.bull.dcf_result


def test_research_wacc_alias_resolves_to_complete_assumptions():
    case = create_scenario_from_base("bear", assumptions(), research_wacc=0.11)
    assert isinstance(case.assumptions, MultiStageDCFAssumptions)
    assert case.assumptions.wacc == pytest.approx(0.11)
    assert case.unavailable_reason is None
