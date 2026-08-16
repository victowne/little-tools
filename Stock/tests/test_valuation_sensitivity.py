from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from Stock.multistage_integration import (
    RealCompanyDCFInputs,
    run_multistage_dcf,
)
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions
from Stock.valuation_sensitivity import build_wacc_terminal_growth_sensitivity
from Stock.valuation_support import FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED


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


def company_inputs(*, net_debt=5_000_000_000.0, shares=10_000_000_000.0):
    normalized = NormalizedShareCount(
        ticker="TEST",
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
        ticker="TEST",
        starting_revenue=100_000_000_000.0,
        starting_revenue_source="ttm",
        starting_revenue_periods=(),
        net_debt=net_debt,
        net_debt_source="fixture",
        net_debt_period=None,
        shares_outstanding=shares,
        normalized_share_count=normalized,
        historical_sales_to_capital_3y=1.4,
        current_accounting_roic=0.30,
    )


def test_default_grid_is_five_by_five_and_uses_relative_axes():
    result = build_wacc_terminal_growth_sensitivity(
        company_inputs(), assumptions()
    )

    assert len(result.points) == 25
    assert result.wacc_values == pytest.approx((0.08, 0.085, 0.09, 0.095, 0.10))
    assert result.terminal_growth_values == pytest.approx(
        (0.02, 0.025, 0.03, 0.035, 0.04)
    )
    assert result.valid_point_count == 25
    assert result.invalid_point_count == 0


def test_base_point_is_included_exactly_once_and_identified():
    model = assumptions(wacc=0.0913, terminal_growth=0.0347)
    result = build_wacc_terminal_growth_sensitivity(company_inputs(), model)

    assert result.base_case_point.wacc == model.wacc
    assert result.base_case_point.terminal_growth == model.terminal_growth
    assert sum(point.is_base_case for point in result.points) == 1


def test_custom_offsets_still_insert_the_exact_base_point():
    model = assumptions()
    result = build_wacc_terminal_growth_sensitivity(
        company_inputs(),
        model,
        wacc_offsets=(-0.005, 0.005),
        terminal_growth_offsets=(-0.005, 0.005),
    )

    assert result.wacc_values == pytest.approx((0.085, 0.09, 0.095))
    assert result.terminal_growth_values == pytest.approx((0.025, 0.03, 0.035))
    assert result.base_case_point.is_base_case


def test_invalid_wacc_not_greater_than_growth_is_preserved_as_unavailable():
    result = build_wacc_terminal_growth_sensitivity(
        company_inputs(),
        assumptions(wacc=0.025, terminal_growth=0.02),
    )
    point = result.point_at(0.015, 0.03)

    assert point is not None
    assert not point.valid
    assert point.intrinsic_value_per_share is None
    assert point.reason == "wacc_not_greater_than_terminal_growth"
    assert result.invalid_point_count > 0


def test_base_cell_reconciles_to_the_standard_full_dcf():
    inputs = company_inputs()
    model = assumptions()
    standard = run_multistage_dcf(inputs, model)
    sensitivity = build_wacc_terminal_growth_sensitivity(inputs, model)
    base = sensitivity.base_case_point

    assert base.intrinsic_value_per_share == pytest.approx(
        standard.per_share_value.intrinsic_value_per_share, rel=1e-12
    )
    assert base.enterprise_value == pytest.approx(
        standard.enterprise_value.enterprise_value, rel=1e-12
    )
    assert base.equity_value == pytest.approx(
        standard.equity_value.equity_value, rel=1e-12
    )
    assert base.terminal_value_share == pytest.approx(
        standard.enterprise_value.terminal_value_share, rel=1e-12
    )


def test_higher_wacc_reduces_value_for_positive_cash_flow_fixture():
    result = build_wacc_terminal_growth_sensitivity(company_inputs(), assumptions())
    values = [
        result.point_at(wacc, 0.03).intrinsic_value_per_share
        for wacc in (0.08, 0.09, 0.10)
    ]

    assert values[0] > values[1] > values[2]


def test_higher_terminal_growth_increases_value_for_high_roic_fixture():
    result = build_wacc_terminal_growth_sensitivity(company_inputs(), assumptions())
    values = [
        result.point_at(0.09, growth).intrinsic_value_per_share
        for growth in (0.025, 0.03, 0.035)
    ]

    assert values[0] < values[1] < values[2]


def test_terminal_growth_changes_reinvestment_rate_and_terminal_fcff():
    result = build_wacc_terminal_growth_sensitivity(company_inputs(), assumptions())
    low = result.point_at(0.09, 0.025)
    high = result.point_at(0.09, 0.035)
    terminal_roic = assumptions().derived_terminal_roic

    assert low.terminal_reinvestment_rate == pytest.approx(0.025 / terminal_roic)
    assert high.terminal_reinvestment_rate == pytest.approx(0.035 / terminal_roic)
    assert high.terminal_reinvestment_rate > low.terminal_reinvestment_rate
    assert high.terminal_fcff != pytest.approx(low.terminal_fcff)


def test_each_cell_reuses_the_identical_company_inputs(monkeypatch):
    import Stock.valuation_sensitivity as module

    inputs = company_inputs()
    seen = []
    original = module.run_multistage_dcf

    def recording_run(received_inputs, received_assumptions):
        seen.append(received_inputs)
        return original(received_inputs, received_assumptions)

    monkeypatch.setattr(module, "run_multistage_dcf", recording_run)
    build_wacc_terminal_growth_sensitivity(inputs, assumptions())

    assert len(seen) == 25
    assert all(received is inputs for received in seen)
    assert all(
        received.normalized_share_count is inputs.normalized_share_count
        for received in seen
    )


def test_each_cell_preserves_all_non_sensitivity_assumptions(monkeypatch):
    import Stock.valuation_sensitivity as module

    model = assumptions()
    seen = []
    original = module.run_multistage_dcf

    def recording_run(received_inputs, received_assumptions):
        seen.append(received_assumptions)
        return original(received_inputs, received_assumptions)

    monkeypatch.setattr(module, "run_multistage_dcf", recording_run)
    build_wacc_terminal_growth_sensitivity(company_inputs(), model)

    unchanged_fields = (
        "forecast_years",
        "near_term_revenue_growth",
        "revenue_fade_years",
        "starting_operating_margin",
        "mature_operating_margin",
        "starting_sales_to_capital",
        "mature_sales_to_capital",
        "operating_tax_rate",
    )
    assert len(seen) == 25
    for point_model in seen:
        for field in unchanged_fields:
            assert getattr(point_model, field) == getattr(model, field)


def test_net_cash_equity_bridge_is_applied_at_every_valid_point():
    inputs = company_inputs(net_debt=-8_000_000_000.0)
    result = build_wacc_terminal_growth_sensitivity(inputs, assumptions())

    for point in result.points:
        assert point.valid
        assert point.equity_value == pytest.approx(
            point.enterprise_value + 8_000_000_000.0
        )
        assert "net_cash_position" in point.warnings


def test_local_impact_reports_absolute_and_percentage_change():
    result = build_wacc_terminal_growth_sensitivity(company_inputs(), assumptions())
    impact = result.impact_at(0.095, 0.03)

    assert impact.point is result.point_at(0.095, 0.03)
    assert impact.absolute_change == pytest.approx(
        impact.point.intrinsic_value_per_share
        - result.base_case_point.intrinsic_value_per_share
    )
    assert impact.percentage_change == pytest.approx(
        impact.absolute_change
        / result.base_case_point.intrinsic_value_per_share
    )


def test_sensitivity_results_are_immutable():
    result = build_wacc_terminal_growth_sensitivity(company_inputs(), assumptions())

    with pytest.raises(FrozenInstanceError):
        result.base_wacc = 0.01
    with pytest.raises(FrozenInstanceError):
        result.points[0].valid = False


def test_unsupported_per_security_bridge_makes_every_grid_value_unavailable():
    unsupported = replace(
        company_inputs(),
        per_security_valuation_supported=False,
        per_security_valuation_unsupported_reason=(
            FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED
        ),
        statement_currency="TWD",
        security_currency="USD",
    )

    result = build_wacc_terminal_growth_sensitivity(
        unsupported, assumptions()
    )

    assert result.valid_point_count == 0
    assert result.invalid_point_count == 25
    assert result.min_value_per_share is None
    assert result.max_value_per_share is None
    assert all(point.intrinsic_value_per_share is None for point in result.points)
    assert all(
        point.reason == FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED
        for point in result.points
    )
    assert all(point.enterprise_value is not None for point in result.points)
