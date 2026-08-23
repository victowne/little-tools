from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from Stock.hybrid_reinvestment_prototype import (
    HybridReinvestmentYearInput,
    build_hybrid_shadow_dcf,
    build_intensity_based_inputs,
    calculate_hybrid_reinvestment_path,
    calculate_hybrid_reinvestment_year,
    compare_hybrid_reinvestment,
    scale_capex_path,
    scale_depreciation_path,
)
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions


def assumptions():
    return MultiStageDCFAssumptions(
        10, (0.20, 0.16, 0.12), 5, 0.03, 0.35, 0.30,
        0.60, 0.80, 0.18, 0.10,
    )


def existing_run():
    shares = NormalizedShareCount(
        "TEST", 10e9, "fixture", pd.Timestamp("2025-12-31"),
        "consolidated_common", "fixture", (), (), True, None,
    )
    inputs = RealCompanyDCFInputs(
        "TEST", 100e9, "ttm", (pd.Timestamp("2025-12-31"),),
        5e9, "fixture", pd.Timestamp("2025-12-31"), 10e9, shares,
        0.7, 0.2, True, None, "USD", "USD",
    )
    return run_multistage_dcf(inputs, assumptions())


def hybrid_inputs():
    return build_intensity_based_inputs(
        existing_run(),
        capex_to_revenue=(0.30, 0.28, 0.25, 0.22, 0.20),
        depreciation_to_revenue=(0.10, 0.12, 0.14, 0.15, 0.16),
        working_capital_to_revenue=(0.01, 0.01, 0.0, 0.0, -0.01),
        rationale="fixture lead-lag path",
    )


def test_positive_capex_and_depreciation_sign_convention_and_fcff_identity():
    item = HybridReinvestmentYearInput(
        1, 100.0, 0.30, 0.20, 25.0, 10.0, 2.0, 0.0, "High"
    )
    result = calculate_hybrid_reinvestment_year(item)

    assert result.nopat == pytest.approx(24.0)
    assert result.net_capex == pytest.approx(15.0)
    assert result.total_reinvestment == pytest.approx(17.0)
    assert result.fcff == pytest.approx(7.0)
    assert result.fcff_margin == pytest.approx(0.07)


def test_negative_yahoo_style_capex_is_rejected():
    with pytest.raises(ValueError, match="positive-outflow convention"):
        calculate_hybrid_reinvestment_year(
            HybridReinvestmentYearInput(1, 100, 0.3, 0.2, -20, 10)
        )


def test_positive_working_capital_is_outflow_and_negative_is_release():
    invested = calculate_hybrid_reinvestment_year(
        HybridReinvestmentYearInput(1, 100, 0.3, 0.2, 20, 10, 5)
    )
    released = calculate_hybrid_reinvestment_year(
        HybridReinvestmentYearInput(1, 100, 0.3, 0.2, 20, 10, -5)
    )

    assert invested.total_reinvestment == pytest.approx(15)
    assert released.total_reinvestment == pytest.approx(5)
    assert released.fcff - invested.fcff == pytest.approx(10)


def test_other_reinvestment_defaults_to_zero():
    result = calculate_hybrid_reinvestment_year(
        HybridReinvestmentYearInput(1, 100, 0.3, 0.2, 20, 10)
    )
    assert result.other_reinvestment == 0


def test_exactly_five_consecutive_years_are_required():
    with pytest.raises(ValueError, match="exactly five years"):
        calculate_hybrid_reinvestment_path(hybrid_inputs()[:4])


def test_depreciation_above_capex_is_retained_and_warned_not_hidden():
    result = calculate_hybrid_reinvestment_year(
        HybridReinvestmentYearInput(1, 100, 0.3, 0.2, 10, 15)
    )
    assert result.net_capex == -5
    assert "negative_net_capex" in result.warnings
    assert "depreciation_exceeds_capex" in result.warnings


def test_hybrid_shadow_preserves_all_non_reinvestment_forecast_values():
    existing = existing_run()
    shadow, results = build_hybrid_shadow_dcf(existing, hybrid_inputs())

    assert shadow.assumptions is existing.assumptions
    assert shadow.forecast_path is existing.forecast_path
    for old, new in zip(
        existing.operating_forecast.years, shadow.operating_forecast.years
    ):
        assert new.revenue == old.revenue
        assert new.operating_margin == old.operating_margin
        assert new.operating_tax_rate == old.operating_tax_rate
        assert new.nopat == old.nopat
    assert shadow.operating_forecast.years[0].fcff == results[0].fcff
    assert shadow.operating_forecast.years[5:] == existing.operating_forecast.years[5:]


def test_cumulative_reinvestment_and_fcff_are_explicitly_reported():
    comparison = compare_hybrid_reinvestment(existing_run(), hybrid_inputs())

    assert comparison.cumulative_sales_to_capital_reinvestment == pytest.approx(
        sum(year.reinvestment for year in comparison.existing_run.operating_forecast.years[:5])
    )
    assert comparison.cumulative_hybrid_reinvestment == pytest.approx(
        sum(year.total_reinvestment for year in comparison.hybrid_years)
    )
    assert comparison.cumulative_hybrid_fcff == pytest.approx(
        sum(year.fcff for year in comparison.hybrid_years)
    )


def test_transition_back_to_sales_to_capital_is_diagnostic_and_visible():
    comparison = compare_hybrid_reinvestment(existing_run(), hybrid_inputs())

    assert comparison.transition.final_explicit_net_capex == pytest.approx(
        comparison.hybrid_years[-1].net_capex
    )
    assert comparison.transition.first_normalized_year_sales_to_capital_reinvestment == pytest.approx(
        comparison.existing_run.operating_forecast.years[5].reinvestment
    )


def test_capex_sensitivity_changes_only_capex():
    base = hybrid_inputs()
    high = scale_capex_path(base, 1.10)

    for old, new in zip(base, high):
        assert new.capex == pytest.approx(old.capex * 1.10)
        assert new.depreciation_amortization == old.depreciation_amortization
        assert new.revenue == old.revenue


def test_depreciation_lag_sensitivity_changes_only_depreciation():
    base = hybrid_inputs()
    slower = scale_depreciation_path(base, 0.90)

    for old, new in zip(base, slower):
        assert new.depreciation_amortization == pytest.approx(
            old.depreciation_amortization * 0.90
        )
        assert new.capex == old.capex


def test_inputs_and_results_are_immutable():
    item = hybrid_inputs()[0]
    result = calculate_hybrid_reinvestment_year(item)
    with pytest.raises(FrozenInstanceError):
        item.capex = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.fcff = 0  # type: ignore[misc]


def test_module_has_no_network_or_streamlit_imports():
    source = Path("Stock/hybrid_reinvestment_prototype.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("import streamlit", "import yfinance", "import requests"):
        assert forbidden not in source


def test_existing_production_run_is_not_mutated():
    existing = existing_run()
    original_fcff = tuple(year.fcff for year in existing.operating_forecast.years)
    compare_hybrid_reinvestment(existing, hybrid_inputs())
    assert tuple(year.fcff for year in existing.operating_forecast.years) == original_fcff
