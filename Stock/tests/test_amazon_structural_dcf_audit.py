from dataclasses import replace
import inspect
from pathlib import Path

import pandas as pd
import pytest

from Stock.amazon_structural_dcf_audit import (
    aggregate_segment_year,
    amazon_segment_evidence,
    amazon_segment_forecast_specs,
    break_even_sales_to_capital,
    build_segment_forecast,
    run_amazon_structural_audit,
)
from Stock.company_profiles import get_company_profile
from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.share_normalization import NormalizedShareCount


def _inputs(revenue=716.924e9, source="annual_fallback"):
    period = pd.Timestamp("2025-12-31")
    shares = NormalizedShareCount(
        "AMZN", 10.8e9, "fixture", period, "consolidated_common",
        "fixture", (), (), True, None,
    )
    return RealCompanyDCFInputs(
        "AMZN", revenue, source, (period,), -55e9, "fixture", period,
        10.8e9, shares, 0.67, 0.18, True, None, "USD", "USD",
    )


@pytest.fixture(scope="module")
def audit():
    return run_amazon_structural_audit(
        _inputs(), 0.12081,
        validated_ttm_inputs=_inputs(775.680e9, "ttm"),
        starting_depreciation_to_revenue=0.075,
    )


def test_segment_revenue_operating_income_and_margin_aggregate_by_identity():
    result = build_segment_forecast(amazon_segment_forecast_specs())
    first = result[0]
    assert first.revenue == pytest.approx(sum(x.revenue for x in first.segments))
    assert first.operating_income == pytest.approx(
        sum(x.operating_income for x in first.segments)
    )
    assert first.operating_margin == pytest.approx(
        first.operating_income / first.revenue
    )
    assert aggregate_segment_year(1, first.segments) == first


def test_segment_evidence_has_three_annual_years_and_recent_quarter():
    evidence = amazon_segment_evidence()
    for segment in ("North America", "International", "AWS", "Advertising overlay"):
        rows = [x for x in evidence if x.segment == segment]
        assert {x.period for x in rows} == {"2023", "2024", "2025", "2026 Q2"}
    assert all(x.operating_income is None for x in evidence if x.segment == "Advertising overlay")


@pytest.mark.parametrize(
    ("delta", "nopat", "expected"),
    [(20.0, 10.0, 2.0), (0.0, 10.0, None), (20.0, 0.0, None)],
)
def test_break_even_sales_to_capital(delta, nopat, expected):
    assert break_even_sales_to_capital(delta, nopat) == expected


def test_ttm_revenue_fix_is_attributed_without_overwriting_baseline(audit):
    assert audit.baseline.run.inputs.starting_revenue == pytest.approx(716.924e9)
    assert audit.revenue_base_fix is not None
    assert audit.revenue_base_fix.run.inputs.starting_revenue == pytest.approx(775.680e9)
    assert audit.baseline.run.assumptions == audit.revenue_base_fix.run.assumptions


def test_ttm_unavailable_keeps_explicit_annual_fallback():
    result = run_amazon_structural_audit(
        _inputs(), 0.12081, validated_ttm_inputs=None,
        starting_depreciation_to_revenue=0.075,
    )
    assert result.revenue_base_fix is None
    assert result.baseline.run.inputs.starting_revenue_source == "annual_fallback"


def test_margin_hybrid_and_combined_models_are_isolated(audit):
    base = audit.baseline.run
    margin = audit.margin_only.run
    hybrid = audit.hybrid_only.run
    combined = audit.margin_hybrid.run
    assert tuple(x.revenue for x in margin.operating_forecast.years) == pytest.approx(
        tuple(x.revenue for x in base.operating_forecast.years)
    )
    assert tuple(x.reinvestment for x in margin.operating_forecast.years) == pytest.approx(
        tuple(x.reinvestment for x in base.operating_forecast.years)
    )
    assert tuple(x.operating_margin for x in hybrid.operating_forecast.years) == pytest.approx(
        tuple(x.operating_margin for x in base.operating_forecast.years)
    )
    assert tuple(x.revenue for x in hybrid.operating_forecast.years) == pytest.approx(
        tuple(x.revenue for x in base.operating_forecast.years)
    )
    assert tuple(x.revenue for x in combined.operating_forecast.years) == pytest.approx(
        tuple(x.revenue for x in base.operating_forecast.years)
    )


def test_growth_monotonicity_diagnostic_exposes_reinvestment_effect(audit):
    points = audit.growth_monotonicity
    assert tuple(x.year3_growth for x in points) == (.10, .12, .14)
    assert points[2].total_reinvestment > points[0].total_reinvestment
    assert all(x.intrinsic_value_per_share is not None for x in points)


def test_waterfall_model_order_and_terminal_economics(audit):
    waterfall = (
        audit.baseline, audit.revenue_base_fix, audit.margin_only,
        audit.hybrid_only, audit.margin_hybrid, audit.segment_shadow,
    )
    assert tuple(x.model for x in waterfall if x is not None) == (
        "baseline_annual_fallback", "validated_ttm_revenue_base",
        "segment_informed_margin_only", "hybrid_reinvestment_only",
        "margin_plus_hybrid", "segment_informed_consolidated_shadow",
    )
    terminal = audit.baseline.run.terminal_value
    assert terminal.derived_terminal_roic == pytest.approx(0.12 * 0.79 * 0.83)
    assert terminal.derived_terminal_roic < audit.baseline.run.assumptions.wacc


def test_no_production_amazon_profile_and_alphabet_profile_is_unchanged():
    before = get_company_profile("GOOGL")
    amazon = get_company_profile("AMZN")
    assert amazon.available is True
    assert amazon.profile.issuer_id == "AMZN"
    assert amazon.reason is None
    after = get_company_profile("GOOGL")
    assert before == after


def test_pure_audit_has_no_network_ui_or_market_price_dependency():
    source = Path("Stock/amazon_structural_dcf_audit.py").read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "import yfinance" not in source
    assert "market_price" not in str(inspect.signature(run_amazon_structural_audit))
    assert "market_price_excluded_from_model_construction" in source


def test_result_is_immutable(audit):
    with pytest.raises(Exception):
        replace(audit.baseline, model="changed").model = "again"
