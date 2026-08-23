from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.nvda_growth_reassessment import (
    ConsensusRevenuePoint,
    build_nvda_growth_reassessment,
    build_run_rate_diagnostics,
    calculate_quarterly_revenue_points,
    compare_growth_duration_dcf,
    forecast_revenue_levels,
)
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions, generate_forecast_path


def baseline_assumptions():
    return MultiStageDCFAssumptions(
        forecast_years=12,
        near_term_revenue_growth=(0.55, 0.40, 0.25),
        revenue_fade_years=9,
        terminal_growth=0.0325,
        starting_operating_margin=0.6402,
        mature_operating_margin=0.45,
        starting_sales_to_capital=1.35,
        mature_sales_to_capital=1.00,
        operating_tax_rate=0.17,
        wacc=0.115,
    )


def consensus_points():
    return (
        ConsensusRevenuePoint(
            "FY2027", "2027-01-31", 395.213e9, 0.8302, 53,
            "fixture", "2026-08-22",
        ),
        ConsensusRevenuePoint(
            "FY2028", "2028-01-31", 568.184e9, 0.4377, 55,
            "fixture", "2026-08-22",
        ),
    )


def reassessment():
    return build_nvda_growth_reassessment(
        baseline_assumptions(), ttm_revenue=253.491e9,
        ttm_period_end="2026-04-30", consensus=consensus_points(),
    )


def inputs():
    shares = NormalizedShareCount(
        ticker="NVDA", shares_outstanding=24.3e9, source="fixture",
        source_period=pd.Timestamp("2026-04-30"),
        scope="consolidated_common", method="fixture", components=(),
        warnings=(), available=True, reason=None,
    )
    return RealCompanyDCFInputs(
        ticker="NVDA", starting_revenue=253.491e9,
        starting_revenue_source="ttm",
        starting_revenue_periods=tuple(pd.to_datetime([
            "2025-07-31", "2025-10-31", "2026-01-31", "2026-04-30",
        ])),
        net_debt=-50e9, net_debt_source="fixture",
        net_debt_period=pd.Timestamp("2026-04-30"),
        shares_outstanding=24.3e9, normalized_share_count=shares,
        historical_sales_to_capital_3y=1.49,
        current_accounting_roic=0.9283,
        statement_currency="USD", security_currency="USD",
    )


def test_quarterly_growth_calculation_sorts_periods_and_calculates_yoy():
    observations = (
        ("Q5", "2025-03-31", 150.0),
        ("Q1", "2024-03-31", 100.0),
        ("Q3", "2024-09-30", 120.0),
        ("Q2", "2024-06-30", 110.0),
        ("Q4", "2024-12-31", 130.0),
    )
    points = calculate_quarterly_revenue_points(observations, source="fixture")

    assert tuple(point.fiscal_quarter for point in points) == (
        "Q1", "Q2", "Q3", "Q4", "Q5"
    )
    assert points[-1].sequential_growth == pytest.approx(150 / 130 - 1)
    assert points[-1].yoy_growth == pytest.approx(0.50)


def test_quarterly_duplicate_period_is_rejected():
    with pytest.raises(ValueError, match="duplicate_quarterly_period"):
        calculate_quarterly_revenue_points(
            (("Q1", "2026-01-31", 1.0), ("duplicate", "2026-01-31", 2.0)),
            source="fixture",
        )


def test_run_rate_diagnostics_do_not_label_annualization_as_forecast():
    result = build_run_rate_diagnostics(
        ttm_revenue=253.491e9, fy2027_consensus_revenue=395.213e9
    )

    assert result.latest_quarter_annualized == pytest.approx(326.46e9)
    assert result.guidance_midpoint_annualized == pytest.approx(364.0e9)
    assert "not_forecasts" in result.warning


def test_reassessment_preserves_current_candidate_and_builds_exact_y1_to_y5_shadow():
    result = reassessment()

    assert result.current_assumptions.near_term_revenue_growth == (0.55, 0.40, 0.25)
    assert tuple(year.growth for year in result.research_path) == (
        0.55, 0.40, 0.30, 0.25, 0.20
    )
    assert result.shadow_assumptions.near_term_revenue_growth == (
        0.55, 0.40, 0.30, 0.25, 0.20
    )
    assert result.shadow_assumptions.revenue_fade_years == 7
    assert result.shadow_assumptions.forecast_years == 12


def test_current_y4_y5_are_documented_from_existing_mathematical_fade():
    result = reassessment()

    assert result.current_implied_first_five_growth == pytest.approx(
        (0.55, 0.40, 0.25, 0.2258333333, 0.2016666667)
    )


def test_shadow_fades_to_terminal_growth_in_year_12():
    path = generate_forecast_path(reassessment().shadow_assumptions)

    assert path.near_term_year_count == 5
    assert path.fade_year_count == 7
    assert path.mature_year_count == 0
    assert path.revenue_growth_path[-1] == pytest.approx(0.0325)


def test_confidence_declines_without_claiming_y4_y5_consensus():
    result = reassessment()

    assert tuple(year.confidence for year in result.research_path) == (
        "High", "High", "Medium", "Low", "Low"
    )
    assert all("consensus" not in item.lower() for item in result.research_path[3].evidence)


def test_period_alignment_distinguishes_dcf_years_from_fiscal_consensus():
    alignments = reassessment().alignments

    assert alignments[0].alignment == "near_aligned"
    assert alignments[1].alignment == "near_aligned"
    assert alignments[2].alignment == "mismatched"
    assert alignments[2].fiscal_consensus_period_end is None


def test_official_quarterly_and_data_center_series_have_eight_points():
    result = reassessment()

    assert len(result.quarterly_revenue) == 8
    assert len(result.data_center_revenue) == 8
    assert result.quarterly_revenue[-1].revenue == pytest.approx(81.615e9)
    assert result.data_center_revenue[-1].revenue == pytest.approx(75.2e9)
    assert result.data_center_revenue[-1].share_of_total_revenue == pytest.approx(
        75.2 / 81.615
    )


def test_product_cycle_timeline_does_not_fabricate_follow_on_date():
    follow_on = reassessment().product_cycles[-1]

    assert follow_on.confidence == "Low"
    assert "No sufficiently precise official" in follow_on.ramp_window


def test_reassessment_decision_does_not_update_stored_candidate():
    current = baseline_assumptions()
    result = build_nvda_growth_reassessment(
        current, ttm_revenue=253.491e9,
        ttm_period_end="2026-04-30", consensus=consensus_points(),
    )

    assert result.decision == "INSUFFICIENT EVIDENCE"
    assert current.near_term_revenue_growth == (0.55, 0.40, 0.25)
    assert "research_shadow_not_a_company_profile_candidate" in result.warnings


def test_primary_dcf_comparison_changes_only_growth_duration():
    result = reassessment()
    comparison = compare_growth_duration_dcf(inputs(), result)

    fixed_fields = (
        "forecast_years", "terminal_growth", "starting_operating_margin",
        "mature_operating_margin", "starting_sales_to_capital",
        "mature_sales_to_capital", "operating_tax_rate", "wacc",
    )
    for field in fixed_fields:
        assert getattr(comparison.existing.assumptions, field) == getattr(
            comparison.shadow.assumptions, field
        )
    assert comparison.existing.assumptions.near_term_revenue_growth != (
        comparison.shadow.assumptions.near_term_revenue_growth
    )


def test_dcf_comparison_does_not_mutate_inputs_or_current_assumptions():
    original_inputs = inputs()
    result = reassessment()
    original_assumptions = result.current_assumptions

    compare_growth_duration_dcf(original_inputs, result)

    assert original_inputs.starting_revenue == 253.491e9
    assert result.current_assumptions is original_assumptions
    assert result.current_assumptions.near_term_revenue_growth == (0.55, 0.40, 0.25)


def test_market_price_is_not_an_input_to_reassessment_or_dcf_comparison():
    reassessment_fields = set(reassessment().__dataclass_fields__)
    input_fields = set(inputs().__dataclass_fields__)

    assert "market_price" not in reassessment_fields
    assert "market_price" not in input_fields


def test_reassessment_result_is_immutable():
    result = reassessment()

    with pytest.raises(FrozenInstanceError):
        result.decision = "KEEP CURRENT"  # type: ignore[misc]


def test_revenue_scale_calculation_is_deterministic():
    levels = forecast_revenue_levels(
        253.491e9, (0.55, 0.40, 0.30, 0.25, 0.20)
    )

    assert levels == pytest.approx((
        392.91105e9,
        550.07547e9,
        715.098111e9,
        893.87263875e9,
        1_072.6471665e9,
    ))
