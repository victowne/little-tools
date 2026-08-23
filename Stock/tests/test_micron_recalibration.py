from dataclasses import replace
import inspect

import pytest

from Stock import valuation
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.company_profile_review import candidate_assumption_signature
from Stock.micron_recalibration import (
    FY2026_CONSENSUS,
    TTM_REVENUE,
    build_micron_period_alignment,
)
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_alphabet_research import current_assumptions
from Stock.tests.test_amazon_research import amazon_history, amazon_inputs
from Stock.unified_company_research import (
    build_apple_research_profile,
    build_broadcom_research_profile,
    build_micron_research_profile,
)


def candidate():
    return build_micron_research_profile(
        current_assumptions(), amazon_history(), retrieved_at="2026-08-23"
    )


def test_current_ttm_period_is_identified_exactly():
    alignment = build_micron_period_alignment()
    assert alignment.ttm_period == "FY2025 Q4–FY2026 Q3 (ended 2026-05-28)"
    assert alignment.ttm_revenue == TTM_REVENUE


def test_fiscal_consensus_is_translated_to_non_overlapping_rolling_years():
    alignment = build_micron_period_alignment()
    y1, y2, y3 = alignment.rolling_years
    assert y1.period == "FY2026 Q4–FY2027 Q3"
    assert y2.period == "FY2027 Q4–FY2028 Q3"
    assert y3.period == "FY2028 Q4–FY2029 Q3"
    assert tuple(label for label, _ in y1.quarters) == (
        "FY2026 Q4", "FY2027 Q1", "FY2027 Q2", "FY2027 Q3"
    )
    assert y1.revenue == pytest.approx(231.46312e9)
    assert y2.revenue == pytest.approx(278.287774e9)
    assert y3.revenue == pytest.approx(321.2215456e9)


def test_old_y1_reproduced_fy2026_and_is_alignment_error():
    alignment = build_micron_period_alignment()
    assert alignment.old_y1_implied_revenue == pytest.approx(130.8973e9)
    assert alignment.old_y1_implied_revenue / FY2026_CONSENSUS - 1 == pytest.approx(
        .0116085, abs=1e-6
    )
    assert alignment.old_y1_alignment_error
    assert alignment.rolling_years[0].revenue > alignment.old_y1_implied_revenue


def test_new_growth_candidate_rounds_aligned_evidence_conservatively():
    alignment = build_micron_period_alignment()
    assert tuple(item.growth for item in alignment.rolling_years) == pytest.approx(
        (1.5640064692, .2022985519, .1542783249)
    )
    assert alignment.candidate_growth == (1.55, .20, .15)
    assert alignment.candidate_growth[2] > 0


def test_profile_uses_new_three_year_path_and_deterministic_fade():
    profile = candidate().lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    assert assumptions.near_term_revenue_growth == (1.55, .20, .15)
    assert assumptions.revenue_fade_years == 8
    assert assumptions.forecast_years == 11
    run = run_multistage_dcf(
        replace(amazon_inputs(), ticker="MU", starting_revenue=TTM_REVENUE),
        assumptions,
    )
    growth = tuple(year.revenue_growth for year in run.forecast_path.years)
    assert growth[:3] == (1.55, .20, .15)
    assert growth[3:5] == pytest.approx((.134375, .11875))
    assert growth[-1] == pytest.approx(.025)


def test_mature_economics_and_risk_inputs_are_preserved():
    profile = candidate().lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    assert assumptions.mature_operating_margin == .28
    assert assumptions.mature_sales_to_capital == .55
    assert assumptions.wacc == .105
    assert assumptions.terminal_growth == .025
    assert assumptions.derived_terminal_roic == pytest.approx(.1309)


def test_candidate_state_and_market_price_independence_are_preserved():
    result = candidate()
    profile = result.lookup.profile
    assert profile.profile_status == "research_in_progress"
    assert profile.last_reviewed_at is None
    assert profile.reinvestment_strategy is None
    assert result.micron_period_alignment is not None
    parameters = inspect.signature(build_micron_research_profile).parameters
    assert "market_price" not in parameters
    assert "current_price" not in parameters


def test_recalibrated_candidate_has_new_candidate_fingerprint():
    profile = candidate().lookup.profile
    revenue = profile.revenue_framework
    old_revenue = replace(
        revenue,
        year1_growth=replace(revenue.year1_growth, value=.45),
        year2_growth=replace(revenue.year2_growth, value=.12),
        year3_growth=replace(revenue.year3_growth, value=-.08),
    )
    old_profile = replace(profile, revenue_framework=old_revenue)
    assert candidate_assumption_signature(profile) != candidate_assumption_signature(
        old_profile
    )


def test_other_company_builders_are_not_changed_by_micron_recalibration():
    apple = build_apple_research_profile(
        current_assumptions(), amazon_history()
    ).lookup.profile
    broadcom = build_broadcom_research_profile(
        current_assumptions(), amazon_history()
    ).lookup.profile
    assert build_multistage_assumptions_from_profile(
        apple
    ).assumptions.near_term_revenue_growth == (.12, .08, .06)
    assert build_multistage_assumptions_from_profile(
        broadcom
    ).assumptions.near_term_revenue_growth == (.35, .22, .15)


def test_standard_valuation_engine_has_no_micron_dispatch():
    source = inspect.getsource(valuation).lower()
    assert "micron" not in source
    assert '"mu"' not in source
