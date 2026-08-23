from dataclasses import fields, replace
import inspect

import pandas as pd
import pytest

from Stock import valuation
from Stock.amazon_research import (
    VALIDATED_TTM_OPERATING_MARGIN,
    VALIDATED_TTM_REVENUE,
    build_amazon_research_profile,
    run_amazon_candidate_preview,
)
from Stock.company_profile_review import (
    candidate_assumption_signature,
    initialize_profile_review,
    mark_profile_reviewed,
    set_review_group,
)
from Stock.company_profiles import (
    CompanyResearchProfile,
    build_multistage_assumptions_from_profile,
)
from Stock.fundamentals import OPERATING_MARGIN, REVENUE, TTMResult
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.share_normalization import NormalizedShareCount
from Stock.tests.test_alphabet_research import current_assumptions, history


def amazon_history():
    base = history()
    periods = tuple(pd.to_datetime([
        "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30",
    ]))
    ttm = dict(base.ttm)
    ttm[REVENUE] = TTMResult(VALIDATED_TTM_REVENUE, True, periods, None)
    ttm[OPERATING_MARGIN] = TTMResult(0.11155296795755199, True, periods, None)
    return replace(base, ttm=ttm)


def amazon_inputs(*, price_is_deliberately_absent=True):
    period = pd.Timestamp("2026-06-30")
    shares = NormalizedShareCount(
        "AMZN", 10.752e9, "fixture", period, "consolidated_common",
        "fixture", (), (), True, None,
    )
    return RealCompanyDCFInputs(
        "AMZN", 716.924e9, "stale_annual_fixture", (pd.Timestamp("2025-12-31"),),
        66.177e9, "fixture", period, 10.752e9, shares, 0.67, 0.18,
        True, None, "USD", "USD",
    )


def candidate():
    return build_amazon_research_profile(
        current_assumptions(), amazon_history(), retrieved_at="2026-08-23"
    )


def test_candidate_exists_and_is_read_only_unreviewed_unapplied():
    result = candidate()
    profile = result.lookup.profile
    assert result.lookup.available
    assert profile is not None
    assert profile.profile_status == "research_in_progress"
    assert profile.last_reviewed_at is None
    assert result.reviewed is False
    assert result.applied is False


def test_candidate_assumptions_and_terminal_economics_are_exact():
    profile = candidate().lookup.profile
    translated = build_multistage_assumptions_from_profile(profile)
    assert translated.available
    assumptions = translated.assumptions
    assert assumptions.near_term_revenue_growth == (0.15, 0.14, 0.12)
    assert assumptions.revenue_fade_years == 8
    assert assumptions.forecast_years == 11
    assert assumptions.mature_operating_margin == pytest.approx(0.1834)
    assert assumptions.mature_sales_to_capital == pytest.approx(0.824)
    assert assumptions.operating_tax_rate == pytest.approx(0.21)
    assert assumptions.wacc == pytest.approx(0.105)
    assert assumptions.terminal_growth == pytest.approx(0.03)
    assert assumptions.derived_terminal_roic == pytest.approx(0.1834 * 0.79 * 0.824)
    assert assumptions.terminal_reinvestment_rate == pytest.approx(
        0.03 / (0.1834 * 0.79 * 0.824)
    )


def test_sec_ttm_revenue_and_margin_do_not_fall_back_to_live_annual_data():
    profile = candidate().lookup.profile
    assert profile.revenue_framework.starting_revenue.value == VALIDATED_TTM_REVENUE
    assert profile.revenue_framework.starting_revenue.period == "TTM ended 2026-06-30"
    assert profile.margin_framework.starting_operating_margin.value == pytest.approx(
        VALIDATED_TTM_OPERATING_MARGIN
    )
    broken = replace(amazon_history(), ttm={REVENUE: amazon_history().ttm[REVENUE]})
    preserved = build_amazon_research_profile(current_assumptions(), broken).lookup.profile
    assert preserved.margin_framework.starting_operating_margin.value == pytest.approx(
        VALIDATED_TTM_OPERATING_MARGIN
    )
    assert preserved.margin_framework.ttm_operating_margin.period == "TTM ended 2026-06-30"


def test_fade_path_is_deterministic_and_has_no_y4_y5_profile_fields():
    profile = candidate().lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    standard = run_multistage_dcf(amazon_inputs(), assumptions)
    growth = tuple(year.revenue_growth for year in standard.forecast_path.years)
    assert growth == pytest.approx((
        .15, .14, .12, .10875, .0975, .08625, .075,
        .06375, .0525, .04125, .03,
    ))
    names = {field.name for field in fields(CompanyResearchProfile)}
    assert not any("year4" in name or "year5" in name for name in names)


def test_amazon_has_no_special_production_strategy():
    profile = candidate().lookup.profile
    assert profile.reinvestment_strategy is None
    assert profile.model_risk == "High"
    limitations = " ".join(profile.uncertainty_notes)
    assert "Unified production methodology" in limitations
    assert "Hybrid research" in limitations


def test_candidate_preview_uses_validated_ttm_and_standard_sc():
    result = candidate()
    profile = result.lookup.profile
    preview = run_amazon_candidate_preview(amazon_inputs(), profile)
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    standard = run_multistage_dcf(replace(
        amazon_inputs(), starting_revenue=VALIDATED_TTM_REVENUE,
    ), assumptions)
    assert preview.inputs.starting_revenue == VALIDATED_TTM_REVENUE
    assert preview.per_share_value.intrinsic_value_per_share == pytest.approx(
        standard.per_share_value.intrinsic_value_per_share
    )
    assert preview.enterprise_value.explicit_forecast_pv == pytest.approx(
        standard.enterprise_value.explicit_forecast_pv
    )


def test_market_price_is_not_an_input_or_candidate_field():
    profile = candidate().lookup.profile
    assert "market_price" not in inspect.signature(build_amazon_research_profile).parameters
    assert "current_price" not in inspect.signature(build_amazon_research_profile).parameters
    assert "market price" in profile.rationale.lower()


def test_builder_does_not_automatically_review_or_apply():
    profile = candidate().lookup.profile
    state = initialize_profile_review(profile)
    assert state.profile_status == "research_in_progress"
    assert state.reviewed_snapshot is None
    assert state.incomplete_groups
    assert profile.last_reviewed_at is None


def test_explicit_future_review_snapshot_captures_exact_strategy_metadata():
    profile = candidate().lookup.profile
    state = initialize_profile_review(profile)
    for group in tuple(item.group for item in state.group_reviews):
        state = set_review_group(state, profile, group, reviewed=True)
    reviewed = mark_profile_reviewed(
        state, profile, reviewed_at="2026-08-23T12:00:00+00:00"
    )
    snapshot_profile = reviewed.reviewed_snapshot.profile
    assert snapshot_profile.reinvestment_strategy == profile.reinvestment_strategy
    assert snapshot_profile.evidence_items == profile.evidence_items
    assert profile.profile_status == "research_in_progress"
    assert profile.last_reviewed_at is None


def test_no_strategy_metadata_participates_in_candidate_signature():
    profile = candidate().lookup.profile
    original = candidate_assumption_signature(profile)
    changed = replace(profile, model_risk="Medium")
    assert candidate_assumption_signature(changed) == original


def test_amazon_candidate_does_not_mutate_other_profiles_or_current_base():
    current = current_assumptions()
    original = replace(current)
    result = build_amazon_research_profile(current, amazon_history())
    assert current == original
    assert result.lookup.profile.issuer_id == "AMZN"
    assert result.lookup.profile.reinvestment_strategy is None


def test_valuation_module_has_no_amazon_or_hybrid_production_logic():
    source = inspect.getsource(valuation).lower()
    assert "amazon" not in source
    assert "hybrid_explicit_with_handoff" not in source
