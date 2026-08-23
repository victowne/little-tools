from dataclasses import replace

import pytest

from Stock.alphabet_reassessment import (
    build_alphabet_growth_economics_reassessment,
    build_terminal_economics_matrix,
)
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_alphabet_research import (
    current_assumptions,
    inputs,
    research,
)


def reassessment():
    return build_alphabet_growth_economics_reassessment(
        0.3311032213642185,
        evidence_as_of="2026-08-21",
    )


def test_baseline_is_preserved_before_revised_candidate():
    result = reassessment()
    assert result.existing_candidate.near_term_revenue_growth == (0.22, 0.17, 0.13)
    assert result.existing_candidate.mature_operating_margin == pytest.approx(0.32)
    assert result.existing_candidate.starting_sales_to_capital == pytest.approx(0.45)
    assert result.existing_candidate.mature_sales_to_capital == pytest.approx(0.60)
    assert result.existing_candidate != result.revised_candidate


def test_revised_candidate_is_exact_and_keeps_independent_inputs_unchanged():
    result = reassessment()
    revised = result.revised_candidate
    assert revised.near_term_revenue_growth == (0.23, 0.20, 0.17)
    assert revised.mature_operating_margin == pytest.approx(0.34)
    assert revised.starting_sales_to_capital == pytest.approx(0.50)
    assert revised.mature_sales_to_capital == pytest.approx(0.75)
    assert revised.revenue_fade_years == 8
    assert revised.forecast_years == 11
    assert revised.wacc == result.existing_candidate.wacc == pytest.approx(0.0975)
    assert revised.terminal_growth == result.existing_candidate.terminal_growth == pytest.approx(0.0325)


def test_eight_quarter_revenue_momentum_is_dated_and_deterministic():
    rows = reassessment().quarterly_revenue
    assert len(rows) == 8
    assert rows[0].quarter == "2024 Q3"
    assert rows[-1].quarter == "2026 Q2"
    assert rows[0].year_over_year_growth == pytest.approx(0.15)
    assert rows[-1].year_over_year_growth == pytest.approx(0.24)
    assert rows[-1].sequential_growth == pytest.approx(119.796 / 109.896 - 1)
    assert all("sec.gov" in row.source for row in rows)


def test_segment_momentum_covers_four_categories_and_eight_quarters():
    rows = reassessment().segment_momentum
    assert len(rows) == 32
    segments = {row.segment for row in rows}
    assert segments == {
        "Google Search & other",
        "YouTube ads",
        "Subscriptions / platforms / devices",
        "Google Cloud",
    }
    latest_cloud = next(
        row for row in rows
        if row.segment == "Google Cloud" and row.quarter == "2026 Q2"
    )
    assert latest_cloud.revenue == pytest.approx(24.768e9)
    assert latest_cloud.year_over_year_growth == pytest.approx(24.768 / 13.624 - 1)


def test_q2_growth_contributions_reconcile_to_consolidated_growth():
    contributions = reassessment().q2_2026_growth_contributions
    assert sum(row.prior_year_revenue_weight for row in contributions) == pytest.approx(1)
    assert sum(row.consolidated_growth_contribution for row in contributions) == pytest.approx(
        119.796 / 96.428 - 1
    )
    cloud = next(row for row in contributions if row.segment == "Google Cloud")
    assert cloud.consolidated_growth_contribution > 0.11


def test_terminal_matrix_reconciles_every_cell():
    matrix = build_terminal_economics_matrix()
    assert len(matrix) == 9
    for point in matrix:
        expected_roic = point.mature_operating_margin * 0.83 * point.mature_sales_to_capital
        assert point.terminal_roic == pytest.approx(expected_roic)
        assert point.terminal_reinvestment_rate == pytest.approx(0.0325 / expected_roic)
        assert point.fcff_to_nopat == pytest.approx(1 - 0.0325 / expected_roic)


def test_growth_ranges_and_references_are_preserved_in_profile():
    result = research()
    assert [(row.low, row.central, row.high) for row in result.growth_ranges] == [
        (0.21, 0.23, 0.25),
        (0.17, 0.20, 0.22),
        (0.14, 0.17, 0.19),
    ]
    profile = result.lookup.profile
    for assumption in (
        profile.revenue_framework.year1_growth,
        profile.revenue_framework.year2_growth,
        profile.revenue_framework.year3_growth,
    ):
        assert assumption.evidence_references


def test_profile_remains_unreviewed_and_does_not_mutate_current_base():
    current = current_assumptions()
    result = research()
    assert result.lookup.profile.profile_status == "research_in_progress"
    assert result.lookup.profile.last_reviewed_at is None
    assert result.current_assumptions == current
    assert current == current_assumptions()


def test_profile_translation_and_preview_use_exact_revision():
    result = research()
    translated = build_multistage_assumptions_from_profile(result.lookup.profile)
    assert translated.assumptions == result.reassessment.revised_candidate
    run = run_multistage_dcf(inputs(), translated.assumptions)
    assert run.assumptions == result.reassessment.revised_candidate
    assert run.forecast_path.revenue_growth_path[:3] == pytest.approx((0.23, 0.20, 0.17))


def test_one_factor_runs_change_only_the_named_operating_inputs():
    result = reassessment()
    growth_only = replace(
        result.existing_candidate,
        near_term_revenue_growth=result.revised_candidate.near_term_revenue_growth,
    )
    assert growth_only.wacc == result.existing_candidate.wacc
    assert growth_only.terminal_growth == result.existing_candidate.terminal_growth
    assert growth_only.mature_operating_margin == result.existing_candidate.mature_operating_margin


def test_reassessment_contains_no_market_targeting_or_review_semantics():
    result = reassessment()
    text = " ".join(
        [result.revision_note]
        + [item.evidence + " " + item.rationale for item in result.revisions]
    ).lower()
    assert "market price" not in text
    assert "target price" not in text
    assert "reviewed" not in text
    assert "approved" not in text
