from dataclasses import replace

import pandas as pd
import pytest

from Stock import stock_valuation_mvp as app
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.company_profile_review import (
    REQUIRED_REVIEW_GROUPS,
    ReviewedCompanyProfileSnapshot,
    initialize_profile_review,
    mark_profile_reviewed,
    reconcile_review_state,
    reopen_profile_review,
    set_overall_review_note,
    set_review_group,
)
from Stock.company_profile_application import (
    assumptions_fingerprint,
    assumptions_match,
    build_profile_apply_plan,
    create_reviewed_profile_application,
)
from Stock.forecast_anchors import ForecastAnchorPoint, RevenueForecastAnchors
from Stock.fundamentals import (
    GROSS_MARGIN,
    OPERATING_MARGIN,
    OPERATING_TAX_RATE,
    REVENUE,
    REVENUE_GROWTH,
    ROIC,
    FundamentalHistory,
    HistoricalDCFAnchors,
    RevenueCAGRResult,
    SalesToCapitalResult,
    TTMResult,
)
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.nvda_research import build_nvda_research_profile
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions


def current_assumptions():
    return MultiStageDCFAssumptions(
        forecast_years=10,
        near_term_revenue_growth=(0.30, 0.25, 0.20),
        revenue_fade_years=7,
        terminal_growth=0.035,
        starting_operating_margin=0.64,
        mature_operating_margin=0.40,
        starting_sales_to_capital=1.5,
        mature_sales_to_capital=1.2,
        operating_tax_rate=0.16,
        wacc=0.09,
    )


def history():
    periods = pd.to_datetime([
        "2022-01-31", "2023-01-31", "2024-01-31",
        "2025-01-31", "2026-01-31",
    ])
    annual = pd.DataFrame(
        {
            REVENUE: [26.9e9, 27.0e9, 60.9e9, 130.5e9, 215.9e9],
            REVENUE_GROWTH: [None, 0.002, 1.26, 1.14, 0.6547],
            GROSS_MARGIN: [0.646, 0.569, 0.727, 0.750, 0.7107],
            OPERATING_MARGIN: [0.372, 0.155, 0.541, 0.623, 0.6038],
            OPERATING_TAX_RATE: [0.019, 0.210, 0.120, 0.135, 0.1512],
            ROIC: [0.35, 0.10, 0.55, 0.78, 0.9283],
        },
        index=periods,
    )
    ttm_periods = tuple(pd.to_datetime([
        "2025-07-31", "2025-10-31", "2026-01-31", "2026-04-30",
    ]))
    anchors = HistoricalDCFAnchors(
        revenue_cagr={
            3: RevenueCAGRResult(
                1.0005, True, periods[1], periods[4], 3, None,
                27.0e9, 215.9e9,
            )
        },
        annual_sales_to_capital={
            periods[4]: SalesToCapitalResult(
                1.11, True, periods[3], periods[4], 1, None,
            )
        },
        normalized_sales_to_capital={
            3: SalesToCapitalResult(
                1.49, True, periods[1], periods[4], 3, None,
            )
        },
    )
    return FundamentalHistory(
        annual=annual,
        ttm={
            REVENUE: TTMResult(253.491e9, True, ttm_periods, None),
            OPERATING_MARGIN: TTMResult(0.6402, True, ttm_periods, None),
        },
        annual_reasons=pd.DataFrame(index=periods),
        dcf_anchors=anchors,
    )


def anchors():
    return RevenueForecastAnchors(
        ticker="NVDA", issuer_ticker="NVDA",
        current_revenue_base=253.491e9,
        base_period=pd.Timestamp("2026-04-30"), base_kind="ttm",
        latest_actual_fiscal_revenue=215.9e9,
        latest_actual_fiscal_period=pd.Timestamp("2026-01-31"),
        points=(
            ForecastAnchorPoint(
                1, pd.Timestamp("2027-01-31"), 393.9e9, 0.824,
                "fixture_consensus", pd.Timestamp("2026-08-17"), 53,
                True, None,
            ),
            ForecastAnchorPoint(
                2, pd.Timestamp("2028-01-31"), 562.1e9, 0.427,
                "fixture_consensus", pd.Timestamp("2026-08-17"), 55,
                True, None,
            ),
            ForecastAnchorPoint(
                3, pd.Timestamp("2029-01-31"), None, None,
                "fixture_consensus", pd.Timestamp("2026-08-17"), None,
                False, "unavailable",
            ),
        ),
        source="fixture_consensus",
        warnings=("ttm_base_not_directly_comparable_to_fiscal_consensus",),
    )


def research():
    return build_nvda_research_profile(
        current_assumptions(), history(), revenue_anchors=anchors(),
        retrieved_at="2026-08-17",
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


def test_nvda_profile_is_research_in_progress_with_multiple_evidence_refs():
    profile = research().lookup.profile
    assert profile.profile_status == "research_in_progress"
    assert profile.last_reviewed_at is None
    assert len(profile.revenue_framework.year1_growth.evidence_references) >= 4
    assert profile.uncertainty_notes


def test_current_and_candidate_assumptions_remain_separate():
    current = current_assumptions()
    result = build_nvda_research_profile(
        current, history(), revenue_anchors=anchors(),
        retrieved_at="2026-08-17",
    )
    translated = build_multistage_assumptions_from_profile(
        result.lookup.profile
    ).assumptions
    assert current.near_term_revenue_growth == (0.30, 0.25, 0.20)
    assert translated.near_term_revenue_growth == pytest.approx((0.55, 0.40, 0.25))
    assert translated is not current


def test_candidate_translates_exactly_and_preserves_provenance():
    profile = research().lookup.profile
    translated = build_multistage_assumptions_from_profile(profile)
    assert translated.available
    assert translated.assumptions.forecast_years == 12
    assert translated.assumptions.revenue_fade_years == 9
    assert translated.assumptions.mature_operating_margin == pytest.approx(0.45)
    assert translated.assumptions.starting_sales_to_capital == pytest.approx(1.35)
    assert translated.assumptions.mature_sales_to_capital == pytest.approx(1.0)
    assert translated.assumptions.operating_tax_rate == pytest.approx(0.17)
    assert translated.assumptions.wacc == pytest.approx(0.115)
    assert profile.wacc_framework.research_wacc.rationale
    assert profile.wacc_framework.research_wacc.last_reviewed_at is None
    evidence_ids = {item.evidence_id for item in profile.evidence_items}
    assumptions = (
        profile.revenue_framework.year1_growth,
        profile.revenue_framework.year2_growth,
        profile.revenue_framework.year3_growth,
        profile.revenue_framework.revenue_fade_years,
        profile.margin_framework.starting_operating_margin,
        profile.margin_framework.mature_operating_margin,
        profile.capital_efficiency_framework.starting_sales_to_capital,
        profile.capital_efficiency_framework.mature_sales_to_capital,
        profile.operating_tax_rate,
        profile.wacc_framework.research_wacc,
        profile.terminal_framework.terminal_growth,
        profile.forecast_years,
    )
    assert all(
        set(item.evidence_references).issubset(evidence_ids)
        for item in assumptions
    )


def test_evidence_change_does_not_auto_update_candidate():
    profile = research().lookup.profile
    changed = replace(
        profile,
        evidence_items=profile.evidence_items + (
            replace(profile.evidence_items[0], evidence_id="new_evidence", value=999e9),
        ),
    )
    before = build_multistage_assumptions_from_profile(profile).assumptions
    after = build_multistage_assumptions_from_profile(changed).assumptions
    assert after == before


def test_missing_candidate_field_causes_incomplete_translation():
    profile = research().lookup.profile
    profile = replace(
        profile,
        revenue_framework=replace(profile.revenue_framework, year2_growth=None),
    )
    translated = build_multistage_assumptions_from_profile(profile)
    assert not translated.available
    assert translated.reason == "research_profile_incomplete"
    assert "revenue_framework.year2_growth" in translated.missing_fields


def test_candidate_preview_runs_full_existing_dcf_chain():
    candidate = build_multistage_assumptions_from_profile(
        research().lookup.profile
    ).assumptions
    preview = run_multistage_dcf(inputs(), candidate)
    assert len(preview.forecast_path.years) == 12
    assert len(preview.operating_forecast.years) == 12
    assert len(preview.discounted_forecast.years) == 12
    assert preview.terminal_value.terminal_value > 0
    assert preview.enterprise_value.enterprise_value > 0
    assert preview.equity_value.equity_value > 0
    assert preview.per_share_value.intrinsic_value_per_share > 0


def test_terminal_roic_and_reinvestment_diagnostics_reconcile():
    profile = research().lookup.profile
    translated = build_multistage_assumptions_from_profile(profile).assumptions
    terminal = profile.terminal_framework
    expected_roic = 0.45 * (1 - 0.17) * 1.0
    assert terminal.terminal_roic == pytest.approx(expected_roic)
    assert terminal.terminal_roic == pytest.approx(translated.derived_terminal_roic)
    assert terminal.terminal_reinvestment_rate == pytest.approx(0.0325 / expected_roic)
    assert terminal.terminal_fcff_conversion == pytest.approx(
        1 - terminal.terminal_reinvestment_rate
    )


def test_revenue_evidence_preserves_periods_dates_and_analyst_counts():
    result = research()
    fy1 = next(row for row in result.revenue_evidence if row.label == "FY1 consensus")
    ttm = next(row for row in result.revenue_evidence if row.label == "Current validated TTM")
    assert fy1.period == "2027-01-31"
    assert fy1.analyst_count == 53
    assert fy1.source_date == "2026-08-17"
    assert "2026-04-30" in ttm.period
    assert len(result.period_reconciliation) >= 4


def test_growth_ranges_are_context_only_and_match_candidate_centers():
    result = research()
    centers = {item.assumption_id: item.central for item in result.growth_ranges}
    assert centers == pytest.approx({
        "year1_growth": 0.55,
        "year2_growth": 0.40,
        "year3_growth": 0.25,
    })
    assert all(item.low < item.central < item.high for item in result.growth_ranges)


def fully_reviewed_state(profile):
    state = initialize_profile_review(profile)
    for group in REQUIRED_REVIEW_GROUPS:
        state = set_review_group(
            state, profile, group, reviewed=True,
            user_note=f"Reviewed {group}",
            reviewed_at="2026-08-17T12:00:00+00:00",
        )
    return state


def test_review_groups_default_unreviewed_and_are_independent():
    profile = research().lookup.profile
    state = initialize_profile_review(profile)
    assert state.profile_status == "research_in_progress"
    assert state.incomplete_groups == REQUIRED_REVIEW_GROUPS

    state = set_review_group(
        state, profile, "revenue", reviewed=True,
        user_note="Revenue path inspected.", reviewed_at="time",
    )
    assert state.group("revenue").reviewed
    assert state.group("revenue").user_note == "Revenue path inspected."
    assert all(
        not state.group(group).reviewed
        for group in REQUIRED_REVIEW_GROUPS if group != "revenue"
    )


def test_all_groups_required_and_reviewed_at_only_set_by_final_action():
    profile = research().lookup.profile
    state = initialize_profile_review(profile)
    state = set_review_group(
        state, profile, "revenue", reviewed=True, reviewed_at="group-time"
    )
    assert state.reviewed_snapshot is None
    with pytest.raises(ValueError, match="review_groups_incomplete"):
        mark_profile_reviewed(
            state, profile, reviewed_at="2026-08-17T13:00:00+00:00"
        )

    state = fully_reviewed_state(profile)
    reviewed = mark_profile_reviewed(
        state, profile, reviewed_at="2026-08-17T13:00:00+00:00"
    )
    assert reviewed.profile_status == "reviewed"
    assert reviewed.reviewed_snapshot.reviewed_at == "2026-08-17T13:00:00+00:00"
    assert reviewed.reviewed_snapshot.profile.last_reviewed_at == "2026-08-17T13:00:00+00:00"


def test_reviewed_snapshot_is_immutable_and_preserves_notes_and_rationale():
    profile = research().lookup.profile
    state = fully_reviewed_state(profile)
    state = set_overall_review_note(
        state, "Accepted current profile after framework review."
    )
    reviewed = mark_profile_reviewed(
        state, profile, reviewed_at="2026-08-17T13:00:00+00:00"
    )
    snapshot = reviewed.reviewed_snapshot
    assert snapshot.overall_review_note == "Accepted current profile after framework review."
    assert snapshot.profile.revenue_framework.year1_growth.rationale == (
        profile.revenue_framework.year1_growth.rationale
    )
    assert snapshot.profile.revenue_framework.year1_growth.status == "reviewed"
    with pytest.raises(Exception):
        snapshot.profile.profile_status = "research_in_progress"


def test_evidence_refresh_does_not_change_reviewed_assumptions():
    profile = research().lookup.profile
    state = mark_profile_reviewed(
        fully_reviewed_state(profile), profile,
        reviewed_at="2026-08-17T13:00:00+00:00",
    )
    refreshed = replace(
        profile,
        evidence_items=(
            replace(profile.evidence_items[0], value=999e9),
        ) + profile.evidence_items[1:],
    )
    reconciled = reconcile_review_state(state, refreshed)
    reviewed_assumptions = build_multistage_assumptions_from_profile(
        reconciled.reviewed_snapshot.profile
    ).assumptions
    assert reviewed_assumptions.near_term_revenue_growth == pytest.approx(
        (0.55, 0.40, 0.25)
    )
    assert "reviewed_profile_evidence_changed" in reconciled.warnings
    assert "review_refresh_recommended" not in reconciled.warnings


def test_candidate_edit_resets_only_affected_review_group():
    profile = research().lookup.profile
    state = fully_reviewed_state(profile)
    edited_y2 = replace(profile.revenue_framework.year2_growth, value=0.38)
    edited = replace(
        profile,
        revenue_framework=replace(
            profile.revenue_framework, year2_growth=edited_y2
        ),
    )
    reconciled = reconcile_review_state(state, edited)
    assert not reconciled.group("revenue").reviewed
    assert all(
        reconciled.group(group).reviewed
        for group in REQUIRED_REVIEW_GROUPS if group != "revenue"
    )
    assert "revenue_review_reset_after_candidate_change" in reconciled.warnings


def test_reviewed_profile_does_not_modify_base_and_preview_reconciles():
    profile = research().lookup.profile
    base = current_assumptions()
    candidate_assumptions = build_multistage_assumptions_from_profile(
        profile
    ).assumptions
    candidate_preview = run_multistage_dcf(inputs(), candidate_assumptions)
    state = mark_profile_reviewed(
        fully_reviewed_state(profile), profile,
        reviewed_at="2026-08-17T13:00:00+00:00",
    )
    reviewed_assumptions = build_multistage_assumptions_from_profile(
        state.reviewed_snapshot.profile
    ).assumptions
    reviewed_preview = run_multistage_dcf(inputs(), reviewed_assumptions)

    assert base == current_assumptions()
    assert reviewed_assumptions == candidate_assumptions
    assert reviewed_preview == candidate_preview


def test_reopen_preserves_prior_snapshot_and_resets_current_groups():
    profile = research().lookup.profile
    reviewed = mark_profile_reviewed(
        fully_reviewed_state(profile), profile,
        reviewed_at="2026-08-17T13:00:00+00:00",
    )
    snapshot = reviewed.reviewed_snapshot
    reopened = reopen_profile_review(
        reviewed, profile, reopened_at="2026-08-18T09:00:00+00:00"
    )
    assert reopened.profile_status == "research_in_progress"
    assert reopened.reviewed_snapshot is snapshot
    assert reopened.reopened_at == "2026-08-18T09:00:00+00:00"
    assert reopened.incomplete_groups == REQUIRED_REVIEW_GROUPS
    assert "review_reopened" in reopened.warnings


def reviewed_state():
    profile = research().lookup.profile
    return mark_profile_reviewed(
        fully_reviewed_state(profile), profile,
        reviewed_at="2026-08-17T13:00:00+00:00",
    )


def test_research_in_progress_snapshot_cannot_apply():
    profile = research().lookup.profile
    invalid_snapshot = ReviewedCompanyProfileSnapshot(
        profile=profile,
        reviewed_at="2026-08-17T13:00:00+00:00",
        group_reviews=fully_reviewed_state(profile).group_reviews,
        overall_review_note="",
        assumption_signature="candidate",
        evidence_signature="evidence",
    )
    plan = build_profile_apply_plan(invalid_snapshot, current_assumptions())
    assert not plan.available
    assert plan.reason == "profile_not_reviewed"


def test_incomplete_reviewed_profile_cannot_apply():
    snapshot = reviewed_state().reviewed_snapshot
    incomplete_profile = replace(
        snapshot.profile,
        revenue_framework=replace(
            snapshot.profile.revenue_framework, year2_growth=None
        ),
    )
    incomplete = replace(snapshot, profile=incomplete_profile)
    plan = build_profile_apply_plan(incomplete, current_assumptions())
    assert not plan.available
    assert plan.reason == "research_profile_incomplete"


def test_reviewed_snapshot_maps_exactly_and_reconciles_dcf_preview():
    snapshot = reviewed_state().reviewed_snapshot
    plan = build_profile_apply_plan(snapshot, current_assumptions())
    translated = build_multistage_assumptions_from_profile(
        snapshot.profile
    ).assumptions
    assert plan.available
    assert plan.assumptions == translated
    assert plan.assumptions.near_term_revenue_growth == (0.55, 0.40, 0.25)
    assert plan.assumptions.starting_operating_margin == pytest.approx(0.6402)
    assert plan.assumptions.wacc == pytest.approx(0.115)
    assert len(plan.changed_fields) > 0
    assert run_multistage_dcf(inputs(), plan.assumptions) == run_multistage_dcf(
        inputs(), translated
    )


def test_already_applied_and_divergence_states_are_deterministic():
    snapshot = reviewed_state().reviewed_snapshot
    initial = build_profile_apply_plan(snapshot, current_assumptions())
    application = create_reviewed_profile_application(
        initial, applied_at="2026-08-18T10:00:00+00:00"
    )
    same = build_profile_apply_plan(
        snapshot, initial.assumptions, previous_application=application
    )
    assert same.already_applied
    assert not same.base_diverged
    assert same.changed_fields == ()

    manually_edited = replace(initial.assumptions, mature_operating_margin=0.44)
    diverged = build_profile_apply_plan(
        snapshot, manually_edited, previous_application=application
    )
    assert not diverged.already_applied
    assert diverged.base_diverged
    assert "base_diverged_from_reviewed_profile" in diverged.warnings
    assert application.assumptions.mature_operating_margin == pytest.approx(0.45)


def test_evidence_refresh_and_reopen_do_not_mutate_applied_assumptions():
    reviewed = reviewed_state()
    snapshot = reviewed.reviewed_snapshot
    plan = build_profile_apply_plan(snapshot, current_assumptions())
    application = create_reviewed_profile_application(
        plan, applied_at="2026-08-18T10:00:00+00:00"
    )
    refreshed_candidate = replace(
        research().lookup.profile,
        evidence_items=(
            replace(research().lookup.profile.evidence_items[0], value=999e9),
        ) + research().lookup.profile.evidence_items[1:],
    )
    reconcile_review_state(reviewed, refreshed_candidate)
    reopen_profile_review(
        reviewed, refreshed_candidate,
        reopened_at="2026-08-18T11:00:00+00:00",
    )
    assert application.assumptions == plan.assumptions


def test_newer_reviewed_snapshot_requires_explicit_new_application():
    first = reviewed_state().reviewed_snapshot
    first_plan = build_profile_apply_plan(first, current_assumptions())
    application = create_reviewed_profile_application(
        first_plan, applied_at="2026-08-18T10:00:00+00:00"
    )
    changed_assumption = replace(
        first.profile.revenue_framework.year1_growth, value=0.56
    )
    newer_profile = replace(
        first.profile,
        revenue_framework=replace(
            first.profile.revenue_framework, year1_growth=changed_assumption
        ),
    )
    newer = replace(
        first, profile=newer_profile,
        reviewed_at="2026-08-19T10:00:00+00:00",
    )
    plan = build_profile_apply_plan(
        newer, application.assumptions, previous_application=application
    )
    assert plan.newer_review_available
    assert not plan.already_applied
    assert application.assumptions.near_term_revenue_growth[0] == 0.55
    assert plan.assumptions.near_term_revenue_growth[0] == 0.56
    assert assumptions_fingerprint(plan.assumptions) != (
        application.snapshot_fingerprint
    )
    assert assumptions_match(application.assumptions, first_plan.assumptions)


def test_explicit_session_apply_updates_complete_base_and_wacc_provenance():
    snapshot = reviewed_state().reviewed_snapshot
    state = {"formula_based_wacc_evidence": 0.091}
    app.initialize_multistage_session_state(state, "NVDA", history())
    wacc_keys = app.research_wacc_session_keys("NVDA")
    state[wacc_keys["rationale"]] = "Existing rationale"

    application = app.apply_reviewed_profile_to_base_session_state(
        state,
        "NVDA",
        snapshot,
        current_assumptions(),
        applied_at="2026-08-18T10:00:00+00:00",
    )
    values = app.initialize_multistage_session_state(state, "NVDA", history())
    applied_base = app.build_multistage_assumptions_from_ui(values)

    assert applied_base == application.assumptions
    assert state[wacc_keys["value"]] == pytest.approx(11.5)
    assert state[wacc_keys["status"]] == "user_reviewed"
    assert state[wacc_keys["created_at"]] == "2026-08-18T10:00:00+00:00"
    assert state[wacc_keys["rationale"]] == "Reviewed wacc"
    assert state["formula_based_wacc_evidence"] == pytest.approx(0.091)
    assert application.source == "reviewed_company_profile"
    assert application.issuer == "NVDA"
    assert application.reviewed_at == snapshot.reviewed_at
    assert application.applied_at != application.reviewed_at


def test_rerun_and_manual_edit_do_not_silently_reapply_and_reapply_restores():
    snapshot = reviewed_state().reviewed_snapshot
    state = {}
    app.initialize_multistage_session_state(state, "NVDA", history())
    application = app.apply_reviewed_profile_to_base_session_state(
        state, "NVDA", snapshot, current_assumptions(),
        applied_at="2026-08-18T10:00:00+00:00",
    )
    prefix = "multistage_NVDA_"
    state[prefix + "year_1_growth"] = 48.0

    rerun_values = app.initialize_multistage_session_state(
        state, "NVDA", history()
    )
    diverged_base = app.build_multistage_assumptions_from_ui(rerun_values)
    assert diverged_base.near_term_revenue_growth[0] == pytest.approx(0.48)
    plan = build_profile_apply_plan(
        snapshot, diverged_base, previous_application=application
    )
    assert plan.base_diverged

    restored = app.apply_reviewed_profile_to_base_session_state(
        state, "NVDA", snapshot, diverged_base,
        applied_at="2026-08-19T10:00:00+00:00",
    )
    restored_values = app.initialize_multistage_session_state(
        state, "NVDA", history()
    )
    restored_base = app.build_multistage_assumptions_from_ui(restored_values)
    assert restored_base == restored.assumptions
    assert restored.applied_at == "2026-08-19T10:00:00+00:00"


def test_reopen_and_evidence_refresh_leave_applied_base_session_values_unchanged():
    reviewed = reviewed_state()
    snapshot = reviewed.reviewed_snapshot
    state = {}
    app.initialize_multistage_session_state(state, "NVDA", history())
    app.apply_reviewed_profile_to_base_session_state(
        state, "NVDA", snapshot, current_assumptions(),
        applied_at="2026-08-18T10:00:00+00:00",
    )
    base_keys = {
        key: value for key, value in state.items()
        if key.startswith("multistage_NVDA_")
        or key.startswith("research_wacc_NVDA_")
        or key == app.base_profile_application_key("NVDA")
    }
    candidate = research().lookup.profile
    refreshed = replace(
        candidate,
        evidence_items=(replace(candidate.evidence_items[0], value=999e9),)
        + candidate.evidence_items[1:],
    )
    reconcile_review_state(reviewed, refreshed)
    reopen_profile_review(
        reviewed, refreshed, reopened_at="2026-08-19T09:00:00+00:00"
    )
    assert {
        key: value for key, value in state.items()
        if key in base_keys
    } == base_keys
