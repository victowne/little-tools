from dataclasses import replace

import pandas as pd
import pytest

from Stock.alphabet_research import build_alphabet_research_profile
from Stock.company_profiles import build_multistage_assumptions_from_profile
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
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions


def current_assumptions():
    return MultiStageDCFAssumptions(
        forecast_years=10,
        near_term_revenue_growth=(0.15, 0.13, 0.11),
        revenue_fade_years=7,
        terminal_growth=0.035,
        starting_operating_margin=0.3311032213642185,
        mature_operating_margin=0.30,
        starting_sales_to_capital=0.8,
        mature_sales_to_capital=0.7,
        operating_tax_rate=0.17,
        wacc=0.085,
    )


def history():
    periods = pd.to_datetime([
        "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31",
    ])
    annual = pd.DataFrame(
        {
            REVENUE: [282.836e9, 307.394e9, 350.018e9, 402.836e9],
            REVENUE_GROWTH: [None, 0.0868, 0.1387, 0.1509],
            GROSS_MARGIN: [0.5538, 0.5694, 0.5813, 0.5965],
            OPERATING_MARGIN: [0.2646, 0.2742, 0.3231, 0.320326],
            OPERATING_TAX_RATE: [0.159, 0.139, 0.164, 0.167831],
            ROIC: [0.22, 0.23, 0.29, 0.2796266],
        },
        index=periods,
    )
    ttm_periods = tuple(pd.to_datetime([
        "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30",
    ]))
    anchors = HistoricalDCFAnchors(
        revenue_cagr={
            3: RevenueCAGRResult(
                0.125117, True, periods[0], periods[3], 3, None,
                282.836e9, 402.836e9,
            )
        },
        annual_sales_to_capital={
            periods[3]: SalesToCapitalResult(
                0.441415, True, periods[2], periods[3], 1, None,
            )
        },
        normalized_sales_to_capital={
            3: SalesToCapitalResult(
                0.667022, True, periods[0], periods[3], 3, None,
            )
        },
    )
    return FundamentalHistory(
        annual=annual,
        ttm={
            REVENUE: TTMResult(445.867e9, True, ttm_periods, None),
            OPERATING_MARGIN: TTMResult(
                0.3311032213642185, True, ttm_periods, None
            ),
        },
        annual_reasons=pd.DataFrame(index=periods),
        dcf_anchors=anchors,
    )


def anchors(ticker="GOOGL"):
    return RevenueForecastAnchors(
        ticker=ticker,
        issuer_ticker="GOOGL",
        current_revenue_base=445.867e9,
        base_period=pd.Timestamp("2026-06-30"),
        base_kind="ttm",
        latest_actual_fiscal_revenue=402.836e9,
        latest_actual_fiscal_period=pd.Timestamp("2025-12-31"),
        points=(
            ForecastAnchorPoint(
                1, pd.Timestamp("2026-12-31"), 497.7158482e9,
                0.2355297, "fixture_consensus", pd.Timestamp("2026-08-19"),
                51, True, None,
            ),
            ForecastAnchorPoint(
                2, pd.Timestamp("2027-12-31"), 607.27471188e9,
                0.2201233, "fixture_consensus", pd.Timestamp("2026-08-19"),
                54, True, None,
            ),
            ForecastAnchorPoint(
                3, pd.Timestamp("2028-12-31"), None, None,
                "fixture_consensus", pd.Timestamp("2026-08-19"), None,
                False, "unavailable",
            ),
        ),
        source="fixture_consensus",
        warnings=("ttm_base_not_directly_comparable_to_fiscal_consensus",),
    )


def research(ticker="GOOGL"):
    return build_alphabet_research_profile(
        current_assumptions(), history(), revenue_anchors=anchors(ticker),
        retrieved_at="2026-08-19",
    )


def inputs(ticker="GOOGL"):
    shares = NormalizedShareCount(
        ticker=ticker, shares_outstanding=12.1e9, source="fixture",
        source_period=pd.Timestamp("2026-06-30"),
        scope="consolidated_common", method="fixture", components=(),
        warnings=("multi_class_issuer",), available=True, reason=None,
    )
    return RealCompanyDCFInputs(
        ticker=ticker, starting_revenue=445.867e9,
        starting_revenue_source="ttm",
        starting_revenue_periods=tuple(pd.to_datetime([
            "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30",
        ])),
        net_debt=-80e9, net_debt_source="fixture",
        net_debt_period=pd.Timestamp("2026-06-30"),
        shares_outstanding=12.1e9, normalized_share_count=shares,
        historical_sales_to_capital_3y=0.667022,
        current_accounting_roic=0.2796266,
        statement_currency="USD", security_currency="USD",
    )


def test_alphabet_candidate_is_issuer_level_research_in_progress():
    googl = research("GOOGL").lookup.profile
    goog = research("GOOG").lookup.profile
    assert googl.issuer_id == goog.issuer_id == "ALPHABET_INC"
    assert googl.profile_status == goog.profile_status == "research_in_progress"
    assert googl.last_reviewed_at is goog.last_reviewed_at is None
    assert build_multistage_assumptions_from_profile(googl).assumptions == (
        build_multistage_assumptions_from_profile(goog).assumptions
    )


def test_candidate_translation_is_exact_and_current_base_is_unchanged():
    current = current_assumptions()
    profile = research().lookup.profile
    translated = build_multistage_assumptions_from_profile(profile)
    assert translated.available
    assert current == current_assumptions()
    assert translated.assumptions.near_term_revenue_growth == (0.22, 0.17, 0.13)
    assert translated.assumptions.revenue_fade_years == 8
    assert translated.assumptions.forecast_years == 11
    assert translated.assumptions.starting_operating_margin == pytest.approx(
        0.3311032213642185
    )
    assert translated.assumptions.mature_operating_margin == pytest.approx(0.32)
    assert translated.assumptions.starting_sales_to_capital == pytest.approx(0.45)
    assert translated.assumptions.mature_sales_to_capital == pytest.approx(0.60)
    assert translated.assumptions.operating_tax_rate == pytest.approx(0.17)
    assert translated.assumptions.wacc == pytest.approx(0.0975)
    assert translated.assumptions.terminal_growth == pytest.approx(0.0325)


def test_evidence_is_auditable_and_does_not_drive_translation_automatically():
    profile = research().lookup.profile
    ids = {item.evidence_id for item in profile.evidence_items}
    assert {
        "q2_2026_revenue", "search_q2_growth", "cloud_q2_growth",
        "h1_2026_capex", "2026_capex_guidance", "h1_2026_depreciation",
        "cloud_backlog", "search_ai_disruption",
    }.issubset(ids)
    assumptions = (
        profile.revenue_framework.year1_growth,
        profile.revenue_framework.year2_growth,
        profile.revenue_framework.year3_growth,
        profile.margin_framework.mature_operating_margin,
        profile.capital_efficiency_framework.starting_sales_to_capital,
        profile.capital_efficiency_framework.mature_sales_to_capital,
        profile.wacc_framework.research_wacc,
    )
    assert all(item.evidence_references for item in assumptions)
    assert all(set(item.evidence_references).issubset(ids) for item in assumptions)
    changed = replace(
        profile,
        evidence_items=(replace(profile.evidence_items[0], value=999e9),)
        + profile.evidence_items[1:],
    )
    assert build_multistage_assumptions_from_profile(changed).assumptions == (
        build_multistage_assumptions_from_profile(profile).assumptions
    )


def test_revenue_and_segment_evidence_tables_cover_required_context():
    result = research()
    annual_rows = [row for row in result.revenue_evidence if row.label.startswith("FY ended")]
    assert len(annual_rows) == 4
    assert any(row.label == "Current validated TTM" for row in result.revenue_evidence)
    assert sum("consensus" in row.label for row in result.revenue_evidence) == 2
    segments = {row.segment for row in result.segment_evidence}
    assert {
        "Google Search & other", "YouTube ads",
        "Subscriptions/platforms/devices", "Google Cloud", "Other Bets",
    }.issubset(segments)


def test_candidate_preview_runs_full_dcf_and_share_classes_reconcile():
    assumptions = build_multistage_assumptions_from_profile(
        research().lookup.profile
    ).assumptions
    googl = run_multistage_dcf(inputs("GOOGL"), assumptions)
    goog = run_multistage_dcf(inputs("GOOG"), assumptions)
    assert len(googl.forecast_path.years) == 11
    assert len(googl.operating_forecast.years) == 11
    assert googl.terminal_value.terminal_value > 0
    assert googl.per_share_value.intrinsic_value_per_share == pytest.approx(
        goog.per_share_value.intrinsic_value_per_share
    )


def test_terminal_economics_reconcile_and_are_not_pathological():
    profile = research().lookup.profile
    terminal = profile.terminal_framework
    expected_roic = 0.32 * (1 - 0.17) * 0.60
    assert terminal.terminal_roic == pytest.approx(expected_roic)
    assert terminal.terminal_reinvestment_rate == pytest.approx(
        0.0325 / expected_roic
    )
    assert terminal.terminal_fcff_conversion == pytest.approx(
        1 - terminal.terminal_reinvestment_rate
    )
    assert 0 < terminal.terminal_reinvestment_rate < 1


def test_profile_contains_no_application_or_recommendation_semantics():
    profile = research().lookup.profile
    text = " ".join((
        profile.business_summary,
        profile.rationale,
        *profile.uncertainty_notes,
        *profile.business_context.major_profile_risks,
    )).lower()
    assert "buy" not in text
    assert "sell" not in text
    assert "recommend" not in text
    assert "approved" not in text
    assert "candidate_not_applied_to_live_dcf" in profile.warnings
