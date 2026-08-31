"""Pure Alphabet issuer-level research-candidate construction.

Live financial history, consensus anchors, and Phase 2 risk evidence are
supplied by callers.  Dated issuer evidence is descriptive and never mutates
the Current Base DCF or performs network access.
"""

from dataclasses import dataclass
import math

import pandas as pd

from Stock.alphabet_reassessment import (
    AlphabetGrowthEconomicsReassessment,
    build_alphabet_growth_economics_reassessment,
)
from Stock.beta_audit import BetaRobustnessAudit
from Stock.bottom_up_beta import BottomUpBetaResult
from Stock.company_profiles import (
    BusinessContext,
    CapitalEfficiencyResearchFramework,
    CompanyProfileLookupResult,
    CompanyResearchProfile,
    MarginResearchFramework,
    ResearchAssumption,
    ResearchEvidenceItem,
    RevenueResearchFramework,
    TerminalResearchFramework,
    WACCResearchFramework,
)
from Stock.company_research_types import (
    ConfidenceAssessment,
    ResearchRange,
    RevenueEvidenceRow,
)
from Stock.forecast_anchors import RevenueForecastAnchors
from Stock.fundamentals import (
    GROSS_MARGIN,
    OPERATING_MARGIN,
    OPERATING_TAX_RATE,
    REVENUE,
    REVENUE_GROWTH,
    ROIC,
    FundamentalHistory,
)
from Stock.valuation import MultiStageDCFAssumptions
from Stock.wacc_audit import WACCAuditResult


ALPHABET_Q2_2026_RELEASE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1652044/"
    "000165204426000066/googexhibit991q22026.htm"
)
ALPHABET_Q2_2026_10Q_URL = (
    "https://www.sec.gov/Archives/edgar/data/1652044/"
    "000165204426000071/goog-20260630.htm"
)
ALPHABET_2025_Q4_CALL_URL = (
    "https://abc.xyz/investor/events/event-details/2026/"
    "2025-Q4-Earnings-Call-2026-Dr_C033hS6/default.aspx"
)


@dataclass(frozen=True)
class SegmentEvidenceRow:
    segment: str
    period: str
    revenue: float | None
    revenue_growth: float | None
    operating_income: float | None
    operating_margin: float | None
    source: str
    notes: str = ""


@dataclass(frozen=True)
class AlphabetResearchProfileResult:
    lookup: CompanyProfileLookupResult
    revenue_evidence: tuple[RevenueEvidenceRow, ...]
    segment_evidence: tuple[SegmentEvidenceRow, ...]
    growth_ranges: tuple[ResearchRange, ...]
    current_assumptions: MultiStageDCFAssumptions
    period_reconciliation: tuple[str, ...]
    reassessment: AlphabetGrowthEconomicsReassessment
    confidence_assessments: tuple[ConfidenceAssessment, ...] = ()
    warnings: tuple[str, ...] = ()


def _finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _period(value) -> str | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    return str(value) if pd.isna(timestamp) else pd.Timestamp(timestamp).date().isoformat()


def _annual_items(
    history: FundamentalHistory,
    metric: str,
    *,
    prefix: str,
    label: str,
    unit: str,
) -> tuple[ResearchEvidenceItem, ...]:
    if history.annual.empty or metric not in history.annual:
        return ()
    items = []
    for raw_period, raw_value in history.annual[metric].items():
        value = _finite(raw_value)
        if value is None:
            continue
        period = _period(raw_period)
        items.append(ResearchEvidenceItem(
            f"{prefix}_{period}", "historical_financial", label, value, unit,
            period, "FundamentalHistory annual",
        ))
    return tuple(items)


def _latest_annual(
    history: FundamentalHistory,
    metric: str,
    evidence_id: str,
    label: str,
    unit: str,
) -> ResearchEvidenceItem | None:
    values = _annual_items(
        history, metric, prefix=evidence_id, label=label, unit=unit
    )
    if not values:
        return None
    latest = values[-1]
    return ResearchEvidenceItem(
        evidence_id, latest.category, latest.label, latest.value, latest.unit,
        latest.period, latest.source,
    )


def _ttm(
    history: FundamentalHistory,
    metric: str,
    evidence_id: str,
    label: str,
    unit: str,
) -> ResearchEvidenceItem | None:
    result = history.ttm.get(metric)
    if result is None or not result.available or _finite(result.value) is None:
        return None
    period = None
    if result.periods_used:
        period = f"{_period(result.periods_used[0])} to {_period(result.periods_used[-1])}"
    return ResearchEvidenceItem(
        evidence_id, "historical_financial", label, float(result.value), unit,
        period, "FundamentalHistory validated TTM",
    )


def _anchor(history: FundamentalHistory, kind: str) -> ResearchEvidenceItem | None:
    if kind == "revenue_cagr_3y":
        result = history.dcf_anchors.revenue_cagr.get(3)
        label, unit = "Historical Revenue CAGR 3Y", "ratio"
    elif kind == "sales_to_capital_3y":
        result = history.dcf_anchors.normalized_sales_to_capital.get(3)
        label, unit = "Normalized Sales-to-Capital 3Y", "multiple"
    else:
        values = history.dcf_anchors.annual_sales_to_capital
        result = values[max(values)] if values else None
        label, unit = "Latest annual Sales-to-Capital", "multiple"
    if result is None or not result.available or _finite(result.value) is None:
        return None
    start, end = _period(result.start_period), _period(result.end_period)
    return ResearchEvidenceItem(
        kind, "historical_financial", label, float(result.value), unit,
        f"{start} to {end}" if start else end,
        "FundamentalHistory historical DCF anchors",
    )


def _assumption(
    assumption_id: str,
    value: float | int | None,
    rationale: str,
    evidence_references: tuple[str, ...],
) -> ResearchAssumption:
    return ResearchAssumption(
        assumption_id, value, "research_in_progress", rationale,
        evidence_references, None,
    )


def _forward_evidence(
    anchors: RevenueForecastAnchors | None,
) -> tuple[ResearchEvidenceItem, ...]:
    if anchors is None:
        return ()
    items = []
    for point in anchors.points:
        if not point.available or _finite(point.revenue_estimate) is None:
            continue
        items.append(ResearchEvidenceItem(
            f"fy{point.forecast_year_index}_consensus_revenue",
            "forward_consensus",
            f"FY{point.forecast_year_index} analyst Revenue consensus",
            float(point.revenue_estimate), "currency_amount",
            _period(point.fiscal_period), point.source,
            source_date=_period(point.source_as_of),
            retrieved_at=_period(point.source_as_of),
            analyst_count=point.analyst_count,
            notes="Fiscal-year evidence; not identical to the TTM-based DCF year.",
        ))
    return tuple(items)


def _external_evidence(retrieved_at: str) -> tuple[ResearchEvidenceItem, ...]:
    release = ALPHABET_Q2_2026_RELEASE_URL
    filing = ALPHABET_Q2_2026_10Q_URL
    call = ALPHABET_2025_Q4_CALL_URL
    return (
        ResearchEvidenceItem("q2_2026_revenue", "historical_financial", "Q2 2026 Revenue", 119.796e9, "currency_amount", "quarter ended 2026-06-30", release, "2026-07-22", retrieved_at, notes="24% year-over-year growth; 23% in constant currency."),
        ResearchEvidenceItem("q2_2026_operating_margin", "historical_financial", "Q2 2026 consolidated Operating Margin", 0.34, "ratio", "quarter ended 2026-06-30", release, "2026-07-22", retrieved_at),
        ResearchEvidenceItem("search_q2_growth", "company_specific_research", "Google Search & other Q2 growth", 0.17, "ratio", "quarter ended 2026-06-30", release, "2026-07-22", retrieved_at, notes="Revenue was $63.271B; paid clicks increased 13% and CPC increased 3%."),
        ResearchEvidenceItem("youtube_q2_growth", "company_specific_research", "YouTube ads Q2 growth", 0.13, "ratio", "quarter ended 2026-06-30", release, "2026-07-22", retrieved_at, notes="Revenue was $11.055B."),
        ResearchEvidenceItem("subscriptions_q2_growth", "company_specific_research", "Subscriptions/platforms/devices Q2 growth", 0.15, "ratio", "quarter ended 2026-06-30", release, "2026-07-22", retrieved_at, notes="Revenue was $12.911B, led by subscriptions."),
        ResearchEvidenceItem("cloud_q2_growth", "company_specific_research", "Google Cloud Q2 growth", 0.82, "ratio", "quarter ended 2026-06-30", release, "2026-07-22", retrieved_at, notes="Revenue was $24.768B; growth reflected enterprise AI infrastructure, enterprise AI solutions, and core GCP."),
        ResearchEvidenceItem("cloud_q2_margin", "company_specific_research", "Google Cloud Q2 Operating Margin", 8.814 / 24.768, "ratio", "quarter ended 2026-06-30", release, "2026-07-22", retrieved_at),
        ResearchEvidenceItem("services_q2_margin", "company_specific_research", "Google Services Q2 Operating Margin", 39.544 / 94.540, "ratio", "quarter ended 2026-06-30", release, "2026-07-22", retrieved_at),
        ResearchEvidenceItem("cloud_backlog", "management_guidance", "Google Cloud revenue backlog", 513.9e9, "currency_amount", "as of 2026-06-30", filing, "2026-07-23", retrieved_at, notes="Alphabet expected just over 50% of total backlog to be recognized over the next 24 months; delivery and utilization remain constraints."),
        ResearchEvidenceItem("h1_2026_capex", "historical_financial", "H1 2026 capital expenditures", 80.6e9, "currency_amount", "six months ended 2026-06-30", filing, "2026-07-23", retrieved_at, notes="Up from $39.6B; driven by servers, networking equipment and data centers."),
        ResearchEvidenceItem("ttm_capex", "historical_financial", "TTM capital expenditures", 132.402e9, "currency_amount", "TTM ended 2026-06-30", release, "2026-07-22", retrieved_at),
        ResearchEvidenceItem("2026_capex_guidance", "management_guidance", "2026 capital expenditure guidance", "Management guided to $175B-$185B, primarily technical infrastructure, and expected supply constraints.", None, "2026", call, "2026-02-04", retrieved_at),
        ResearchEvidenceItem("h1_2026_depreciation", "historical_financial", "H1 2026 depreciation", 13.586e9, "currency_amount", "six months ended 2026-06-30", filing, "2026-07-23", retrieved_at, notes="Up from $9.485B; infrastructure not yet in service will add future depreciation."),
        ResearchEvidenceItem("technical_infrastructure_commitments", "company_specific_research", "Technical infrastructure commitments", 811.0e9, "currency_amount", "as of 2026-06-30", filing, "2026-07-23", retrieved_at, notes="Purchase commitments include technical infrastructure, inventory, content and energy; not all are near-term CapEx."),
        ResearchEvidenceItem("search_ai_monetization", "company_specific_research", "Search and AI monetization evidence", "Popular AI features were disclosed as driving Search query growth; revenue growth also reflected advertiser spending and ad-delivery improvements. Usage evidence is kept separate from monetization and no parity assumption is made.", None, "2026", release, "2026-07-22", retrieved_at, notes="Gemini models processed 22B API tokens per minute and the Gemini app had 950M monthly active users; narrative adoption evidence is not converted directly into a growth rate."),
        ResearchEvidenceItem("search_ai_disruption", "company_specific_research", "Search AI disruption risk", "Alternative AI interfaces, query mix shifts, cannibalization, regulation and different monetization models may pressure Search economics.", None, "long horizon", filing, "2026-07-23", retrieved_at),
        ResearchEvidenceItem("other_bets_drag", "historical_financial", "Other Bets and shared AI R&D drag", -7.588e9, "currency_amount", "quarter ended 2026-06-30", filing, "2026-07-23", retrieved_at, notes="Other Bets loss $1.799B plus Alphabet-level activities loss $5.789B; the latter primarily shared AI R&D."),
        ResearchEvidenceItem("global_nominal_growth_framework", "industry_reference", "Mature nominal-growth framework", "Long-run nominal global growth anchor", None, "terminal period", "Research framework", None, retrieved_at),
    )


def _wacc_evidence(
    wacc_audit: WACCAuditResult | None,
    beta_audit: BetaRobustnessAudit | None,
    bottom_up_beta: BottomUpBetaResult | None,
    retrieved_at: str,
) -> tuple[ResearchEvidenceItem, ...]:
    items = []
    if wacc_audit is not None and wacc_audit.available:
        items.extend((
            ResearchEvidenceItem("formula_based_wacc", "market_risk", "Formula-Based WACC", wacc_audit.calculated_wacc, "ratio", wacc_audit.risk_free_period, "Phase 2 WACC audit", wacc_audit.risk_free_period, retrieved_at, notes=f"Rf {wacc_audit.risk_free_rate:.4%}; ERP {wacc_audit.equity_risk_premium:.4%}; ticker beta {wacc_audit.beta:.3f}."),
            ResearchEvidenceItem("historical_raw_beta", "market_risk", "Historical Raw Beta", wacc_audit.beta, "beta", wacc_audit.risk_free_period, wacc_audit.beta_source or "Phase 2 beta audit", wacc_audit.risk_free_period, retrieved_at),
        ))
    if beta_audit is not None and beta_audit.production_estimate.available:
        estimate = beta_audit.production_estimate
        items.append(ResearchEvidenceItem("historical_adjusted_beta", "market_risk", "Historical Adjusted Beta", estimate.adjusted_beta, "beta", _period(estimate.end_date), "Phase 2 beta robustness audit", _period(estimate.end_date), retrieved_at, notes="Ticker-specific evidence; not a separate issuer WACC assumption."))
    if bottom_up_beta is not None:
        items.extend((
            ResearchEvidenceItem("bottom_up_beta_median", "market_risk", "Bottom-Up Beta median", bottom_up_beta.relevered_beta_median, "beta", retrieved_at, "Phase 2 bottom-up beta audit", retrieved_at, retrieved_at),
            ResearchEvidenceItem("bottom_up_adjusted_beta_median", "market_risk", "Adjusted Bottom-Up Beta median", bottom_up_beta.adjusted_relevered_beta_median, "beta", retrieved_at, "Phase 2 bottom-up beta audit", retrieved_at, retrieved_at),
        ))
    present = {item.evidence_id for item in items}
    for evidence_id, label in (
        ("formula_based_wacc", "Formula-Based WACC"),
        ("historical_raw_beta", "Historical Raw Beta"),
        ("historical_adjusted_beta", "Historical Adjusted Beta"),
        ("bottom_up_beta_median", "Bottom-Up Beta median"),
        ("bottom_up_adjusted_beta_median", "Adjusted Bottom-Up Beta median"),
    ):
        if evidence_id not in present:
            items.append(ResearchEvidenceItem(evidence_id, "market_risk", label, None, None, None, "Phase 2 WACC evidence unavailable in this run", retrieved_at=retrieved_at, available=False))
    return tuple(items)


def _revenue_rows(
    history: FundamentalHistory,
    anchors: RevenueForecastAnchors | None,
    external: tuple[ResearchEvidenceItem, ...],
    retrieved_at: str,
) -> tuple[RevenueEvidenceRow, ...]:
    rows = []
    revenues = _annual_items(history, REVENUE, prefix="annual_revenue", label="Annual Revenue", unit="currency_amount")
    growth = {item.period: item.value for item in _annual_items(history, REVENUE_GROWTH, prefix="annual_growth", label="Annual Revenue growth", unit="ratio")}
    for item in revenues[-4:]:
        rows.append(RevenueEvidenceRow(f"FY ended {item.period}", item.period, _finite(item.value), _finite(growth.get(item.period)), item.source, item.period, retrieved_at))
    ttm = _ttm(history, REVENUE, "ttm_revenue", "TTM Revenue", "currency_amount")
    if ttm is not None:
        rows.append(RevenueEvidenceRow("Current validated TTM", ttm.period, _finite(ttm.value), None, ttm.source, ttm.period, retrieved_at, notes="Starting base for DCF Year 1; not a fiscal-year endpoint."))
    if anchors is not None:
        for point in anchors.points[:2]:
            rows.append(RevenueEvidenceRow(f"FY{point.forecast_year_index} consensus", _period(point.fiscal_period), _finite(point.revenue_estimate), _finite(point.implied_revenue_growth), point.source, _period(point.source_as_of), _period(point.source_as_of), point.analyst_count, "Fiscal consensus; period differs from the TTM-based DCF year."))
    quarter = next(item for item in external if item.evidence_id == "q2_2026_revenue")
    rows.append(RevenueEvidenceRow("Latest reported quarter", quarter.period, _finite(quarter.value), 0.24, quarter.source, quarter.source_date, quarter.retrieved_at, notes=quarter.notes))
    cagr = _anchor(history, "revenue_cagr_3y")
    if cagr is not None:
        rows.append(RevenueEvidenceRow("Historical Revenue CAGR 3Y", cagr.period, None, _finite(cagr.value), cagr.source, cagr.period, retrieved_at))
    return tuple(rows)


def _segment_rows() -> tuple[SegmentEvidenceRow, ...]:
    source = ALPHABET_Q2_2026_RELEASE_URL
    return (
        SegmentEvidenceRow("Google Search & other", "Q2 2026", 63.271e9, 0.17, None, None, source, "Paid clicks +13%; CPC +3%."),
        SegmentEvidenceRow("YouTube ads", "Q2 2026", 11.055e9, 0.13, None, None, source, "Direct-response followed by brand advertising drove growth."),
        SegmentEvidenceRow("Subscriptions/platforms/devices", "Q2 2026", 12.911e9, 0.15, None, None, source, "Growth led by subscriptions."),
        SegmentEvidenceRow("Google Services", "Q2 2026", 94.540e9, 0.15, 39.544e9, 39.544 / 94.540, source, "Search economics remain the consolidated profit foundation."),
        SegmentEvidenceRow("Google Cloud", "Q2 2026", 24.768e9, 0.82, 8.814e9, 8.814 / 24.768, source, "Enterprise AI infrastructure/solutions and core GCP drove acceleration."),
        SegmentEvidenceRow("Other Bets", "Q2 2026", 0.382e9, 0.024, -1.799e9, None, source, "Small Revenue scale with continuing operating loss."),
    )


def build_alphabet_research_profile(
    current_assumptions: MultiStageDCFAssumptions,
    history: FundamentalHistory,
    *,
    revenue_anchors: RevenueForecastAnchors | None = None,
    wacc_audit: WACCAuditResult | None = None,
    beta_audit: BetaRobustnessAudit | None = None,
    bottom_up_beta: BottomUpBetaResult | None = None,
    retrieved_at: str = "2026-08-19",
) -> AlphabetResearchProfileResult:
    """Build one unreviewed ALPHABET_INC candidate for GOOG and GOOGL."""
    annual_revenue = _latest_annual(history, REVENUE, "latest_annual_revenue", "Latest annual Revenue", "currency_amount")
    ttm_revenue = _ttm(history, REVENUE, "ttm_revenue", "TTM Revenue", "currency_amount")
    annual_growth = _latest_annual(history, REVENUE_GROWTH, "latest_annual_growth", "Latest annual Revenue growth", "ratio")
    cagr = _anchor(history, "revenue_cagr_3y")
    latest_margin = _latest_annual(history, OPERATING_MARGIN, "latest_annual_operating_margin", "Latest annual Operating Margin", "ratio")
    ttm_margin = _ttm(history, OPERATING_MARGIN, "ttm_operating_margin", "TTM Operating Margin", "ratio")
    latest_stc = _anchor(history, "latest_sales_to_capital")
    normalized_stc = _anchor(history, "sales_to_capital_3y")
    accounting_roic = _latest_annual(history, ROIC, "accounting_roic", "Accounting ROIC", "ratio")
    annual_tax = _latest_annual(history, OPERATING_TAX_RATE, "latest_operating_tax_rate", "Latest annual Operating Tax Rate", "ratio")
    external = _external_evidence(retrieved_at)
    forward = _forward_evidence(revenue_anchors)
    risk = _wacc_evidence(wacc_audit, beta_audit, bottom_up_beta, retrieved_at)

    starting_margin_value = _finite(ttm_margin.value) if ttm_margin else None
    if starting_margin_value is None:
        starting_margin_value = current_assumptions.starting_operating_margin
    reassessment = build_alphabet_growth_economics_reassessment(
        starting_margin_value, evidence_as_of=retrieved_at
    )
    revised = reassessment.revised_candidate
    y1 = _assumption("year1_growth", revised.near_term_revenue_growth[0], "The revised 23% DCF Year 1 rate is supported by the eight-quarter acceleration to 24%, current fiscal consensus and continued Search/Cloud/subscription momentum. It still does not annualize the latest quarter and explicitly reconciles the June TTM DCF year to December fiscal estimates.", ("ttm_revenue", "fy1_consensus_revenue", "fy2_consensus_revenue", "q2_2026_revenue", "search_q2_growth", "cloud_q2_growth"))
    y2 = _assumption("year2_growth", revised.near_term_revenue_growth[1], "The revised 20% Year 2 slows less abruptly than the prior 17%. Cloud backlog and capacity installation, Search query/monetization evidence and FY2027 consensus support durability, while scale and execution risk still require normalization.", ("fy2_consensus_revenue", "cloud_backlog", "search_ai_monetization", "2026_capex_guidance"))
    y3 = _assumption("year3_growth", revised.near_term_revenue_growth[2], "The revised 17% Year 3 recognizes that Cloud mix, subscriptions and AI-enabled Search can remain material for several years, but stays below current consolidated growth because competition, comparisons and infrastructure intensity constrain extrapolation.", ("revenue_cagr_3y", "search_ai_disruption", "cloud_q2_growth", "technical_infrastructure_commitments"))
    fade = _assumption("revenue_fade_years", 8, "Eight fade years after three explicit years produce an 11-year horizon: long enough for Cloud/AI adoption and infrastructure utilization to mature without treating current acceleration as permanent.", ("cloud_backlog", "2026_capex_guidance", "search_ai_monetization", "search_ai_disruption"))
    terminal_growth = _assumption("terminal_growth", 0.0325, "3.25% reflects mature nominal global growth plus durable digital-advertising and cloud exposure, not current AI or Cloud growth, and remains well below Research WACC.", ("global_nominal_growth_framework", "search_ai_disruption"))
    starting_margin = _assumption("starting_operating_margin", starting_margin_value, "Starting margin is the exact validated TTM consolidated operating margin, not a discretionary research override.", ("ttm_operating_margin", "q2_2026_operating_margin"))
    mature_margin = _assumption("mature_operating_margin", revised.mature_operating_margin, "The revised 34% mature margin sits between current consolidated economics and Google Services/Cloud segment margins. It recognizes mix and operating leverage while retaining structural AI R&D, depreciation, infrastructure operating costs and Other Bets burdens.", ("latest_annual_operating_margin", "ttm_operating_margin", "q2_2026_operating_margin", "services_q2_margin", "cloud_q2_margin", "h1_2026_depreciation", "other_bets_drag"))
    starting_stc = _assumption("starting_sales_to_capital", revised.starting_sales_to_capital, "The revised 0.50x forward incremental input remains close to the latest 0.44x accounting result, but distinguishes contemporaneous capital under construction from capacity monetized over the forecast year.", ("latest_sales_to_capital", "sales_to_capital_3y", "h1_2026_capex", "ttm_capex", "h1_2026_depreciation", "cloud_backlog"))
    mature_stc = _assumption("mature_sales_to_capital", revised.mature_sales_to_capital, "The final 0.75x mature input modestly exceeds the normalized 3Y anchor as installed AI capacity utilization improves. It still reflects a mixed issuer—asset-light Search/YouTube plus capital-intensive Cloud/AI—and does not assume a return to pure-platform economics.", ("sales_to_capital_3y", "h1_2026_capex", "2026_capex_guidance", "technical_infrastructure_commitments", "cloud_backlog"))
    operating_tax = _assumption("operating_tax_rate", 0.17, "17% closely matches the latest annual operating tax rate and remains separate from the debt tax shield used by WACC.", ("latest_operating_tax_rate",))
    research_wacc = _assumption("research_wacc", 0.0975, "9.75% is an issuer-level long-horizon judgment supported by Formula WACC and historical/bottom-up beta evidence. It recognizes platform, regulatory, AI-capital and disruption risks without creating separate GOOG/GOOGL WACCs or mechanically selecting one beta method.", ("formula_based_wacc", "historical_raw_beta", "historical_adjusted_beta", "bottom_up_beta_median", "bottom_up_adjusted_beta_median"))
    horizon = _assumption("forecast_years", 11, "Eleven years exactly covers three explicit growth years plus the eight-year convergence period.", ("cloud_backlog", "2026_capex_guidance", "search_ai_disruption"))

    terminal_roic = revised.derived_terminal_roic
    terminal_reinvestment = 0.0325 / terminal_roic
    evidence = tuple(item for item in (annual_revenue, ttm_revenue, annual_growth, cagr, latest_margin, ttm_margin, latest_stc, normalized_stc, accounting_roic, annual_tax) if item is not None) + external + forward + risk
    profile = CompanyResearchProfile(
        ticker="GOOGL", issuer_id="ALPHABET_INC", company_name="Alphabet Inc.", profile_status="research_in_progress",
        business_summary="Alphabet issuer-level research candidate spanning Search, YouTube, subscriptions/platforms/devices, Google Cloud, AI infrastructure and Other Bets; no segment DCF is implied.",
        business_context=BusinessContext(
            business_model_summary="Advertising-funded Google Services remain the profit core; Cloud and subscriptions add recurring growth while Alphabet funds unusually large AI infrastructure and shared-model R&D.",
            primary_growth_drivers=("Search query and monetization growth", "YouTube advertising", "Google Cloud AI infrastructure and solutions", "subscriptions", "AI-enabled product surfaces"),
            primary_margin_drivers=("Search operating leverage", "Cloud margin expansion", "AI monetization", "infrastructure depreciation", "shared AI R&D", "Other Bets losses"),
            capital_intensity_notes=("H1 2026 CapEx reached $80.6B and TTM CapEx $132.4B.", "Servers, networking, data centers, energy and long-term capacity commitments make current Alphabet materially more capital intensive."),
            cyclicality_notes=("Advertising demand remains macro-sensitive.", "Cloud infrastructure demand and capacity deployment can be lumpy."),
            competitive_structure_notes=("Search faces alternative AI interfaces and monetization/cannibalization uncertainty.", "Cloud competes with AWS, Microsoft and specialized AI infrastructure providers."),
            major_profile_risks=("AI Search monetization and cannibalization", "Cloud competition", "AI CapEx and depreciation intensity", "regulatory pressure", "Other Bets losses", "custom infrastructure efficiency"),
        ),
        revenue_framework=RevenueResearchFramework(starting_revenue=ttm_revenue, latest_annual_revenue=annual_revenue, ttm_revenue=ttm_revenue, latest_annual_growth=annual_growth, historical_3y_cagr=cagr, forward_revenue_anchors=revenue_anchors, year1_growth=y1, year2_growth=y2, year3_growth=y3, revenue_fade_years=fade, terminal_growth=terminal_growth, near_term_growth_rationale="DCF years start from the validated TTM through June 2026. Fiscal consensus endpoints are level evidence and are conservatively translated rather than copied as same-period growth.", fade_rationale=fade.rationale, terminal_growth_rationale=terminal_growth.rationale, warnings=("ttm_and_fiscal_consensus_periods_differ",)),
        margin_framework=MarginResearchFramework(latest_annual_operating_margin=latest_margin, ttm_operating_margin=ttm_margin, historical_operating_margin=_annual_items(history, OPERATING_MARGIN, prefix="operating_margin", label="Operating Margin", unit="ratio"), historical_gross_margin=_annual_items(history, GROSS_MARGIN, prefix="gross_margin", label="Gross Margin", unit="ratio"), starting_operating_margin=starting_margin, mature_operating_margin=mature_margin, current_margin_rationale="Current consolidated margin benefits from Search economics and rapidly improving Cloud profitability while absorbing rising shared AI R&D and depreciation.", mature_margin_rationale=mature_margin.rationale),
        capital_efficiency_framework=CapitalEfficiencyResearchFramework(latest_sales_to_capital=latest_stc, normalized_3y_sales_to_capital=normalized_stc, accounting_roic=accounting_roic, starting_sales_to_capital=starting_stc, mature_sales_to_capital=mature_stc, implied_starting_roic=(starting_margin_value * (1 - 0.17) * revised.starting_sales_to_capital if starting_margin_value is not None else None), implied_terminal_roic=terminal_roic, starting_s2c_rationale=starting_stc.rationale, mature_s2c_rationale=mature_stc.rationale, warnings=("ai_infrastructure_cycle_materially_reduces_current_capital_efficiency",)),
        wacc_framework=WACCResearchFramework(wacc_audit=wacc_audit, research_wacc=research_wacc, rationale=research_wacc.rationale, warnings=("research_wacc_candidate_not_reviewed", "ticker_beta_is_security_specific_evidence")),
        terminal_framework=TerminalResearchFramework(terminal_growth=terminal_growth, mature_operating_margin=mature_margin, mature_sales_to_capital=mature_stc, terminal_roic=terminal_roic, terminal_reinvestment_rate=terminal_reinvestment, terminal_fcff_conversion=1 - terminal_reinvestment, terminal_growth_rationale=terminal_growth.rationale, mature_margin_rationale=mature_margin.rationale, mature_capital_efficiency_rationale=mature_stc.rationale, evidence_references=("global_nominal_growth_framework", "h1_2026_capex", "cloud_backlog", "search_ai_disruption")),
        operating_tax_rate=operating_tax, forecast_years=horizon,
        rationale="Unreviewed Alphabet issuer-level research candidate revised from the preserved Phase 3C baseline after a dated growth-durability and mature-economics reassessment; valuation is output only and market price was not used.",
        warnings=("research_candidate_not_reviewed", "candidate_not_applied_to_live_dcf", "multi_class_security_structure"), last_reviewed_at=None,
        evidence_items=evidence,
        uncertainty_notes=("AI Search monetization and potential cannibalization", "competition from alternative AI interfaces", "Cloud growth and margin durability", "AI infrastructure CapEx, depreciation and utilization", "regulatory and antitrust outcomes", "Other Bets losses", "efficiency of custom TPU/server/network investments"),
        future_scenario_drivers=("near-term consolidated growth", "Search/Cloud mix", "mature operating margin", "starting and mature Sales-to-Capital", "Research WACC"),
    )
    range_inputs = reassessment.growth_ranges
    ranges = (
        ResearchRange("year1_growth", range_inputs[0][1], range_inputs[0][2], range_inputs[0][3], "Eight-quarter acceleration and fiscal-period alignment support a low-to-mid 20s evidence range without extrapolating one quarter.", y1.evidence_references),
        ResearchRange("year2_growth", range_inputs[1][1], range_inputs[1][2], range_inputs[1][3], "Cloud backlog/capacity and FY2027 consensus support high-teens to low-20s durability while scale constrains the upper bound.", y2.evidence_references),
        ResearchRange("year3_growth", range_inputs[2][1], range_inputs[2][2], range_inputs[2][3], "No reliable third consensus year is used; segment mix supports mid/high teens but Search and AI competition widen uncertainty.", y3.evidence_references),
    )
    reconciliation = (
        "Current validated TTM ends 2026-06-30 and is the DCF starting Revenue.",
        "FY2026 consensus ends 2026-12-31, six months before DCF Year 1 ends 2027-06-30.",
        "FY2027 consensus ends 2027-12-31, six months after DCF Year 1 and six months before DCF Year 2.",
        "Candidate Y1/Y2 rates translate fiscal consensus levels to TTM-based DCF periods without claiming exact alignment.",
    )
    return AlphabetResearchProfileResult(
        CompanyProfileLookupResult(profile, True, None),
        _revenue_rows(history, revenue_anchors, external, retrieved_at),
        _segment_rows(), ranges, current_assumptions, reconciliation,
        reassessment,
        (
            ConfidenceAssessment("Y1 Growth", "High", "Eight-quarter trend and FY1 consensus are directly observable."),
            ConfidenceAssessment("Y2 Growth", "Medium", "Cloud backlog and FY2 consensus support durability, but conversion timing varies."),
            ConfidenceAssessment("Y3 Growth", "Low", "No dependable third-year consensus endpoint."),
            ConfidenceAssessment("Mature Margin", "Medium", "Segment economics are observable; long-run AI cost mix is not."),
            ConfidenceAssessment("Mature S/C", "Low", "Current capacity buildout obscures normalized incremental efficiency."),
            ConfidenceAssessment("WACC", "Medium", "Existing issuer risk framework retained without balancing."),
            ConfidenceAssessment("Terminal Economics", "Low", "Long-horizon mix and infrastructure replacement remain uncertain."),
        ),
        ("live_forward_consensus_is_supporting_evidence_not_an_assumption",),
    )
