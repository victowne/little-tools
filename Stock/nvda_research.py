"""Pure NVDA research-candidate construction for the Company Profile layer.

Reported history and analyst anchors are supplied by callers.  Date-stamped
management and industry evidence below is deliberately descriptive: it never
changes a DCF assumption by itself and it performs no network access.
"""

from dataclasses import dataclass
import math

import pandas as pd

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


NVIDIA_Q1_FY27_URL = (
    "https://investor.nvidia.com/news/press-release-details/2026/"
    "NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2027/"
    "default.aspx"
)
NVIDIA_FY26_10K_URL = (
    "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/"
    "e361e58a-7483-44f5-bc62-a9080ae6ec72.pdf"
)
MICROSOFT_FY26_Q3_URL = (
    "https://www.microsoft.com/en-us/Investor/events/FY-2026/"
    "earnings-fy-2026-q3"
)
ALPHABET_2025_Q4_URL = (
    "https://abc.xyz/investor/events/event-details/2026/"
    "2025-Q4-Earnings-Call-2026-Dr_C033hS6/default.aspx"
)
META_2026_Q2_URL = (
    "https://investor.atmeta.com/investor-news/press-release-details/2026/"
    "Meta-Reports-Second-Quarter-2026-Results/default.aspx"
)
AMAZON_2025_Q4_URL = (
    "https://ir.aboutamazon.com/news-release/news-release-details/2026/"
    "Amazon-com-Announces-Fourth-Quarter-Results/default.aspx"
)
AMD_2025_ANNUAL_URL = (
    "https://ir.amd.com/financial-information/sec-filings/content/"
    "0001193125-26-129106/0001193125-26-129106.pdf"
)


@dataclass(frozen=True)
class ResearchRange:
    assumption_id: str
    low: float
    central: float
    high: float
    rationale: str
    evidence_references: tuple[str, ...]


@dataclass(frozen=True)
class RevenueEvidenceRow:
    label: str
    period: str | None
    revenue: float | None
    growth: float | None
    source: str
    source_date: str | None
    retrieved_at: str | None
    analyst_count: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class NVDAResearchProfileResult:
    lookup: CompanyProfileLookupResult
    revenue_evidence: tuple[RevenueEvidenceRow, ...]
    growth_ranges: tuple[ResearchRange, ...]
    current_assumptions: MultiStageDCFAssumptions
    period_reconciliation: tuple[str, ...]
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
        latest.period, latest.source, latest.source_date, latest.retrieved_at,
        latest.analyst_count, latest.notes, latest.available,
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


def _anchor(
    history: FundamentalHistory,
    *,
    kind: str,
) -> ResearchEvidenceItem | None:
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
        assumption_id=assumption_id,
        value=value,
        status="research_in_progress",
        rationale=rationale,
        evidence_references=evidence_references,
        last_reviewed_at=None,
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


def _dated_external_evidence(retrieved_at: str) -> tuple[ResearchEvidenceItem, ...]:
    return (
        ResearchEvidenceItem(
            "q1_fy27_revenue", "management_guidance", "Q1 FY2027 Revenue",
            81.615e9, "currency_amount", "quarter ended 2026-04-26",
            NVIDIA_Q1_FY27_URL, "2026-05-20", retrieved_at,
            notes="20% sequential and 85% year-over-year growth.",
        ),
        ResearchEvidenceItem(
            "q2_fy27_revenue_guidance", "management_guidance",
            "Q2 FY2027 Revenue guidance midpoint", 91.0e9, "currency_amount",
            "quarter ending July 2026", NVIDIA_Q1_FY27_URL, "2026-05-20",
            retrieved_at, notes="Plus or minus 2%; no China Data Center compute Revenue assumed.",
        ),
        ResearchEvidenceItem(
            "q2_fy27_sequential_growth", "management_guidance",
            "Q2 FY2027 guidance implied sequential growth",
            91.0 / 81.615 - 1, "ratio", "Q2 FY2027 vs Q1 FY2027",
            NVIDIA_Q1_FY27_URL, "2026-05-20", retrieved_at,
        ),
        ResearchEvidenceItem(
            "q2_fy27_yoy_growth", "management_guidance",
            "Q2 FY2027 guidance implied year-over-year growth",
            91.0 / 46.743 - 1, "ratio", "Q2 FY2027 vs Q2 FY2026",
            NVIDIA_Q1_FY27_URL, "2026-05-20", retrieved_at,
            notes="Uses the $91B midpoint and NVIDIA-reported Q2 FY2026 Revenue.",
        ),
        ResearchEvidenceItem(
            "q1_fy27_gross_margin", "management_guidance",
            "Q1 FY2027 GAAP Gross Margin", 0.749, "ratio",
            "quarter ended 2026-04-26", NVIDIA_Q1_FY27_URL, "2026-05-20",
            retrieved_at,
        ),
        ResearchEvidenceItem(
            "q1_fy27_operating_margin", "historical_financial",
            "Q1 FY2027 GAAP Operating Margin", 53.536 / 81.615, "ratio",
            "quarter ended 2026-04-26", NVIDIA_Q1_FY27_URL, "2026-05-20",
            retrieved_at, notes="Calculated from reported GAAP operating income and Revenue.",
        ),
        ResearchEvidenceItem(
            "q2_fy27_gross_margin_guidance", "management_guidance",
            "Q2 FY2027 GAAP Gross Margin guidance", 0.749, "ratio",
            "quarter ending July 2026", NVIDIA_Q1_FY27_URL, "2026-05-20",
            retrieved_at, notes="Plus or minus 50 basis points.",
        ),
        ResearchEvidenceItem(
            "fy27_tax_guidance", "management_guidance",
            "FY2027 GAAP tax-rate guidance midpoint", 0.17, "ratio",
            "FY2027", NVIDIA_Q1_FY27_URL, "2026-05-20", retrieved_at,
            notes="Management range is 16% to 18%, excluding discrete items.",
        ),
        ResearchEvidenceItem(
            "rubin_product_cycle", "company_specific_research",
            "Blackwell-to-Rubin product-cycle evidence",
            "Blackwell demand remains strong; Vera Rubin platform and networking broaden the ramp.",
            None, "FY2027 product cycle", NVIDIA_Q1_FY27_URL, "2026-05-20",
            retrieved_at,
        ),
        ResearchEvidenceItem(
            "microsoft_ai_capex", "company_specific_research",
            "Microsoft AI infrastructure demand",
            "FY2026 capex expected near $190B; capacity constrained through at least 2026.",
            None, "Microsoft FY2026", MICROSOFT_FY26_Q3_URL, "2026-04-29",
            retrieved_at,
        ),
        ResearchEvidenceItem(
            "alphabet_ai_capex", "company_specific_research",
            "Alphabet AI infrastructure demand",
            "2026 capex guidance $175B-$185B; management expected supply constraints and described both NVIDIA Rubin GPUs and its own TPUs.",
            None, "Alphabet 2026", ALPHABET_2025_Q4_URL, "2026-02-04",
            retrieved_at,
        ),
        ResearchEvidenceItem(
            "meta_ai_capex", "company_specific_research",
            "Meta AI infrastructure demand",
            "2026 capex guidance $130B-$145B after $31.08B in Q2.",
            None, "Meta 2026", META_2026_Q2_URL, "2026-07-29", retrieved_at,
        ),
        ResearchEvidenceItem(
            "amazon_ai_capex", "company_specific_research",
            "Amazon AI infrastructure demand",
            "Amazon indicated roughly $200B of 2026 capex while its Trainium and Graviton businesses continued rapid growth.",
            None, "Amazon 2026", AMAZON_2025_Q4_URL, "2026-02-05", retrieved_at,
        ),
        ResearchEvidenceItem(
            "amd_competition", "industry_reference", "AMD accelerator competition",
            "AMD targets MI400/MI450 and Helios production in 2H 2026 with large customer deployments and a growing ROCm ecosystem.",
            None, "2026 accelerator roadmap", AMD_2025_ANNUAL_URL, "2026-05-01",
            retrieved_at,
        ),
        ResearchEvidenceItem(
            "custom_silicon_competition", "industry_reference",
            "Hyperscaler custom-silicon competition",
            "Microsoft continues deploying first-party accelerators alongside NVIDIA and AMD hardware.",
            None, "2026 demand environment", MICROSOFT_FY26_Q3_URL,
            "2026-04-29", retrieved_at,
        ),
        ResearchEvidenceItem(
            "export_restrictions", "company_specific_research",
            "Export-control exposure",
            "Q2 FY2027 guidance assumes no China Data Center compute Revenue.",
            None, "Q2 FY2027 guidance", NVIDIA_Q1_FY27_URL, "2026-05-20",
            retrieved_at,
        ),
        ResearchEvidenceItem(
            "fabless_and_working_capital", "historical_financial",
            "Fabless model and working-capital evidence",
            "NVIDIA is fabless, but inventory and receivables expanded during the systems ramp and supplier commitments remain economically relevant.",
            None, "FY2026 to Q1 FY2027", NVIDIA_FY26_10K_URL, "2026-02-25",
            retrieved_at,
        ),
    )


def _revenue_rows(
    history: FundamentalHistory,
    anchors: RevenueForecastAnchors | None,
    external: tuple[ResearchEvidenceItem, ...],
) -> tuple[RevenueEvidenceRow, ...]:
    rows = []
    revenues = _annual_items(
        history, REVENUE, prefix="annual_revenue", label="Annual Revenue",
        unit="currency_amount",
    )
    growth_by_period = {
        item.period: item.value for item in _annual_items(
            history, REVENUE_GROWTH, prefix="annual_growth",
            label="Annual Revenue growth", unit="ratio",
        )
    }
    for item in revenues[-3:]:
        rows.append(RevenueEvidenceRow(
            f"FY ended {item.period}", item.period, _finite(item.value),
            _finite(growth_by_period.get(item.period)), item.source,
            item.source_date, item.retrieved_at,
        ))
    ttm = _ttm(history, REVENUE, "ttm_revenue", "TTM Revenue", "currency_amount")
    if ttm is not None:
        rows.append(RevenueEvidenceRow(
            "Current validated TTM", ttm.period, _finite(ttm.value), None,
            ttm.source, ttm.source_date, ttm.retrieved_at,
            notes="Starting base for DCF Year 1; not a fiscal-year endpoint.",
        ))
    if anchors is not None:
        for point in anchors.points[:2]:
            rows.append(RevenueEvidenceRow(
                f"FY{point.forecast_year_index} consensus", _period(point.fiscal_period),
                _finite(point.revenue_estimate), _finite(point.implied_revenue_growth),
                point.source, _period(point.source_as_of), _period(point.source_as_of),
                point.analyst_count,
                "Fiscal consensus; period differs from the TTM-based DCF year.",
            ))
    guidance = next(item for item in external if item.evidence_id == "q2_fy27_revenue_guidance")
    rows.append(RevenueEvidenceRow(
        "Management next-quarter guidance", guidance.period,
        _finite(guidance.value), 91.0 / 46.743 - 1, guidance.source,
        guidance.source_date, guidance.retrieved_at,
        notes="Growth is midpoint YoY; do not annualize one quarter.",
    ))
    cagr = _anchor(history, kind="revenue_cagr_3y")
    if cagr is not None:
        rows.append(RevenueEvidenceRow(
            "Historical Revenue CAGR 3Y", cagr.period, None,
            _finite(cagr.value), cagr.source, cagr.source_date, cagr.retrieved_at,
        ))
    trend = next(item for item in external if item.evidence_id == "q1_fy27_revenue")
    rows.append(RevenueEvidenceRow(
        "Latest reported quarter", trend.period, _finite(trend.value), 0.85,
        trend.source, trend.source_date, trend.retrieved_at,
        notes="Reported 20% sequential growth and 85% YoY growth.",
    ))
    return tuple(rows)


def build_nvda_research_profile(
    current_assumptions: MultiStageDCFAssumptions,
    history: FundamentalHistory,
    *,
    revenue_anchors: RevenueForecastAnchors | None = None,
    wacc_audit: WACCAuditResult | None = None,
    beta_audit: BetaRobustnessAudit | None = None,
    bottom_up_beta: BottomUpBetaResult | None = None,
    retrieved_at: str = "2026-08-17",
) -> NVDAResearchProfileResult:
    """Build the unreviewed NVDA candidate without mutating current assumptions."""
    annual_revenue = _latest_annual(
        history, REVENUE, "latest_annual_revenue", "Latest annual Revenue",
        "currency_amount",
    )
    ttm_revenue = _ttm(
        history, REVENUE, "ttm_revenue", "TTM Revenue", "currency_amount"
    )
    annual_growth = _latest_annual(
        history, REVENUE_GROWTH, "latest_annual_growth",
        "Latest annual Revenue growth", "ratio",
    )
    cagr = _anchor(history, kind="revenue_cagr_3y")
    latest_margin = _latest_annual(
        history, OPERATING_MARGIN, "latest_annual_operating_margin",
        "Latest annual Operating Margin", "ratio",
    )
    ttm_margin = _ttm(
        history, OPERATING_MARGIN, "ttm_operating_margin",
        "TTM Operating Margin", "ratio",
    )
    latest_stc = _anchor(history, kind="latest_sales_to_capital")
    normalized_stc = _anchor(history, kind="sales_to_capital_3y")
    accounting_roic = _latest_annual(
        history, ROIC, "accounting_roic", "Accounting ROIC", "ratio"
    )
    annual_tax = _latest_annual(
        history, OPERATING_TAX_RATE, "latest_operating_tax_rate",
        "Latest annual Operating Tax Rate", "ratio",
    )
    external = _dated_external_evidence(retrieved_at)
    forward = _forward_evidence(revenue_anchors)

    wacc_evidence = []
    if wacc_audit is not None and wacc_audit.available:
        wacc_evidence.extend((
            ResearchEvidenceItem(
                "formula_based_wacc", "market_risk", "Formula-Based WACC",
                wacc_audit.calculated_wacc, "ratio", wacc_audit.risk_free_period,
                "Phase 2 WACC audit", wacc_audit.risk_free_period, retrieved_at,
                notes=f"Rf {wacc_audit.risk_free_rate:.4%}; ERP {wacc_audit.equity_risk_premium:.4%}; raw beta {wacc_audit.beta:.3f}.",
            ),
            ResearchEvidenceItem(
                "historical_raw_beta", "market_risk", "Historical Raw Beta",
                wacc_audit.beta, "beta", wacc_audit.risk_free_period,
                wacc_audit.beta_source or "Phase 2 beta audit",
                wacc_audit.risk_free_period, retrieved_at,
            ),
        ))
    if beta_audit is not None and beta_audit.production_estimate.available:
        production = beta_audit.production_estimate
        wacc_evidence.append(ResearchEvidenceItem(
            "historical_adjusted_beta", "market_risk", "Historical Adjusted Beta",
            production.adjusted_beta, "beta", _period(production.end_date),
            "Phase 2 beta robustness audit", _period(production.end_date), retrieved_at,
            notes="Blume adjustment is diagnostic, not an automatic WACC input.",
        ))
    if bottom_up_beta is not None:
        wacc_evidence.extend((
            ResearchEvidenceItem(
                "bottom_up_beta_median", "market_risk", "Bottom-Up Beta median",
                bottom_up_beta.relevered_beta_median, "beta", retrieved_at,
                "Phase 2 bottom-up beta audit", retrieved_at, retrieved_at,
            ),
            ResearchEvidenceItem(
                "bottom_up_adjusted_beta_median", "market_risk",
                "Adjusted Bottom-Up Beta median",
                bottom_up_beta.adjusted_relevered_beta_median, "beta", retrieved_at,
                "Phase 2 bottom-up beta audit", retrieved_at, retrieved_at,
            ),
        ))
    present_wacc_ids = {item.evidence_id for item in wacc_evidence}
    for evidence_id, label in (
        ("formula_based_wacc", "Formula-Based WACC"),
        ("historical_raw_beta", "Historical Raw Beta"),
        ("historical_adjusted_beta", "Historical Adjusted Beta"),
        ("bottom_up_beta_median", "Bottom-Up Beta median"),
        ("bottom_up_adjusted_beta_median", "Adjusted Bottom-Up Beta median"),
    ):
        if evidence_id not in present_wacc_ids:
            wacc_evidence.append(ResearchEvidenceItem(
                evidence_id, "market_risk", label, None, None, None,
                "Phase 2 WACC evidence unavailable in this run",
                retrieved_at=retrieved_at, available=False,
            ))

    y1 = _assumption(
        "year1_growth", 0.60,
        "A 55% TTM-to-DCF-Year-1 step produces a Revenue level close to the live FY2027 consensus, while recognizing that the DCF year ends three months later; Q1 results and Q2 guidance support a strong near-term run-rate without annualizing one quarter.",
        ("ttm_revenue", "fy1_consensus_revenue", "q1_fy27_revenue", "q2_fy27_revenue_guidance", "rubin_product_cycle"),
    )
    y2 = _assumption(
        "year2_growth", 0.45,
        "FY2028 consensus, continuing hyperscaler capacity constraints, and the Rubin ramp support another high-growth year, but the larger base warrants a step down from Year 1.",
        ("fy2_consensus_revenue", "microsoft_ai_capex", "alphabet_ai_capex", "meta_ai_capex", "amazon_ai_capex", "rubin_product_cycle"),
    )
    y3 = _assumption(
        "year3_growth", 0.25,
        "Year 3 begins explicit normalization as scale, AMD/custom silicon, export controls, and customer concentration make current growth rates progressively harder to sustain.",
        ("revenue_cagr_3y", "amd_competition", "custom_silicon_competition", "export_restrictions"),
    )
    fade = _assumption(
        "revenue_fade_years", 9,
        "A nine-year fade after three explicit years gives a 12-year path for extraordinary AI-infrastructure economics to converge rather than forcing rapid reversion during a multi-generation platform buildout.",
        ("rubin_product_cycle", "microsoft_ai_capex", "alphabet_ai_capex", "amd_competition", "custom_silicon_competition"),
    )
    terminal_growth = _assumption(
        "terminal_growth", 0.0325,
        "3.25% reflects mature nominal global growth and technology exposure, not current AI-cycle growth, and remains well below the Research WACC candidate.",
        ("global_nominal_growth_framework", "global_end_market_exposure"),
    )
    starting_margin_value = _finite(ttm_margin.value) if ttm_margin else None
    starting_margin = _assumption(
        "starting_operating_margin", starting_margin_value,
        "Starting margin is the current validated TTM operating margin, not a research override.",
        ("ttm_operating_margin", "q1_fy27_operating_margin"),
    )
    mature_margin = _assumption(
        "mature_operating_margin", 0.45,
        "A 45% mature margin preserves durable CUDA/platform and fabless scale economics while allowing substantial normalization from current scarcity pricing and operating leverage as systems mix and competition broaden.",
        ("ttm_operating_margin", "latest_annual_operating_margin", "q1_fy27_operating_margin", "q2_fy27_gross_margin_guidance", "amd_competition", "custom_silicon_competition"),
    )
    starting_stc = _assumption(
        "starting_sales_to_capital", 1.35,
        "1.35x sits between the latest annual and normalized 3Y accounting anchors. It preserves fabless efficiency but recognizes inventory, receivables, supplier commitments, and rack-scale systems working capital.",
        ("latest_sales_to_capital", "sales_to_capital_3y", "accounting_roic", "fabless_and_working_capital"),
    )
    mature_stc = _assumption(
        "mature_sales_to_capital", 1.00,
        "1.0x assumes long-run incremental efficiency remains above capital-intensive foundries and hyperscalers, but falls as a mature NVIDIA finances broader systems, networking, inventory, and working capital.",
        ("latest_sales_to_capital", "sales_to_capital_3y", "fabless_and_working_capital"),
    )
    operating_tax = _assumption(
        "operating_tax_rate", 0.17,
        "17% is the midpoint of current FY2027 management guidance and is kept separate from the WACC tax-shield input.",
        ("latest_operating_tax_rate", "fy27_tax_guidance"),
    )
    research_wacc = _assumption(
        "research_wacc", 0.115,
        "11.5% is a long-horizon judgment: below the unusually high raw-beta Formula WACC, but near adjusted historical/peer evidence and above the former 9% development default. It retains cyclicality, concentration, export and competition risk without mechanically averaging beta methods.",
        ("formula_based_wacc", "historical_raw_beta", "historical_adjusted_beta", "bottom_up_beta_median", "bottom_up_adjusted_beta_median"),
    )
    horizon = _assumption(
        "forecast_years", 12,
        "Twelve years accommodates three explicit high-growth years plus the full nine-year convergence to mature economics.",
        ("rubin_product_cycle", "microsoft_ai_capex", "alphabet_ai_capex", "amd_competition"),
    )

    terminal_roic = 0.45 * (1 - 0.17) * 1.0
    terminal_reinvestment = 0.0325 / terminal_roic
    evidence = tuple(item for item in (
        annual_revenue, ttm_revenue, annual_growth, cagr, latest_margin,
        ttm_margin, latest_stc, normalized_stc, accounting_roic, annual_tax,
    ) if item is not None) + external + forward + tuple(wacc_evidence)
    evidence += (
        ResearchEvidenceItem(
            "global_nominal_growth_framework", "industry_reference",
            "Mature nominal-growth framework", "Long-run nominal global growth anchor",
            None, "terminal period", "Research framework", None, retrieved_at,
        ),
        ResearchEvidenceItem(
            "global_end_market_exposure", "company_specific_research",
            "Global end-market exposure", "Data Center and Edge platforms serve global end markets",
            None, "terminal period", NVIDIA_FY26_10K_URL, "2026-02-25", retrieved_at,
        ),
    )

    profile = CompanyResearchProfile(
        ticker="NVDA", issuer_id="NVDA", company_name="NVIDIA Corporation",
        profile_status="research_in_progress",
        business_summary="NVDA research candidate: accelerated-computing platform with fabless economics, unusually strong current AI demand, and material cycle, concentration, export and competitive uncertainty.",
        business_context=BusinessContext(
            business_model_summary="Full-stack accelerated-computing platform spanning GPUs, networking, systems and CUDA software; semiconductor manufacturing is outsourced.",
            primary_growth_drivers=("Blackwell/Rubin ramp", "hyperscaler and sovereign AI capex", "inference and agentic AI", "networking and rack-scale systems"),
            primary_margin_drivers=("platform pricing power", "software ecosystem", "product and systems mix", "supply availability", "operating leverage"),
            capital_intensity_notes=("Fabless structure limits owned-fab capital needs.", "Inventory, receivables, supplier commitments and systems integration still consume capital."),
            cyclicality_notes=("Customer capex and accelerator replacement cycles can make growth and margins cyclical.",),
            competitive_structure_notes=("AMD accelerators and hyperscaler ASICs/TPUs are credible alternatives.", "CUDA and full-stack networking remain important differentiation."),
            major_profile_risks=("export restrictions", "customer concentration", "custom silicon penetration", "product-transition execution", "margin normalization"),
        ),
        revenue_framework=RevenueResearchFramework(
            starting_revenue=ttm_revenue, latest_annual_revenue=annual_revenue,
            ttm_revenue=ttm_revenue, latest_annual_growth=annual_growth,
            historical_3y_cagr=cagr, forward_revenue_anchors=revenue_anchors,
            year1_growth=y1, year2_growth=y2, year3_growth=y3,
            revenue_fade_years=fade, terminal_growth=terminal_growth,
            near_term_growth_rationale="DCF years begin from the current validated TTM base; fiscal consensus is translated as level evidence, not copied as an identical-period growth rate.",
            fade_rationale=fade.rationale,
            terminal_growth_rationale=terminal_growth.rationale,
            warnings=("ttm_and_fiscal_consensus_periods_differ",),
        ),
        margin_framework=MarginResearchFramework(
            latest_annual_operating_margin=latest_margin,
            ttm_operating_margin=ttm_margin,
            historical_operating_margin=_annual_items(history, OPERATING_MARGIN, prefix="operating_margin", label="Operating Margin", unit="ratio"),
            historical_gross_margin=_annual_items(history, GROSS_MARGIN, prefix="gross_margin", label="Gross Margin", unit="ratio"),
            starting_operating_margin=starting_margin,
            mature_operating_margin=mature_margin,
            current_margin_rationale="Current TTM and Q1 FY2027 results show sustained mid-60s operating margin, but scarcity, mix and operating leverage are unusually favorable.",
            mature_margin_rationale=mature_margin.rationale,
        ),
        capital_efficiency_framework=CapitalEfficiencyResearchFramework(
            latest_sales_to_capital=latest_stc,
            normalized_3y_sales_to_capital=normalized_stc,
            accounting_roic=accounting_roic,
            starting_sales_to_capital=starting_stc,
            mature_sales_to_capital=mature_stc,
            implied_starting_roic=(starting_margin_value * (1 - 0.17) * 1.35 if starting_margin_value is not None else None),
            implied_terminal_roic=terminal_roic,
            starting_s2c_rationale=starting_stc.rationale,
            mature_s2c_rationale=mature_stc.rationale,
            warnings=("historical_sales_to_capital_is_accounting_anchor_not_forecast",),
        ),
        wacc_framework=WACCResearchFramework(
            wacc_audit=wacc_audit, research_wacc=research_wacc,
            rationale=research_wacc.rationale,
            warnings=("research_wacc_candidate_not_reviewed", "bottom_up_beta_is_peer_sensitive"),
        ),
        terminal_framework=TerminalResearchFramework(
            terminal_growth=terminal_growth,
            mature_operating_margin=mature_margin,
            mature_sales_to_capital=mature_stc,
            terminal_roic=terminal_roic,
            terminal_reinvestment_rate=terminal_reinvestment,
            terminal_fcff_conversion=1 - terminal_reinvestment,
            terminal_growth_rationale=terminal_growth.rationale,
            mature_margin_rationale=mature_margin.rationale,
            mature_capital_efficiency_rationale=mature_stc.rationale,
            evidence_references=("global_nominal_growth_framework", "global_end_market_exposure", "fabless_and_working_capital"),
        ),
        operating_tax_rate=operating_tax,
        forecast_years=horizon,
        rationale="Unreviewed evidence-driven NVDA candidate; valuation is an output and market price was not used.",
        warnings=("research_candidate_not_reviewed", "candidate_not_applied_to_live_dcf"),
        last_reviewed_at=None,
        evidence_items=evidence,
        uncertainty_notes=(
            "Blackwell/Rubin ramp timing and supply execution",
            "hyperscaler capex durability after the current capacity buildout",
            "China/export restrictions and geographic mix",
            "AMD and custom ASIC/TPU penetration, especially in inference",
            "gross-margin and operating-margin normalization as systems mix expands",
        ),
        future_scenario_drivers=(
            "near-term growth and fade duration", "mature operating margin",
            "starting and mature Sales-to-Capital", "Research WACC",
        ),
    )

    ranges = (
        ResearchRange("year1_growth", 0.45, 0.55, 0.65,
                      "TTM/FY-period mismatch and the current product ramp create a wide but evidence-bounded range.",
                      y1.evidence_references),
        ResearchRange("year2_growth", 0.30, 0.40, 0.48,
                      "Consensus and capex evidence remain strong, while scale and supply/customer timing widen outcomes.",
                      y2.evidence_references),
        ResearchRange("year3_growth", 0.15, 0.25, 0.35,
                      "No reliable third fiscal-year consensus is available; normalization and competition dominate the range.",
                      y3.evidence_references),
    )
    reconciliation = (
        "FY2026 actual is the fiscal year ended 2026-01-31.",
        "Current TTM sums the four validated quarters through 2026-04-30 and is the DCF starting Revenue.",
        "FY2027 consensus ends 2027-01-31, three months before DCF Year 1 ends 2027-04-30.",
        "FY2028 consensus ends 2028-01-31, three months before DCF Year 2 ends 2028-04-30.",
        "Candidate growth rates translate fiscal levels into TTM-based model periods without claiming exact alignment.",
    )
    return NVDAResearchProfileResult(
        CompanyProfileLookupResult(profile, True, None),
        _revenue_rows(history, revenue_anchors, external), ranges,
        current_assumptions, reconciliation,
        ("live_forward_consensus_is_supporting_evidence_not_an_assumption",),
    )
