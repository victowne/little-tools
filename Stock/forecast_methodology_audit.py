"""Pure cross-company forecast-methodology audit helpers.

This module is deliberately separated from production Company Profiles.  The
temporary audit candidates below are research instruments: they are not
registered profiles, cannot be reviewed/applied, use no market prices, and
perform no network access.
"""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.valuation import MultiStageDCFAssumptions


MethodologyFit = Literal["FIT", "FIT WITH CAUTION", "NEEDS ADAPTATION", "POOR FIT"]
EvidenceConfidence = Literal["High", "Medium", "Low"]
ExplicitPeriodAssessment = Literal[
    "sufficient", "possibly too short", "clearly too short", "not the main issue"
]
FiveYearAssessment = Literal[
    "strong case", "moderate case", "little benefit", "potentially harmful"
]
HybridAssessment = Literal["strong candidate", "possible", "unnecessary"]


@dataclass(frozen=True)
class AuditCandidateSpec:
    ticker: str
    issuer: str
    archetype: str
    near_term_growth: tuple[float, float, float]
    shadow_year4_growth: float
    shadow_year5_growth: float
    fade_years: int
    mature_margin: float
    starting_sales_to_capital: float
    mature_sales_to_capital: float
    normalized_starting_sales_to_capital: float | None
    normalized_mature_sales_to_capital: float | None
    normalized_mature_margin: float
    operating_tax_rate: float
    wacc: float
    terminal_growth: float
    candidate_status: str
    explicit_period_assessment: ExplicitPeriodAssessment
    five_year_assessment: FiveYearAssessment
    hybrid_reinvestment_assessment: HybridAssessment
    methodology_fit: MethodologyFit
    growth_method: str
    sales_to_capital_method: str
    mature_margin_assessment: str
    methodology_risk: str
    growth_confidence: EvidenceConfidence
    margin_confidence: EvidenceConfidence
    capital_efficiency_confidence: EvidenceConfidence
    wacc_confidence: EvidenceConfidence
    terminal_economics_confidence: EvidenceConfidence


@dataclass(frozen=True)
class ShadowFiveYearPath:
    ticker: str
    three_year_growth: tuple[float, float, float]
    five_year_growth: tuple[float, float, float, float, float]
    confidence: tuple[EvidenceConfidence, ...]
    assumptions: MultiStageDCFAssumptions
    purpose: str = "research_only_not_a_production_profile"


@dataclass(frozen=True)
class TerminalEconomicsAudit:
    terminal_roic: float
    terminal_reinvestment_rate: float
    terminal_fcff_to_nopat: float


@dataclass(frozen=True)
class HypothesisAudit:
    hypothesis_id: str
    result: str
    evidence_for: tuple[str, ...]
    evidence_against: tuple[str, ...]


def audit_candidate_specs() -> tuple[AuditCandidateSpec, ...]:
    """Return eight fixed audit candidates, never production profile defaults."""
    return (
        AuditCandidateSpec(
            "NVDA", "NVIDIA", "asset-light / fabless growth",
            (0.55, 0.40, 0.25), 0.20, 0.15, 8, 0.45, 1.35, 1.00,
            1.35, 1.10, 0.45, 0.17, 0.115, 0.0325,
            "Research Candidate (unchanged)", "possibly too short", "moderate case",
            "unnecessary", "FIT WITH CAUTION", "product-cycle and customer-demand path",
            "forward incremental S/C with working-capital caution",
            "justified normalization", "customer concentration and product-cycle fade",
            "High", "High", "Medium", "Medium", "Medium",
        ),
        AuditCandidateSpec(
            "GOOGL", "Alphabet", "advertising + Cloud + AI infrastructure",
            (0.23, 0.20, 0.17), 0.16, 0.14, 8, 0.34, 0.50, 0.70,
            0.55, 0.75, 0.34, 0.17, 0.0975, 0.0325,
            "Research Candidate (unchanged)", "possibly too short", "moderate case",
            "strong candidate", "NEEDS ADAPTATION", "issuer growth with slower AI/Cloud fade",
            "transition S/C is affected by front-loaded infrastructure",
            "uncertain", "CapEx-to-revenue lead-lag and Search/Cloud economic mixing",
            "High", "Medium", "Low", "Medium", "Medium",
        ),
        AuditCandidateSpec(
            "META", "Meta", "advertising + AI infrastructure + Reality Labs",
            (0.24, 0.20, 0.16), 0.14, 0.12, 8, 0.35, 0.47, 0.66,
            0.55, 0.80, 0.38, 0.16, 0.0975, 0.0325,
            "Audit Candidate only", "possibly too short", "moderate case",
            "strong candidate", "NEEDS ADAPTATION", "ad monetization plus AI investment path",
            "trailing S/C understates capacity not yet monetized",
            "likely too conservative", "AI CapEx/depreciation and Reality Labs obscure steady state",
            "High", "Medium", "Low", "Medium", "Low",
        ),
        AuditCandidateSpec(
            "MSFT", "Microsoft", "software + Azure + AI infrastructure",
            (0.18, 0.18, 0.15), 0.14, 0.12, 8, 0.42, 0.48, 0.49,
            0.55, 0.75, 0.45, 0.19, 0.0925, 0.0325,
            "Audit Candidate only", "possibly too short", "moderate case",
            "strong candidate", "NEEDS ADAPTATION", "contracted cloud plus recurring-software path",
            "consolidated S/C mixes software and infrastructure",
            "likely too conservative", "large RPO coexists with multi-year capacity installation",
            "High", "High", "Low", "Medium", "Medium",
        ),
        AuditCandidateSpec(
            "MU", "Micron", "cyclical capital-intensive memory semiconductor",
            (0.45, 0.20, -0.10), -0.15, 0.00, 7, 0.25, 1.59, 0.62,
            0.70, 0.65, 0.25, 0.15, 0.14, 0.03,
            "Cycle-aware Audit Candidate only", "not the main issue", "potentially harmful",
            "possible", "POOR FIT", "cycle-aware peak-to-trough path",
            "single-period S/C is unstable across the memory cycle",
            "uncertain", "linear margin/S-C fade cannot represent peak, trough and recovery",
            "Low", "Low", "Low", "Medium", "Low",
        ),
        AuditCandidateSpec(
            "AMZN", "Amazon", "retail + AWS + advertising + logistics",
            (0.15, 0.14, 0.12), 0.10, 0.09, 8, 0.12, 0.57, 0.83,
            0.65, 0.85, 0.14, 0.21, 0.105, 0.03,
            "Audit Candidate only", "possibly too short", "moderate case",
            "strong candidate", "NEEDS ADAPTATION", "mixed issuer-level growth path",
            "one consolidated S/C is coarse across retail, AWS and ads",
            "likely too conservative", "segment mixing makes consolidated reinvestment hard to interpret",
            "High", "Medium", "Low", "Medium", "Low",
        ),
        AuditCandidateSpec(
            "AVGO", "Broadcom", "fabless semiconductors + infrastructure software + acquisitions",
            (0.50, 0.35, 0.20), 0.15, 0.10, 8, 0.40, 2.83, 0.38,
            0.70, 0.80, 0.42, 0.18, 0.1075, 0.0325,
            "Acquisition-aware Audit Candidate only", "not the main issue", "potentially harmful",
            "possible", "POOR FIT", "AI/custom-silicon path with acquisition-base caveat",
            "accounting invested capital is distorted by acquisition goodwill/amortization",
            "uncertain", "VMware acquisition accounting breaks direct S/C comparability",
            "Medium", "Medium", "Low", "Medium", "Low",
        ),
        AuditCandidateSpec(
            "AAPL", "Apple", "mature hardware ecosystem + services",
            (0.12, 0.08, 0.06), 0.05, 0.04, 7, 0.32, 8.91, 1.00,
            1.50, 1.50, 0.32, 0.16, 0.0925, 0.0275,
            "Capital-structure-aware Audit Candidate only", "sufficient", "little benefit",
            "unnecessary", "NEEDS ADAPTATION", "mature product/services path",
            "buybacks and low/negative invested capital make accounting S/C unstable",
            "justified normalization", "capital-return policy distorts ROIC and S/C denominators",
            "Medium", "High", "Low", "Medium", "Medium",
        ),
    )


def spec_for_ticker(ticker: str) -> AuditCandidateSpec:
    normalized = ticker.strip().upper()
    if normalized == "GOOG":
        normalized = "GOOGL"
    for spec in audit_candidate_specs():
        if spec.ticker == normalized:
            return spec
    raise ValueError("ticker is outside the eight-issuer audit universe")


def build_audit_candidate(
    spec: AuditCandidateSpec,
    starting_operating_margin: float,
) -> MultiStageDCFAssumptions:
    """Build one temporary, non-registered audit candidate."""
    return MultiStageDCFAssumptions(
        forecast_years=3 + spec.fade_years,
        near_term_revenue_growth=spec.near_term_growth,
        revenue_fade_years=spec.fade_years,
        terminal_growth=spec.terminal_growth,
        starting_operating_margin=starting_operating_margin,
        mature_operating_margin=spec.mature_margin,
        starting_sales_to_capital=spec.starting_sales_to_capital,
        mature_sales_to_capital=spec.mature_sales_to_capital,
        operating_tax_rate=spec.operating_tax_rate,
        wacc=spec.wacc,
    )


def build_five_year_shadow(
    spec: AuditCandidateSpec,
    starting_operating_margin: float,
) -> ShadowFiveYearPath:
    """Add low-confidence Y4/Y5 research points before the unchanged fade.

    The two added years extend the diagnostic horizon by two years so the same
    number of fade transitions remains visible.  This helper never updates a
    Company Profile or UI/session state.
    """
    base = build_audit_candidate(spec, starting_operating_margin)
    growth = spec.near_term_growth + (
        spec.shadow_year4_growth, spec.shadow_year5_growth
    )
    assumptions = replace(
        base,
        forecast_years=base.forecast_years + 2,
        near_term_revenue_growth=growth,
    )
    return ShadowFiveYearPath(
        ticker=spec.ticker,
        three_year_growth=spec.near_term_growth,
        five_year_growth=growth,
        confidence=("High", "High", "Medium", "Low", "Low"),
        assumptions=assumptions,
    )


def build_capital_efficiency_normalization(
    assumptions: MultiStageDCFAssumptions,
    spec: AuditCandidateSpec,
) -> MultiStageDCFAssumptions | None:
    """Return a one-factor normalized S/C diagnostic when defensible."""
    start = spec.normalized_starting_sales_to_capital
    mature = spec.normalized_mature_sales_to_capital
    if start is None or mature is None or start <= 1e-9 or mature <= 1e-9:
        return None
    return replace(
        assumptions,
        starting_sales_to_capital=start,
        mature_sales_to_capital=mature,
    )


def build_mature_margin_normalization(
    assumptions: MultiStageDCFAssumptions,
    spec: AuditCandidateSpec,
) -> MultiStageDCFAssumptions:
    """Return the audit's one-factor mature-margin diagnostic."""
    return replace(
        assumptions,
        mature_operating_margin=spec.normalized_mature_margin,
    )


def terminal_economics(
    assumptions: MultiStageDCFAssumptions,
) -> TerminalEconomicsAudit:
    roic = assumptions.derived_terminal_roic
    reinvestment = assumptions.terminal_reinvestment_rate
    if reinvestment is None or not math.isfinite(reinvestment):
        raise ValueError("terminal reinvestment is unavailable")
    return TerminalEconomicsAudit(roic, reinvestment, 1 - reinvestment)


def classify_quarterly_growth(
    growth_rates: tuple[float, ...],
    *,
    cyclical: bool = False,
) -> str:
    """Conservative descriptive classification, not a forecast generator."""
    finite = tuple(float(value) for value in growth_rates if math.isfinite(value))
    if len(finite) < 4:
        return "insufficient_data"
    if cyclical:
        return "cyclical/rebounding"
    recent = finite[-4:]
    if all(later >= earlier for earlier, later in zip(recent, recent[1:])):
        return "accelerating"
    if all(later <= earlier for earlier, later in zip(recent, recent[1:])):
        return "decelerating"
    if max(recent) - min(recent) <= 0.05:
        return "stable"
    return "mixed"


def methodology_hypotheses() -> tuple[HypothesisAudit, ...]:
    """Return explicit balanced conclusions for hypotheses H1-H8."""
    return (
        HypothesisAudit("H1", "partially supported", ("Alphabet, MSFT and META have multi-year demand/capacity evidence.",), ("The existing fade already models post-Y3 growth; NVDA and AVGO shadow Y4/Y5 may be below linear fade.",)),
        HypothesisAudit("H2", "supported", ("Alphabet, META and MSFT trailing S/C fell during large front-loaded AI CapEx.", "Backlog and capacity constraints indicate revenue can lag installed capital."), ("Not every CapEx dollar will monetize efficiently; depreciation is structurally real.",)),
        HypothesisAudit("H3", "partially supported", ("Generic 20% mature margin is inappropriate for MSFT, META and platform economics.",), ("Company-specific NVDA and Alphabet margins already preserve stronger mature economics.",)),
        HypothesisAudit("H4", "supported with caution", ("NVDA remains fabless with materially higher S/C than hyperscalers.",), ("Inventory, receivables, supplier commitments and systems mix still require reinvestment.",)),
        HypothesisAudit("H5", "supported", ("Hyperscaler lead-lag, MU cyclicality, AVGO acquisitions and AAPL buybacks create different denominator problems.",), ("One Revenue/margin/ROIC identity remains useful in normalized steady state.",)),
        HypothesisAudit("H6", "supported", ("MU spans negative margins, rebound and exceptional current growth within a short history.",), ("S/C can still be used after a separately estimated mid-cycle state.",)),
        HypothesisAudit("H7", "supported", ("VMware goodwill, amortization, debt and changing consolidation base distort accounting invested capital.",), ("An adjusted operating-capital bridge could restore usefulness later.",)),
        HypothesisAudit("H8", "supported", ("AAPL's normalized 3Y S/C is negative while latest annual S/C is extremely high.",), ("Its operating cash generation and mature Revenue path remain modelable when capital structure is separated.",)),
    )

