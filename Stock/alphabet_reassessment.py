"""Pure Alphabet growth-durability and mature-economics reassessment.

The module contains dated, issuer-level research evidence and deterministic
diagnostics.  It performs no network access, imports no UI framework, does not
use market price, and never mutates the Current Base or scenario state.
"""

from dataclasses import dataclass, replace

from Stock.valuation import MultiStageDCFAssumptions


SEC_ARCHIVE_ROOT = "https://www.sec.gov/Archives/edgar/data/1652044"


@dataclass(frozen=True)
class QuarterlyRevenueMomentum:
    quarter: str
    revenue: float
    year_over_year_growth: float
    sequential_growth: float | None
    source: str


@dataclass(frozen=True)
class SegmentMomentum:
    segment: str
    quarter: str
    revenue: float
    year_over_year_growth: float
    source: str


@dataclass(frozen=True)
class GrowthContribution:
    segment: str
    prior_year_revenue_weight: float
    year_over_year_growth: float
    consolidated_growth_contribution: float


@dataclass(frozen=True)
class TerminalEconomicsPoint:
    mature_operating_margin: float
    mature_sales_to_capital: float
    terminal_roic: float
    terminal_reinvestment_rate: float
    fcff_to_nopat: float


@dataclass(frozen=True)
class CandidateRevision:
    parameter: str
    existing_value: float | int
    revised_value: float | int
    unit: str
    evidence: str
    rationale: str


@dataclass(frozen=True)
class AlphabetGrowthEconomicsReassessment:
    existing_candidate: MultiStageDCFAssumptions
    revised_candidate: MultiStageDCFAssumptions
    quarterly_revenue: tuple[QuarterlyRevenueMomentum, ...]
    segment_momentum: tuple[SegmentMomentum, ...]
    q2_2026_growth_contributions: tuple[GrowthContribution, ...]
    terminal_economics_matrix: tuple[TerminalEconomicsPoint, ...]
    revisions: tuple[CandidateRevision, ...]
    growth_ranges: tuple[tuple[str, float, float, float], ...]
    capex_lead_lag: tuple[str, ...]
    evidence_as_of: str
    revision_note: str


def _earnings_source(accession: str, exhibit: str) -> str:
    return f"{SEC_ARCHIVE_ROOT}/{accession}/{exhibit}"


_QUARTERS = (
    ("2024 Q3", 88.268e9, 0.15, "000165204424000115", "googexhibit991q32024.htm"),
    ("2024 Q4", 96.469e9, 0.12, "000165204425000010", "googexhibit991q42024.htm"),
    ("2025 Q1", 90.234e9, 0.12, "000165204425000040", "googexhibit991q12025.htm"),
    ("2025 Q2", 96.428e9, 0.14, "000165204425000056", "googexhibit991q22025.htm"),
    ("2025 Q3", 102.346e9, 0.16, "000165204425000087", "googexhibit991q32025.htm"),
    ("2025 Q4", 113.828e9, 0.18, "000165204426000012", "googexhibit991q42025.htm"),
    ("2026 Q1", 109.896e9, 0.22, "000165204426000043", "googexhibit991q12026.htm"),
    ("2026 Q2", 119.796e9, 0.24, "000165204426000066", "googexhibit991q22026.htm"),
)


_SEGMENTS = {
    "Google Search & other": (
        (49.385e9, 44.026e9), (54.034e9, 48.020e9),
        (50.702e9, 46.156e9), (54.190e9, 48.509e9),
        (56.567e9, 49.385e9), (63.073e9, 54.034e9),
        (60.399e9, 50.702e9), (63.271e9, 54.190e9),
    ),
    "YouTube ads": (
        (8.921e9, 7.952e9), (10.473e9, 9.200e9),
        (8.927e9, 8.090e9), (9.796e9, 8.663e9),
        (10.261e9, 8.921e9), (11.383e9, 10.473e9),
        (9.883e9, 8.927e9), (11.055e9, 9.796e9),
    ),
    "Subscriptions / platforms / devices": (
        (10.656e9, 8.339e9), (11.633e9, 10.794e9),
        (10.379e9, 8.739e9), (11.203e9, 9.312e9),
        (12.870e9, 10.656e9), (13.578e9, 11.633e9),
        (12.384e9, 10.379e9), (12.911e9, 11.203e9),
    ),
    "Google Cloud": (
        (11.353e9, 8.411e9), (11.955e9, 9.192e9),
        (12.260e9, 9.574e9), (13.624e9, 10.347e9),
        (15.157e9, 11.353e9), (17.664e9, 11.955e9),
        (20.028e9, 12.260e9), (24.768e9, 13.624e9),
    ),
}


def _quarterly_rows() -> tuple[QuarterlyRevenueMomentum, ...]:
    rows = []
    previous_revenue = None
    for quarter, revenue, growth, accession, exhibit in _QUARTERS:
        sequential = (
            revenue / previous_revenue - 1 if previous_revenue is not None else None
        )
        rows.append(QuarterlyRevenueMomentum(
            quarter, revenue, growth, sequential,
            _earnings_source(accession, exhibit),
        ))
        previous_revenue = revenue
    return tuple(rows)


def _segment_rows() -> tuple[SegmentMomentum, ...]:
    rows = []
    for segment, values in _SEGMENTS.items():
        for quarter_data, (revenue, prior_year_revenue) in zip(_QUARTERS, values):
            quarter, _, _, accession, exhibit = quarter_data
            rows.append(SegmentMomentum(
                segment=segment,
                quarter=quarter,
                revenue=revenue,
                year_over_year_growth=revenue / prior_year_revenue - 1,
                source=_earnings_source(accession, exhibit),
            ))
    return tuple(rows)


def _growth_contributions() -> tuple[GrowthContribution, ...]:
    prior_total = 96.428e9
    components = (
        ("Google Search & other", 54.190e9, 63.271e9),
        ("YouTube ads", 9.796e9, 11.055e9),
        ("Subscriptions / platforms / devices", 11.203e9, 12.911e9),
        ("Google Cloud", 13.624e9, 24.768e9),
    )
    prior_disclosed = sum(item[1] for item in components)
    current_disclosed = sum(item[2] for item in components)
    components += ((
        "Other / hedging / residual",
        prior_total - prior_disclosed,
        119.796e9 - current_disclosed,
    ),)
    return tuple(
        GrowthContribution(
            segment=segment,
            prior_year_revenue_weight=prior_revenue / prior_total,
            year_over_year_growth=current_revenue / prior_revenue - 1,
            consolidated_growth_contribution=(current_revenue - prior_revenue) / prior_total,
        )
        for segment, prior_revenue, current_revenue in components
    )


def build_terminal_economics_matrix(
    *,
    margins: tuple[float, ...] = (0.32, 0.34, 0.36),
    sales_to_capital_values: tuple[float, ...] = (0.60, 0.70, 0.80),
    operating_tax_rate: float = 0.17,
    terminal_growth: float = 0.0325,
) -> tuple[TerminalEconomicsPoint, ...]:
    """Return exact terminal ROIC/reinvestment diagnostics for each combination."""
    points = []
    for margin in margins:
        for sales_to_capital in sales_to_capital_values:
            roic = margin * (1 - operating_tax_rate) * sales_to_capital
            reinvestment_rate = terminal_growth / roic
            points.append(TerminalEconomicsPoint(
                margin, sales_to_capital, roic, reinvestment_rate,
                1 - reinvestment_rate,
            ))
    return tuple(points)


def build_alphabet_growth_economics_reassessment(
    starting_operating_margin: float,
    *,
    evidence_as_of: str = "2026-08-21",
) -> AlphabetGrowthEconomicsReassessment:
    """Build the preserved Phase 3C baseline and evidence-supported revision."""
    existing = MultiStageDCFAssumptions(
        forecast_years=11,
        near_term_revenue_growth=(0.22, 0.17, 0.13),
        revenue_fade_years=8,
        terminal_growth=0.0325,
        starting_operating_margin=starting_operating_margin,
        mature_operating_margin=0.32,
        starting_sales_to_capital=0.45,
        mature_sales_to_capital=0.60,
        operating_tax_rate=0.17,
        wacc=0.0975,
    )
    revised = replace(
        existing,
        near_term_revenue_growth=(0.23, 0.20, 0.17),
        mature_operating_margin=0.34,
        starting_sales_to_capital=0.50,
        mature_sales_to_capital=0.75,
    )
    revisions = (
        CandidateRevision("Y1 Growth", 0.22, 0.23, "percent", "Eight-quarter consolidated trend; FY2026/FY2027 consensus", "23% remains below the latest 24% quarter and aligns the June-TTM DCF year with fiscal consensus levels."),
        CandidateRevision("Y2 Growth", 0.17, 0.20, "percent", "Cloud growth/backlog; Search and subscription durability", "A drop to 17% by June 2028 was inconsistent with the current 22% FY2027 consensus and capacity-led Cloud realization."),
        CandidateRevision("Y3 Growth", 0.13, 0.17, "percent", "Segment mix and multi-year Cloud capacity deployment", "Mid-to-high teens is supported, but still assumes meaningful normalization from current consolidated growth."),
        CandidateRevision("Fade Years", 8, 8, "integer", "Convergence check", "Unchanged: eight fade years still bring growth exactly to terminal growth in Year 11."),
        CandidateRevision("Forecast Horizon", 11, 11, "integer", "Convergence check", "Unchanged with the existing three explicit years plus eight fade years."),
        CandidateRevision("Mature Margin", 0.32, 0.34, "percent", "Services >40%; Cloud mid-30s; shared AI costs", "34% recognizes stronger mix economics while retaining structural depreciation, R&D and Other Bets burdens."),
        CandidateRevision("Starting S/C", 0.45, 0.50, "multiple", "Latest 0.44x; normalized 3Y 0.67x; CapEx lead-lag", "0.50x remains close to depressed accounting efficiency but treats it as a forward incremental input, not a contemporaneous balance-sheet observation."),
        CandidateRevision("Mature S/C", 0.60, 0.75, "multiple", "Mixed Search/YouTube and Cloud economics; 3Y 0.67x anchor", "0.75x modestly exceeds the depressed historical anchor as installed AI capacity normalizes, while remaining far below pure-platform economics."),
        CandidateRevision("WACC", 0.0975, 0.0975, "percent", "No new WACC-specific evidence", "Unchanged and independent of operating-assumption revisions."),
        CandidateRevision("Terminal Growth", 0.0325, 0.0325, "percent", "No new terminal macro evidence", "Unchanged; it was not used to raise valuation."),
    )
    return AlphabetGrowthEconomicsReassessment(
        existing_candidate=existing,
        revised_candidate=revised,
        quarterly_revenue=_quarterly_rows(),
        segment_momentum=_segment_rows(),
        q2_2026_growth_contributions=_growth_contributions(),
        terminal_economics_matrix=build_terminal_economics_matrix(),
        revisions=revisions,
        growth_ranges=(
            ("Y1 Growth", 0.21, 0.23, 0.25),
            ("Y2 Growth", 0.17, 0.20, 0.22),
            ("Y3 Growth", 0.14, 0.17, 0.19),
        ),
        capex_lead_lag=(
            "Capital is committed before equipment and data-center construction is complete.",
            "Assets under construction and installed capacity depress current accounting Sales-to-Capital before monetization.",
            "Depreciation begins when capacity is placed in service, while customer onboarding and utilization ramp later.",
            "Cloud backlog and capacity constraints support a revenue lead-lag, but do not guarantee efficient conversion of every CapEx dollar.",
        ),
        evidence_as_of=evidence_as_of,
        revision_note=(
            "Phase 3C baseline 22%/17%/13%, 32% margin and 0.45x/0.60x "
            "Sales-to-Capital was revised after eight-quarter momentum, consensus, "
            "Cloud backlog/capacity and CapEx lead-lag evidence. WACC, terminal "
            "growth, fade years and horizon were deliberately unchanged."
        ),
    )
