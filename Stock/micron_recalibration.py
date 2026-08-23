"""Pure Micron rolling-period alignment for the Phase 4.1 research profile."""

from dataclasses import dataclass


TTM_END = "2026-05-28"
TTM_REVENUE = 90.274e9
FY2025_Q4_REVENUE = 11.315e9
FY2026_YTD_REVENUE = 78.959e9
FY2026_Q4_CONSENSUS = 50.41324e9
FY2027_Q1_CONSENSUS = 56.30485e9
FY2026_CONSENSUS = 129.39528e9
FY2027_CONSENSUS = 249.49006e9
FY2028_CONSENSUS = 281.30581e9
FY2029_CONSENSUS = 344.814e9


@dataclass(frozen=True)
class FiscalRevenueConsensus:
    fiscal_year: str
    revenue: float
    growth: float | None
    analyst_count: int | None
    source: str
    retrieved_at: str


@dataclass(frozen=True)
class RollingForecastYear:
    year_index: int
    period: str
    revenue: float
    growth: float
    quarters: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GrowthPathCase:
    name: str
    growth: tuple[float, float, float]
    rationale: str


@dataclass(frozen=True)
class MicronPeriodAlignment:
    ttm_period: str
    ttm_revenue: float
    fiscal_consensus: tuple[FiscalRevenueConsensus, ...]
    rolling_years: tuple[RollingForecastYear, ...]
    candidate_growth: tuple[float, float, float]
    growth_cases: tuple[GrowthPathCase, ...]
    old_y1_implied_revenue: float
    old_y1_alignment_error: bool
    interpolation_method: str


def _linear_quarters_from_known_first(
    first_quarter: float, annual_total: float
) -> tuple[float, float, float, float]:
    """Use the unique linear path whose Q1 and four-quarter sum are known."""
    step = (annual_total - 4 * first_quarter) / 6
    return tuple(first_quarter + index * step for index in range(4))


def _linear_quarters_after_prior_q4(
    prior_q4: float, annual_total: float
) -> tuple[float, float, float, float]:
    """Linearly extend from prior Q4 while matching the next fiscal total."""
    step = (annual_total - 4 * prior_q4) / 10
    return tuple(prior_q4 + (index + 1) * step for index in range(4))


def build_micron_period_alignment(
    *, retrieved_at: str = "2026-08-23"
) -> MicronPeriodAlignment:
    source = "FactSet consensus via finanzen.net"
    consensus = (
        FiscalRevenueConsensus("FY2026", FY2026_CONSENSUS, None, 45, source, retrieved_at),
        FiscalRevenueConsensus(
            "FY2027", FY2027_CONSENSUS,
            FY2027_CONSENSUS / FY2026_CONSENSUS - 1, 45, source, retrieved_at,
        ),
        FiscalRevenueConsensus(
            "FY2028", FY2028_CONSENSUS,
            FY2028_CONSENSUS / FY2027_CONSENSUS - 1, None, source, retrieved_at,
        ),
    )
    fy27 = _linear_quarters_from_known_first(
        FY2027_Q1_CONSENSUS, FY2027_CONSENSUS
    )
    fy28 = _linear_quarters_after_prior_q4(fy27[-1], FY2028_CONSENSUS)
    fy29 = _linear_quarters_after_prior_q4(fy28[-1], FY2029_CONSENSUS)
    y1_revenue = FY2026_Q4_CONSENSUS + sum(fy27[:3])
    y2_revenue = fy27[-1] + sum(fy28[:3])
    y3_revenue = fy28[-1] + sum(fy29[:3])
    rolling = (
        RollingForecastYear(
            1, "FY2026 Q4–FY2027 Q3", y1_revenue,
            y1_revenue / TTM_REVENUE - 1,
            (("FY2026 Q4", FY2026_Q4_CONSENSUS),)
            + tuple((f"FY2027 Q{i}", value) for i, value in enumerate(fy27[:3], 1)),
        ),
        RollingForecastYear(
            2, "FY2027 Q4–FY2028 Q3", y2_revenue,
            y2_revenue / y1_revenue - 1,
            (("FY2027 Q4", fy27[-1]),)
            + tuple((f"FY2028 Q{i}", value) for i, value in enumerate(fy28[:3], 1)),
        ),
        RollingForecastYear(
            3, "FY2028 Q4–FY2029 Q3", y3_revenue,
            y3_revenue / y2_revenue - 1,
            (("FY2028 Q4", fy28[-1]),)
            + tuple((f"FY2029 Q{i}", value) for i, value in enumerate(fy29[:3], 1)),
        ),
    )
    cases = (
        GrowthPathCase(
            "Conservative", (1.40, .15, .05),
            "Haircuts the aligned revenue ramp and assumes earlier pricing normalization.",
        ),
        GrowthPathCase(
            "Central", (1.55, .20, .15),
            "Rounds the aligned 156.4% / 20.2% / 15.4% rolling path conservatively.",
        ),
        GrowthPathCase(
            "High", (1.65, .25, .20),
            "Allows stronger pricing, HBM mix and contracted-volume realization.",
        ),
    )
    old_implied = TTM_REVENUE * 1.45
    return MicronPeriodAlignment(
        "FY2025 Q4–FY2026 Q3 (ended 2026-05-28)", TTM_REVENUE,
        consensus, rolling, cases[1].growth, cases, old_implied,
        abs(old_implied / FY2026_CONSENSUS - 1) < .02,
        "Known FY2026 Q4 and FY2027 Q1 consensus anchor a linear quarterly path; "
        "each later fiscal-year quarterly path extends linearly from the prior Q4 "
        "and is constrained to its fiscal consensus total. FY2029 is supplemental, "
        "low-confidence evidence used only to align DCF Y3.",
    )
