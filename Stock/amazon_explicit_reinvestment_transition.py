"""Pure Phase 3F.3 Amazon explicit-reinvestment transition research.

The module is intentionally price-, network-, and Streamlit-free.  It keeps
cash PP&E purchases, economic PP&E additions, and PP&E depreciation separate:
economic additions drive the capacity/depreciation cohorts and replace (rather
than supplement) Sales-to-Capital reinvestment during the explicit transition.
"""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.forecast_methodology_audit import build_audit_candidate, spec_for_ticker
from Stock.multistage_integration import MultiStageDCFRunResult, RealCompanyDCFInputs, run_multistage_dcf
from Stock.valuation import (
    MultiStageOperatingForecast,
    aggregate_enterprise_value,
    bridge_enterprise_to_equity_value,
    calculate_intrinsic_value_per_share,
    calculate_terminal_value,
    discount_operating_forecast,
)


TransitionName = Literal["slow_normalization", "central_normalization", "fast_normalization"]
Confidence = Literal["High", "Medium", "Low"]

FROZEN_MATURE_MARGIN = 0.1834
FROZEN_MATURE_SALES_TO_CAPITAL = 0.824
FROZEN_TAX_RATE = 0.21
FROZEN_WACC = 0.105
FROZEN_TERMINAL_GROWTH = 0.03
FROZEN_NEAR_TERM_GROWTH = (0.15, 0.14, 0.12)
FCFF_JUMP_THRESHOLD = 0.50

AMAZON_2022_10K = "https://www.sec.gov/Archives/edgar/data/1018724/000101872423000004/amzn-20221231.htm"
AMAZON_2023_10K = "https://www.sec.gov/Archives/edgar/data/1018724/000101872424000008/amzn-20231231.htm"
AMAZON_2024_10K = "https://www.sec.gov/Archives/edgar/data/1018724/000101872425000004/amzn-20241231.htm"
AMAZON_2025_10K = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm"
AMAZON_2026_Q2_10Q = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000026/amzn-20260630.htm"


@dataclass(frozen=True)
class FrozenMatureControls:
    mature_operating_margin: float = FROZEN_MATURE_MARGIN
    mature_sales_to_capital: float = FROZEN_MATURE_SALES_TO_CAPITAL
    operating_tax_rate: float = FROZEN_TAX_RATE
    wacc: float = FROZEN_WACC
    terminal_growth: float = FROZEN_TERMINAL_GROWTH

    @property
    def terminal_roic(self) -> float:
        return self.mature_operating_margin * (1 - self.operating_tax_rate) * self.mature_sales_to_capital

    @property
    def terminal_reinvestment_rate(self) -> float:
        return self.terminal_growth / self.terminal_roic


@dataclass(frozen=True)
class CapexTaxonomyItem:
    code: str
    metric: str
    accounting_source: str
    cash: bool | None
    adds_ppe: bool
    cash_flow_analysis: bool
    economic_investment_analysis: bool
    depreciation_bridge: bool
    included_in_hybrid_proxy: bool
    double_counting_risk: str
    notes: str


@dataclass(frozen=True)
class HistoricalCapitalEvidence:
    period: str
    revenue: float
    cash_ppe_purchases: float
    ppe_sale_proceeds_and_incentives: float
    cash_capex: float
    economic_ppe_additions: float
    finance_lease_additions: float
    build_to_suit_additions: float | None
    net_ppe: float
    construction_in_progress: float | None
    ppe_depreciation: float
    total_da_cashflow: float
    operating_cash_flow: float
    free_cash_flow: float
    change_operating_working_capital: float
    source: str

    @property
    def net_capex(self) -> float:
        return self.economic_ppe_additions - self.ppe_depreciation

    @property
    def cash_capex_to_revenue(self) -> float:
        return self.cash_capex / self.revenue

    @property
    def economic_capex_to_revenue(self) -> float:
        return self.economic_ppe_additions / self.revenue

    @property
    def depreciation_to_revenue(self) -> float:
        return self.ppe_depreciation / self.revenue

    @property
    def depreciation_to_capex(self) -> float:
        return self.ppe_depreciation / self.economic_ppe_additions


@dataclass(frozen=True)
class UsefulLifeEvidence:
    asset_class: str
    useful_life_years: str
    depreciation_method: str
    evidence: str
    source: str


@dataclass(frozen=True)
class TransitionCaseSpec:
    name: TransitionName
    economic_capex_to_revenue: tuple[float, float, float, float, float]
    working_capital_to_delta_revenue: float
    utilization_ramp: tuple[float, ...]
    placed_in_service_ramp: tuple[float, ...]
    server_useful_life: float
    legacy_depreciation_decay: float
    handoff_years: int
    confidence: Confidence
    rationale: str


@dataclass(frozen=True)
class CapitalCohortState:
    cohort_year: int
    forecast_year: int
    age: int
    installed_capital: float
    placed_in_service_fraction: float
    utilization_fraction: float
    utilized_capital: float
    depreciation: float


@dataclass(frozen=True)
class ExplicitTransitionYear:
    year: int
    revenue: float
    revenue_growth: float
    operating_margin: float
    nopat: float
    capex: float
    depreciation_amortization: float
    net_capex: float
    change_in_working_capital: float
    other_reinvestment: float
    total_reinvestment: float
    fcff: float
    capex_to_revenue: float
    depreciation_to_revenue: float
    depreciation_to_capex: float
    fcff_to_nopat: float | None
    sales_to_capital_reinvestment: float
    hybrid_minus_sales_to_capital: float
    implied_sales_to_capital: float | None
    installed_capital: float
    utilized_capital: float
    unutilized_capital: float
    capacity_absorbed: float
    delta_revenue_to_capacity_absorbed: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class HandoffYearDiagnostic:
    year: int
    handoff_weight_sales_to_capital: float
    reinvestment: float
    reinvestment_to_nopat: float | None
    implied_sales_to_capital: float | None
    fcff: float
    fcff_change: float
    fcff_change_ratio: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TransitionValuationResult:
    model: str
    run: MultiStageDCFRunResult
    explicit_years: tuple[ExplicitTransitionYear, ...]
    handoff: tuple[HandoffYearDiagnostic, ...]
    cumulative_explicit_reinvestment: float
    cumulative_explicit_fcff: float
    warnings: tuple[str, ...]


def frozen_mature_controls() -> FrozenMatureControls:
    return FrozenMatureControls()


def capex_taxonomy() -> tuple[CapexTaxonomyItem, ...]:
    """Accounting/economic roles; items are never blindly added together."""
    return (
        CapexTaxonomyItem("A", "Cash purchases of PP&E", "investing cash flow", True, True, True, False, False, False, "overlaps total additions", "Gross cash purchases; cash FCF uses net purchases after proceeds/incentives."),
        CapexTaxonomyItem("B", "Total net PP&E additions", "segment PP&E footnote", None, True, False, True, True, True, "already contains C/D and payable timing", "Primary economic-capacity and Hybrid proxy."),
        CapexTaxonomyItem("C", "Finance-lease additions", "supplemental cash flow/PP&E footnote", False, True, False, True, True, False, "included in B", "Reconciliation only when B is used."),
        CapexTaxonomyItem("D", "Other financed equipment", "financing/build-to-suit footnote", False, True, False, True, True, False, "included in B where controlled", "No separate current amount is invented."),
        CapexTaxonomyItem("E", "Construction in progress", "PP&E balance footnote", False, True, False, False, False, False, "stock, not period investment", "Utilization proxy only."),
        CapexTaxonomyItem("F", "Servers/networking", "PP&E class footnote", None, True, False, False, True, False, "subset of B", "Used only to support cohort lives/mix."),
        CapexTaxonomyItem("G", "Fulfillment/logistics facilities", "segment/PP&E discussion", None, True, False, False, True, False, "subset of B", "Used only to support cohort lives/mix."),
        CapexTaxonomyItem("H", "Land/buildings", "PP&E class footnote", None, True, False, False, True, False, "subset of B", "Used only to support cohort lives/mix."),
        CapexTaxonomyItem("I", "Other capitalized infrastructure", "PP&E footnote", None, True, False, False, True, False, "subset of B", "No extra Hybrid charge without separate evidence."),
    )


def historical_capital_evidence() -> tuple[HistoricalCapitalEvidence, ...]:
    """Fixed issuer disclosures in USD; TTM is FY2025 + H1'26 - H1'25."""
    rows = (
        ("2021", 469.822, 61.053, 5.653, 55.400, 72.325, 7.061, 9.251, 160.281, 24.895, 22.909, 34.433, 46.327, -9.073, 10.593, AMAZON_2022_10K),
        ("2022", 513.983, 63.645, 5.345, 58.300, 60.836, .675, 3.220, 186.715, 30.020, 24.924, 41.921, 46.752, -11.548, 7.611, AMAZON_2022_10K),
        ("2023", 574.785, 52.729, 4.596, 48.133, 48.344, .642, .357, 204.177, 28.840, 30.225, 48.663, 84.946, 36.813, -.724, AMAZON_2023_10K),
        ("2024", 637.959, 82.999, 5.341, 77.658, 85.752, .854, .097, 252.665, 46.636, 32.067, 52.795, 115.877, 38.219, 1.058, AMAZON_2024_10K),
        ("2025", 716.924, 131.819, 3.499, 128.320, 142.352, 2.911, .441, 357.025, 71.745, 41.860, 65.756, 139.514, 11.194, 4.337, AMAZON_2025_10K),
        ("TTM 2026-06-30", 775.680, 173.028, 4.021, 169.007, 202.790, 4.048, None, 446.046, None, 49.741, 75.200, 161.403, -7.604, 12.228, AMAZON_2025_10K + " + " + AMAZON_2026_Q2_10Q),
    )
    return tuple(HistoricalCapitalEvidence(
        period, *(value * 1e9 if isinstance(value, (int, float)) else value for value in values[:-1]), values[-1]
    ) for period, *values in rows)


def useful_life_evidence() -> tuple[UsefulLifeEvidence, ...]:
    return (
        UsefulLifeEvidence("Servers and networking", "5–6", "straight-line", "A subset changed from six to five years in 2025; 2024 had extended servers from five to six.", AMAZON_2025_10K),
        UsefulLifeEvidence("Heavy equipment", "10–13", "straight-line", "Heavy equipment primarily supports fulfillment and data-center infrastructure.", AMAZON_2025_10K),
        UsefulLifeEvidence("Other equipment", "3–10", "straight-line", "Primarily fulfillment equipment.", AMAZON_2025_10K),
        UsefulLifeEvidence("Buildings", "lesser of 40 or remaining building life", "straight-line", "Long-lived data-center/fulfillment shell component.", AMAZON_2025_10K),
    )


def transition_case_specs() -> tuple[TransitionCaseSpec, ...]:
    """Research paths chosen before valuation; ratios use economic additions."""
    return (
        TransitionCaseSpec("slow_normalization", (.255, .245, .225, .200, .175), .07, (.08, .32, .62, .85, 1.0), (.12, .42, .72, .92, 1.0), 6.0, .07, 3, "Low", "Investment remains near the current economic-additions intensity and capacity activates slowly."),
        TransitionCaseSpec("central_normalization", (.245, .220, .195, .170, .150), .05, (.15, .48, .76, .94, 1.0), (.25, .65, .90, 1.0), 5.5, .09, 3, "Medium", "2026 cash guidance plus the disclosed non-cash/timing premium fades as installed AI/AWS capacity is absorbed."),
        TransitionCaseSpec("fast_normalization", (.235, .200, .170, .150, .140), .03, (.25, .62, .88, 1.0), (.40, .80, 1.0), 5.0, .11, 3, "Low", "Faster placement, utilization, and CapEx normalization; retained as an upside diagnostic."),
    )


def build_frozen_assumptions(starting_operating_margin: float):
    base = build_audit_candidate(spec_for_ticker("AMZN"), starting_operating_margin)
    assumptions = replace(
        base,
        near_term_revenue_growth=FROZEN_NEAR_TERM_GROWTH,
        mature_operating_margin=FROZEN_MATURE_MARGIN,
        mature_sales_to_capital=FROZEN_MATURE_SALES_TO_CAPITAL,
        operating_tax_rate=FROZEN_TAX_RATE,
        wacc=FROZEN_WACC,
        terminal_growth=FROZEN_TERMINAL_GROWTH,
    )
    assert_frozen_controls(assumptions)
    return assumptions


def assert_frozen_controls(assumptions) -> None:
    expected = frozen_mature_controls()
    checks = (
        (assumptions.near_term_revenue_growth, FROZEN_NEAR_TERM_GROWTH, "near-term growth"),
        (assumptions.mature_operating_margin, expected.mature_operating_margin, "mature margin"),
        (assumptions.mature_sales_to_capital, expected.mature_sales_to_capital, "mature Sales-to-Capital"),
        (assumptions.operating_tax_rate, expected.operating_tax_rate, "tax"),
        (assumptions.wacc, expected.wacc, "WACC"),
        (assumptions.terminal_growth, expected.terminal_growth, "terminal growth"),
    )
    for actual, target, name in checks:
        if isinstance(target, tuple):
            valid = actual == target
        else:
            valid = math.isclose(actual, target, rel_tol=0, abs_tol=1e-12)
        if not valid:
            raise ValueError(f"Phase 3F.3 frozen control changed: {name}")


def _ramp_value(ramp: tuple[float, ...], age: int) -> float:
    return ramp[min(age, len(ramp) - 1)]


def _cohort_states(
    capex_values: tuple[float, ...],
    forecast_year: int,
    spec: TransitionCaseSpec,
) -> tuple[CapitalCohortState, ...]:
    weighted_rate = .68 / spec.server_useful_life + .20 / 11.5 + .12 / 25.0
    states = []
    for cohort_year in range(1, forecast_year + 1):
        capex = capex_values[cohort_year - 1]
        age = forecast_year - cohort_year
        placed = _ramp_value(spec.placed_in_service_ramp, age)
        utilization = min(placed, _ramp_value(spec.utilization_ramp, age))
        states.append(CapitalCohortState(
            cohort_year, forecast_year, age, capex, placed, utilization,
            capex * utilization, capex * weighted_rate * placed,
        ))
    return tuple(states)


def build_explicit_transition_path(
    standard_run: MultiStageDCFRunResult,
    spec: TransitionCaseSpec,
    *,
    starting_ppe_depreciation: float,
) -> tuple[ExplicitTransitionYear, ...]:
    assert_frozen_controls(standard_run.assumptions)
    if standard_run.inputs.ticker != "AMZN":
        raise ValueError("Amazon transition requires AMZN inputs")
    if starting_ppe_depreciation <= 0 or not math.isfinite(starting_ppe_depreciation):
        raise ValueError("starting PP&E depreciation must be finite and positive")
    years = standard_run.operating_forecast.years[:5]
    if len(years) != 5:
        raise ValueError("Amazon transition requires five explicit years")
    capex_values = tuple(year.revenue * ratio for year, ratio in zip(years, spec.economic_capex_to_revenue))
    output = []
    prior_utilized = 0.0
    for index, (year, capex) in enumerate(zip(years, capex_values), start=1):
        cohorts = _cohort_states(capex_values, index, spec)
        legacy_da = starting_ppe_depreciation * (1 - spec.legacy_depreciation_decay) ** index
        depreciation = legacy_da + sum(item.depreciation for item in cohorts)
        net_capex = capex - depreciation
        working_capital = year.delta_revenue * spec.working_capital_to_delta_revenue
        reinvestment = net_capex + working_capital
        fcff = year.nopat - reinvestment
        installed = sum(item.installed_capital for item in cohorts)
        utilized = sum(item.utilized_capital for item in cohorts)
        capacity_absorbed = utilized - prior_utilized
        prior_utilized = utilized
        implied_sc = None if abs(reinvestment) <= 1e-9 else year.delta_revenue / reinvestment
        capacity_productivity = None if abs(capacity_absorbed) <= 1e-9 else year.delta_revenue / capacity_absorbed
        warnings = []
        if reinvestment < 0:
            warnings.append("negative_total_reinvestment")
        if capacity_productivity is not None and capacity_productivity < .15:
            warnings.append("capacity_overbuild_proxy")
        if implied_sc is not None and implied_sc < 0:
            warnings.append("negative_implied_sales_to_capital")
        output.append(ExplicitTransitionYear(
            index, year.revenue, year.revenue_growth, year.operating_margin,
            year.nopat, capex, depreciation, net_capex, working_capital, 0.0,
            reinvestment, fcff, capex / year.revenue,
            depreciation / year.revenue, depreciation / capex,
            None if abs(year.nopat) <= 1e-9 else fcff / year.nopat,
            year.reinvestment, reinvestment - year.reinvestment, implied_sc,
            installed, utilized, installed - utilized, capacity_absorbed,
            capacity_productivity, tuple(warnings),
        ))
    return tuple(output)


def _rebuild_run(
    existing_run: MultiStageDCFRunResult,
    operating: MultiStageOperatingForecast,
) -> MultiStageDCFRunResult:
    assumptions = existing_run.assumptions
    discounted = discount_operating_forecast(operating, assumptions)
    terminal = calculate_terminal_value(operating, discounted, assumptions)
    enterprise = aggregate_enterprise_value(discounted, terminal, assumptions)
    equity = bridge_enterprise_to_equity_value(enterprise, existing_run.inputs.net_debt)
    if existing_run.inputs.shares_outstanding is None:
        per_share = None
        reason = existing_run.per_share_unavailable_reason
    else:
        per_share = calculate_intrinsic_value_per_share(equity, existing_run.inputs.shares_outstanding)
        reason = None
    return MultiStageDCFRunResult(
        existing_run.inputs, assumptions, existing_run.forecast_path, operating,
        discounted, terminal, enterprise, equity, per_share, reason,
    )


def apply_explicit_transition(
    standard_run: MultiStageDCFRunResult,
    explicit_years: tuple[ExplicitTransitionYear, ...],
    *,
    handoff_years: int,
    model_name: str,
) -> TransitionValuationResult:
    """Replace, never add, Hybrid reinvestment; then blend to production S/C."""
    assert_frozen_controls(standard_run.assumptions)
    if len(explicit_years) != 5:
        raise ValueError("exactly five explicit transition years are required")
    if handoff_years not in (1, 2, 3):
        raise ValueError("handoff_years must be one, two, or three")
    years = list(standard_run.operating_forecast.years)
    for index, explicit in enumerate(explicit_years):
        year = years[index]
        if not math.isclose(explicit.nopat, year.nopat, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("transition changes fixed operating economics")
        years[index] = replace(year, reinvestment=explicit.total_reinvestment, fcff=explicit.fcff)

    last = explicit_years[-1]
    if abs(last.revenue - explicit_years[-2].revenue) <= 1e-9:
        raise ValueError("cannot infer handoff capital productivity")
    anchor_inverse_sc = last.total_reinvestment / (last.revenue - explicit_years[-2].revenue)
    diagnostics = []
    previous_fcff = last.fcff
    for step in range(1, handoff_years + 1):
        index = 5 + step - 1
        if index >= len(years):
            raise ValueError("forecast is too short for requested handoff")
        current = years[index]
        direct_anchor = current.delta_revenue * anchor_inverse_sc
        weight = step / handoff_years
        reinvestment = direct_anchor * (1 - weight) + current.reinvestment * weight
        fcff = current.nopat - reinvestment
        change = fcff - previous_fcff
        denominator = max(abs(previous_fcff), abs(fcff), 1.0)
        change_ratio = abs(change) / denominator
        warnings = ("artificial_fcff_jump",) if change_ratio > FCFF_JUMP_THRESHOLD else ()
        implied_sc = None if abs(reinvestment) <= 1e-9 else current.delta_revenue / reinvestment
        diagnostics.append(HandoffYearDiagnostic(
            current.year_index, weight, reinvestment,
            None if abs(current.nopat) <= 1e-9 else reinvestment / current.nopat,
            implied_sc, fcff, change, change_ratio, warnings,
        ))
        years[index] = replace(current, reinvestment=reinvestment, fcff=fcff)
        previous_fcff = fcff

    operating = MultiStageOperatingForecast(standard_run.operating_forecast.starting_revenue, tuple(years))
    run = _rebuild_run(standard_run, operating)
    warnings = tuple(dict.fromkeys(
        warning
        for row in explicit_years for warning in row.warnings
    ) | dict.fromkeys(
        warning
        for row in diagnostics for warning in row.warnings
    ))
    return TransitionValuationResult(
        model_name, run, explicit_years, tuple(diagnostics),
        sum(row.total_reinvestment for row in explicit_years),
        sum(row.fcff for row in explicit_years), warnings,
    )


def run_transition_case(
    standard_run: MultiStageDCFRunResult,
    spec: TransitionCaseSpec,
    *,
    starting_ppe_depreciation: float,
    handoff_years: int | None = None,
) -> TransitionValuationResult:
    explicit = build_explicit_transition_path(
        standard_run, spec, starting_ppe_depreciation=starting_ppe_depreciation
    )
    return apply_explicit_transition(
        standard_run, explicit,
        handoff_years=spec.handoff_years if handoff_years is None else handoff_years,
        model_name=f"H1_{spec.name}",
    )


def run_frozen_standard_model(
    inputs: RealCompanyDCFInputs,
    *,
    starting_operating_margin: float,
) -> MultiStageDCFRunResult:
    assumptions = build_frozen_assumptions(starting_operating_margin)
    return run_multistage_dcf(inputs, assumptions)


def capex_definition_sensitivity(period: HistoricalCapitalEvidence) -> tuple[tuple[str, float], ...]:
    """Alternative definitions are displayed, not added into a preferred total."""
    return (
        ("cash_ppe_net", period.cash_capex),
        ("cash_plus_finance_lease", period.cash_capex + period.finance_lease_additions),
        ("total_economic_ppe_additions", period.economic_ppe_additions),
    )


def historical_lead_lag_evidence() -> tuple[tuple[str, int, float | None], ...]:
    """Small-sample directional correlations; never an assumption generator."""
    rows = historical_capital_evidence()[:-1]
    capex_growth = tuple(rows[i].economic_ppe_additions / rows[i - 1].economic_ppe_additions - 1 for i in range(1, len(rows)))
    revenue_growth = tuple(rows[i].revenue / rows[i - 1].revenue - 1 for i in range(1, len(rows)))
    da_growth = tuple(rows[i].ppe_depreciation / rows[i - 1].ppe_depreciation - 1 for i in range(1, len(rows)))

    def correlation(left, right):
        if len(left) < 2:
            return None
        mean_l, mean_r = sum(left) / len(left), sum(right) / len(right)
        numerator = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
        denominator = math.sqrt(sum((a - mean_l) ** 2 for a in left) * sum((b - mean_r) ** 2 for b in right))
        return None if denominator <= 1e-12 else numerator / denominator

    output = []
    for lag in (1, 2, 3):
        output.append(("capex_growth_to_revenue_growth", lag, correlation(capex_growth[:-lag], revenue_growth[lag:])))
        output.append(("capex_growth_to_da_growth", lag, correlation(capex_growth[:-lag], da_growth[lag:])))
    return tuple(output)
