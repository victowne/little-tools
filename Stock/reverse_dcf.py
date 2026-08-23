"""Pure single-variable Reverse DCF over the production valuation engine.

Each solve changes exactly one assumption and reruns ``run_multistage_dcf``.
Market price is a target for diagnostics only; it never mutates a Company
Profile, a reviewed application, or the supplied research Base.
"""

from dataclasses import dataclass, replace
import math
import re
from typing import Callable, Literal

from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.valuation import MultiStageDCFAssumptions
from Stock.company_profiles import CompanyResearchProfile


GROWTH_UPLIFT = "near_term_growth_uplift"
MATURE_MARGIN = "mature_operating_margin"
MATURE_SALES_TO_CAPITAL = "mature_sales_to_capital"
WACC = "wacc"

ReverseVariable = Literal[
    "near_term_growth_uplift",
    "mature_operating_margin",
    "mature_sales_to_capital",
    "wacc",
]
ReverseStatus = Literal[
    "SOLVED",
    "NO_BRACKET",
    "OUTSIDE_REASONABLE_RANGE",
    "INVALID_BASE_ASSUMPTIONS",
    "VALUATION_FAILED",
    "MARKET_PRICE_UNAVAILABLE",
    "NON_MONOTONIC",
    "AMBIGUOUS",
]

SOLVED = "SOLVED"
NO_BRACKET = "NO_BRACKET"
OUTSIDE_REASONABLE_RANGE = "OUTSIDE_REASONABLE_RANGE"
INVALID_BASE_ASSUMPTIONS = "INVALID_BASE_ASSUMPTIONS"
VALUATION_FAILED = "VALUATION_FAILED"
MARKET_PRICE_UNAVAILABLE = "MARKET_PRICE_UNAVAILABLE"
NON_MONOTONIC = "NON_MONOTONIC"
AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class ReverseResearchRange:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("research range bounds must be finite")
        if self.lower > self.upper:
            raise ValueError("research range lower bound must not exceed upper bound")


@dataclass(frozen=True)
class ReverseVariableConfig:
    variable: ReverseVariable
    lower_bound: float
    upper_bound: float
    expected_direction: Literal["increasing", "decreasing"]


@dataclass(frozen=True)
class ReverseDCFResult:
    variable: ReverseVariable
    status: ReverseStatus
    market_price: float | None
    research_value: float | tuple[float, ...]
    implied_value: float | None
    research_growth_path: tuple[float, ...]
    implied_growth_path: tuple[float, ...] | None
    implied_dcf_value: float | None
    residual: float | None
    lower_bound: float
    upper_bound: float
    lower_bound_dcf: float | None
    upper_bound_dcf: float | None
    iterations: int
    monotonic: bool | None
    root_interval_count: int
    research_range: ReverseResearchRange | None
    range_relation: str
    expectation_gap: str
    enterprise_value: float | None
    equity_value: float | None
    terminal_value_share: float | None
    reason: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReverseDCFAnalysis:
    ticker: str
    base_source: str
    base_dcf_per_share: float | None
    market_price: float | None
    price_to_base_dcf: float | None
    results: tuple[ReverseDCFResult, ...]
    warnings: tuple[str, ...] = ()

    def result_for(self, variable: ReverseVariable) -> ReverseDCFResult | None:
        return next((item for item in self.results if item.variable == variable), None)


@dataclass(frozen=True)
class _Evaluation:
    x: float
    value: float | None
    enterprise_value: float | None = None
    equity_value: float | None = None
    terminal_value_share: float | None = None
    warnings: tuple[str, ...] = ()
    reason: str | None = None


def default_reverse_configs(
    assumptions: MultiStageDCFAssumptions,
) -> tuple[ReverseVariableConfig, ...]:
    """Transparent reasonable search ranges; percentage bounds are decimals."""
    return (
        ReverseVariableConfig(GROWTH_UPLIFT, -0.30, 0.50, "increasing"),
        ReverseVariableConfig(MATURE_MARGIN, 0.0, 0.80, "increasing"),
        ReverseVariableConfig(MATURE_SALES_TO_CAPITAL, 0.10, 5.0, "increasing"),
        ReverseVariableConfig(
            WACC,
            assumptions.terminal_growth + 0.005,
            0.20,
            "decreasing",
        ),
    )


def research_ranges_from_profile(
    profile: CompanyResearchProfile | None,
) -> dict[ReverseVariable, ReverseResearchRange]:
    """Read only explicitly stored margin/S-C evidence ranges.

    This intentionally does not infer a range from point assumptions. Supported
    product evidence is either a dedicated range item or the legacy Amazon
    bridge note retained from Phase 3F research.
    """
    if profile is None:
        return {}
    found: dict[ReverseVariable, ReverseResearchRange] = {}
    for evidence in profile.evidence_items:
        if not evidence.available:
            continue
        variable = None
        is_percent = False
        if evidence.evidence_id == "mature_margin_range":
            variable, is_percent = MATURE_MARGIN, True
        elif evidence.evidence_id == "mature_sc_range":
            variable = MATURE_SALES_TO_CAPITAL
        elif evidence.evidence_id == "mature_margin_bridge" and "range" in evidence.notes.lower():
            variable, is_percent = MATURE_MARGIN, True
        elif evidence.evidence_id == "mature_sc_bridge" and "range" in evidence.notes.lower():
            variable = MATURE_SALES_TO_CAPITAL
        if variable is None:
            continue
        text = f"{evidence.value or ''} {evidence.notes or ''}"
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        # Dedicated values may precede the range in a legacy note; locate the
        # two numbers after the word "range" when it is present.
        range_text = re.split(r"range", text, flags=re.IGNORECASE)[-1]
        range_numbers = re.findall(r"\d+(?:\.\d+)?", range_text)
        if len(range_numbers) >= 2:
            numbers = range_numbers
        if len(numbers) < 2:
            continue
        lower, upper = float(numbers[0]), float(numbers[1])
        if is_percent:
            lower, upper = lower / 100, upper / 100
        try:
            found[variable] = ReverseResearchRange(lower, upper)
        except ValueError:
            continue
    return found


def assumptions_for_reverse_value(
    base: MultiStageDCFAssumptions,
    variable: ReverseVariable,
    value: float,
) -> MultiStageDCFAssumptions:
    """Return a new assumptions object with exactly one reverse lever changed."""
    if variable == GROWTH_UPLIFT:
        return replace(
            base,
            near_term_revenue_growth=tuple(
                growth + value for growth in base.near_term_revenue_growth
            ),
        )
    if variable == MATURE_MARGIN:
        return replace(base, mature_operating_margin=value)
    if variable == MATURE_SALES_TO_CAPITAL:
        return replace(base, mature_sales_to_capital=value)
    if variable == WACC:
        return replace(base, wacc=value)
    raise ValueError(f"unsupported reverse variable: {variable}")


def _research_value(
    assumptions: MultiStageDCFAssumptions,
    variable: ReverseVariable,
) -> float | tuple[float, ...]:
    if variable == GROWTH_UPLIFT:
        return assumptions.near_term_revenue_growth
    return float(getattr(assumptions, variable))


def _range_relation(
    implied: float | None,
    research_range: ReverseResearchRange | None,
) -> str:
    if implied is None or research_range is None:
        return "not_available"
    if implied < research_range.lower:
        return "below_research_range"
    if implied > research_range.upper:
        return "above_research_range"
    return "within_research_range"


def _gap_classification(
    variable: ReverseVariable,
    base: MultiStageDCFAssumptions,
    implied: float | None,
    status: ReverseStatus,
) -> str:
    if status == OUTSIDE_REASONABLE_RANGE:
        return "extreme_outside_search_range"
    if status in {NON_MONOTONIC, AMBIGUOUS}:
        return "ambiguous"
    if status != SOLVED or implied is None:
        return "unresolved"
    if variable == GROWTH_UPLIFT:
        gap = implied
    else:
        gap = implied - float(getattr(base, variable))
    tolerance = 0.0025 if variable != MATURE_SALES_TO_CAPITAL else 0.05
    if abs(gap) <= tolerance:
        return "within_research_base_tolerance"
    material_threshold = (
        0.05 if variable == GROWTH_UPLIFT
        else 0.05 if variable in {MATURE_MARGIN, WACC}
        else 0.50
    )
    magnitude = "materially" if abs(gap) > material_threshold else "moderately"
    direction = "below" if gap < 0 else "above"
    return f"{magnitude}_{direction}_research_base"


def _invalid_result(
    config: ReverseVariableConfig,
    base: MultiStageDCFAssumptions,
    market_price: float | None,
    status: ReverseStatus,
    reason: str,
    *,
    research_range: ReverseResearchRange | None = None,
) -> ReverseDCFResult:
    return ReverseDCFResult(
        variable=config.variable,
        status=status,
        market_price=market_price,
        research_value=_research_value(base, config.variable),
        implied_value=None,
        research_growth_path=base.near_term_revenue_growth,
        implied_growth_path=None,
        implied_dcf_value=None,
        residual=None,
        lower_bound=config.lower_bound,
        upper_bound=config.upper_bound,
        lower_bound_dcf=None,
        upper_bound_dcf=None,
        iterations=0,
        monotonic=None,
        root_interval_count=0,
        research_range=research_range,
        range_relation="not_available",
        expectation_gap=_gap_classification(config.variable, base, None, status),
        enterprise_value=None,
        equity_value=None,
        terminal_value_share=None,
        reason=reason,
    )


def _sample_points(lower: float, upper: float, count: int) -> tuple[float, ...]:
    if count < 3:
        raise ValueError("sample_count must be at least 3")
    step = (upper - lower) / (count - 1)
    return tuple(lower + step * index for index in range(count))


def solve_reverse_variable(
    base: MultiStageDCFAssumptions,
    market_price: float | None,
    config: ReverseVariableConfig,
    value_function: Callable[[MultiStageDCFAssumptions], float],
    *,
    research_range: ReverseResearchRange | None = None,
    sample_count: int = 101,
    max_iterations: int = 100,
    value_tolerance: float | None = None,
) -> ReverseDCFResult:
    """Solve one lever after scanning the full range for monotonicity/roots.

    The scan is deliberate: bisection is used only when the valid sampled path
    is monotonic and contains exactly one root interval. This prevents a
    plausible-looking answer when an economic path has multiple solutions.
    """
    if market_price is None or not math.isfinite(float(market_price)) or market_price <= 0:
        return _invalid_result(
            config, base, market_price, MARKET_PRICE_UNAVAILABLE,
            "positive_finite_market_price_required", research_range=research_range,
        )
    target = float(market_price)
    # Solver precision is deliberately much tighter than UI currency rounding.
    # A one-cent valuation tolerance can map to a materially wide assumption
    # interval when the selected lever has a shallow valuation slope.
    tolerance = value_tolerance or max(1e-8, target * 1e-10)

    def evaluate(x: float) -> _Evaluation:
        try:
            candidate = assumptions_for_reverse_value(base, config.variable, x)
            value = float(value_function(candidate))
            if not math.isfinite(value):
                return _Evaluation(x, None, reason="non_finite_valuation")
            return _Evaluation(x, value)
        except (TypeError, ValueError, ArithmeticError, OverflowError) as exc:
            return _Evaluation(x, None, reason=f"valuation_failed:{exc}")

    sampled = tuple(evaluate(x) for x in _sample_points(
        config.lower_bound, config.upper_bound, sample_count
    ))
    valid = tuple(point for point in sampled if point.value is not None)
    lower_eval, upper_eval = sampled[0], sampled[-1]
    if len(valid) < 2:
        return replace(
            _invalid_result(
                config, base, target, VALUATION_FAILED,
                "insufficient_valid_valuations", research_range=research_range,
            ),
            lower_bound_dcf=lower_eval.value,
            upper_bound_dcf=upper_eval.value,
        )

    scale = max(1.0, max(abs(point.value or 0.0) for point in valid))
    monotonic_tolerance = scale * 1e-10
    changes = tuple(
        right.value - left.value
        for left, right in zip(valid, valid[1:])
        if left.value is not None and right.value is not None
    )
    increasing = all(change >= -monotonic_tolerance for change in changes)
    decreasing = all(change <= monotonic_tolerance for change in changes)
    monotonic = increasing or decreasing

    exact = [point for point in valid if abs((point.value or 0.0) - target) <= tolerance]
    intervals: list[tuple[_Evaluation, _Evaluation]] = []
    for left, right in zip(sampled, sampled[1:]):
        if left.value is None or right.value is None:
            continue
        left_residual = left.value - target
        right_residual = right.value - target
        if left_residual * right_residual < 0:
            intervals.append((left, right))

    # Adjacent exact samples describe one root neighborhood, not many roots.
    root_count = len(intervals)
    if exact and not intervals:
        exact_groups = 1
        for left, right in zip(exact, exact[1:]):
            if right.x - left.x > (config.upper_bound - config.lower_bound) / (sample_count - 1) * 1.5:
                exact_groups += 1
        root_count = exact_groups

    if not monotonic:
        status: ReverseStatus = AMBIGUOUS if root_count > 1 else NON_MONOTONIC
        return replace(
            _invalid_result(
                config, base, target, status,
                "sampled_valuation_path_is_not_monotonic",
                research_range=research_range,
            ),
            lower_bound_dcf=lower_eval.value,
            upper_bound_dcf=upper_eval.value,
            monotonic=False,
            root_interval_count=root_count,
        )
    if root_count > 1:
        return replace(
            _invalid_result(
                config, base, target, AMBIGUOUS,
                "multiple_root_intervals_detected", research_range=research_range,
            ),
            lower_bound_dcf=lower_eval.value,
            upper_bound_dcf=upper_eval.value,
            monotonic=True,
            root_interval_count=root_count,
        )

    if exact:
        best = min(exact, key=lambda item: abs((item.value or 0.0) - target))
        implied = best.x
        final = best
        iterations = 0
    elif len(intervals) == 1:
        left, right = intervals[0]
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            midpoint = evaluate((left.x + right.x) / 2)
            if midpoint.value is None:
                return replace(
                    _invalid_result(
                        config, base, target, VALUATION_FAILED,
                        midpoint.reason or "valuation_failed_inside_bracket",
                        research_range=research_range,
                    ),
                    lower_bound_dcf=lower_eval.value,
                    upper_bound_dcf=upper_eval.value,
                    monotonic=True,
                    root_interval_count=1,
                )
            if abs(midpoint.value - target) <= tolerance:
                left = right = midpoint
                break
            if (left.value - target) * (midpoint.value - target) <= 0:
                right = midpoint
            else:
                left = midpoint
        final = min((left, right), key=lambda item: abs((item.value or 0.0) - target))
        implied = final.x
    else:
        values = tuple(point.value for point in valid if point.value is not None)
        outside = target < min(values) - tolerance or target > max(values) + tolerance
        status = OUTSIDE_REASONABLE_RANGE if outside else NO_BRACKET
        return replace(
            _invalid_result(
                config, base, target, status,
                "market_price_not_bracketed_by_reasonable_range",
                research_range=research_range,
            ),
            lower_bound_dcf=lower_eval.value,
            upper_bound_dcf=upper_eval.value,
            monotonic=True,
            root_interval_count=0,
        )

    implied_assumptions = assumptions_for_reverse_value(base, config.variable, implied)
    relation = _range_relation(implied, research_range)
    return ReverseDCFResult(
        variable=config.variable,
        status=SOLVED,
        market_price=target,
        research_value=_research_value(base, config.variable),
        implied_value=implied,
        research_growth_path=base.near_term_revenue_growth,
        implied_growth_path=(
            implied_assumptions.near_term_revenue_growth
            if config.variable == GROWTH_UPLIFT else None
        ),
        implied_dcf_value=final.value,
        residual=(final.value - target) if final.value is not None else None,
        lower_bound=config.lower_bound,
        upper_bound=config.upper_bound,
        lower_bound_dcf=lower_eval.value,
        upper_bound_dcf=upper_eval.value,
        iterations=iterations,
        monotonic=True,
        root_interval_count=1,
        research_range=research_range,
        range_relation=relation,
        expectation_gap=_gap_classification(config.variable, base, implied, SOLVED),
        enterprise_value=None,
        equity_value=None,
        terminal_value_share=None,
        reason=None,
    )


def run_reverse_dcf(
    inputs: RealCompanyDCFInputs,
    base_assumptions: MultiStageDCFAssumptions,
    market_price: float | None,
    *,
    ticker: str | None = None,
    base_source: str = "Current Manual Base",
    research_ranges: dict[ReverseVariable, ReverseResearchRange] | None = None,
) -> ReverseDCFAnalysis:
    """Run all four one-at-a-time market-implied expectation diagnostics."""
    if not isinstance(inputs, RealCompanyDCFInputs):
        raise TypeError("inputs must be RealCompanyDCFInputs")
    if not isinstance(base_assumptions, MultiStageDCFAssumptions):
        raise TypeError("base_assumptions must be MultiStageDCFAssumptions")
    ranges = research_ranges or {}
    try:
        base_run = run_multistage_dcf(inputs, base_assumptions)
        base_value = (
            base_run.per_share_value.intrinsic_value_per_share
            if base_run.per_share_value is not None else None
        )
    except (TypeError, ValueError, ArithmeticError, OverflowError) as exc:
        configs = default_reverse_configs(base_assumptions)
        return ReverseDCFAnalysis(
            ticker=ticker or inputs.ticker,
            base_source=base_source,
            base_dcf_per_share=None,
            market_price=market_price,
            price_to_base_dcf=None,
            results=tuple(
                _invalid_result(
                    config, base_assumptions, market_price,
                    INVALID_BASE_ASSUMPTIONS, f"base_valuation_failed:{exc}",
                    research_range=ranges.get(config.variable),
                )
                for config in configs
            ),
            warnings=("base_valuation_failed",),
        )

    def full_model_value(assumptions: MultiStageDCFAssumptions) -> float:
        result = run_multistage_dcf(inputs, assumptions)
        if result.per_share_value is None:
            raise ValueError(result.per_share_unavailable_reason or "per_share_value_unavailable")
        return result.per_share_value.intrinsic_value_per_share

    results = []
    for config in default_reverse_configs(base_assumptions):
        solved = solve_reverse_variable(
            base_assumptions,
            market_price,
            config,
            full_model_value,
            research_range=ranges.get(config.variable),
        )
        if solved.status == SOLVED and solved.implied_value is not None:
            implied_run = run_multistage_dcf(
                inputs,
                assumptions_for_reverse_value(
                    base_assumptions, solved.variable, solved.implied_value
                ),
            )
            solved = replace(
                solved,
                enterprise_value=implied_run.enterprise_value.enterprise_value,
                equity_value=implied_run.equity_value.equity_value,
                terminal_value_share=implied_run.enterprise_value.terminal_value_share,
                warnings=implied_run.warnings,
            )
        results.append(solved)

    price = (
        float(market_price)
        if market_price is not None and math.isfinite(float(market_price)) and market_price > 0
        else None
    )
    unresolved = sum(result.status != SOLVED for result in results)
    outside_research = sum(
        result.range_relation in {"below_research_range", "above_research_range"}
        for result in results
    )
    warnings_list = ["single_variable_results_are_not_joint_requirements"]
    if unresolved >= 2:
        warnings_list.append("multiple_reverse_dimensions_unresolved")
    if outside_research >= 2:
        warnings_list.append("multiple_implied_values_outside_research_ranges")
    return ReverseDCFAnalysis(
        ticker=ticker or inputs.ticker,
        base_source=base_source,
        base_dcf_per_share=base_value,
        market_price=price,
        price_to_base_dcf=(price / base_value if price is not None and base_value not in (None, 0) else None),
        results=tuple(results),
        warnings=tuple(warnings_list),
    )
