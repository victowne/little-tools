"""Pure assumptions and validation for the future multi-stage DCF.

This module intentionally contains no valuation engine, market-data access, or
UI integration. Historical anchors are evidence for later judgment and are not
used to populate assumptions automatically.
"""

from dataclasses import dataclass
import math
from typing import Literal


SALES_TO_CAPITAL_EPSILON = 1e-9
ROIC_EPSILON = 1e-12
ForecastStage = Literal["near_term", "fade", "mature"]


def _require_finite(name: str, value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _require_non_negative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class MultiStageDCFAssumptions:
    """Economic inputs for a future Revenue-driven multi-stage FCFF model.

    ``near_term_revenue_growth`` contains explicit annual rates. After its last
    value, a future engine will fade Revenue growth toward ``terminal_growth``
    over exactly ``revenue_fade_years`` annual transitions. Consequently,
    ``forecast_years`` must cover ``near_term_years + revenue_fade_years``.
    Any remaining forecast years represent years at mature assumptions.

    Margins and Sales-to-Capital paths are not generated in this phase. A
    future engine can fade their starting values toward their mature values
    over the forecast horizon without requiring annual manual inputs.
    """

    forecast_years: int
    near_term_revenue_growth: tuple[float, ...]
    revenue_fade_years: int
    terminal_growth: float
    starting_operating_margin: float
    mature_operating_margin: float
    starting_sales_to_capital: float
    mature_sales_to_capital: float
    operating_tax_rate: float
    wacc: float

    def __post_init__(self) -> None:
        if isinstance(self.forecast_years, bool) or not isinstance(
            self.forecast_years, int
        ) or self.forecast_years <= 0:
            raise ValueError("forecast_years must be a positive integer")
        _require_non_negative_integer("revenue_fade_years", self.revenue_fade_years)

        try:
            growth_rates = tuple(self.near_term_revenue_growth)
        except TypeError as exc:
            raise ValueError(
                "near_term_revenue_growth must contain at least one annual rate"
            ) from exc
        if not growth_rates:
            raise ValueError(
                "near_term_revenue_growth must contain at least one annual rate"
            )
        if len(growth_rates) > 5:
            raise ValueError(
                "near_term_revenue_growth must contain no more than five annual rates"
            )
        normalized_growth = tuple(
            _require_finite(f"near_term_revenue_growth[{index}]", growth)
            for index, growth in enumerate(growth_rates)
        )
        if any(growth <= -1 for growth in normalized_growth):
            raise ValueError("near-term Revenue growth must be greater than -100%")
        object.__setattr__(self, "near_term_revenue_growth", normalized_growth)

        for name in (
            "terminal_growth",
            "starting_operating_margin",
            "mature_operating_margin",
            "starting_sales_to_capital",
            "mature_sales_to_capital",
            "operating_tax_rate",
            "wacc",
        ):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))

        if self.terminal_growth <= -1:
            raise ValueError("terminal_growth must be greater than -100%")
        if self.wacc <= self.terminal_growth:
            raise ValueError("wacc must be greater than terminal_growth")
        if self.wacc <= 0:
            raise ValueError("wacc must be positive")
        if not 0 <= self.operating_tax_rate <= 1:
            raise ValueError("operating_tax_rate must be between 0 and 100%")

        if abs(self.starting_sales_to_capital) <= SALES_TO_CAPITAL_EPSILON:
            raise ValueError("starting_sales_to_capital must not be zero or near zero")
        if self.mature_sales_to_capital <= SALES_TO_CAPITAL_EPSILON:
            raise ValueError("mature_sales_to_capital must be positive")

        required_years = self.near_term_years + self.fade_years
        if self.forecast_years < required_years:
            raise ValueError(
                "forecast_years must cover all near-term and Revenue-fade years"
            )

        if (
            abs(self.derived_terminal_roic) <= ROIC_EPSILON
            and self.terminal_growth != 0
        ):
            raise ValueError(
                "non-zero terminal_growth requires non-zero derived terminal ROIC"
            )

    @property
    def near_term_years(self) -> int:
        """Number of explicitly supplied annual Revenue-growth assumptions."""
        return len(self.near_term_revenue_growth)

    @property
    def fade_years(self) -> int:
        """Number of later annual growth transitions toward terminal growth."""
        return self.revenue_fade_years

    @property
    def total_forecast_years(self) -> int:
        return self.forecast_years

    @property
    def mature_state_years(self) -> int:
        """Forecast years remaining after explicit growth and its fade."""
        return self.forecast_years - self.near_term_years - self.fade_years

    @property
    def after_tax_mature_operating_margin(self) -> float:
        return self.mature_operating_margin * (1 - self.operating_tax_rate)

    @property
    def derived_terminal_roic(self) -> float:
        """Diagnostic after-tax mature margin × mature Sales-to-Capital."""
        return (
            self.after_tax_mature_operating_margin
            * self.mature_sales_to_capital
        )

    @property
    def terminal_reinvestment_rate(self) -> float | None:
        """Diagnostic terminal growth / derived terminal ROIC, when defined."""
        if abs(self.derived_terminal_roic) <= ROIC_EPSILON:
            return None
        return self.terminal_growth / self.derived_terminal_roic

    @property
    def validation_warnings(self) -> tuple[str, ...]:
        """Descriptive diagnostics that do not make investment judgments."""
        warnings: list[str] = []
        reinvestment_rate = self.terminal_reinvestment_rate
        if self.derived_terminal_roic < 0:
            warnings.append("derived_terminal_roic_is_negative")
        if reinvestment_rate is not None and reinvestment_rate > 1:
            warnings.append("terminal_reinvestment_rate_exceeds_100_percent")
        if reinvestment_rate is not None and reinvestment_rate < 0:
            warnings.append("terminal_reinvestment_rate_is_negative")
        if self.terminal_growth > self.near_term_revenue_growth[-1]:
            warnings.append("terminal_growth_exceeds_final_near_term_growth")
        return tuple(warnings)


@dataclass(frozen=True)
class ForecastYearAssumptions:
    """The deterministic economic assumptions assigned to one forecast year."""

    year_index: int
    stage: ForecastStage
    revenue_growth: float
    operating_margin: float
    sales_to_capital: float


@dataclass(frozen=True)
class MultiStageForecastPath:
    """Immutable annual paths generated from ``MultiStageDCFAssumptions``."""

    years: tuple[ForecastYearAssumptions, ...]

    @property
    def near_term_year_count(self) -> int:
        return sum(year.stage == "near_term" for year in self.years)

    @property
    def fade_year_count(self) -> int:
        return sum(year.stage == "fade" for year in self.years)

    @property
    def mature_year_count(self) -> int:
        return sum(year.stage == "mature" for year in self.years)

    @property
    def revenue_growth_path(self) -> tuple[float, ...]:
        return tuple(year.revenue_growth for year in self.years)

    @property
    def operating_margin_path(self) -> tuple[float, ...]:
        return tuple(year.operating_margin for year in self.years)

    @property
    def sales_to_capital_path(self) -> tuple[float, ...]:
        return tuple(year.sales_to_capital for year in self.years)

    @property
    def starting_values(self) -> tuple[float, float, float]:
        first = self.years[0]
        return (
            first.revenue_growth,
            first.operating_margin,
            first.sales_to_capital,
        )

    @property
    def ending_values(self) -> tuple[float, float, float]:
        last = self.years[-1]
        return (
            last.revenue_growth,
            last.operating_margin,
            last.sales_to_capital,
        )


def _linear_transition_value(
    start: float,
    end: float,
    year_index: int,
    transition_years: int,
) -> float:
    """Interpolate from start in Year 1 to end in the final transition year.

    For transition horizons longer than one year, Year ``i`` uses
    ``start + (end - start) * (i - 1) / (transition_years - 1)``. If the
    horizon is only one year, Year 1 remains at the required starting value;
    any later mature year uses the ending value directly.
    """
    if year_index > transition_years:
        return end
    if transition_years == 1:
        return start
    if year_index == transition_years:
        return end
    return start + (end - start) * (year_index - 1) / (transition_years - 1)


def _validate_sales_to_capital_path(values: tuple[float, ...]) -> None:
    for year_index, value in enumerate(values, start=1):
        if abs(value) <= SALES_TO_CAPITAL_EPSILON:
            raise ValueError(
                "generated Sales-to-Capital is zero or near zero "
                f"in forecast year {year_index}"
            )
    for year_index, (previous, current) in enumerate(
        zip(values, values[1:]), start=2
    ):
        if (previous < 0 < current) or (current < 0 < previous):
            raise ValueError(
                "generated Sales-to-Capital crosses zero between forecast "
                f"years {year_index - 1} and {year_index}"
            )


def generate_forecast_path(
    assumptions: MultiStageDCFAssumptions,
) -> MultiStageForecastPath:
    """Generate annual growth, margin, and capital-efficiency assumptions only.

    Explicit Revenue growth occupies Years 1..N. Fade Year ``k`` then uses
    ``last_explicit + (terminal - last_explicit) * k / F`` for ``k=1..F``;
    the final fade year therefore equals terminal growth without repeating the
    last explicit rate. Remaining years are mature.

    Operating margin and Sales-to-Capital share an economic transition horizon
    of ``N + F`` years. They start at their starting values in Year 1 and reach
    their mature values in the final transition year, then remain constant.
    No Revenue dollars or valuation cash flows are calculated here.
    """
    if not isinstance(assumptions, MultiStageDCFAssumptions):
        raise TypeError("assumptions must be MultiStageDCFAssumptions")

    near_term_years = assumptions.near_term_years
    fade_years = assumptions.fade_years
    transition_years = near_term_years + fade_years
    final_explicit_growth = assumptions.near_term_revenue_growth[-1]
    generated: list[ForecastYearAssumptions] = []

    for year_index in range(1, assumptions.forecast_years + 1):
        if year_index <= near_term_years:
            stage: ForecastStage = "near_term"
            revenue_growth = assumptions.near_term_revenue_growth[year_index - 1]
        elif year_index <= transition_years:
            stage = "fade"
            fade_index = year_index - near_term_years
            if fade_index == fade_years:
                revenue_growth = assumptions.terminal_growth
            else:
                revenue_growth = final_explicit_growth + (
                    assumptions.terminal_growth - final_explicit_growth
                ) * fade_index / fade_years
        else:
            stage = "mature"
            revenue_growth = assumptions.terminal_growth

        operating_margin = _linear_transition_value(
            assumptions.starting_operating_margin,
            assumptions.mature_operating_margin,
            year_index,
            transition_years,
        )
        sales_to_capital = _linear_transition_value(
            assumptions.starting_sales_to_capital,
            assumptions.mature_sales_to_capital,
            year_index,
            transition_years,
        )
        values = (revenue_growth, operating_margin, sales_to_capital)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"generated non-finite value in forecast year {year_index}")
        generated.append(
            ForecastYearAssumptions(
                year_index=year_index,
                stage=stage,
                revenue_growth=revenue_growth,
                operating_margin=operating_margin,
                sales_to_capital=sales_to_capital,
            )
        )

    path = MultiStageForecastPath(tuple(generated))
    _validate_sales_to_capital_path(path.sales_to_capital_path)

    if len(path.years) != assumptions.forecast_years:
        raise RuntimeError("forecast path length invariant failed")
    if path.near_term_year_count != near_term_years:
        raise RuntimeError("near-term stage-count invariant failed")
    if path.fade_year_count != fade_years:
        raise RuntimeError("fade stage-count invariant failed")
    if path.mature_year_count != assumptions.mature_state_years:
        raise RuntimeError("mature stage-count invariant failed")
    if fade_years and path.years[transition_years - 1].revenue_growth != (
        assumptions.terminal_growth
    ):
        raise RuntimeError("final fade growth invariant failed")
    return path


@dataclass(frozen=True)
class OperatingForecastYear:
    """Operating economics for one forecast year, before any discounting."""

    year_index: int
    stage: ForecastStage
    revenue_growth: float
    revenue: float
    operating_margin: float
    operating_income: float
    operating_tax_rate: float
    nopat: float
    sales_to_capital: float
    delta_revenue: float
    reinvestment: float
    fcff: float


@dataclass(frozen=True)
class MultiStageOperatingForecast:
    """Immutable operating forecast and arithmetic, non-valuation summaries."""

    starting_revenue: float
    years: tuple[OperatingForecastYear, ...]

    @property
    def ending_revenue(self) -> float:
        return self.years[-1].revenue

    @property
    def cumulative_revenue_growth(self) -> float:
        return self.ending_revenue / self.starting_revenue - 1

    @property
    def total_nopat(self) -> float:
        return sum(year.nopat for year in self.years)

    @property
    def total_reinvestment(self) -> float:
        return sum(year.reinvestment for year in self.years)

    @property
    def total_fcff(self) -> float:
        return sum(year.fcff for year in self.years)


def _expected_stage(
    year_index: int,
    assumptions: MultiStageDCFAssumptions,
) -> ForecastStage:
    if year_index <= assumptions.near_term_years:
        return "near_term"
    if year_index <= assumptions.near_term_years + assumptions.fade_years:
        return "fade"
    return "mature"


def _validate_supplied_forecast_path(
    path: MultiStageForecastPath,
    assumptions: MultiStageDCFAssumptions,
) -> None:
    if len(path.years) != assumptions.forecast_years:
        raise ValueError("forecast_path length must equal forecast_years")
    sales_to_capital_values: list[float] = []
    for expected_index, year in enumerate(path.years, start=1):
        if year.year_index != expected_index:
            raise ValueError("forecast_path year indexes must be consecutive from 1")
        if year.stage != _expected_stage(expected_index, assumptions):
            raise ValueError(
                f"forecast_path stage does not match assumptions in year {expected_index}"
            )
        values = (
            year.revenue_growth,
            year.operating_margin,
            year.sales_to_capital,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"forecast_path contains a non-finite value in year {expected_index}"
            )
        if year.revenue_growth <= -1:
            raise ValueError(
                f"forecast_path Revenue growth must be greater than -100% in year {expected_index}"
            )
        sales_to_capital_values.append(year.sales_to_capital)
    _validate_sales_to_capital_path(tuple(sales_to_capital_values))


def build_operating_forecast(
    starting_revenue: float,
    assumptions: MultiStageDCFAssumptions,
    forecast_path: MultiStageForecastPath | None = None,
) -> MultiStageOperatingForecast:
    """Convert an assumptions path into annual operating economics.

    ``starting_revenue`` is the latest actual Revenue immediately before
    forecast Year 1 and must be finite and strictly positive. Forecast FCFF is
    an operating-economic measure, ``NOPAT - Reinvestment``. It is deliberately
    distinct from the legacy cash-flow-statement FCFF based on CFO, CapEx, and
    after-tax interest.

    This function forecasts no separate CapEx, D&A, working capital, or
    acquisitions and performs no discounting or terminal-value calculation.
    """
    if isinstance(starting_revenue, bool):
        raise ValueError("starting_revenue must be a finite positive number")
    revenue_base = _require_finite("starting_revenue", starting_revenue)
    if revenue_base <= 0:
        raise ValueError("starting_revenue must be greater than zero")
    if not isinstance(assumptions, MultiStageDCFAssumptions):
        raise TypeError("assumptions must be MultiStageDCFAssumptions")

    if forecast_path is None:
        path = generate_forecast_path(assumptions)
    elif isinstance(forecast_path, MultiStageForecastPath):
        path = forecast_path
    else:
        raise TypeError("forecast_path must be MultiStageForecastPath")
    _validate_supplied_forecast_path(path, assumptions)

    rows: list[OperatingForecastYear] = []
    prior_revenue = revenue_base
    for path_year in path.years:
        revenue = prior_revenue * (1 + path_year.revenue_growth)
        if not math.isfinite(revenue):
            raise ValueError(
                f"generated Revenue is non-finite in year {path_year.year_index}"
            )
        if revenue < 0:
            raise ValueError(
                f"generated Revenue is negative in year {path_year.year_index}"
            )
        delta_revenue = revenue - prior_revenue
        operating_income = revenue * path_year.operating_margin
        nopat = operating_income * (1 - assumptions.operating_tax_rate)
        if abs(path_year.sales_to_capital) <= SALES_TO_CAPITAL_EPSILON:
            raise ValueError(
                "forecast Sales-to-Capital is zero or near zero "
                f"in year {path_year.year_index}"
            )
        reinvestment = delta_revenue / path_year.sales_to_capital
        fcff = nopat - reinvestment
        generated_values = (
            revenue,
            delta_revenue,
            operating_income,
            nopat,
            reinvestment,
            fcff,
        )
        if not all(math.isfinite(value) for value in generated_values):
            raise ValueError(
                f"generated non-finite operating value in year {path_year.year_index}"
            )
        rows.append(
            OperatingForecastYear(
                year_index=path_year.year_index,
                stage=path_year.stage,
                revenue_growth=path_year.revenue_growth,
                revenue=revenue,
                operating_margin=path_year.operating_margin,
                operating_income=operating_income,
                operating_tax_rate=assumptions.operating_tax_rate,
                nopat=nopat,
                sales_to_capital=path_year.sales_to_capital,
                delta_revenue=delta_revenue,
                reinvestment=reinvestment,
                fcff=fcff,
            )
        )
        prior_revenue = revenue

    forecast = MultiStageOperatingForecast(revenue_base, tuple(rows))
    if len(forecast.years) != len(path.years):
        raise RuntimeError("operating forecast length invariant failed")
    for operating_year, path_year in zip(forecast.years, path.years):
        if operating_year.year_index != path_year.year_index:
            raise RuntimeError("operating forecast year-index invariant failed")
        if operating_year.stage != path_year.stage:
            raise RuntimeError("operating forecast stage invariant failed")
    return forecast


@dataclass(frozen=True)
class DiscountedForecastYear:
    """One explicit forecast year's FCFF discounted at the year end."""

    year_index: int
    stage: ForecastStage
    fcff: float
    discount_factor: float
    present_value_fcff: float


@dataclass(frozen=True)
class MultiStageDiscountedForecast:
    """Immutable present values for the explicit forecast period only."""

    wacc: float
    years: tuple[DiscountedForecastYear, ...]

    @property
    def total_undiscounted_fcff(self) -> float:
        return sum(year.fcff for year in self.years)

    @property
    def total_present_value_fcff(self) -> float:
        return sum(year.present_value_fcff for year in self.years)

    @property
    def explicit_pv_to_undiscounted_fcff(self) -> float | None:
        """Descriptive ratio; unavailable when undiscounted FCFF sums to zero."""
        total = self.total_undiscounted_fcff
        if abs(total) <= ROIC_EPSILON:
            return None
        return self.total_present_value_fcff / total


def discount_operating_forecast(
    operating_forecast: MultiStageOperatingForecast,
    assumptions: MultiStageDCFAssumptions,
) -> MultiStageDiscountedForecast:
    """Discount explicit annual FCFF at each forecast year end.

    Forecast Year ``t`` uses ``1 / (1 + assumptions.wacc) ** t``. This is
    standard year-end discounting, not a mid-year convention. The function
    consumes existing FCFF values and does not recompute operating economics,
    terminal value, or any enterprise/equity bridge.
    """
    if not isinstance(operating_forecast, MultiStageOperatingForecast):
        raise TypeError("operating_forecast must be MultiStageOperatingForecast")
    if not isinstance(assumptions, MultiStageDCFAssumptions):
        raise TypeError("assumptions must be MultiStageDCFAssumptions")
    if not operating_forecast.years:
        raise ValueError("operating_forecast must contain at least one year")

    wacc = _require_finite("wacc", assumptions.wacc)
    if wacc <= -1:
        raise ValueError("wacc must be greater than -100%")
    if len(operating_forecast.years) != assumptions.forecast_years:
        raise ValueError(
            "operating_forecast length must equal assumptions.forecast_years"
        )

    discounted_years: list[DiscountedForecastYear] = []
    for expected_index, forecast_year in enumerate(
        operating_forecast.years, start=1
    ):
        if forecast_year.year_index != expected_index:
            raise ValueError(
                "operating_forecast year indexes must be consecutive from 1"
            )
        expected_stage = _expected_stage(expected_index, assumptions)
        if forecast_year.stage != expected_stage:
            raise ValueError(
                "operating_forecast stage does not match assumptions "
                f"in year {expected_index}"
            )
        fcff = _require_finite(f"fcff[{expected_index}]", forecast_year.fcff)
        discount_factor = 1 / (1 + wacc) ** expected_index
        present_value_fcff = fcff * discount_factor
        if not math.isfinite(discount_factor) or not math.isfinite(
            present_value_fcff
        ):
            raise ValueError(
                f"generated non-finite discount value in year {expected_index}"
            )
        discounted_years.append(
            DiscountedForecastYear(
                year_index=expected_index,
                stage=forecast_year.stage,
                fcff=fcff,
                discount_factor=discount_factor,
                present_value_fcff=present_value_fcff,
            )
        )

    result = MultiStageDiscountedForecast(wacc, tuple(discounted_years))
    if len(result.years) != len(operating_forecast.years):
        raise RuntimeError("discounted forecast length invariant failed")
    return result


TERMINAL_CONSISTENCY_REL_TOLERANCE = 1e-9
TERMINAL_CONSISTENCY_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TerminalValueResult:
    """Steady-state Year N+1 economics and value measured at Year N."""

    terminal_growth: float
    wacc: float
    mature_operating_margin: float
    operating_tax_rate: float
    mature_sales_to_capital: float
    derived_terminal_roic: float
    terminal_reinvestment_rate: float
    terminal_fcff_to_nopat: float
    final_forecast_revenue: float
    terminal_year_revenue: float
    terminal_operating_income: float
    terminal_nopat: float
    terminal_reinvestment: float
    terminal_fcff: float
    terminal_value: float
    terminal_discount_factor: float
    present_value_terminal_value: float
    warnings: tuple[str, ...]


def _terminal_values_close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=TERMINAL_CONSISTENCY_REL_TOLERANCE,
        abs_tol=TERMINAL_CONSISTENCY_ABS_TOLERANCE,
    )


def calculate_terminal_value(
    operating_forecast: MultiStageOperatingForecast,
    discounted_forecast: MultiStageDiscountedForecast,
    assumptions: MultiStageDCFAssumptions,
) -> TerminalValueResult:
    """Calculate sustainable terminal economics and their present value.

    The explicit forecast ends at Year N. Terminal economics are constructed
    for Year N+1, and their Gordon-growth value is measured at the end of Year
    N. Terminal reinvestment uses ``growth / derived ROIC`` rather than the
    explicit-period delta-Revenue formula. For the otherwise ambiguous case of
    zero growth and zero ROIC, terminal reinvestment is explicitly defined as
    zero: no growth requires no steady-state growth investment.
    """
    if not isinstance(operating_forecast, MultiStageOperatingForecast):
        raise TypeError("operating_forecast must be MultiStageOperatingForecast")
    if not isinstance(discounted_forecast, MultiStageDiscountedForecast):
        raise TypeError("discounted_forecast must be MultiStageDiscountedForecast")
    if not isinstance(assumptions, MultiStageDCFAssumptions):
        raise TypeError("assumptions must be MultiStageDCFAssumptions")
    if not operating_forecast.years:
        raise ValueError("operating_forecast must contain at least one year")
    if not discounted_forecast.years:
        raise ValueError("discounted_forecast must contain at least one year")
    if len(operating_forecast.years) != assumptions.forecast_years:
        raise ValueError(
            "operating_forecast length must equal assumptions.forecast_years"
        )
    if len(discounted_forecast.years) != assumptions.forecast_years:
        raise ValueError(
            "discounted_forecast length must equal assumptions.forecast_years"
        )
    if not _terminal_values_close(discounted_forecast.wacc, assumptions.wacc):
        raise ValueError("discounted_forecast WACC does not match assumptions")

    for expected_index, (operating_year, discounted_year) in enumerate(
        zip(operating_forecast.years, discounted_forecast.years), start=1
    ):
        if operating_year.year_index != expected_index:
            raise ValueError(
                "operating_forecast year indexes must be consecutive from 1"
            )
        if discounted_year.year_index != expected_index:
            raise ValueError(
                "discounted_forecast year indexes must be consecutive from 1"
            )
        expected_stage = _expected_stage(expected_index, assumptions)
        if operating_year.stage != expected_stage:
            raise ValueError(
                f"operating_forecast stage mismatch in year {expected_index}"
            )
        if discounted_year.stage != operating_year.stage:
            raise ValueError(
                f"discounted_forecast stage mismatch in year {expected_index}"
            )
        if not _terminal_values_close(discounted_year.fcff, operating_year.fcff):
            raise ValueError(
                f"discounted_forecast FCFF mismatch in year {expected_index}"
            )

    final_year = operating_forecast.years[-1]
    if final_year.year_index != assumptions.forecast_years:
        raise ValueError("final forecast year index must equal forecast_years")
    steady_state_checks = (
        (
            "Revenue growth",
            final_year.revenue_growth,
            assumptions.terminal_growth,
        ),
        (
            "operating margin",
            final_year.operating_margin,
            assumptions.mature_operating_margin,
        ),
        (
            "Sales-to-Capital",
            final_year.sales_to_capital,
            assumptions.mature_sales_to_capital,
        ),
    )
    for name, actual, expected in steady_state_checks:
        if not _terminal_values_close(actual, expected):
            raise ValueError(
                f"final forecast {name} has not reached its mature assumption"
            )

    wacc = _require_finite("wacc", assumptions.wacc)
    terminal_growth = _require_finite(
        "terminal_growth", assumptions.terminal_growth
    )
    denominator = wacc - terminal_growth
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("WACC minus terminal growth must be finite and positive")

    terminal_revenue = final_year.revenue * (1 + terminal_growth)
    terminal_operating_income = (
        terminal_revenue * assumptions.mature_operating_margin
    )
    terminal_nopat = terminal_operating_income * (
        1 - assumptions.operating_tax_rate
    )
    derived_roic = assumptions.derived_terminal_roic
    if abs(derived_roic) <= ROIC_EPSILON:
        if terminal_growth != 0:
            raise ValueError(
                "non-zero terminal growth requires non-zero derived terminal ROIC"
            )
        terminal_reinvestment_rate = 0.0
    else:
        terminal_reinvestment_rate = terminal_growth / derived_roic
    terminal_conversion = 1 - terminal_reinvestment_rate
    terminal_reinvestment = terminal_nopat * terminal_reinvestment_rate
    terminal_fcff = terminal_nopat - terminal_reinvestment
    terminal_value = terminal_fcff / denominator

    final_discounted_year = discounted_forecast.years[-1]
    expected_discount_factor = 1 / (1 + wacc) ** assumptions.forecast_years
    if not _terminal_values_close(
        final_discounted_year.discount_factor, expected_discount_factor
    ):
        raise ValueError(
            "final explicit discount factor is inconsistent with WACC and horizon"
        )
    terminal_discount_factor = final_discounted_year.discount_factor
    present_value_terminal_value = terminal_value * terminal_discount_factor

    generated_values = (
        terminal_revenue,
        terminal_operating_income,
        terminal_nopat,
        derived_roic,
        terminal_reinvestment_rate,
        terminal_conversion,
        terminal_reinvestment,
        terminal_fcff,
        terminal_value,
        terminal_discount_factor,
        present_value_terminal_value,
    )
    if not all(math.isfinite(value) for value in generated_values):
        raise ValueError("terminal calculation produced a non-finite value")

    warnings = list(assumptions.validation_warnings)
    if terminal_fcff < 0 and "negative_terminal_fcff" not in warnings:
        warnings.append("negative_terminal_fcff")
    if terminal_value < 0 and "negative_terminal_value" not in warnings:
        warnings.append("negative_terminal_value")

    return TerminalValueResult(
        terminal_growth=terminal_growth,
        wacc=wacc,
        mature_operating_margin=assumptions.mature_operating_margin,
        operating_tax_rate=assumptions.operating_tax_rate,
        mature_sales_to_capital=assumptions.mature_sales_to_capital,
        derived_terminal_roic=derived_roic,
        terminal_reinvestment_rate=terminal_reinvestment_rate,
        terminal_fcff_to_nopat=terminal_conversion,
        final_forecast_revenue=final_year.revenue,
        terminal_year_revenue=terminal_revenue,
        terminal_operating_income=terminal_operating_income,
        terminal_nopat=terminal_nopat,
        terminal_reinvestment=terminal_reinvestment,
        terminal_fcff=terminal_fcff,
        terminal_value=terminal_value,
        terminal_discount_factor=terminal_discount_factor,
        present_value_terminal_value=present_value_terminal_value,
        warnings=tuple(warnings),
    )


ENTERPRISE_VALUE_SHARE_EPSILON = 1e-12
TERMINAL_VALUE_DOMINANCE_THRESHOLD = 0.80


@dataclass(frozen=True)
class EnterpriseValueResult:
    """Aggregation of explicit and terminal operating-asset present values."""

    explicit_forecast_pv: float
    terminal_value_pv: float
    enterprise_value: float
    terminal_value_share: float | None
    explicit_value_share: float | None
    forecast_years: int
    wacc: float
    terminal_growth: float
    warnings: tuple[str, ...]


def aggregate_enterprise_value(
    discounted_forecast: MultiStageDiscountedForecast,
    terminal_result: TerminalValueResult,
    assumptions: MultiStageDCFAssumptions,
) -> EnterpriseValueResult:
    """Add already-calculated explicit and terminal present values.

    Component shares are valuation-dependency diagnostics only. They are
    available only when Enterprise Value is positive and greater than the
    numerical denominator tolerance. A terminal-value share strictly above
    80% produces a descriptive dependency warning, not a validation failure.
    """
    if not isinstance(discounted_forecast, MultiStageDiscountedForecast):
        raise TypeError("discounted_forecast must be MultiStageDiscountedForecast")
    if not isinstance(terminal_result, TerminalValueResult):
        raise TypeError("terminal_result must be TerminalValueResult")
    if not isinstance(assumptions, MultiStageDCFAssumptions):
        raise TypeError("assumptions must be MultiStageDCFAssumptions")
    if not discounted_forecast.years:
        raise ValueError("discounted_forecast must contain at least one year")
    if len(discounted_forecast.years) != assumptions.forecast_years:
        raise ValueError(
            "discounted_forecast length must equal assumptions.forecast_years"
        )
    if not _terminal_values_close(discounted_forecast.wacc, assumptions.wacc):
        raise ValueError("discounted_forecast WACC does not match assumptions")
    if not _terminal_values_close(terminal_result.wacc, assumptions.wacc):
        raise ValueError("terminal_result WACC does not match assumptions")
    if not _terminal_values_close(
        terminal_result.terminal_growth, assumptions.terminal_growth
    ):
        raise ValueError(
            "terminal_result terminal growth does not match assumptions"
        )

    final_discounted_year = discounted_forecast.years[-1]
    if final_discounted_year.year_index != assumptions.forecast_years:
        raise ValueError("final discounted year index must equal forecast_years")
    expected_discount_factor = 1 / (
        1 + assumptions.wacc
    ) ** assumptions.forecast_years
    if not _terminal_values_close(
        final_discounted_year.discount_factor, expected_discount_factor
    ):
        raise ValueError(
            "final explicit discount factor is inconsistent with WACC and horizon"
        )
    if not _terminal_values_close(
        terminal_result.terminal_discount_factor,
        final_discounted_year.discount_factor,
    ):
        raise ValueError(
            "terminal discount factor does not match final explicit year"
        )
    if not math.isfinite(terminal_result.final_forecast_revenue):
        raise ValueError("terminal_result final forecast Revenue must be finite")

    explicit_pv = discounted_forecast.total_present_value_fcff
    terminal_pv = terminal_result.present_value_terminal_value
    if not math.isfinite(explicit_pv):
        raise ValueError("explicit forecast present value must be finite")
    if not math.isfinite(terminal_pv):
        raise ValueError("terminal present value must be finite")
    enterprise_value = explicit_pv + terminal_pv
    if not math.isfinite(enterprise_value):
        raise ValueError("enterprise value must be finite")

    warnings = list(dict.fromkeys(terminal_result.warnings))
    if enterprise_value < -ENTERPRISE_VALUE_SHARE_EPSILON:
        terminal_share = None
        explicit_share = None
        if "negative_enterprise_value" not in warnings:
            warnings.append("negative_enterprise_value")
    elif enterprise_value <= ENTERPRISE_VALUE_SHARE_EPSILON:
        terminal_share = None
        explicit_share = None
        if "zero_enterprise_value" not in warnings:
            warnings.append("zero_enterprise_value")
    else:
        terminal_share = terminal_pv / enterprise_value
        explicit_share = explicit_pv / enterprise_value
        if (
            terminal_share > TERMINAL_VALUE_DOMINANCE_THRESHOLD
            and "terminal_value_dominates_enterprise_value" not in warnings
        ):
            warnings.append("terminal_value_dominates_enterprise_value")

    return EnterpriseValueResult(
        explicit_forecast_pv=explicit_pv,
        terminal_value_pv=terminal_pv,
        enterprise_value=enterprise_value,
        terminal_value_share=terminal_share,
        explicit_value_share=explicit_share,
        forecast_years=assumptions.forecast_years,
        wacc=assumptions.wacc,
        terminal_growth=assumptions.terminal_growth,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class EquityValueResult:
    """Pure signed-net-debt bridge from Enterprise Value to Equity Value."""

    enterprise_value: float
    net_debt: float
    equity_value: float
    net_debt_to_enterprise_value: float | None
    warnings: tuple[str, ...]


def bridge_enterprise_to_equity_value(
    enterprise_result: EnterpriseValueResult,
    net_debt: float,
) -> EquityValueResult:
    """Subtract explicit signed net debt from an existing Enterprise Value.

    ``net_debt`` follows ``Debt - Cash``: positive values reduce Equity Value,
    while negative values represent net cash and increase Equity Value. Missing
    values are never replaced with zero. No per-share or market-price result is
    calculated here.
    """
    if not isinstance(enterprise_result, EnterpriseValueResult):
        raise TypeError("enterprise_result must be EnterpriseValueResult")
    enterprise_value = _require_finite(
        "enterprise_value", enterprise_result.enterprise_value
    )
    if isinstance(net_debt, bool):
        raise ValueError("net_debt must be a finite number")
    validated_net_debt = _require_finite("net_debt", net_debt)

    equity_value = enterprise_value - validated_net_debt
    if not math.isfinite(equity_value):
        raise ValueError("equity value must be finite")
    if enterprise_value > ENTERPRISE_VALUE_SHARE_EPSILON:
        net_debt_to_ev = validated_net_debt / enterprise_value
    else:
        net_debt_to_ev = None

    warnings = list(dict.fromkeys(enterprise_result.warnings))
    if validated_net_debt < 0 and "net_cash_position" not in warnings:
        warnings.append("net_cash_position")
    if (
        enterprise_value > 0
        and validated_net_debt > enterprise_value
        and "net_debt_exceeds_enterprise_value" not in warnings
    ):
        warnings.append("net_debt_exceeds_enterprise_value")
    if equity_value < -ENTERPRISE_VALUE_SHARE_EPSILON:
        if "negative_equity_value" not in warnings:
            warnings.append("negative_equity_value")
    elif abs(equity_value) <= ENTERPRISE_VALUE_SHARE_EPSILON:
        if "zero_equity_value" not in warnings:
            warnings.append("zero_equity_value")

    return EquityValueResult(
        enterprise_value=enterprise_value,
        net_debt=validated_net_debt,
        equity_value=equity_value,
        net_debt_to_enterprise_value=net_debt_to_ev,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class PerShareValueResult:
    """Intrinsic equity value divided by current common shares outstanding."""

    equity_value: float
    shares_outstanding: float
    intrinsic_value_per_share: float
    warnings: tuple[str, ...]


def calculate_intrinsic_value_per_share(
    equity_result: EquityValueResult,
    shares_outstanding: float,
) -> PerShareValueResult:
    """Calculate intrinsic value per current common share.

    The denominator is the current common shares outstanding, not an accounting
    weighted-average basic/diluted EPS denominator and not an estimate of future
    dilution. The engine is unit-agnostic: Equity Value and shares must be
    supplied in compatible units (for example, billions and billions). No unit
    conversion, dilution model, or market-price comparison is performed here.
    """
    if not isinstance(equity_result, EquityValueResult):
        raise TypeError("equity_result must be EquityValueResult")
    equity_value = _require_finite("equity_value", equity_result.equity_value)
    if isinstance(shares_outstanding, bool):
        raise ValueError("shares_outstanding must be a finite positive number")
    validated_shares = _require_finite(
        "shares_outstanding", shares_outstanding
    )
    if validated_shares <= 0:
        raise ValueError("shares_outstanding must be greater than zero")

    intrinsic_value = equity_value / validated_shares
    if not math.isfinite(intrinsic_value):
        raise ValueError("intrinsic value per share must be finite")

    warnings = list(dict.fromkeys(equity_result.warnings))
    if intrinsic_value < 0:
        if "negative_intrinsic_value_per_share" not in warnings:
            warnings.append("negative_intrinsic_value_per_share")
    elif intrinsic_value == 0:
        if "zero_intrinsic_value_per_share" not in warnings:
            warnings.append("zero_intrinsic_value_per_share")

    return PerShareValueResult(
        equity_value=equity_value,
        shares_outstanding=validated_shares,
        intrinsic_value_per_share=intrinsic_value,
        warnings=tuple(warnings),
    )
