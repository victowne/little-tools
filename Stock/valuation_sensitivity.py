"""Pure WACC x terminal-growth sensitivity for the multi-stage DCF.

The module changes only the two sensitivity assumptions.  Every valid point
is evaluated through ``run_multistage_dcf`` so WACC affects explicit-period
discounting as well as terminal value, while terminal growth also updates the
steady-state reinvestment requirement.
"""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.multistage_integration import (
    RealCompanyDCFInputs,
    run_multistage_dcf,
)
from Stock.valuation import MultiStageDCFAssumptions


DEFAULT_AXIS_OFFSETS = (-0.01, -0.005, 0.0, 0.005, 0.01)
SensitivityAxisName = Literal["wacc", "terminal_growth"]


@dataclass(frozen=True)
class SensitivityAxis:
    """One immutable sensitivity axis centered on its exact base value."""

    name: SensitivityAxisName
    base_value: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class SensitivityPoint:
    """Full-model result for one WACC and terminal-growth combination."""

    wacc: float
    terminal_growth: float
    is_base_case: bool
    valid: bool
    intrinsic_value_per_share: float | None
    enterprise_value: float | None
    equity_value: float | None
    terminal_value_share: float | None
    terminal_reinvestment_rate: float | None
    terminal_fcff: float | None
    reason: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SensitivityImpact:
    """Difference between one neighboring point and the base point."""

    point: SensitivityPoint | None
    absolute_change: float | None
    percentage_change: float | None


@dataclass(frozen=True)
class WACCTerminalGrowthSensitivity:
    """Immutable rectangular sensitivity grid and objective diagnostics."""

    base_wacc: float
    base_terminal_growth: float
    wacc_axis: SensitivityAxis
    terminal_growth_axis: SensitivityAxis
    points: tuple[SensitivityPoint, ...]
    base_case_point: SensitivityPoint
    min_value_per_share: float | None
    max_value_per_share: float | None
    valid_point_count: int
    invalid_point_count: int
    warnings: tuple[str, ...]

    @property
    def wacc_values(self) -> tuple[float, ...]:
        return self.wacc_axis.values

    @property
    def terminal_growth_values(self) -> tuple[float, ...]:
        return self.terminal_growth_axis.values

    def point_at(
        self,
        wacc: float,
        terminal_growth: float,
    ) -> SensitivityPoint | None:
        """Return the point at exact economic coordinates, within float noise."""
        for point in self.points:
            if math.isclose(point.wacc, wacc, rel_tol=0.0, abs_tol=1e-12) and math.isclose(
                point.terminal_growth,
                terminal_growth,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return point
        return None

    def impact_at(
        self,
        wacc: float,
        terminal_growth: float,
    ) -> SensitivityImpact:
        """Compare one grid point with the intrinsic value at the base cell."""
        point = self.point_at(wacc, terminal_growth)
        base_value = self.base_case_point.intrinsic_value_per_share
        if (
            point is None
            or not point.valid
            or point.intrinsic_value_per_share is None
            or base_value is None
        ):
            return SensitivityImpact(point, None, None)
        change = point.intrinsic_value_per_share - base_value
        percentage = change / base_value if base_value != 0 else None
        return SensitivityImpact(point, change, percentage)


def _validated_axis_values(
    base_value: float,
    offsets: tuple[float, ...],
) -> tuple[float, ...]:
    if not math.isfinite(base_value):
        raise ValueError("sensitivity base value must be finite")
    try:
        numeric_offsets = tuple(float(offset) for offset in offsets)
    except (TypeError, ValueError) as exc:
        raise ValueError("sensitivity offsets must be finite numbers") from exc
    if not numeric_offsets or not all(math.isfinite(value) for value in numeric_offsets):
        raise ValueError("sensitivity offsets must be finite numbers")

    # Always insert the original float object, rather than a rounded equivalent,
    # so the currently selected base assumption is represented exactly.
    values = [base_value]
    values.extend(
        base_value if offset == 0.0 else base_value + offset
        for offset in numeric_offsets
    )
    return tuple(sorted(set(values)))


def _invalid_reason(exc: Exception) -> str:
    message = str(exc)
    if "wacc must be greater than terminal_growth" in message:
        return "wacc_not_greater_than_terminal_growth"
    if "wacc must be positive" in message:
        return "wacc_not_positive"
    return f"invalid_assumptions:{message}"


def _calculate_point(
    inputs: RealCompanyDCFInputs,
    base_assumptions: MultiStageDCFAssumptions,
    wacc: float,
    terminal_growth: float,
) -> SensitivityPoint:
    is_base = wacc == base_assumptions.wacc and terminal_growth == base_assumptions.terminal_growth
    try:
        assumptions = replace(
            base_assumptions,
            wacc=wacc,
            terminal_growth=terminal_growth,
        )
        run = run_multistage_dcf(inputs, assumptions)
    except (TypeError, ValueError) as exc:
        return SensitivityPoint(
            wacc=wacc,
            terminal_growth=terminal_growth,
            is_base_case=is_base,
            valid=False,
            intrinsic_value_per_share=None,
            enterprise_value=None,
            equity_value=None,
            terminal_value_share=None,
            terminal_reinvestment_rate=None,
            terminal_fcff=None,
            reason=_invalid_reason(exc),
            warnings=(),
        )

    if run.per_share_value is None:
        return SensitivityPoint(
            wacc=wacc,
            terminal_growth=terminal_growth,
            is_base_case=is_base,
            valid=False,
            intrinsic_value_per_share=None,
            enterprise_value=run.enterprise_value.enterprise_value,
            equity_value=run.equity_value.equity_value,
            terminal_value_share=run.enterprise_value.terminal_value_share,
            terminal_reinvestment_rate=run.terminal_value.terminal_reinvestment_rate,
            terminal_fcff=run.terminal_value.terminal_fcff,
            reason=run.per_share_unavailable_reason or "per_share_value_unavailable",
            warnings=run.warnings,
        )

    return SensitivityPoint(
        wacc=wacc,
        terminal_growth=terminal_growth,
        is_base_case=is_base,
        valid=True,
        intrinsic_value_per_share=run.per_share_value.intrinsic_value_per_share,
        enterprise_value=run.enterprise_value.enterprise_value,
        equity_value=run.equity_value.equity_value,
        terminal_value_share=run.enterprise_value.terminal_value_share,
        terminal_reinvestment_rate=run.terminal_value.terminal_reinvestment_rate,
        terminal_fcff=run.terminal_value.terminal_fcff,
        reason=None,
        warnings=run.warnings,
    )


def build_wacc_terminal_growth_sensitivity(
    inputs: RealCompanyDCFInputs,
    base_assumptions: MultiStageDCFAssumptions,
    *,
    wacc_offsets: tuple[float, ...] = DEFAULT_AXIS_OFFSETS,
    terminal_growth_offsets: tuple[float, ...] = DEFAULT_AXIS_OFFSETS,
) -> WACCTerminalGrowthSensitivity:
    """Run the complete multi-stage DCF at every sensitivity coordinate."""
    if not isinstance(inputs, RealCompanyDCFInputs):
        raise TypeError("inputs must be RealCompanyDCFInputs")
    if not isinstance(base_assumptions, MultiStageDCFAssumptions):
        raise TypeError("base_assumptions must be MultiStageDCFAssumptions")

    wacc_values = _validated_axis_values(base_assumptions.wacc, wacc_offsets)
    growth_values = _validated_axis_values(
        base_assumptions.terminal_growth,
        terminal_growth_offsets,
    )
    points = tuple(
        _calculate_point(inputs, base_assumptions, wacc, growth)
        for wacc in wacc_values
        for growth in growth_values
    )
    base_points = tuple(point for point in points if point.is_base_case)
    if len(base_points) != 1:
        raise RuntimeError("sensitivity grid must contain exactly one base point")

    valid_values = tuple(
        point.intrinsic_value_per_share
        for point in points
        if point.valid and point.intrinsic_value_per_share is not None
    )
    valid_count = sum(point.valid for point in points)
    invalid_count = len(points) - valid_count
    warnings = (
        ("invalid_sensitivity_points_present",) if invalid_count else ()
    )
    return WACCTerminalGrowthSensitivity(
        base_wacc=base_assumptions.wacc,
        base_terminal_growth=base_assumptions.terminal_growth,
        wacc_axis=SensitivityAxis("wacc", base_assumptions.wacc, wacc_values),
        terminal_growth_axis=SensitivityAxis(
            "terminal_growth",
            base_assumptions.terminal_growth,
            growth_values,
        ),
        points=points,
        base_case_point=base_points[0],
        min_value_per_share=min(valid_values) if valid_values else None,
        max_value_per_share=max(valid_values) if valid_values else None,
        valid_point_count=valid_count,
        invalid_point_count=invalid_count,
        warnings=warnings,
    )
