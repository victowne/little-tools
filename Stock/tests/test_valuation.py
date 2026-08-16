from dataclasses import FrozenInstanceError

import pytest

from Stock.valuation import (
    ForecastYearAssumptions,
    MultiStageDCFAssumptions,
    MultiStageForecastPath,
    generate_forecast_path,
)


def assumptions(**overrides):
    values = {
        "forecast_years": 10,
        "near_term_revenue_growth": (0.12, 0.10, 0.08),
        "revenue_fade_years": 5,
        "terminal_growth": 0.025,
        "starting_operating_margin": 0.25,
        "mature_operating_margin": 0.22,
        "starting_sales_to_capital": 1.8,
        "mature_sales_to_capital": 1.5,
        "operating_tax_rate": 0.21,
        "wacc": 0.09,
    }
    values.update(overrides)
    return MultiStageDCFAssumptions(**values)


def test_valid_ordinary_mature_company():
    model = assumptions()

    assert model.near_term_years == 3
    assert model.fade_years == 5
    assert model.total_forecast_years == 10
    assert model.mature_state_years == 2


def test_valid_high_growth_technology_example():
    model = assumptions(
        near_term_revenue_growth=(0.30, 0.25, 0.20),
        starting_operating_margin=0.50,
        mature_operating_margin=0.35,
        starting_sales_to_capital=2.5,
        mature_sales_to_capital=1.5,
        terminal_growth=0.025,
    )

    assert model.near_term_revenue_growth == (0.30, 0.25, 0.20)
    assert model.mature_operating_margin < model.starting_operating_margin
    assert model.mature_sales_to_capital < model.starting_sales_to_capital


def test_valid_mature_platform_example():
    model = assumptions(
        near_term_revenue_growth=(0.13, 0.11),
        revenue_fade_years=4,
        starting_operating_margin=0.32,
        mature_operating_margin=0.30,
        starting_sales_to_capital=1.3,
        mature_sales_to_capital=1.1,
    )

    assert model.near_term_years == 2
    assert model.mature_state_years == 4


def test_valid_shrinking_company_with_negative_near_term_growth():
    model = assumptions(near_term_revenue_growth=(-0.20, -0.05, 0.01))

    assert model.near_term_revenue_growth[0] == -0.20


def test_empty_near_term_growth_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        assumptions(near_term_revenue_growth=())


def test_more_than_five_near_term_growth_rates_are_rejected():
    with pytest.raises(ValueError, match="no more than five"):
        assumptions(near_term_revenue_growth=(0.20,) * 6)


@pytest.mark.parametrize("growth", [-1.0, -1.01])
def test_growth_at_or_below_negative_100_percent_is_rejected(growth):
    with pytest.raises(ValueError, match="greater than -100%"):
        assumptions(near_term_revenue_growth=(growth,))


@pytest.mark.parametrize("growth", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_near_term_growth_is_rejected(growth):
    with pytest.raises(ValueError, match="finite"):
        assumptions(near_term_revenue_growth=(growth,))


def test_very_high_but_finite_near_term_growth_is_allowed():
    assert assumptions(near_term_revenue_growth=(5.0,)).near_term_revenue_growth == (5.0,)


@pytest.mark.parametrize("terminal_growth", [-1.0, -1.2])
def test_terminal_growth_at_or_below_negative_100_percent_is_rejected(
    terminal_growth,
):
    with pytest.raises(ValueError, match="greater than -100%"):
        assumptions(terminal_growth=terminal_growth)


@pytest.mark.parametrize("terminal_growth", [float("nan"), float("inf")])
def test_non_finite_terminal_growth_is_rejected(terminal_growth):
    with pytest.raises(ValueError, match="finite"):
        assumptions(terminal_growth=terminal_growth)


@pytest.mark.parametrize("terminal_growth", [0.09, 0.10])
def test_terminal_growth_must_be_below_wacc(terminal_growth):
    with pytest.raises(ValueError, match="wacc must be greater"):
        assumptions(terminal_growth=terminal_growth, wacc=0.09)


def test_exact_fit_near_term_plus_fade_is_valid():
    model = assumptions(
        forecast_years=6,
        near_term_revenue_growth=(0.20, 0.15),
        revenue_fade_years=4,
    )

    assert model.mature_state_years == 0


def test_insufficient_forecast_horizon_is_rejected():
    with pytest.raises(ValueError, match="cover all near-term"):
        assumptions(
            forecast_years=5,
            near_term_revenue_growth=(0.20, 0.15),
            revenue_fade_years=4,
        )


def test_zero_fade_years_is_valid():
    model = assumptions(
        forecast_years=3,
        near_term_revenue_growth=(0.12, 0.10, 0.08),
        revenue_fade_years=0,
    )

    assert model.fade_years == 0
    assert model.mature_state_years == 0


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_invalid_forecast_years_are_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        assumptions(forecast_years=value)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_fade_years_are_rejected(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        assumptions(revenue_fade_years=value)


@pytest.mark.parametrize(
    ("starting", "mature"),
    [(0.20, 0.25), (0.30, 0.20), (-0.20, -0.05), (-0.10, 0.15)],
)
def test_positive_negative_expanding_and_contracting_margins_are_allowed(
    starting, mature
):
    model = assumptions(
        starting_operating_margin=starting,
        mature_operating_margin=mature,
    )

    assert model.starting_operating_margin == starting
    assert model.mature_operating_margin == mature


@pytest.mark.parametrize(
    "field", ["starting_operating_margin", "mature_operating_margin"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_margin_is_rejected(field, value):
    with pytest.raises(ValueError, match="finite"):
        assumptions(**{field: value})


def test_negative_starting_sales_to_capital_is_preserved():
    model = assumptions(starting_sales_to_capital=-1.25)

    assert model.starting_sales_to_capital == -1.25


@pytest.mark.parametrize("value", [0.0, 1e-10, -1e-10])
def test_zero_or_near_zero_starting_sales_to_capital_is_rejected(value):
    with pytest.raises(ValueError, match="zero or near zero"):
        assumptions(starting_sales_to_capital=value)


@pytest.mark.parametrize("value", [0.0, 1e-10, -1.0])
def test_non_positive_mature_sales_to_capital_is_rejected(value):
    with pytest.raises(ValueError, match="must be positive"):
        assumptions(mature_sales_to_capital=value)


def test_terminal_roic_and_reinvestment_rate_are_derived():
    model = assumptions(
        mature_operating_margin=0.20,
        operating_tax_rate=0.25,
        mature_sales_to_capital=2.0,
        terminal_growth=0.03,
    )

    assert model.after_tax_mature_operating_margin == pytest.approx(0.15)
    assert model.derived_terminal_roic == pytest.approx(0.30)
    assert model.terminal_reinvestment_rate == pytest.approx(0.10)
    assert "terminal_reinvestment_rate_exceeds_100_percent" not in model.validation_warnings


def test_zero_terminal_roic_with_nonzero_growth_is_rejected():
    with pytest.raises(ValueError, match="non-zero terminal_growth"):
        assumptions(mature_operating_margin=0.0, terminal_growth=0.02)


def test_zero_roic_and_zero_terminal_growth_has_unavailable_reinvestment_rate():
    model = assumptions(mature_operating_margin=0.0, terminal_growth=0.0)

    assert model.derived_terminal_roic == 0.0
    assert model.terminal_reinvestment_rate is None


def test_negative_terminal_roic_is_preserved_with_diagnostics():
    model = assumptions(mature_operating_margin=-0.10, terminal_growth=0.02)

    assert model.derived_terminal_roic < 0
    assert model.terminal_reinvestment_rate < 0
    assert "derived_terminal_roic_is_negative" in model.validation_warnings
    assert "terminal_reinvestment_rate_is_negative" in model.validation_warnings


def test_terminal_reinvestment_rate_above_100_percent_is_warning_not_error():
    model = assumptions(
        mature_operating_margin=0.02,
        operating_tax_rate=0.25,
        mature_sales_to_capital=1.0,
        terminal_growth=0.02,
    )

    assert model.terminal_reinvestment_rate == pytest.approx(4 / 3)
    assert "terminal_reinvestment_rate_exceeds_100_percent" in model.validation_warnings


def test_zero_terminal_growth_produces_zero_reinvestment_for_nonzero_roic():
    model = assumptions(terminal_growth=0.0)

    assert model.terminal_reinvestment_rate == 0.0


def test_terminal_growth_above_final_near_term_growth_is_diagnostic():
    model = assumptions(near_term_revenue_growth=(0.01,), terminal_growth=0.02)

    assert "terminal_growth_exceeds_final_near_term_growth" in model.validation_warnings


@pytest.mark.parametrize("tax_rate", [-0.01, 1.01, float("nan")])
def test_invalid_operating_tax_rate_is_rejected(tax_rate):
    with pytest.raises(ValueError):
        assumptions(operating_tax_rate=tax_rate)


def test_assumptions_are_immutable():
    model = assumptions()

    with pytest.raises(FrozenInstanceError):
        model.wacc = 0.08


def test_standard_revenue_growth_path_does_not_repeat_last_explicit_rate():
    path = generate_forecast_path(
        assumptions(
            forecast_years=10,
            near_term_revenue_growth=(0.30, 0.25, 0.20),
            revenue_fade_years=4,
            terminal_growth=0.04,
        )
    )

    assert path.revenue_growth_path == pytest.approx(
        (0.30, 0.25, 0.20, 0.16, 0.12, 0.08, 0.04, 0.04, 0.04, 0.04)
    )
    assert tuple(year.stage for year in path.years) == (
        "near_term", "near_term", "near_term",
        "fade", "fade", "fade", "fade",
        "mature", "mature", "mature",
    )


def test_one_near_term_year_and_additional_mature_years():
    path = generate_forecast_path(
        assumptions(
            forecast_years=5,
            near_term_revenue_growth=(0.20,),
            revenue_fade_years=2,
            terminal_growth=0.02,
        )
    )

    assert path.revenue_growth_path == pytest.approx((0.20, 0.11, 0.02, 0.02, 0.02))
    assert (path.near_term_year_count, path.fade_year_count, path.mature_year_count) == (1, 2, 2)


def test_five_explicit_near_term_growth_rates_are_used_exactly():
    explicit = (0.50, 0.40, 0.30, 0.20, 0.10)
    path = generate_forecast_path(
        assumptions(
            forecast_years=7,
            near_term_revenue_growth=explicit,
            revenue_fade_years=2,
            terminal_growth=0.02,
        )
    )

    assert path.revenue_growth_path[:5] == explicit
    assert path.revenue_growth_path[5:] == pytest.approx((0.06, 0.02))


def test_no_revenue_fade_moves_next_year_immediately_to_terminal_growth():
    path = generate_forecast_path(
        assumptions(
            forecast_years=5,
            near_term_revenue_growth=(0.20, 0.10),
            revenue_fade_years=0,
            terminal_growth=0.03,
        )
    )

    assert path.revenue_growth_path == (0.20, 0.10, 0.03, 0.03, 0.03)
    assert path.fade_year_count == 0


def test_exact_fit_near_term_and_fade_has_no_mature_years():
    path = generate_forecast_path(
        assumptions(
            forecast_years=5,
            near_term_revenue_growth=(0.20, 0.15),
            revenue_fade_years=3,
            terminal_growth=0.03,
        )
    )

    assert len(path.years) == 5
    assert path.mature_year_count == 0
    assert path.years[-1].stage == "fade"
    assert path.years[-1].revenue_growth == 0.03


def test_negative_growth_fades_toward_positive_terminal_growth():
    path = generate_forecast_path(
        assumptions(
            near_term_revenue_growth=(-0.20,),
            revenue_fade_years=3,
            terminal_growth=0.01,
        )
    )

    assert path.revenue_growth_path[:4] == pytest.approx((-0.20, -0.13, -0.06, 0.01))


def test_positive_growth_fades_toward_negative_terminal_growth():
    path = generate_forecast_path(
        assumptions(
            near_term_revenue_growth=(0.20,),
            revenue_fade_years=2,
            terminal_growth=-0.02,
        )
    )

    assert path.revenue_growth_path[:3] == pytest.approx((0.20, 0.09, -0.02))


def test_operating_margin_contraction_is_smooth_over_transition_horizon():
    path = generate_forecast_path(
        assumptions(
            forecast_years=6,
            near_term_revenue_growth=(0.20, 0.15),
            revenue_fade_years=2,
            starting_operating_margin=0.60,
            mature_operating_margin=0.30,
        )
    )

    assert path.operating_margin_path == pytest.approx(
        (0.60, 0.50, 0.40, 0.30, 0.30, 0.30)
    )


def test_operating_margin_expansion_reaches_exact_mature_margin():
    path = generate_forecast_path(
        assumptions(
            forecast_years=4,
            near_term_revenue_growth=(0.20,),
            revenue_fade_years=3,
            starting_operating_margin=0.10,
            mature_operating_margin=0.40,
        )
    )

    assert path.operating_margin_path == pytest.approx((0.10, 0.20, 0.30, 0.40))
    assert path.operating_margin_path[-1] == 0.40


@pytest.mark.parametrize(
    ("start", "mature"),
    [(-0.20, 0.20), (0.20, -0.20)],
)
def test_operating_margin_can_cross_zero(start, mature):
    path = generate_forecast_path(
        assumptions(
            forecast_years=4,
            near_term_revenue_growth=(0.20, 0.15),
            revenue_fade_years=2,
            starting_operating_margin=start,
            mature_operating_margin=mature,
            terminal_growth=0.0,
        )
    )

    assert path.operating_margin_path[0] == start
    assert path.operating_margin_path[-1] == mature


def test_no_fade_margin_reaches_mature_by_last_near_term_year():
    path = generate_forecast_path(
        assumptions(
            forecast_years=5,
            near_term_revenue_growth=(0.20, 0.15, 0.10),
            revenue_fade_years=0,
            starting_operating_margin=0.20,
            mature_operating_margin=0.30,
        )
    )

    assert path.operating_margin_path == pytest.approx((0.20, 0.25, 0.30, 0.30, 0.30))


def test_one_year_transition_preserves_start_then_mature_value():
    path = generate_forecast_path(
        assumptions(
            forecast_years=3,
            near_term_revenue_growth=(0.10,),
            revenue_fade_years=0,
            starting_operating_margin=0.20,
            mature_operating_margin=0.30,
            starting_sales_to_capital=2.0,
            mature_sales_to_capital=1.5,
        )
    )

    assert path.operating_margin_path == (0.20, 0.30, 0.30)
    assert path.sales_to_capital_path == (2.0, 1.5, 1.5)


@pytest.mark.parametrize(
    ("start", "mature", "expected"),
    [
        (2.0, 1.0, (2.0, 5 / 3, 4 / 3, 1.0, 1.0)),
        (1.0, 2.0, (1.0, 4 / 3, 5 / 3, 2.0, 2.0)),
        (1.5, 1.5, (1.5, 1.5, 1.5, 1.5, 1.5)),
    ],
)
def test_sales_to_capital_positive_contraction_expansion_and_constant(
    start, mature, expected
):
    path = generate_forecast_path(
        assumptions(
            forecast_years=5,
            near_term_revenue_growth=(0.20, 0.15),
            revenue_fade_years=2,
            starting_sales_to_capital=start,
            mature_sales_to_capital=mature,
        )
    )

    assert path.sales_to_capital_path == pytest.approx(expected)


def test_negative_starting_sales_to_capital_crossing_zero_is_rejected():
    with pytest.raises(ValueError, match="crosses zero"):
        generate_forecast_path(
            assumptions(
                forecast_years=4,
                near_term_revenue_growth=(0.20, 0.15),
                revenue_fade_years=2,
                starting_sales_to_capital=-1.0,
                mature_sales_to_capital=1.0,
            )
        )


def test_sales_to_capital_touching_zero_is_rejected():
    with pytest.raises(ValueError, match="zero or near zero"):
        generate_forecast_path(
            assumptions(
                forecast_years=3,
                near_term_revenue_growth=(0.20,),
                revenue_fade_years=2,
                starting_sales_to_capital=-1.0,
                mature_sales_to_capital=1.0,
            )
        )


def test_sales_to_capital_approaching_near_zero_is_rejected():
    with pytest.raises(ValueError, match="zero or near zero"):
        generate_forecast_path(
            assumptions(
                forecast_years=3,
                near_term_revenue_growth=(0.20,),
                revenue_fade_years=2,
                starting_sales_to_capital=-1.0,
                mature_sales_to_capital=1.000000001,
            )
        )


def test_worked_ten_year_structural_path_and_diagnostics():
    path = generate_forecast_path(
        assumptions(
            forecast_years=10,
            near_term_revenue_growth=(0.30, 0.25, 0.20),
            revenue_fade_years=5,
            terminal_growth=0.04,
            starting_operating_margin=0.60,
            mature_operating_margin=0.40,
            starting_sales_to_capital=2.0,
            mature_sales_to_capital=1.2,
        )
    )

    assert path.revenue_growth_path == pytest.approx(
        (0.30, 0.25, 0.20, 0.168, 0.136, 0.104, 0.072, 0.04, 0.04, 0.04)
    )
    assert (path.near_term_year_count, path.fade_year_count, path.mature_year_count) == (3, 5, 2)
    assert path.operating_margin_path[0] == 0.60
    assert path.operating_margin_path[7:] == (0.40, 0.40, 0.40)
    assert path.sales_to_capital_path[0] == 2.0
    assert path.sales_to_capital_path[7:] == (1.2, 1.2, 1.2)
    assert path.starting_values == (0.30, 0.60, 2.0)
    assert path.ending_values == (0.04, 0.40, 1.2)


def test_forecast_result_structures_are_immutable():
    year = ForecastYearAssumptions(1, "near_term", 0.10, 0.20, 1.5)
    path = MultiStageForecastPath((year,))

    with pytest.raises(FrozenInstanceError):
        year.stage = "mature"
    with pytest.raises(FrozenInstanceError):
        path.years = ()


def test_generator_requires_valid_assumptions_object():
    with pytest.raises(TypeError, match="MultiStageDCFAssumptions"):
        generate_forecast_path(object())
