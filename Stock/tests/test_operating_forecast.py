from dataclasses import FrozenInstanceError

import pytest

from Stock.valuation import (
    ForecastYearAssumptions,
    MultiStageDCFAssumptions,
    MultiStageForecastPath,
    OperatingForecastYear,
    build_operating_forecast,
)


def assumptions(**overrides):
    values = {
        "forecast_years": 3,
        "near_term_revenue_growth": (0.20, 0.10, 0.05),
        "revenue_fade_years": 0,
        "terminal_growth": 0.02,
        "starting_operating_margin": 0.30,
        "mature_operating_margin": 0.30,
        "starting_sales_to_capital": 2.0,
        "mature_sales_to_capital": 2.0,
        "operating_tax_rate": 0.20,
        "wacc": 0.09,
    }
    values.update(overrides)
    return MultiStageDCFAssumptions(**values)


def supplied_path(rows):
    return MultiStageForecastPath(tuple(ForecastYearAssumptions(*row) for row in rows))


def test_basic_revenue_compounding():
    forecast = build_operating_forecast(100.0, assumptions())

    assert tuple(year.revenue for year in forecast.years) == pytest.approx(
        (120.0, 132.0, 138.6)
    )
    assert tuple(year.delta_revenue for year in forecast.years) == pytest.approx(
        (20.0, 12.0, 6.6)
    )


def test_zero_growth_preserves_revenue_and_zero_reinvestment():
    model = assumptions(near_term_revenue_growth=(0.0, 0.0, 0.0))
    forecast = build_operating_forecast(100.0, model)

    assert tuple(year.revenue for year in forecast.years) == (100.0, 100.0, 100.0)
    assert tuple(year.delta_revenue for year in forecast.years) == (0.0, 0.0, 0.0)
    assert tuple(year.reinvestment for year in forecast.years) == (0.0, 0.0, 0.0)


def test_negative_growth_compounds_and_releases_capital():
    model = assumptions(near_term_revenue_growth=(-0.10, -0.20, -0.05))
    forecast = build_operating_forecast(100.0, model)

    assert tuple(year.revenue for year in forecast.years) == pytest.approx(
        (90.0, 72.0, 68.4)
    )
    assert tuple(year.reinvestment for year in forecast.years) == pytest.approx(
        (-5.0, -9.0, -1.8)
    )


def test_alternating_positive_and_negative_growth():
    model = assumptions(near_term_revenue_growth=(0.20, -0.10, 0.05))
    forecast = build_operating_forecast(100.0, model)

    assert tuple(year.revenue for year in forecast.years) == pytest.approx(
        (120.0, 108.0, 113.4)
    )
    assert tuple(year.delta_revenue for year in forecast.years) == pytest.approx(
        (20.0, -12.0, 5.4)
    )


def test_growth_close_to_negative_100_percent_remains_nonnegative():
    model = assumptions(
        forecast_years=1,
        near_term_revenue_growth=(-0.999999,),
    )
    forecast = build_operating_forecast(100.0, model)

    assert forecast.ending_revenue == pytest.approx(0.0001)
    assert forecast.ending_revenue >= 0


@pytest.mark.parametrize("starting_revenue", [-1.0, 0.0, float("nan"), float("inf")])
def test_invalid_starting_revenue_is_rejected(starting_revenue):
    with pytest.raises(ValueError):
        build_operating_forecast(starting_revenue, assumptions())


def test_standard_operating_income_nopat_reinvestment_and_fcff():
    model = assumptions(forecast_years=1, near_term_revenue_growth=(0.20,))
    forecast = build_operating_forecast(100.0, model)
    year = forecast.years[0]

    assert year.revenue == pytest.approx(120.0)
    assert year.operating_income == pytest.approx(36.0)
    assert year.nopat == pytest.approx(28.8)
    assert year.delta_revenue == pytest.approx(20.0)
    assert year.reinvestment == pytest.approx(10.0)
    assert year.fcff == pytest.approx(18.8)


@pytest.mark.parametrize(
    ("margin", "expected_income", "expected_nopat"),
    [(0.25, 25.0, 20.0), (0.0, 0.0, 0.0), (-0.25, -25.0, -20.0)],
)
def test_positive_zero_and_negative_operating_margin(
    margin, expected_income, expected_nopat
):
    model = assumptions(
        forecast_years=1,
        near_term_revenue_growth=(0.0,),
        starting_operating_margin=margin,
        mature_operating_margin=margin,
        terminal_growth=0.0,
    )
    year = build_operating_forecast(100.0, model).years[0]

    assert year.operating_income == pytest.approx(expected_income)
    assert year.nopat == pytest.approx(expected_nopat)


@pytest.mark.parametrize(
    ("start_margin", "mature_margin"),
    [(0.10, 0.30), (0.30, 0.10)],
)
def test_margin_expansion_and_contraction_flow_into_operating_income(
    start_margin, mature_margin
):
    model = assumptions(
        near_term_revenue_growth=(0.0, 0.0, 0.0),
        starting_operating_margin=start_margin,
        mature_operating_margin=mature_margin,
        terminal_growth=0.0,
    )
    forecast = build_operating_forecast(100.0, model)

    assert forecast.years[0].operating_income == pytest.approx(100 * start_margin)
    assert forecast.years[-1].operating_income == pytest.approx(100 * mature_margin)


@pytest.mark.parametrize(
    ("tax_rate", "expected_nopat"),
    [(0.0, 30.0), (0.80, 6.0), (1.0, 0.0)],
)
def test_zero_high_and_full_valid_tax_rates(tax_rate, expected_nopat):
    model = assumptions(
        forecast_years=1,
        near_term_revenue_growth=(0.0,),
        operating_tax_rate=tax_rate,
        terminal_growth=0.0,
    )
    year = build_operating_forecast(100.0, model).years[0]

    assert year.operating_tax_rate == tax_rate
    assert year.nopat == pytest.approx(expected_nopat)


def test_positive_growth_reinvestment_formula():
    model = assumptions(forecast_years=1, near_term_revenue_growth=(0.20,))
    year = build_operating_forecast(100.0, model).years[0]

    assert year.delta_revenue == pytest.approx(20.0)
    assert year.reinvestment == pytest.approx(10.0)


def test_negative_growth_reinvestment_is_negative_capital_release():
    model = assumptions(forecast_years=1, near_term_revenue_growth=(-0.10,))
    year = build_operating_forecast(100.0, model).years[0]

    assert year.delta_revenue == pytest.approx(-10.0)
    assert year.reinvestment == pytest.approx(-5.0)


def test_changing_sales_to_capital_changes_reinvestment():
    model = assumptions(
        forecast_years=3,
        near_term_revenue_growth=(0.10,),
        revenue_fade_years=2,
        terminal_growth=0.10,
        starting_sales_to_capital=1.0,
        mature_sales_to_capital=2.0,
        wacc=0.20,
    )
    forecast = build_operating_forecast(100.0, model)

    assert tuple(year.sales_to_capital for year in forecast.years) == pytest.approx(
        (1.0, 1.5, 2.0)
    )
    assert tuple(year.reinvestment for year in forecast.years) == pytest.approx(
        (10.0, 11.0 / 1.5, 12.1 / 2.0)
    )


@pytest.mark.parametrize(
    ("sales_to_capital", "expected_reinvestment"),
    [(1_000_000.0, 0.00002), (1e-6, 20_000_000.0)],
)
def test_very_high_and_low_valid_sales_to_capital(
    sales_to_capital, expected_reinvestment
):
    model = assumptions(
        forecast_years=1,
        near_term_revenue_growth=(0.20,),
        starting_sales_to_capital=sales_to_capital,
        mature_sales_to_capital=max(sales_to_capital, 1e-6),
    )
    year = build_operating_forecast(100.0, model).years[0]

    assert year.reinvestment == pytest.approx(expected_reinvestment)


@pytest.mark.parametrize("bad_denominator", [0.0, 1e-10])
def test_defensive_rejection_of_zero_or_near_zero_supplied_path_denominator(
    bad_denominator
):
    model = assumptions(forecast_years=1, near_term_revenue_growth=(0.10,))
    path = supplied_path([(1, "near_term", 0.10, 0.30, bad_denominator)])

    with pytest.raises(ValueError, match="zero or near zero"):
        build_operating_forecast(100.0, model, path)


def test_high_growth_can_create_negative_fcff_from_large_reinvestment():
    model = assumptions(
        forecast_years=1,
        near_term_revenue_growth=(1.0,),
        starting_operating_margin=0.10,
        mature_operating_margin=0.10,
        starting_sales_to_capital=0.50,
        mature_sales_to_capital=0.50,
    )
    year = build_operating_forecast(100.0, model).years[0]

    assert year.nopat == pytest.approx(16.0)
    assert year.reinvestment == pytest.approx(200.0)
    assert year.fcff == pytest.approx(-184.0)


def test_low_growth_produces_higher_fcff_conversion():
    high_growth = assumptions(forecast_years=1, near_term_revenue_growth=(0.20,))
    low_growth = assumptions(forecast_years=1, near_term_revenue_growth=(0.02,))

    high = build_operating_forecast(100.0, high_growth).years[0]
    low = build_operating_forecast(100.0, low_growth).years[0]

    assert low.fcff / low.nopat > high.fcff / high.nopat


def test_negative_growth_capital_release_increases_fcff():
    model = assumptions(forecast_years=1, near_term_revenue_growth=(-0.10,))
    year = build_operating_forecast(100.0, model).years[0]

    assert year.reinvestment == -5.0
    assert year.fcff == pytest.approx(year.nopat + 5.0)


def test_negative_nopat_and_negative_fcff_are_preserved():
    model = assumptions(
        forecast_years=1,
        near_term_revenue_growth=(0.10,),
        starting_operating_margin=-0.20,
        mature_operating_margin=-0.20,
        terminal_growth=-0.01,
    )
    year = build_operating_forecast(100.0, model).years[0]

    assert year.nopat < 0
    assert year.fcff < 0


def test_fcff_can_be_exactly_zero():
    model = assumptions(
        forecast_years=1,
        near_term_revenue_growth=(0.20,),
        starting_operating_margin=0.10416666666666667,
        mature_operating_margin=0.10416666666666667,
    )
    year = build_operating_forecast(100.0, model).years[0]

    assert year.nopat == pytest.approx(10.0)
    assert year.reinvestment == pytest.approx(10.0)
    assert year.fcff == pytest.approx(0.0)


def test_worked_three_year_example_compounds_and_changes_efficiency():
    model = assumptions(
        forecast_years=3,
        near_term_revenue_growth=(0.20,),
        revenue_fade_years=2,
        terminal_growth=0.10,
        starting_operating_margin=0.30,
        mature_operating_margin=0.25,
        starting_sales_to_capital=2.0,
        mature_sales_to_capital=1.5,
        wacc=0.20,
    )
    forecast = build_operating_forecast(100.0, model)

    first, second, third = forecast.years
    assert (
        first.revenue,
        first.operating_income,
        first.nopat,
        first.delta_revenue,
        first.reinvestment,
        first.fcff,
    ) == pytest.approx((120.0, 36.0, 28.8, 20.0, 10.0, 18.8))
    assert second.revenue == pytest.approx(138.0)
    assert second.sales_to_capital == pytest.approx(1.75)
    assert second.reinvestment == pytest.approx(18.0 / 1.75)
    assert third.revenue == pytest.approx(151.8)
    assert third.sales_to_capital == 1.5
    assert third.reinvestment == pytest.approx(13.8 / 1.5)


def test_arithmetic_forecast_diagnostics():
    forecast = build_operating_forecast(100.0, assumptions())

    assert forecast.ending_revenue == pytest.approx(138.6)
    assert forecast.cumulative_revenue_growth == pytest.approx(0.386)
    assert forecast.total_nopat == pytest.approx(
        sum(year.nopat for year in forecast.years)
    )
    assert forecast.total_reinvestment == pytest.approx(
        sum(year.reinvestment for year in forecast.years)
    )
    assert forecast.total_fcff == pytest.approx(
        forecast.total_nopat - forecast.total_reinvestment
    )


def test_supplied_path_must_match_length_indexes_and_stages():
    model = assumptions()
    bad_paths = (
        supplied_path([(1, "near_term", 0.20, 0.30, 2.0)]),
        supplied_path([
            (1, "near_term", 0.20, 0.30, 2.0),
            (3, "near_term", 0.10, 0.30, 2.0),
            (4, "near_term", 0.05, 0.30, 2.0),
        ]),
        supplied_path([
            (1, "near_term", 0.20, 0.30, 2.0),
            (2, "mature", 0.10, 0.30, 2.0),
            (3, "near_term", 0.05, 0.30, 2.0),
        ]),
    )

    for path in bad_paths:
        with pytest.raises(ValueError):
            build_operating_forecast(100.0, model, path)


def test_supplied_path_non_finite_value_is_rejected():
    model = assumptions(forecast_years=1, near_term_revenue_growth=(0.10,))
    path = supplied_path([(1, "near_term", 0.10, float("nan"), 2.0)])

    with pytest.raises(ValueError, match="non-finite"):
        build_operating_forecast(100.0, model, path)


def test_operating_forecast_structures_are_immutable():
    forecast = build_operating_forecast(100.0, assumptions())
    assert isinstance(forecast.years[0], OperatingForecastYear)

    with pytest.raises(FrozenInstanceError):
        forecast.starting_revenue = 200.0
    with pytest.raises(FrozenInstanceError):
        forecast.years[0].fcff = 0.0

