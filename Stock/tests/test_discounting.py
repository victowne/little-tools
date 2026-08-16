from dataclasses import FrozenInstanceError, replace

import pytest

from Stock.valuation import (
    DiscountedForecastYear,
    MultiStageDCFAssumptions,
    MultiStageDiscountedForecast,
    MultiStageOperatingForecast,
    OperatingForecastYear,
    build_operating_forecast,
    discount_operating_forecast,
)


def assumptions(*, years=1, growth=None, wacc=0.10, **overrides):
    growth_path = growth if growth is not None else (0.0,) * years
    values = {
        "forecast_years": years,
        "near_term_revenue_growth": tuple(growth_path),
        "revenue_fade_years": 0,
        "terminal_growth": 0.0,
        "starting_operating_margin": 0.30,
        "mature_operating_margin": 0.30,
        "starting_sales_to_capital": 2.0,
        "mature_sales_to_capital": 2.0,
        "operating_tax_rate": 0.20,
        "wacc": wacc,
    }
    values.update(overrides)
    return MultiStageDCFAssumptions(**values)


def operating_forecast_with_fcff(fcff_values, model):
    rows = tuple(
        OperatingForecastYear(
            year_index=index,
            stage="near_term",
            revenue_growth=0.0,
            revenue=100.0,
            operating_margin=0.30,
            operating_income=30.0,
            operating_tax_rate=0.20,
            nopat=24.0,
            sales_to_capital=2.0,
            delta_revenue=0.0,
            reinvestment=0.0,
            fcff=fcff,
        )
        for index, fcff in enumerate(fcff_values, start=1)
    )
    return MultiStageOperatingForecast(100.0, rows)


def test_one_year_fcff_is_discounted_at_year_end():
    model = assumptions()
    result = discount_operating_forecast(
        operating_forecast_with_fcff((100.0,), model), model
    )
    year = result.years[0]

    assert year.discount_factor == pytest.approx(1 / 1.10)
    assert year.present_value_fcff == pytest.approx(100 / 1.10)


def test_two_year_equal_fcff_uses_consecutive_year_end_exponents():
    model = assumptions(years=2)
    result = discount_operating_forecast(
        operating_forecast_with_fcff((100.0, 100.0), model), model
    )

    assert tuple(year.discount_factor for year in result.years) == pytest.approx(
        (1 / 1.10, 1 / 1.10**2)
    )
    assert tuple(year.present_value_fcff for year in result.years) == pytest.approx(
        (100 / 1.10, 100 / 1.10**2)
    )


def test_different_fcff_values_are_preserved_and_discounted():
    model = assumptions(years=3)
    result = discount_operating_forecast(
        operating_forecast_with_fcff((50.0, 100.0, 150.0), model), model
    )

    assert tuple(year.fcff for year in result.years) == (50.0, 100.0, 150.0)
    assert tuple(year.present_value_fcff for year in result.years) == pytest.approx(
        (50 / 1.1, 100 / 1.1**2, 150 / 1.1**3)
    )


@pytest.mark.parametrize(
    ("fcff", "expected_pv"),
    [(0.0, 0.0), (-100.0, -100 / 1.1)],
)
def test_zero_and_negative_fcff_are_preserved(fcff, expected_pv):
    model = assumptions()
    result = discount_operating_forecast(
        operating_forecast_with_fcff((fcff,), model), model
    )

    assert result.years[0].fcff == fcff
    assert result.years[0].present_value_fcff == pytest.approx(expected_pv)


@pytest.mark.parametrize("wacc", [1e-9, 1.0, 5.0])
def test_very_low_and_high_valid_wacc(wacc):
    model = assumptions(wacc=wacc)
    result = discount_operating_forecast(
        operating_forecast_with_fcff((100.0,), model), model
    )

    assert result.years[0].discount_factor == pytest.approx(1 / (1 + wacc))
    assert 0 < result.years[0].discount_factor < 1


def test_positive_wacc_discount_factors_decline_monotonically():
    model = assumptions(years=5)
    result = discount_operating_forecast(
        operating_forecast_with_fcff((10.0,) * 5, model), model
    )
    factors = tuple(year.discount_factor for year in result.years)

    assert factors[0] < 1
    assert all(later < earlier for earlier, later in zip(factors, factors[1:]))


def test_total_explicit_present_value_is_sum_of_annual_present_values():
    model = assumptions(years=3)
    result = discount_operating_forecast(
        operating_forecast_with_fcff((20.0, 30.0, 40.0), model), model
    )

    assert result.total_present_value_fcff == pytest.approx(
        20 / 1.1 + 30 / 1.1**2 + 40 / 1.1**3
    )


def test_structural_integrity_and_source_forecast_are_preserved():
    model = assumptions(
        years=3,
        growth=(0.20,),
        revenue_fade_years=1,
        terminal_growth=0.05,
    )
    operating = build_operating_forecast(100.0, model)
    before = operating.years
    result = discount_operating_forecast(operating, model)

    assert len(result.years) == len(operating.years)
    assert tuple(year.year_index for year in result.years) == tuple(
        year.year_index for year in operating.years
    )
    assert tuple(year.stage for year in result.years) == tuple(
        year.stage for year in operating.years
    )
    assert tuple(year.fcff for year in result.years) == tuple(
        year.fcff for year in operating.years
    )
    assert operating.years == before


def test_discounting_result_structures_are_immutable():
    model = assumptions()
    result = discount_operating_forecast(
        operating_forecast_with_fcff((100.0,), model), model
    )
    assert isinstance(result, MultiStageDiscountedForecast)
    assert isinstance(result.years[0], DiscountedForecastYear)

    with pytest.raises(FrozenInstanceError):
        result.wacc = 0.20
    with pytest.raises(FrozenInstanceError):
        result.years[0].present_value_fcff = 0.0


def test_arithmetic_diagnostics():
    model = assumptions(years=3)
    result = discount_operating_forecast(
        operating_forecast_with_fcff((20.0, 30.0, 40.0), model), model
    )

    assert result.total_undiscounted_fcff == 90.0
    assert result.explicit_pv_to_undiscounted_fcff == pytest.approx(
        result.total_present_value_fcff / 90.0
    )


def test_zero_total_undiscounted_fcff_has_unavailable_ratio():
    model = assumptions(years=2)
    result = discount_operating_forecast(
        operating_forecast_with_fcff((10.0, -10.0), model), model
    )

    assert result.total_undiscounted_fcff == 0.0
    assert result.explicit_pv_to_undiscounted_fcff is None


def test_worked_three_year_operating_example_at_ten_percent_wacc():
    expected_fcff = (18.8, 20.074285714285715, 21.16)
    model = assumptions(years=3, wacc=0.10)
    operating = operating_forecast_with_fcff(expected_fcff, model)
    result = discount_operating_forecast(operating, model)
    expected_factors = (1 / 1.1, 1 / 1.1**2, 1 / 1.1**3)
    expected_pv = tuple(
        fcff * factor for fcff, factor in zip(expected_fcff, expected_factors)
    )

    assert tuple(year.fcff for year in result.years) == pytest.approx(expected_fcff)
    assert tuple(year.discount_factor for year in result.years) == pytest.approx(
        expected_factors
    )
    assert tuple(year.present_value_fcff for year in result.years) == pytest.approx(
        expected_pv
    )
    assert result.total_present_value_fcff == pytest.approx(sum(expected_pv))


def test_empty_forecast_is_rejected():
    model = assumptions()
    empty = MultiStageOperatingForecast(100.0, ())

    with pytest.raises(ValueError, match="at least one"):
        discount_operating_forecast(empty, model)


def test_forecast_length_must_match_assumptions():
    model = assumptions(years=2)
    short = operating_forecast_with_fcff((100.0,), model)

    with pytest.raises(ValueError, match="length"):
        discount_operating_forecast(short, model)


def test_year_indexes_must_be_consecutive_from_one():
    model = assumptions(years=2)
    operating = operating_forecast_with_fcff((100.0, 100.0), model)
    bad = replace(operating, years=(operating.years[0], replace(operating.years[1], year_index=3)))

    with pytest.raises(ValueError, match="consecutive"):
        discount_operating_forecast(bad, model)


def test_stage_must_match_assumptions():
    model = assumptions(years=2)
    operating = operating_forecast_with_fcff((100.0, 100.0), model)
    bad = replace(operating, years=(replace(operating.years[0], stage="mature"), operating.years[1]))

    with pytest.raises(ValueError, match="stage"):
        discount_operating_forecast(bad, model)


@pytest.mark.parametrize("fcff", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_fcff_is_rejected(fcff):
    model = assumptions()
    operating = operating_forecast_with_fcff((fcff,), model)

    with pytest.raises(ValueError, match="finite"):
        discount_operating_forecast(operating, model)


def test_input_types_are_validated():
    model = assumptions()
    operating = operating_forecast_with_fcff((100.0,), model)

    with pytest.raises(TypeError, match="operating_forecast"):
        discount_operating_forecast(object(), model)
    with pytest.raises(TypeError, match="assumptions"):
        discount_operating_forecast(operating, object())
