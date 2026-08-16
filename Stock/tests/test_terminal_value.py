from dataclasses import FrozenInstanceError, replace

import pytest

from Stock.valuation import (
    MultiStageDCFAssumptions,
    TerminalValueResult,
    build_operating_forecast,
    calculate_terminal_value,
    discount_operating_forecast,
)


def assumptions(**overrides):
    values = {
        "forecast_years": 1,
        "near_term_revenue_growth": (0.03,),
        "revenue_fade_years": 0,
        "terminal_growth": 0.03,
        "starting_operating_margin": 0.25,
        "mature_operating_margin": 0.25,
        "starting_sales_to_capital": 1.5,
        "mature_sales_to_capital": 1.5,
        "operating_tax_rate": 0.20,
        "wacc": 0.09,
    }
    values.update(overrides)
    return MultiStageDCFAssumptions(**values)


def terminal_inputs(model, starting_revenue=100.0):
    operating = build_operating_forecast(starting_revenue, model)
    discounted = discount_operating_forecast(operating, model)
    return operating, discounted


def test_standard_positive_terminal_economics():
    model = assumptions()
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    expected_terminal_revenue = operating.ending_revenue * 1.03
    expected_nopat = expected_terminal_revenue * 0.25 * 0.80
    assert result.terminal_year_revenue == pytest.approx(expected_terminal_revenue)
    assert result.terminal_nopat == pytest.approx(expected_nopat)
    assert result.derived_terminal_roic == pytest.approx(0.30)
    assert result.terminal_reinvestment_rate == pytest.approx(0.10)
    assert result.terminal_reinvestment == pytest.approx(expected_nopat * 0.10)
    assert result.terminal_fcff == pytest.approx(expected_nopat * 0.90)
    assert result.terminal_value == pytest.approx(result.terminal_fcff / 0.06)


def test_zero_terminal_growth_uses_zero_reinvestment_rate():
    model = assumptions(
        near_term_revenue_growth=(0.0,),
        terminal_growth=0.0,
    )
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    assert result.terminal_reinvestment_rate == 0.0
    assert result.terminal_reinvestment == 0.0
    assert result.terminal_fcff == pytest.approx(result.terminal_nopat)
    assert result.terminal_value == pytest.approx(result.terminal_fcff / model.wacc)


def test_zero_growth_and_zero_roic_has_explicit_zero_reinvestment():
    model = assumptions(
        near_term_revenue_growth=(0.0,),
        terminal_growth=0.0,
        starting_operating_margin=0.0,
        mature_operating_margin=0.0,
    )
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    assert result.derived_terminal_roic == 0.0
    assert result.terminal_reinvestment_rate == 0.0
    assert result.terminal_nopat == 0.0
    assert result.terminal_fcff == 0.0
    assert result.terminal_value == 0.0


def test_high_terminal_roic_produces_low_reinvestment_rate():
    model = assumptions(
        starting_operating_margin=0.40,
        mature_operating_margin=0.40,
        starting_sales_to_capital=3.0,
        mature_sales_to_capital=3.0,
    )
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    assert result.derived_terminal_roic == pytest.approx(0.96)
    assert result.terminal_reinvestment_rate == pytest.approx(0.03 / 0.96)
    assert result.terminal_fcff_to_nopat > 0.90


def test_low_terminal_roic_produces_high_reinvestment_rate():
    model = assumptions(
        terminal_growth=0.02,
        near_term_revenue_growth=(0.02,),
        starting_operating_margin=0.04,
        mature_operating_margin=0.04,
        starting_sales_to_capital=1.0,
        mature_sales_to_capital=1.0,
    )
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    assert result.derived_terminal_roic == pytest.approx(0.032)
    assert result.terminal_reinvestment_rate == pytest.approx(0.625)


def test_reinvestment_rate_above_100_percent_and_negative_fcff_are_preserved():
    model = assumptions(
        terminal_growth=0.02,
        near_term_revenue_growth=(0.02,),
        starting_operating_margin=0.02,
        mature_operating_margin=0.02,
        starting_sales_to_capital=1.0,
        mature_sales_to_capital=1.0,
    )
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    assert result.terminal_reinvestment_rate == pytest.approx(1.25)
    assert result.terminal_fcff_to_nopat == pytest.approx(-0.25)
    assert result.terminal_fcff < 0
    assert result.terminal_value < 0
    assert "terminal_reinvestment_rate_exceeds_100_percent" in result.warnings
    assert "negative_terminal_fcff" in result.warnings
    assert "negative_terminal_value" in result.warnings


def test_negative_mature_margin_is_mathematically_preserved_with_warnings():
    model = assumptions(
        terminal_growth=0.02,
        near_term_revenue_growth=(0.02,),
        starting_operating_margin=-0.10,
        mature_operating_margin=-0.10,
    )
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    assert result.terminal_operating_income < 0
    assert result.terminal_nopat < 0
    assert result.derived_terminal_roic < 0
    assert result.terminal_reinvestment_rate < 0
    assert result.terminal_fcff < 0
    assert "derived_terminal_roic_is_negative" in result.warnings
    assert "terminal_reinvestment_rate_is_negative" in result.warnings
    assert "negative_terminal_fcff" in result.warnings


def test_wacc_only_slightly_above_terminal_growth_is_supported():
    model = assumptions(wacc=0.030001)
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    assert result.terminal_value == pytest.approx(result.terminal_fcff / 0.000001)
    assert result.terminal_value > 0


def test_exact_fit_final_fade_year_can_support_terminal_value():
    model = assumptions(
        forecast_years=3,
        near_term_revenue_growth=(0.20,),
        revenue_fade_years=2,
        terminal_growth=0.03,
        starting_operating_margin=0.35,
        mature_operating_margin=0.25,
        starting_sales_to_capital=2.0,
        mature_sales_to_capital=1.5,
    )
    operating, discounted = terminal_inputs(model)
    assert operating.years[-1].stage == "fade"

    result = calculate_terminal_value(operating, discounted, model)

    assert operating.years[-1].revenue_growth == model.terminal_growth
    assert operating.years[-1].operating_margin == model.mature_operating_margin
    assert operating.years[-1].sales_to_capital == model.mature_sales_to_capital
    assert result.terminal_value > 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revenue_growth", 0.04, "Revenue growth"),
        ("operating_margin", 0.30, "operating margin"),
        ("sales_to_capital", 1.6, "Sales-to-Capital"),
    ],
)
def test_forecast_not_at_mature_assumption_is_rejected(field, value, message):
    model = assumptions()
    operating, discounted = terminal_inputs(model)
    changed_final = replace(operating.years[-1], **{field: value})
    changed = replace(operating, years=(changed_final,))

    with pytest.raises(ValueError, match=message):
        calculate_terminal_value(changed, discounted, model)


def test_terminal_value_is_discounted_by_final_explicit_year_factor():
    model = assumptions(
        forecast_years=3,
        near_term_revenue_growth=(0.03, 0.03, 0.03),
    )
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)

    expected_factor = 1 / (1 + model.wacc) ** 3
    assert result.terminal_discount_factor == pytest.approx(expected_factor)
    assert result.terminal_discount_factor == pytest.approx(
        discounted.years[-1].discount_factor
    )
    assert result.present_value_terminal_value == pytest.approx(
        result.terminal_value / (1 + model.wacc) ** 3
    )


def test_inconsistent_final_discount_factor_is_rejected():
    model = assumptions()
    operating, discounted = terminal_inputs(model)
    bad_year = replace(discounted.years[-1], discount_factor=0.50)
    bad_discounted = replace(discounted, years=(bad_year,))

    with pytest.raises(ValueError, match="discount factor"):
        calculate_terminal_value(operating, bad_discounted, model)


def test_discounted_fcff_must_match_operating_forecast():
    model = assumptions()
    operating, discounted = terminal_inputs(model)
    bad_year = replace(discounted.years[-1], fcff=discounted.years[-1].fcff + 1)
    bad_discounted = replace(discounted, years=(bad_year,))

    with pytest.raises(ValueError, match="FCFF mismatch"):
        calculate_terminal_value(operating, bad_discounted, model)


def test_worked_terminal_example():
    model = assumptions()
    operating, discounted = terminal_inputs(model, starting_revenue=150 / 1.03)
    result = calculate_terminal_value(operating, discounted, model)

    assert result.final_forecast_revenue == pytest.approx(150.0)
    assert result.terminal_year_revenue == pytest.approx(154.5)
    assert result.terminal_operating_income == pytest.approx(38.625)
    assert result.terminal_nopat == pytest.approx(30.9)
    assert result.derived_terminal_roic == pytest.approx(0.30)
    assert result.terminal_reinvestment_rate == pytest.approx(0.10)
    assert result.terminal_reinvestment == pytest.approx(3.09)
    assert result.terminal_fcff == pytest.approx(27.81)
    assert result.terminal_value == pytest.approx(463.5)
    assert result.present_value_terminal_value == pytest.approx(463.5 / 1.09)


def test_terminal_result_is_immutable():
    model = assumptions()
    operating, discounted = terminal_inputs(model)
    result = calculate_terminal_value(operating, discounted, model)
    assert isinstance(result, TerminalValueResult)

    with pytest.raises(FrozenInstanceError):
        result.terminal_value = 0.0


def test_input_types_and_empty_forecasts_are_rejected():
    model = assumptions()
    operating, discounted = terminal_inputs(model)

    with pytest.raises(TypeError, match="operating_forecast"):
        calculate_terminal_value(object(), discounted, model)
    with pytest.raises(TypeError, match="discounted_forecast"):
        calculate_terminal_value(operating, object(), model)
    with pytest.raises(TypeError, match="assumptions"):
        calculate_terminal_value(operating, discounted, object())

