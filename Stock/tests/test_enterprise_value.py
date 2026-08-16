from dataclasses import FrozenInstanceError, replace

import pytest

from Stock.valuation import (
    EnterpriseValueResult,
    MultiStageDCFAssumptions,
    aggregate_enterprise_value,
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


def upstream_results(model=None):
    model = model or assumptions()
    operating = build_operating_forecast(100.0, model)
    discounted = discount_operating_forecast(operating, model)
    terminal = calculate_terminal_value(operating, discounted, model)
    return model, discounted, terminal


def with_component_values(discounted, terminal, explicit_pv, terminal_pv):
    first = replace(discounted.years[0], present_value_fcff=explicit_pv)
    changed_discounted = replace(discounted, years=(first,))
    changed_terminal = replace(
        terminal, present_value_terminal_value=terminal_pv
    )
    return changed_discounted, changed_terminal


def test_standard_positive_enterprise_value_and_component_shares():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, 200.0, 300.0
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.explicit_forecast_pv == 200.0
    assert result.terminal_value_pv == 300.0
    assert result.enterprise_value == 500.0
    assert result.explicit_value_share == pytest.approx(0.40)
    assert result.terminal_value_share == pytest.approx(0.60)
    assert result.explicit_value_share + result.terminal_value_share == pytest.approx(1)


def test_negative_explicit_pv_with_positive_terminal_pv_is_preserved():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, -100.0, 300.0
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.enterprise_value == 200.0
    assert result.explicit_value_share == pytest.approx(-0.50)
    assert result.terminal_value_share == pytest.approx(1.50)
    assert "terminal_value_dominates_enterprise_value" in result.warnings


def test_positive_explicit_pv_with_negative_terminal_pv_is_preserved():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, 300.0, -100.0
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.enterprise_value == 200.0
    assert result.explicit_value_share == pytest.approx(1.50)
    assert result.terminal_value_share == pytest.approx(-0.50)


def test_both_negative_components_produce_negative_ev_without_shares():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, -100.0, -200.0
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.enterprise_value == -300.0
    assert result.explicit_value_share is None
    assert result.terminal_value_share is None
    assert "negative_enterprise_value" in result.warnings


def test_exactly_zero_ev_has_unavailable_shares():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, 100.0, -100.0
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.enterprise_value == 0.0
    assert result.explicit_value_share is None
    assert result.terminal_value_share is None
    assert "zero_enterprise_value" in result.warnings


def test_positive_near_zero_ev_is_not_used_as_share_denominator():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, 1e-12, -0.5e-12
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.enterprise_value == pytest.approx(0.5e-12)
    assert result.explicit_value_share is None
    assert result.terminal_value_share is None
    assert "zero_enterprise_value" in result.warnings


def test_terminal_value_above_80_percent_adds_dependency_warning():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, 10.0, 90.0
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.terminal_value_share == pytest.approx(0.90)
    assert "terminal_value_dominates_enterprise_value" in result.warnings


def test_terminal_value_exactly_80_percent_does_not_warn():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, 20.0, 80.0
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.terminal_value_share == pytest.approx(0.80)
    assert "terminal_value_dominates_enterprise_value" not in result.warnings


def test_zero_terminal_value_and_zero_explicit_value_cases():
    model, discounted, terminal = upstream_results()
    no_terminal_discounted, no_terminal = with_component_values(
        discounted, terminal, 100.0, 0.0
    )
    no_explicit_discounted, no_explicit = with_component_values(
        discounted, terminal, 0.0, 100.0
    )

    no_terminal_result = aggregate_enterprise_value(
        no_terminal_discounted, no_terminal, model
    )
    no_explicit_result = aggregate_enterprise_value(
        no_explicit_discounted, no_explicit, model
    )

    assert no_terminal_result.explicit_value_share == 1.0
    assert no_terminal_result.terminal_value_share == 0.0
    assert no_explicit_result.explicit_value_share == 0.0
    assert no_explicit_result.terminal_value_share == 1.0


def test_worked_arithmetic_example():
    model, discounted, terminal = upstream_results()
    discounted, terminal = with_component_values(
        discounted, terminal, 250.0, 750.0
    )

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.enterprise_value == 1000.0
    assert result.explicit_value_share == 0.25
    assert result.terminal_value_share == 0.75


def test_real_upstream_results_are_aggregated_without_recomputation():
    model, discounted, terminal = upstream_results()

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert result.explicit_forecast_pv == discounted.total_present_value_fcff
    assert result.terminal_value_pv == terminal.present_value_terminal_value
    assert result.enterprise_value == pytest.approx(
        discounted.total_present_value_fcff
        + terminal.present_value_terminal_value
    )
    assert result.forecast_years == model.forecast_years


def test_wacc_mismatches_are_rejected():
    model, discounted, terminal = upstream_results()

    with pytest.raises(ValueError, match="discounted_forecast WACC"):
        aggregate_enterprise_value(
            replace(discounted, wacc=0.10), terminal, model
        )
    with pytest.raises(ValueError, match="terminal_result WACC"):
        aggregate_enterprise_value(
            discounted, replace(terminal, wacc=0.10), model
        )


def test_terminal_growth_mismatch_is_rejected():
    model, discounted, terminal = upstream_results()

    with pytest.raises(ValueError, match="terminal growth"):
        aggregate_enterprise_value(
            discounted,
            replace(terminal, terminal_growth=0.02),
            model,
        )


def test_forecast_year_mismatch_is_rejected():
    model, discounted, terminal = upstream_results()
    extra_year = replace(discounted.years[0], year_index=2)

    with pytest.raises(ValueError, match="length"):
        aggregate_enterprise_value(
            replace(discounted, years=discounted.years + (extra_year,)),
            terminal,
            model,
        )


def test_terminal_discount_factor_must_match_final_explicit_factor():
    model, discounted, terminal = upstream_results()

    with pytest.raises(ValueError, match="terminal discount factor"):
        aggregate_enterprise_value(
            discounted,
            replace(terminal, terminal_discount_factor=0.50),
            model,
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_non_finite_explicit_pv_is_rejected(bad_value):
    model, discounted, terminal = upstream_results()
    bad_year = replace(
        discounted.years[0], present_value_fcff=bad_value
    )

    with pytest.raises(ValueError, match="explicit forecast present value"):
        aggregate_enterprise_value(
            replace(discounted, years=(bad_year,)), terminal, model
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_non_finite_terminal_pv_is_rejected(bad_value):
    model, discounted, terminal = upstream_results()

    with pytest.raises(ValueError, match="terminal present value"):
        aggregate_enterprise_value(
            discounted,
            replace(terminal, present_value_terminal_value=bad_value),
            model,
        )


def test_terminal_warnings_are_preserved():
    model, discounted, terminal = upstream_results()
    terminal = replace(terminal, warnings=("negative_terminal_fcff",))

    result = aggregate_enterprise_value(discounted, terminal, model)

    assert "negative_terminal_fcff" in result.warnings


def test_enterprise_value_result_is_immutable():
    model, discounted, terminal = upstream_results()
    result = aggregate_enterprise_value(discounted, terminal, model)
    assert isinstance(result, EnterpriseValueResult)

    with pytest.raises(FrozenInstanceError):
        result.enterprise_value = 0.0


def test_input_types_are_validated():
    model, discounted, terminal = upstream_results()

    with pytest.raises(TypeError, match="discounted_forecast"):
        aggregate_enterprise_value(object(), terminal, model)
    with pytest.raises(TypeError, match="terminal_result"):
        aggregate_enterprise_value(discounted, object(), model)
    with pytest.raises(TypeError, match="assumptions"):
        aggregate_enterprise_value(discounted, terminal, object())

