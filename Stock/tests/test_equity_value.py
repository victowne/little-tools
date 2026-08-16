from dataclasses import FrozenInstanceError

import pytest

from Stock.valuation import (
    EnterpriseValueResult,
    EquityValueResult,
    bridge_enterprise_to_equity_value,
)


def enterprise_result(value, warnings=()):
    return EnterpriseValueResult(
        explicit_forecast_pv=value * 0.40,
        terminal_value_pv=value * 0.60,
        enterprise_value=value,
        terminal_value_share=0.60 if value > 1e-12 else None,
        explicit_value_share=0.40 if value > 1e-12 else None,
        forecast_years=10,
        wacc=0.09,
        terminal_growth=0.03,
        warnings=tuple(warnings),
    )


def test_standard_positive_net_debt_bridge():
    result = bridge_enterprise_to_equity_value(
        enterprise_result(500.0), 50.0
    )

    assert result.enterprise_value == 500.0
    assert result.net_debt == 50.0
    assert result.equity_value == 450.0
    assert result.net_debt_to_enterprise_value == pytest.approx(0.10)


def test_zero_net_debt_preserves_enterprise_value():
    result = bridge_enterprise_to_equity_value(
        enterprise_result(500.0), 0.0
    )

    assert result.equity_value == 500.0
    assert result.net_debt_to_enterprise_value == 0.0


def test_net_cash_increases_equity_value_and_adds_warning():
    result = bridge_enterprise_to_equity_value(
        enterprise_result(500.0), -100.0
    )

    assert result.equity_value == 600.0
    assert result.net_debt_to_enterprise_value == pytest.approx(-0.20)
    assert "net_cash_position" in result.warnings


def test_net_debt_equal_to_ev_produces_zero_equity_value():
    result = bridge_enterprise_to_equity_value(
        enterprise_result(500.0), 500.0
    )

    assert result.equity_value == 0.0
    assert result.net_debt_to_enterprise_value == 1.0
    assert "zero_equity_value" in result.warnings
    assert "net_debt_exceeds_enterprise_value" not in result.warnings


def test_net_debt_above_ev_produces_negative_equity_value():
    result = bridge_enterprise_to_equity_value(
        enterprise_result(500.0), 600.0
    )

    assert result.equity_value == -100.0
    assert result.net_debt_to_enterprise_value == pytest.approx(1.20)
    assert "net_debt_exceeds_enterprise_value" in result.warnings
    assert "negative_equity_value" in result.warnings


def test_negative_ev_with_positive_net_debt_is_preserved():
    result = bridge_enterprise_to_equity_value(
        enterprise_result(-100.0, ("negative_enterprise_value",)), 50.0
    )

    assert result.enterprise_value == -100.0
    assert result.equity_value == -150.0
    assert result.net_debt_to_enterprise_value is None
    assert "negative_enterprise_value" in result.warnings
    assert "negative_equity_value" in result.warnings


def test_negative_ev_with_net_cash_can_produce_positive_equity_value():
    result = bridge_enterprise_to_equity_value(
        enterprise_result(-100.0, ("negative_enterprise_value",)), -200.0
    )

    assert result.equity_value == 100.0
    assert result.net_debt_to_enterprise_value is None
    assert "negative_enterprise_value" in result.warnings
    assert "net_cash_position" in result.warnings
    assert "negative_equity_value" not in result.warnings


@pytest.mark.parametrize("ev", [0.0, 0.5e-12, -0.5e-12, -100.0])
def test_nonpositive_or_near_zero_ev_has_unavailable_leverage_ratio(ev):
    result = bridge_enterprise_to_equity_value(
        enterprise_result(ev), 10.0
    )

    assert result.net_debt_to_enterprise_value is None


@pytest.mark.parametrize(
    "net_debt", [None, float("nan"), float("inf"), -float("inf"), True]
)
def test_missing_nonfinite_and_boolean_net_debt_are_rejected(net_debt):
    with pytest.raises(ValueError, match="net_debt"):
        bridge_enterprise_to_equity_value(
            enterprise_result(500.0), net_debt
        )


@pytest.mark.parametrize("bad_ev", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_enterprise_value_is_rejected(bad_ev):
    with pytest.raises(ValueError, match="enterprise_value"):
        bridge_enterprise_to_equity_value(
            enterprise_result(bad_ev), 50.0
        )


def test_upstream_value_and_warnings_are_preserved_without_mutation():
    upstream = enterprise_result(
        500.0, ("terminal_value_dominates_enterprise_value",)
    )
    original_warnings = upstream.warnings

    result = bridge_enterprise_to_equity_value(upstream, -100.0)

    assert result.enterprise_value == upstream.enterprise_value
    assert "terminal_value_dominates_enterprise_value" in result.warnings
    assert "net_cash_position" in result.warnings
    assert upstream.warnings == original_warnings


def test_upstream_warnings_are_deduplicated():
    upstream = enterprise_result(
        500.0, ("net_cash_position", "net_cash_position")
    )

    result = bridge_enterprise_to_equity_value(upstream, -100.0)

    assert result.warnings.count("net_cash_position") == 1


def test_worked_debt_and_net_cash_examples():
    debt = bridge_enterprise_to_equity_value(
        enterprise_result(1000.0), 150.0
    )
    net_cash = bridge_enterprise_to_equity_value(
        enterprise_result(1000.0), -200.0
    )

    assert debt.equity_value == 850.0
    assert debt.net_debt_to_enterprise_value == pytest.approx(0.15)
    assert net_cash.equity_value == 1200.0
    assert net_cash.net_debt_to_enterprise_value == pytest.approx(-0.20)


def test_equity_result_is_immutable():
    result = bridge_enterprise_to_equity_value(
        enterprise_result(500.0), 50.0
    )
    assert isinstance(result, EquityValueResult)

    with pytest.raises(FrozenInstanceError):
        result.equity_value = 0.0


def test_invalid_enterprise_result_type_is_rejected():
    with pytest.raises(TypeError, match="enterprise_result"):
        bridge_enterprise_to_equity_value(object(), 50.0)

