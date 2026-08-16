from dataclasses import FrozenInstanceError

import pytest

from Stock.valuation import (
    EquityValueResult,
    PerShareValueResult,
    calculate_intrinsic_value_per_share,
)


def equity_result(value, warnings=(), enterprise_value=None, net_debt=0.0):
    ev = value + net_debt if enterprise_value is None else enterprise_value
    ratio = net_debt / ev if ev > 1e-12 else None
    return EquityValueResult(
        enterprise_value=ev,
        net_debt=net_debt,
        equity_value=value,
        net_debt_to_enterprise_value=ratio,
        warnings=tuple(warnings),
    )


@pytest.mark.parametrize(
    ("equity_value", "shares", "expected"),
    [(500.0, 10.0, 50.0), (850.0, 17.0, 50.0), (1200.0, 24.0, 50.0)],
)
def test_standard_per_share_cases(equity_value, shares, expected):
    result = calculate_intrinsic_value_per_share(
        equity_result(equity_value), shares
    )

    assert result.equity_value == equity_value
    assert result.shares_outstanding == shares
    assert result.intrinsic_value_per_share == pytest.approx(expected)


@pytest.mark.parametrize(
    "shares", [None, 0.0, -10.0, float("nan"), float("inf"), -float("inf"), True]
)
def test_invalid_shares_are_rejected(shares):
    with pytest.raises(ValueError, match="shares_outstanding"):
        calculate_intrinsic_value_per_share(equity_result(500.0), shares)


def test_zero_equity_value_produces_zero_per_share_value():
    result = calculate_intrinsic_value_per_share(equity_result(0.0), 10.0)

    assert result.intrinsic_value_per_share == 0.0
    assert "zero_intrinsic_value_per_share" in result.warnings


def test_negative_equity_value_is_preserved_per_share():
    result = calculate_intrinsic_value_per_share(
        equity_result(-100.0, ("negative_equity_value",)), 10.0
    )

    assert result.intrinsic_value_per_share == -10.0
    assert "negative_equity_value" in result.warnings
    assert "negative_intrinsic_value_per_share" in result.warnings


def test_very_small_positive_equity_value_is_not_rounded_or_marked_zero():
    result = calculate_intrinsic_value_per_share(equity_result(1e-15), 10.0)

    assert result.intrinsic_value_per_share == pytest.approx(1e-16)
    assert "zero_intrinsic_value_per_share" not in result.warnings


def test_very_large_equity_value_is_not_rounded():
    result = calculate_intrinsic_value_per_share(
        equity_result(1e18), 3.0
    )

    assert result.intrinsic_value_per_share == pytest.approx(1e18 / 3)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_equity_value_is_rejected_defensively(bad_value):
    with pytest.raises(ValueError, match="equity_value"):
        calculate_intrinsic_value_per_share(equity_result(bad_value), 10.0)


def test_upstream_result_is_not_modified_and_warnings_are_preserved():
    upstream = equity_result(
        850.0,
        ("terminal_value_dominates_enterprise_value",),
        enterprise_value=1000.0,
        net_debt=150.0,
    )
    original_warnings = upstream.warnings

    result = calculate_intrinsic_value_per_share(upstream, 17.0)

    assert result.equity_value == upstream.equity_value
    assert result.intrinsic_value_per_share == 50.0
    assert "terminal_value_dominates_enterprise_value" in result.warnings
    assert upstream.warnings == original_warnings


def test_upstream_warnings_are_deduplicated():
    upstream = equity_result(
        -100.0,
        ("negative_intrinsic_value_per_share",) * 2,
    )

    result = calculate_intrinsic_value_per_share(upstream, 10.0)

    assert result.warnings.count("negative_intrinsic_value_per_share") == 1


def test_worked_debt_example():
    upstream = equity_result(
        850.0,
        enterprise_value=1000.0,
        net_debt=150.0,
    )

    result = calculate_intrinsic_value_per_share(upstream, 17.0)

    assert result.equity_value == 850.0
    assert result.intrinsic_value_per_share == 50.0


def test_worked_net_cash_example():
    upstream = equity_result(
        1200.0,
        ("net_cash_position",),
        enterprise_value=1000.0,
        net_debt=-200.0,
    )

    result = calculate_intrinsic_value_per_share(upstream, 24.0)

    assert result.equity_value == 1200.0
    assert result.intrinsic_value_per_share == 50.0
    assert "net_cash_position" in result.warnings


def test_unit_consistency_is_caller_controlled():
    billions = calculate_intrinsic_value_per_share(
        equity_result(500.0), 10.0
    )
    raw_units = calculate_intrinsic_value_per_share(
        equity_result(500e9), 10e9
    )

    assert billions.intrinsic_value_per_share == 50.0
    assert raw_units.intrinsic_value_per_share == 50.0


def test_per_share_result_is_immutable():
    result = calculate_intrinsic_value_per_share(equity_result(500.0), 10.0)
    assert isinstance(result, PerShareValueResult)

    with pytest.raises(FrozenInstanceError):
        result.intrinsic_value_per_share = 0.0


def test_invalid_equity_result_type_is_rejected():
    with pytest.raises(TypeError, match="equity_result"):
        calculate_intrinsic_value_per_share(object(), 10.0)

