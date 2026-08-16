from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from Stock.beta_audit import (
    BetaWACCContext,
    build_beta_robustness_audit,
    calculate_beta_estimate,
    calculate_rolling_beta,
    implied_beta_from_target_wacc,
    resample_adjusted_prices,
    wacc_from_beta,
)


def prices_from_returns(dates, returns, initial=100.0):
    values = [initial]
    for value in returns:
        values.append(values[-1] * (1 + value))
    return pd.Series(values, index=pd.DatetimeIndex(dates))


def regression_prices(beta=1.7, alpha=0.002, periods=72, frequency="ME"):
    dates = pd.date_range("2020-01-31", periods=periods + 1, freq=frequency, tz="UTC")
    market_returns = np.array(
        [0.01 + 0.025 * np.sin(index * 0.7) for index in range(periods)]
    )
    stock_returns = alpha + beta * market_returns
    return (
        prices_from_returns(dates, stock_returns),
        prices_from_returns(dates, market_returns),
    )


def context():
    return BetaWACCContext(
        risk_free_rate=0.04,
        equity_risk_premium=0.05,
        after_tax_cost_of_debt=0.035,
        equity_weight=0.9,
        debt_weight=0.1,
    )


def estimate(stock, market, **overrides):
    values = {
        "lookback_years": 5,
        "frequency": "monthly",
        "minimum_observations": 24,
        "wacc_context": context(),
    }
    values.update(overrides)
    return calculate_beta_estimate("TEST", "^GSPC", stock, market, **values)


def test_ols_beta_covariance_and_alpha_reconcile():
    stock, market = regression_prices(beta=1.7, alpha=0.002)
    result = estimate(stock, market)

    assert result.available
    assert result.raw_beta == pytest.approx(1.7, abs=1e-12)
    assert result.alpha == pytest.approx(0.002, abs=1e-12)
    assert result.r_squared == pytest.approx(1.0)


def test_adjusted_beta_uses_blume_formula():
    stock, market = regression_prices(beta=1.9)
    result = estimate(stock, market)

    assert result.adjusted_beta == pytest.approx((2 / 3) * 1.9 + 1 / 3)


def test_alignment_uses_only_common_dates_without_forward_fill():
    stock, market = regression_prices()
    stock = stock.drop(stock.index[[10, 20]])
    result = estimate(stock, market)

    assert result.available
    assert result.observation_count < result.market_return_observations
    assert result.dropped_for_alignment > 0


def test_insufficient_observations_is_explicit():
    stock, market = regression_prices(periods=12)
    result = estimate(stock, market, minimum_observations=24)

    assert not result.available
    assert result.reason == "insufficient_observations"
    assert result.raw_beta is None


def test_weekly_resampling_uses_friday_week_end():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    prices = pd.Series(np.arange(20) + 100.0, index=dates)
    weekly = resample_adjusted_prices(prices, "weekly")

    assert all(date.weekday() == 4 for date in weekly.index)
    assert weekly.iloc[0] == 101.0


def test_monthly_resampling_uses_last_available_price():
    dates = pd.date_range("2026-01-01", "2026-02-28", freq="D")
    prices = pd.Series(np.arange(len(dates)) + 100.0, index=dates)
    monthly = resample_adjusted_prices(prices, "monthly")

    assert list(monthly.index.day) == [31, 28]
    assert monthly.iloc[0] == prices.loc["2026-01-31"]


def test_beta_decomposition_reconstructs_covariance_beta():
    stock, market = regression_prices(beta=1.6, alpha=0.001)
    result = estimate(stock, market)

    assert result.reconstructed_beta == pytest.approx(result.raw_beta)
    assert result.reconstructed_beta == pytest.approx(
        result.correlation * result.volatility_ratio
    )


def test_beta_standard_error_and_confidence_interval_are_exposed():
    stock, market = regression_prices(beta=1.7)
    result = estimate(stock, market)

    assert result.beta_standard_error == pytest.approx(0.0, abs=1e-12)
    assert result.confidence_interval_low == pytest.approx(1.7, abs=1e-12)
    assert result.confidence_interval_high == pytest.approx(1.7, abs=1e-12)


def test_constant_market_returns_are_invalid():
    dates = pd.date_range("2020-01-31", periods=73, freq="ME")
    market = prices_from_returns(dates, np.repeat(0.01, 72))
    stock = prices_from_returns(dates, np.repeat(0.02, 72))
    result = estimate(stock, market)

    assert not result.available
    assert result.reason == "constant_market_returns"


def test_rolling_beta_tracks_changing_beta_regimes():
    dates = pd.date_range("2018-01-31", periods=85, freq="ME", tz="UTC")
    market_returns = np.array([0.01 + 0.02 * np.sin(i) for i in range(84)])
    stock_returns = np.concatenate(
        [0.8 * market_returns[:42], 2.0 * market_returns[42:]]
    )
    stock = prices_from_returns(dates, stock_returns)
    market = prices_from_returns(dates, market_returns)
    result = calculate_rolling_beta(stock, market, window_observations=36)

    assert len(result.points) == 49
    assert result.minimum < result.maximum
    assert result.latest > result.points[0].raw_beta
    assert result.standard_deviation > 0


def test_wacc_translation_changes_only_cost_of_equity_beta_component():
    assert wacc_from_beta(1.5, context()) == pytest.approx(
        0.9 * (0.04 + 1.5 * 0.05) + 0.1 * 0.035
    )


def test_implied_beta_reconciles_target_wacc():
    target = 0.095
    beta = implied_beta_from_target_wacc(target, context())

    assert wacc_from_beta(beta, context()) == pytest.approx(target)


def test_implied_beta_is_unavailable_without_equity_erp_exposure():
    no_exposure = BetaWACCContext(0.04, 0.0, 0.03, 1.0, 0.0)
    assert implied_beta_from_target_wacc(0.08, no_exposure) is None


def test_robustness_flags_detect_lookback_frequency_and_adjustment_sensitivity():
    monthly_stock, monthly_market = regression_prices(beta=1.8, periods=72)
    weekly_stock, weekly_market = regression_prices(
        beta=1.1, periods=312, frequency="W-FRI"
    )
    audit = build_beta_robustness_audit(
        "TEST",
        monthly_stock,
        monthly_market,
        weekly_stock,
        weekly_market,
        wacc_context=context(),
        current_dcf_wacc=0.09,
    )

    assert "beta_sensitive_to_frequency" in audit.flags
    assert "adjusted_beta_materially_below_raw_beta" in audit.flags
    assert audit.classification in {
        "moderately_specification_sensitive",
        "highly_specification_sensitive",
    }


def test_missing_price_observations_do_not_become_returns():
    stock, market = regression_prices()
    stock.iloc[15] = np.nan
    result = estimate(stock, market)

    assert result.available
    assert result.observation_count < result.market_return_observations


def test_results_are_immutable():
    stock, market = regression_prices()
    audit = build_beta_robustness_audit(
        "TEST",
        stock,
        market,
        *regression_prices(beta=1.7, periods=312, frequency="W-FRI"),
        wacc_context=context(),
        current_dcf_wacc=0.09,
    )

    with pytest.raises(FrozenInstanceError):
        audit.classification = "robust"
    with pytest.raises(FrozenInstanceError):
        audit.production_estimate.raw_beta = 0.0
