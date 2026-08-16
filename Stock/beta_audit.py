"""Pure diagnostics for historical regression-beta robustness.

The calculations in this module do not fetch data and do not select a
production beta. Percentage adjusted-price returns are aligned on common dates
without forward filling. Adjusted beta is diagnostic Blume shrinkage only.
"""

from dataclasses import dataclass
import math
import statistics
from typing import Literal

import pandas as pd


Frequency = Literal["monthly", "weekly"]
ANNUALIZATION = {"monthly": 12, "weekly": 52}
LOOKBACK_SENSITIVITY_THRESHOLD = 0.30
FREQUENCY_SENSITIVITY_THRESHOLD = 0.20
RAW_ADJUSTED_DIFFERENCE_THRESHOLD = 0.20
LOW_R_SQUARED_THRESHOLD = 0.25
WIDE_CONFIDENCE_INTERVAL_THRESHOLD = 0.50


@dataclass(frozen=True)
class BetaWACCContext:
    risk_free_rate: float
    equity_risk_premium: float
    after_tax_cost_of_debt: float
    equity_weight: float
    debt_weight: float


@dataclass(frozen=True)
class BetaEstimate:
    ticker: str
    benchmark: str
    lookback_years: int
    frequency: Frequency
    available: bool
    reason: str | None
    raw_beta: float | None
    adjusted_beta: float | None
    alpha: float | None
    r_squared: float | None
    correlation: float | None
    observation_count: int
    stock_return_observations: int
    market_return_observations: int
    dropped_for_alignment: int
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    annualized_volatility_stock: float | None
    annualized_volatility_market: float | None
    volatility_ratio: float | None
    reconstructed_beta: float | None
    beta_standard_error: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    implied_wacc_raw: float | None
    implied_wacc_adjusted: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RollingBetaPoint:
    period_end: pd.Timestamp
    raw_beta: float


@dataclass(frozen=True)
class RollingBetaSummary:
    window_observations: int
    frequency: Frequency
    points: tuple[RollingBetaPoint, ...]
    latest: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    standard_deviation: float | None


@dataclass(frozen=True)
class BetaRobustnessAudit:
    ticker: str
    primary_benchmark: str
    production_estimate: BetaEstimate
    estimates: tuple[BetaEstimate, ...]
    alternative_benchmark_estimate: BetaEstimate | None
    rolling_beta: RollingBetaSummary
    minimum_raw_beta: float | None
    maximum_raw_beta: float | None
    median_raw_beta: float | None
    raw_beta_dispersion: float | None
    implied_beta_for_current_dcf_wacc: float | None
    current_dcf_wacc: float
    flags: tuple[str, ...]
    classification: str

    def estimate_at(self, years: int, frequency: Frequency) -> BetaEstimate | None:
        return next(
            (
                estimate
                for estimate in self.estimates
                if estimate.lookback_years == years
                and estimate.frequency == frequency
            ),
            None,
        )


def _numeric_prices(prices: pd.Series) -> pd.Series:
    values = pd.to_numeric(pd.Series(prices), errors="coerce")
    dates = pd.to_datetime(values.index, errors="coerce", utc=True)
    frame = pd.DataFrame({"date": dates, "value": values.to_numpy()})
    frame = frame.dropna(subset=["date", "value"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    result = pd.Series(frame["value"].to_numpy(), index=pd.DatetimeIndex(frame["date"]))
    return result[result > 0]


def resample_adjusted_prices(prices: pd.Series, frequency: Frequency) -> pd.Series:
    """Resample adjusted prices to Friday week-end or calendar month-end."""
    values = _numeric_prices(prices)
    if frequency == "weekly":
        weekly = values.resample("W-FRI").last().dropna()
        # Exclude the currently incomplete week: pandas labels that partial
        # bin with the coming Friday, which otherwise looks like future data.
        return weekly[weekly.index <= values.index.max().normalize()]
    if frequency == "monthly":
        try:
            return values.resample("ME").last().dropna()
        except ValueError:  # pandas < 2.2 compatibility
            return values.resample("M").last().dropna()
    raise ValueError("frequency must be monthly or weekly")


def _unavailable_estimate(
    ticker: str,
    benchmark: str,
    years: int,
    frequency: Frequency,
    reason: str,
    *,
    stock_observations: int = 0,
    market_observations: int = 0,
    overlap: int = 0,
    dropped: int = 0,
) -> BetaEstimate:
    return BetaEstimate(
        ticker=ticker,
        benchmark=benchmark,
        lookback_years=years,
        frequency=frequency,
        available=False,
        reason=reason,
        raw_beta=None,
        adjusted_beta=None,
        alpha=None,
        r_squared=None,
        correlation=None,
        observation_count=overlap,
        stock_return_observations=stock_observations,
        market_return_observations=market_observations,
        dropped_for_alignment=dropped,
        start_date=None,
        end_date=None,
        annualized_volatility_stock=None,
        annualized_volatility_market=None,
        volatility_ratio=None,
        reconstructed_beta=None,
        beta_standard_error=None,
        confidence_interval_low=None,
        confidence_interval_high=None,
        implied_wacc_raw=None,
        implied_wacc_adjusted=None,
        warnings=(reason,),
    )


def wacc_from_beta(beta: float, context: BetaWACCContext) -> float:
    cost_equity = context.risk_free_rate + beta * context.equity_risk_premium
    return (
        context.equity_weight * cost_equity
        + context.debt_weight * context.after_tax_cost_of_debt
    )


def implied_beta_from_target_wacc(
    target_wacc: float,
    context: BetaWACCContext,
) -> float | None:
    denominator = context.equity_weight * context.equity_risk_premium
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return None
    numerator = (
        target_wacc
        - context.equity_weight * context.risk_free_rate
        - context.debt_weight * context.after_tax_cost_of_debt
    )
    result = numerator / denominator
    return result if math.isfinite(result) else None


def calculate_beta_estimate(
    ticker: str,
    benchmark: str,
    stock_prices: pd.Series,
    market_prices: pd.Series,
    *,
    lookback_years: int,
    frequency: Frequency,
    minimum_observations: int,
    wacc_context: BetaWACCContext | None = None,
) -> BetaEstimate:
    """Calculate OLS beta from already-frequency-aligned adjusted prices."""
    stock = _numeric_prices(stock_prices)
    market = _numeric_prices(market_prices)
    if stock.empty or market.empty:
        return _unavailable_estimate(
            ticker, benchmark, lookback_years, frequency, "price_data_unavailable"
        )
    common_end = min(stock.index.max(), market.index.max())
    start = common_end - pd.DateOffset(years=lookback_years)
    stock = stock[(stock.index >= start) & (stock.index <= common_end)]
    market = market[(market.index >= start) & (market.index <= common_end)]
    # Production uses percentage price returns, not log returns. No forward fill.
    stock_returns = stock.pct_change(fill_method=None).dropna()
    market_returns = market.pct_change(fill_method=None).dropna()
    aligned = pd.concat(
        {"stock": stock_returns, "market": market_returns}, axis=1, sort=True
    ).dropna()
    union_count = len(stock_returns.index.union(market_returns.index))
    dropped = union_count - len(aligned)
    if len(aligned) < minimum_observations:
        return _unavailable_estimate(
            ticker,
            benchmark,
            lookback_years,
            frequency,
            "insufficient_observations",
            stock_observations=len(stock_returns),
            market_observations=len(market_returns),
            overlap=len(aligned),
            dropped=dropped,
        )

    x = aligned["market"]
    y = aligned["stock"]
    market_variance = float(x.var(ddof=1))
    if not math.isfinite(market_variance) or market_variance <= 0:
        return _unavailable_estimate(
            ticker,
            benchmark,
            lookback_years,
            frequency,
            "constant_market_returns",
            stock_observations=len(stock_returns),
            market_observations=len(market_returns),
            overlap=len(aligned),
            dropped=dropped,
        )
    covariance = float(y.cov(x))
    beta = covariance / market_variance
    alpha = float(y.mean() - beta * x.mean())
    correlation = float(y.corr(x))
    r_squared = correlation ** 2
    periods = ANNUALIZATION[frequency]
    stock_vol = float(y.std(ddof=1) * math.sqrt(periods))
    market_vol = float(x.std(ddof=1) * math.sqrt(periods))
    volatility_ratio = stock_vol / market_vol
    reconstructed = correlation * volatility_ratio
    fitted = alpha + beta * x
    residuals = y - fitted
    centered_market_ss = float(((x - x.mean()) ** 2).sum())
    if len(aligned) > 2 and centered_market_ss > 0:
        residual_variance = float((residuals ** 2).sum()) / (len(aligned) - 2)
        standard_error = math.sqrt(residual_variance / centered_market_ss)
        ci_low = beta - 1.96 * standard_error
        ci_high = beta + 1.96 * standard_error
    else:
        standard_error = ci_low = ci_high = None
    adjusted_beta = (2 / 3) * beta + (1 / 3)
    warnings = []
    if r_squared < LOW_R_SQUARED_THRESHOLD:
        warnings.append("low_beta_regression_r_squared")
    if abs(beta - adjusted_beta) >= RAW_ADJUSTED_DIFFERENCE_THRESHOLD:
        warnings.append("adjusted_beta_materially_differs_from_raw_beta")
    if ci_low is not None and ci_high - ci_low >= WIDE_CONFIDENCE_INTERVAL_THRESHOLD:
        warnings.append("wide_beta_confidence_interval")

    return BetaEstimate(
        ticker=ticker,
        benchmark=benchmark,
        lookback_years=lookback_years,
        frequency=frequency,
        available=True,
        reason=None,
        raw_beta=beta,
        adjusted_beta=adjusted_beta,
        alpha=alpha,
        r_squared=r_squared,
        correlation=correlation,
        observation_count=len(aligned),
        stock_return_observations=len(stock_returns),
        market_return_observations=len(market_returns),
        dropped_for_alignment=dropped,
        start_date=pd.Timestamp(aligned.index.min()),
        end_date=pd.Timestamp(aligned.index.max()),
        annualized_volatility_stock=stock_vol,
        annualized_volatility_market=market_vol,
        volatility_ratio=volatility_ratio,
        reconstructed_beta=reconstructed,
        beta_standard_error=standard_error,
        confidence_interval_low=ci_low,
        confidence_interval_high=ci_high,
        implied_wacc_raw=(wacc_from_beta(beta, wacc_context) if wacc_context else None),
        implied_wacc_adjusted=(
            wacc_from_beta(adjusted_beta, wacc_context) if wacc_context else None
        ),
        warnings=tuple(warnings),
    )


def calculate_rolling_beta(
    stock_prices: pd.Series,
    market_prices: pd.Series,
    *,
    window_observations: int = 36,
) -> RollingBetaSummary:
    stock_returns = _numeric_prices(stock_prices).pct_change(fill_method=None)
    market_returns = _numeric_prices(market_prices).pct_change(fill_method=None)
    aligned = pd.concat(
        {"stock": stock_returns, "market": market_returns}, axis=1, sort=True
    ).dropna()
    points = []
    for end_index in range(window_observations, len(aligned) + 1):
        window = aligned.iloc[end_index - window_observations:end_index]
        variance = float(window["market"].var(ddof=1))
        if variance <= 0 or not math.isfinite(variance):
            continue
        beta = float(window["stock"].cov(window["market"]) / variance)
        points.append(RollingBetaPoint(pd.Timestamp(window.index[-1]), beta))
    values = tuple(point.raw_beta for point in points)
    return RollingBetaSummary(
        window_observations=window_observations,
        frequency="monthly",
        points=tuple(points),
        latest=values[-1] if values else None,
        median=statistics.median(values) if values else None,
        minimum=min(values) if values else None,
        maximum=max(values) if values else None,
        standard_deviation=(statistics.stdev(values) if len(values) > 1 else None),
    )


def build_beta_robustness_audit(
    ticker: str,
    monthly_stock_prices: pd.Series,
    monthly_market_prices: pd.Series,
    weekly_stock_prices: pd.Series,
    weekly_market_prices: pd.Series,
    *,
    wacc_context: BetaWACCContext,
    current_dcf_wacc: float,
    alternative_benchmark_prices: pd.Series | None = None,
    alternative_benchmark: str = "VTI",
) -> BetaRobustnessAudit:
    estimates = []
    for years in (2, 3, 5):
        estimates.append(
            calculate_beta_estimate(
                ticker,
                "^GSPC",
                monthly_stock_prices,
                monthly_market_prices,
                lookback_years=years,
                frequency="monthly",
                minimum_observations=24,
                wacc_context=wacc_context,
            )
        )
        estimates.append(
            calculate_beta_estimate(
                ticker,
                "^GSPC",
                weekly_stock_prices,
                weekly_market_prices,
                lookback_years=years,
                frequency="weekly",
                minimum_observations=52,
                wacc_context=wacc_context,
            )
        )
    production = next(
        estimate
        for estimate in estimates
        if estimate.lookback_years == 5 and estimate.frequency == "monthly"
    )
    alternative = None
    if alternative_benchmark_prices is not None:
        alternative = calculate_beta_estimate(
            ticker,
            alternative_benchmark,
            monthly_stock_prices,
            alternative_benchmark_prices,
            lookback_years=5,
            frequency="monthly",
            minimum_observations=24,
            wacc_context=wacc_context,
        )
    rolling = calculate_rolling_beta(
        monthly_stock_prices, monthly_market_prices, window_observations=36
    )
    valid_betas = tuple(
        estimate.raw_beta
        for estimate in estimates
        if estimate.available and estimate.raw_beta is not None
    )
    minimum = min(valid_betas) if valid_betas else None
    maximum = max(valid_betas) if valid_betas else None
    median = statistics.median(valid_betas) if valid_betas else None
    dispersion = maximum - minimum if valid_betas else None
    flags = []
    for frequency in ("monthly", "weekly"):
        values = [
            estimate.raw_beta
            for estimate in estimates
            if estimate.frequency == frequency and estimate.available
        ]
        if values and max(values) - min(values) >= LOOKBACK_SENSITIVITY_THRESHOLD:
            flags.append("beta_sensitive_to_lookback")
            break
    for years in (2, 3, 5):
        monthly = next(e for e in estimates if e.lookback_years == years and e.frequency == "monthly")
        weekly = next(e for e in estimates if e.lookback_years == years and e.frequency == "weekly")
        if (
            monthly.available
            and weekly.available
            and abs(monthly.raw_beta - weekly.raw_beta)
            >= FREQUENCY_SENSITIVITY_THRESHOLD
        ):
            flags.append("beta_sensitive_to_frequency")
            break
    if production.available:
        if abs(production.raw_beta - production.adjusted_beta) >= RAW_ADJUSTED_DIFFERENCE_THRESHOLD:
            flags.append("adjusted_beta_materially_below_raw_beta")
        if production.r_squared < LOW_R_SQUARED_THRESHOLD:
            flags.append("low_beta_regression_r_squared")
        if (
            production.confidence_interval_low is not None
            and production.confidence_interval_high - production.confidence_interval_low
            >= WIDE_CONFIDENCE_INTERVAL_THRESHOLD
        ):
            flags.append("wide_beta_confidence_interval")
    flags = list(dict.fromkeys(flags))
    specification_flags = {
        "beta_sensitive_to_lookback",
        "beta_sensitive_to_frequency",
    }
    if specification_flags.issubset(flags) or (
        dispersion is not None and dispersion >= 0.50
    ):
        classification = "highly_specification_sensitive"
    elif any(
        flag in specification_flags
        or flag in {"low_beta_regression_r_squared", "wide_beta_confidence_interval"}
        for flag in flags
    ):
        classification = "moderately_specification_sensitive"
    else:
        classification = "robust_within_tested_specifications"
    return BetaRobustnessAudit(
        ticker=ticker,
        primary_benchmark="^GSPC",
        production_estimate=production,
        estimates=tuple(estimates),
        alternative_benchmark_estimate=alternative,
        rolling_beta=rolling,
        minimum_raw_beta=minimum,
        maximum_raw_beta=maximum,
        median_raw_beta=median,
        raw_beta_dispersion=dispersion,
        implied_beta_for_current_dcf_wacc=implied_beta_from_target_wacc(
            current_dcf_wacc, wacc_context
        ),
        current_dcf_wacc=current_dcf_wacc,
        flags=tuple(flags),
        classification=classification,
    )
