from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TTMResult:
    """Validated trailing-twelve-month result and its supporting periods."""

    value: float | None
    available: bool
    periods_used: tuple[pd.Timestamp, ...]
    reason: str | None


@dataclass(frozen=True)
class RevenueCAGRResult:
    """Revenue CAGR over an exact number of consecutive fiscal-year intervals."""

    value: float | None
    available: bool
    start_period: pd.Timestamp | None
    end_period: pd.Timestamp | None
    years: int
    reason: str | None
    start_revenue: float | None = None
    end_revenue: float | None = None


@dataclass(frozen=True)
class SalesToCapitalResult:
    """Historical accounting capital-efficiency anchor and its components.

    This is not a causal efficiency estimate. Accounting cash is fully
    deducted; excess cash is not estimated; R&D remains expensed; acquisitions,
    goodwill, buybacks, and working-capital timing can make the result noisy.
    """

    value: float | None
    available: bool
    start_period: pd.Timestamp | None
    end_period: pd.Timestamp | None
    years: int
    reason: str | None
    start_revenue: float | None = None
    end_revenue: float | None = None
    delta_revenue: float | None = None
    start_invested_capital: float | None = None
    end_invested_capital: float | None = None
    delta_invested_capital: float | None = None


@dataclass(frozen=True)
class HistoricalDCFAnchors:
    """Transparent historical evidence for later DCF assumption design."""

    revenue_cagr: dict[int, RevenueCAGRResult] = field(default_factory=dict)
    annual_sales_to_capital: dict[pd.Timestamp, SalesToCapitalResult] = field(
        default_factory=dict
    )
    normalized_sales_to_capital: dict[int, SalesToCapitalResult] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class FundamentalHistory:
    """Annual fundamental amounts/ratios plus independently validated TTM metrics.

    Reported inputs and calculated metrics use separate columns in ``annual``.
    Pandas represents unavailable cells as NaN; callers should not interpret
    those cells as zero. Each TTM metric carries explicit availability metadata.
    """

    annual: pd.DataFrame
    ttm: dict[str, TTMResult]
    annual_reasons: pd.DataFrame
    dcf_anchors: HistoricalDCFAnchors = field(default_factory=HistoricalDCFAnchors)


@dataclass(frozen=True)
class MetricCalculation:
    value: float | None
    available: bool
    reason: str | None


@dataclass(frozen=True)
class NOPATCalculation:
    nopat: float | None
    tax_rate: float | None
    available: bool
    reason: str | None
    assumption_used: bool = False


REVENUE = "Revenue"
REVENUE_GROWTH = "Revenue Growth"
GROSS_PROFIT = "Gross Profit"
GROSS_MARGIN = "Gross Margin"
OPERATING_INCOME = "Operating Income"
OPERATING_MARGIN = "Operating Margin"
CFO = "CFO"
CAPEX = "CapEx"
FCF = "FCF"
FCF_MARGIN = "FCF Margin"
PRETAX_INCOME = "Pretax Income"
TAX_PROVISION = "Tax Provision"
OPERATING_TAX_RATE = "Operating Tax Rate"
NOPAT = "NOPAT"
TOTAL_EQUITY = "Total Equity"
TOTAL_DEBT = "Total Debt"
CASH = "Cash And Cash Equivalents"
INVESTED_CAPITAL = "Invested Capital"
AVERAGE_INVESTED_CAPITAL = "Average Invested Capital"
ROIC = "ROIC"
DEPRECIATION_AND_AMORTIZATION = "Depreciation And Amortization"
NET_INVESTMENT = "Simplified Net Investment"
REINVESTMENT_RATE = "Reinvestment Rate"
FUNDAMENTAL_GROWTH_CAPACITY = "Fundamental Growth Capacity"
DELTA_REVENUE = "Delta Revenue"
DELTA_INVESTED_CAPITAL = "Delta Invested Capital"
SALES_TO_CAPITAL = "Sales-to-Capital"
APPROXIMATE_ROIC = "Approximate ROIC"


def _valid_number(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _latest_fiscal_window(values: pd.Series, years: int):
    """Return the latest N+1 fiscal periods, preserving missing endpoint values.

    Fiscal continuity uses adjacent reporting-year identities (year + 1). Yahoo
    annual statement dates may move by a few days under 52/53-week calendars,
    so exact day spacing is less reliable. If duplicate dates exist, normalized
    statement data has already retained the last occurrence deterministically.
    """
    normalized = _normalize_series(values)
    if len(normalized) < years + 1:
        return normalized, None, "insufficient_history"
    window = normalized.iloc[-(years + 1):]
    periods = list(window.index)
    if any(
        current.year - previous.year != 1
        for previous, current in zip(periods, periods[1:])
    ):
        return window, periods, "non_consecutive_fiscal_years"
    return window, periods, None


def calculate_revenue_cagr(
    revenue: pd.Series | None,
    years: int,
) -> RevenueCAGRResult:
    """Calculate CAGR from the latest N+1 consecutive annual observations.

    FY2022 to FY2025 is three elapsed fiscal-year intervals and therefore a 3Y
    CAGR. A missing latest endpoint is never replaced with an older year.
    """
    empty = RevenueCAGRResult(None, False, None, None, years, "insufficient_history")
    if revenue is None or years <= 0:
        return empty
    window, periods, reason = _latest_fiscal_window(revenue, years)
    if reason is not None:
        start = window.index[0] if len(window) else None
        end = window.index[-1] if len(window) else None
        return RevenueCAGRResult(None, False, start, end, years, reason)
    start_period, end_period = periods[0], periods[-1]
    start = _valid_number(window.iloc[0])
    end = _valid_number(window.iloc[-1])
    if start is None:
        return RevenueCAGRResult(
            None, False, start_period, end_period, years, "missing_start_revenue"
        )
    if end is None:
        return RevenueCAGRResult(
            None, False, start_period, end_period, years, "missing_end_revenue",
            start_revenue=start,
        )
    if any(_valid_number(value) is None for value in window.iloc[1:-1]):
        return RevenueCAGRResult(
            None, False, start_period, end_period, years,
            "missing_intermediate_revenue", start, end,
        )
    if start <= 0:
        return RevenueCAGRResult(
            None, False, start_period, end_period, years,
            "non_positive_start_revenue", start, end,
        )
    if end < 0:
        return RevenueCAGRResult(
            None, False, start_period, end_period, years,
            "negative_end_revenue", start, end,
        )
    return RevenueCAGRResult(
        float((end / start) ** (1 / years) - 1),
        True, start_period, end_period, years, None, start, end,
    )


def _capital_delta_is_near_zero(delta: float, start: float, end: float) -> bool:
    """Reject numerical-zero capital deltas without suppressing real changes.

    Tolerance is the larger of 1e-9 currency units and 1e-12 of the endpoint
    capital scale. This catches floating-point residue while retaining small but
    genuinely reported capital changes. Ratios are never clamped.
    """
    tolerance = max(1e-9, max(abs(start), abs(end)) * 1e-12)
    return abs(delta) <= tolerance


def calculate_sales_to_capital(
    start_revenue: float | None,
    end_revenue: float | None,
    start_invested_capital: float | None,
    end_invested_capital: float | None,
    start_period,
    end_period,
    *,
    years: int = 1,
) -> SalesToCapitalResult:
    """Calculate cumulative delta Revenue / delta accounting Invested Capital."""
    start_date = pd.Timestamp(start_period)
    end_date = pd.Timestamp(end_period)
    base = dict(start_period=start_date, end_period=end_date, years=years)
    sr = _valid_number(start_revenue)
    er = _valid_number(end_revenue)
    sic = _valid_number(start_invested_capital)
    eic = _valid_number(end_invested_capital)
    components = dict(
        start_revenue=sr, end_revenue=er,
        start_invested_capital=sic, end_invested_capital=eic,
    )
    if end_date.year - start_date.year != years:
        return SalesToCapitalResult(
            None, False, reason="non_consecutive_fiscal_years", **base, **components
        )
    for value, reason in (
        (sr, "missing_start_revenue"), (er, "missing_end_revenue"),
        (sic, "missing_start_invested_capital"),
        (eic, "missing_end_invested_capital"),
    ):
        if value is None:
            return SalesToCapitalResult(
                None, False, reason=reason, **base, **components
            )
    delta_revenue = er - sr
    delta_capital = eic - sic
    components.update(
        delta_revenue=delta_revenue,
        delta_invested_capital=delta_capital,
    )
    if _capital_delta_is_near_zero(delta_capital, sic, eic):
        return SalesToCapitalResult(
            None, False, reason="zero_or_near_zero_delta_invested_capital",
            **base, **components,
        )
    return SalesToCapitalResult(
        float(delta_revenue / delta_capital), True, reason=None,
        **base, **components,
    )


def calculate_normalized_sales_to_capital(
    revenue: pd.Series | None,
    invested_capital: pd.Series | None,
    years: int,
) -> SalesToCapitalResult:
    """Calculate latest N-year cumulative, not average annual, capital efficiency."""
    unavailable = SalesToCapitalResult(
        None, False, None, None, years, "insufficient_history"
    )
    if revenue is None or invested_capital is None or years <= 0:
        return unavailable
    revenue_normalized = _normalize_series(revenue)
    capital_normalized = _normalize_series(invested_capital)
    period_axis = revenue_normalized.index.union(capital_normalized.index).sort_values()
    if len(period_axis) < years + 1:
        return unavailable
    window = period_axis[-(years + 1):]
    if any(b.year - a.year != 1 for a, b in zip(window, window[1:])):
        return SalesToCapitalResult(
            None, False, window[0], window[-1], years,
            "non_consecutive_fiscal_years",
        )
    revenue_aligned = revenue_normalized.reindex(window)
    capital_aligned = capital_normalized.reindex(window)
    return calculate_sales_to_capital(
        revenue_aligned.iloc[0], revenue_aligned.iloc[-1],
        capital_aligned.iloc[0], capital_aligned.iloc[-1],
        window[0], window[-1], years=years,
    )


def calculate_approximate_roic(
    operating_margin: float | None,
    operating_tax_rate: float | None,
    sales_to_capital: float | None,
) -> MetricCalculation:
    """Diagnostic only: after-tax operating margin times Sales-to-Capital."""
    margin = _valid_number(operating_margin)
    tax_rate = _valid_number(operating_tax_rate)
    efficiency = _valid_number(sales_to_capital)
    if margin is None:
        return MetricCalculation(None, False, "missing_operating_margin")
    if tax_rate is None:
        return MetricCalculation(None, False, "missing_operating_tax_rate")
    if efficiency is None:
        return MetricCalculation(None, False, "missing_sales_to_capital")
    if not 0 <= tax_rate <= 0.50:
        return MetricCalculation(None, False, "unreasonable_tax_rate")
    return MetricCalculation(float(margin * (1 - tax_rate) * efficiency), True, None)


def calculate_nopat(
    operating_income: float | None,
    pretax_income: float | None,
    tax_provision: float | None,
) -> NOPATCalculation:
    """Calculate NOPAT without a fallback tax assumption.

    A reported effective rate is accepted only when pretax income is positive,
    tax provision is non-negative, and the un-clamped rate is between 0% and
    50%. This deliberately differs from the compatibility clamp used by FCFF.
    """
    if operating_income is None or pd.isna(operating_income):
        return NOPATCalculation(None, None, False, "missing_operating_income")
    if (
        pretax_income is None
        or pd.isna(pretax_income)
        or tax_provision is None
        or pd.isna(tax_provision)
    ):
        return NOPATCalculation(None, None, False, "missing_tax_inputs")
    if pretax_income <= 0:
        return NOPATCalculation(None, None, False, "non_positive_pretax_income")
    tax_rate = tax_provision / pretax_income
    if tax_provision < 0 or not np.isfinite(tax_rate) or not 0 <= tax_rate <= 0.50:
        return NOPATCalculation(None, None, False, "unreasonable_tax_rate")
    return NOPATCalculation(
        float(operating_income * (1 - tax_rate)),
        float(tax_rate),
        True,
        None,
    )


def calculate_invested_capital(
    total_equity: float | None,
    total_debt: float | None,
    cash: float | None,
) -> MetricCalculation:
    """Accounting invested capital = equity + debt - cash."""
    for value, reason in (
        (total_equity, "missing_total_equity"),
        (total_debt, "missing_total_debt"),
        (cash, "missing_cash"),
    ):
        if value is None or pd.isna(value):
            return MetricCalculation(None, False, reason)
    return MetricCalculation(float(total_equity + total_debt - cash), True, None)


def calculate_average_invested_capital(
    current: float | None,
    previous: float | None,
    current_period,
    previous_period,
) -> MetricCalculation:
    if current is None or pd.isna(current):
        return MetricCalculation(None, False, "missing_current_invested_capital")
    if previous is None or pd.isna(previous):
        return MetricCalculation(None, False, "missing_prior_invested_capital")
    current_date = pd.Timestamp(current_period)
    previous_date = pd.Timestamp(previous_period)
    if current_date.year - previous_date.year != 1:
        return MetricCalculation(None, False, "non_consecutive_fiscal_years")
    return MetricCalculation(float((current + previous) / 2), True, None)


def calculate_roic(
    nopat: float | None,
    average_invested_capital: float | None,
) -> MetricCalculation:
    if nopat is None or pd.isna(nopat):
        return MetricCalculation(None, False, "missing_nopat")
    if average_invested_capital is None or pd.isna(average_invested_capital):
        return MetricCalculation(None, False, "missing_average_invested_capital")
    if average_invested_capital <= 0:
        return MetricCalculation(
            None, False, "non_positive_average_invested_capital"
        )
    return MetricCalculation(float(nopat / average_invested_capital), True, None)


def calculate_simplified_net_investment(
    yahoo_capex: float | None,
    depreciation_and_amortization: float | None,
) -> MetricCalculation:
    """Simplified reinvestment = CapEx cash outlay - D&A.

    Yahoo normally reports CapEx as a negative cash-flow amount, so its economic
    outlay is ``-yahoo_capex``. Working-capital investment is intentionally not
    estimated in this first version.
    """
    if yahoo_capex is None or pd.isna(yahoo_capex):
        return MetricCalculation(None, False, "missing_capex")
    if depreciation_and_amortization is None or pd.isna(
        depreciation_and_amortization
    ):
        return MetricCalculation(None, False, "missing_depreciation_amortization")
    return MetricCalculation(
        float(-yahoo_capex - depreciation_and_amortization), True, None
    )


def calculate_reinvestment_rate(
    net_investment: float | None,
    nopat: float | None,
) -> MetricCalculation:
    if net_investment is None or pd.isna(net_investment):
        return MetricCalculation(None, False, "missing_net_investment")
    if nopat is None or pd.isna(nopat):
        return MetricCalculation(None, False, "missing_nopat")
    if nopat <= 0:
        return MetricCalculation(None, False, "non_positive_nopat")
    return MetricCalculation(float(net_investment / nopat), True, None)


def calculate_fundamental_growth_capacity(
    roic: float | None,
    reinvestment_rate: float | None,
) -> MetricCalculation:
    if roic is None or pd.isna(roic):
        return MetricCalculation(None, False, "missing_roic")
    if reinvestment_rate is None or pd.isna(reinvestment_rate):
        return MetricCalculation(None, False, "missing_reinvestment_rate")
    return MetricCalculation(float(roic * reinvestment_rate), True, None)


def build_validated_ttm(values: pd.Series, expected_periods=None) -> TTMResult:
    """Build TTM only from four distinct, consecutive fiscal quarters.

    Yahoo statement dates are normalized to calendar-quarter identities rather
    than checked with an approximate 90-day interval. This tolerates month-end
    differences and 52/53-week fiscal calendars while still detecting a skipped
    quarter. Within one calendar quarter, the latest reporting date wins; for an
    identical duplicate date, the last input occurrence wins deterministically.

    ``expected_periods`` should be the raw quarterly statement columns. Keeping
    this period axis separate from usable values prevents a NaN quarter from
    being dropped and silently replaced by an older observation.
    """
    values = values if values is not None else pd.Series(dtype=float)
    expected = (
        list(expected_periods)
        if expected_periods is not None
        else list(values.index)
    )

    value_dates = pd.to_datetime(pd.Index(values.index), errors="coerce")
    expected_dates = pd.to_datetime(pd.Index(expected), errors="coerce")
    if value_dates.isna().any() or expected_dates.isna().any():
        return TTMResult(None, False, (), "invalid_dates")

    observations = pd.DataFrame(
        {
            "date": value_dates,
            "value": pd.to_numeric(
                pd.Series(values).reset_index(drop=True), errors="coerce"
            ),
            "order": np.arange(len(values)),
        }
    )
    observations["quarter"] = observations["date"].dt.to_period("Q")
    observations = observations.sort_values(["date", "order"]).drop_duplicates(
        "quarter", keep="last"
    )

    period_axis = pd.DataFrame(
        {"date": expected_dates, "order": np.arange(len(expected_dates))}
    )
    period_axis["quarter"] = period_axis["date"].dt.to_period("Q")
    period_axis = (
        period_axis.sort_values(["date", "order"])
        .drop_duplicates("quarter", keep="last")
        .sort_values("quarter")
    )

    if observations.empty or observations["value"].isna().all():
        return TTMResult(None, False, (), "all_values_missing")
    if len(period_axis) < 4:
        periods = tuple(pd.Timestamp(date) for date in period_axis["date"])
        return TTMResult(None, False, periods, "fewer_than_four_quarters")

    candidates = period_axis.iloc[-4:].copy()
    periods = tuple(pd.Timestamp(date) for date in candidates["date"])
    quarter_ordinals = [period.ordinal for period in candidates["quarter"]]
    if any(
        current - previous != 1
        for previous, current in zip(quarter_ordinals, quarter_ordinals[1:])
    ):
        return TTMResult(None, False, periods, "non_consecutive_quarters")

    values_by_quarter = observations.set_index("quarter")["value"]
    candidate_values = values_by_quarter.reindex(candidates["quarter"])
    if candidate_values.isna().any():
        return TTMResult(None, False, periods, "missing_quarter_value")
    return TTMResult(float(candidate_values.sum()), True, periods, None)


def _normalize_series(values: pd.Series | None) -> pd.Series:
    if values is None or len(values) == 0:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(pd.Index(values.index), errors="coerce")
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": pd.to_numeric(
                pd.Series(values).reset_index(drop=True), errors="coerce"
            ),
            "order": np.arange(len(values)),
        }
    ).dropna(subset=["date"])
    if frame.empty:
        return pd.Series(dtype=float)
    frame = frame.sort_values(["date", "order"]).drop_duplicates(
        "date", keep="last"
    )
    return pd.Series(frame["value"].to_numpy(), index=pd.DatetimeIndex(frame["date"]))


def _normalize_periods(periods, series: tuple[pd.Series, ...]) -> pd.DatetimeIndex:
    if periods is None:
        raw_periods = [date for values in series for date in values.index]
    else:
        raw_periods = list(periods)
    parsed = pd.to_datetime(pd.Index(raw_periods), errors="coerce").dropna()
    return pd.DatetimeIndex(parsed).drop_duplicates().sort_values()


def build_period_fundamentals(
    *,
    revenue: pd.Series | None,
    gross_profit: pd.Series | None,
    operating_income: pd.Series | None,
    cfo: pd.Series | None,
    capex: pd.Series | None,
    periods=None,
    include_revenue_growth: bool = True,
) -> pd.DataFrame:
    """Align reported amounts by period and calculate pure fundamental metrics."""
    normalized = {
        REVENUE: _normalize_series(revenue),
        GROSS_PROFIT: _normalize_series(gross_profit),
        OPERATING_INCOME: _normalize_series(operating_income),
        CFO: _normalize_series(cfo),
        CAPEX: _normalize_series(capex),
    }
    period_index = _normalize_periods(periods, tuple(normalized.values()))
    frame = pd.DataFrame(index=period_index)
    for name, values in normalized.items():
        frame[name] = values.reindex(period_index)

    complete_fcf = frame[CFO].notna() & frame[CAPEX].notna()
    frame[FCF] = np.nan
    frame.loc[complete_fcf, FCF] = frame.loc[complete_fcf, CFO] + frame.loc[
        complete_fcf, CAPEX
    ]

    valid_revenue = frame[REVENUE].notna() & (frame[REVENUE] != 0)
    frame[GROSS_MARGIN] = np.nan
    frame.loc[valid_revenue & frame[GROSS_PROFIT].notna(), GROSS_MARGIN] = (
        frame[GROSS_PROFIT] / frame[REVENUE]
    )
    frame[OPERATING_MARGIN] = np.nan
    frame.loc[
        valid_revenue & frame[OPERATING_INCOME].notna(), OPERATING_MARGIN
    ] = frame[OPERATING_INCOME] / frame[REVENUE]
    frame[FCF_MARGIN] = np.nan
    frame.loc[valid_revenue & frame[FCF].notna(), FCF_MARGIN] = (
        frame[FCF] / frame[REVENUE]
    )

    if include_revenue_growth:
        frame[REVENUE_GROWTH] = np.nan
        for previous_date, current_date in zip(period_index, period_index[1:]):
            if current_date.year - previous_date.year != 1:
                continue
            previous = frame.at[previous_date, REVENUE]
            current = frame.at[current_date, REVENUE]
            if pd.notna(previous) and previous > 0 and pd.notna(current):
                frame.at[current_date, REVENUE_GROWTH] = current / previous - 1

    return frame


def _ttm_ratio(numerator: TTMResult, denominator: TTMResult) -> TTMResult:
    if not numerator.available:
        return TTMResult(None, False, numerator.periods_used, numerator.reason)
    if not denominator.available:
        return TTMResult(None, False, denominator.periods_used, denominator.reason)
    numerator_quarters = tuple(period.to_period("Q") for period in numerator.periods_used)
    denominator_quarters = tuple(
        period.to_period("Q") for period in denominator.periods_used
    )
    if numerator_quarters != denominator_quarters:
        return TTMResult(None, False, (), "period_mismatch")
    if denominator.value == 0:
        return TTMResult(None, False, denominator.periods_used, "zero_revenue")
    return TTMResult(
        float(numerator.value / denominator.value),
        True,
        denominator.periods_used,
        None,
    )


def build_fundamental_history(
    *,
    annual_revenue: pd.Series | None,
    annual_gross_profit: pd.Series | None,
    annual_operating_income: pd.Series | None,
    annual_cfo: pd.Series | None,
    annual_capex: pd.Series | None,
    quarterly_revenue: pd.Series | None,
    quarterly_gross_profit: pd.Series | None,
    quarterly_operating_income: pd.Series | None,
    quarterly_cfo: pd.Series | None,
    quarterly_capex: pd.Series | None,
    annual_pretax_income: pd.Series | None = None,
    annual_tax_provision: pd.Series | None = None,
    annual_total_equity: pd.Series | None = None,
    annual_total_debt: pd.Series | None = None,
    annual_cash: pd.Series | None = None,
    annual_depreciation_amortization: pd.Series | None = None,
    annual_periods=None,
    quarterly_income_periods=None,
    quarterly_cashflow_periods=None,
) -> FundamentalHistory:
    """Calculate annual and validated TTM fundamental performance metrics."""
    annual = build_period_fundamentals(
        revenue=annual_revenue,
        gross_profit=annual_gross_profit,
        operating_income=annual_operating_income,
        cfo=annual_cfo,
        capex=annual_capex,
        periods=annual_periods,
        include_revenue_growth=True,
    )
    additional_reported = {
        PRETAX_INCOME: _normalize_series(annual_pretax_income),
        TAX_PROVISION: _normalize_series(annual_tax_provision),
        TOTAL_EQUITY: _normalize_series(annual_total_equity),
        TOTAL_DEBT: _normalize_series(annual_total_debt),
        CASH: _normalize_series(annual_cash),
        DEPRECIATION_AND_AMORTIZATION: _normalize_series(
            annual_depreciation_amortization
        ),
    }
    for name, values in additional_reported.items():
        annual[name] = values.reindex(annual.index)

    calculated_columns = (
        OPERATING_TAX_RATE,
        NOPAT,
        INVESTED_CAPITAL,
        AVERAGE_INVESTED_CAPITAL,
        ROIC,
        NET_INVESTMENT,
        REINVESTMENT_RATE,
        FUNDAMENTAL_GROWTH_CAPACITY,
        DELTA_REVENUE,
        DELTA_INVESTED_CAPITAL,
        SALES_TO_CAPITAL,
        APPROXIMATE_ROIC,
    )
    for name in calculated_columns:
        annual[name] = np.nan
    annual_reasons = pd.DataFrame(
        [[None] * len(calculated_columns) for _ in annual.index],
        index=annual.index,
        columns=calculated_columns,
        dtype=object,
    )

    def optional(row: pd.Series, name: str) -> float | None:
        value = row[name]
        return None if pd.isna(value) else float(value)

    for period, row in annual.iterrows():
        nopat = calculate_nopat(
            optional(row, OPERATING_INCOME),
            optional(row, PRETAX_INCOME),
            optional(row, TAX_PROVISION),
        )
        if nopat.available:
            annual.at[period, OPERATING_TAX_RATE] = nopat.tax_rate
            annual.at[period, NOPAT] = nopat.nopat
        else:
            annual_reasons.at[period, OPERATING_TAX_RATE] = nopat.reason
            annual_reasons.at[period, NOPAT] = nopat.reason

        invested_capital = calculate_invested_capital(
            optional(row, TOTAL_EQUITY),
            optional(row, TOTAL_DEBT),
            optional(row, CASH),
        )
        if invested_capital.available:
            annual.at[period, INVESTED_CAPITAL] = invested_capital.value
        else:
            annual_reasons.at[period, INVESTED_CAPITAL] = invested_capital.reason

        net_investment = calculate_simplified_net_investment(
            optional(row, CAPEX),
            optional(row, DEPRECIATION_AND_AMORTIZATION),
        )
        if net_investment.available:
            annual.at[period, NET_INVESTMENT] = net_investment.value
        else:
            annual_reasons.at[period, NET_INVESTMENT] = net_investment.reason

    annual_sales_to_capital: dict[pd.Timestamp, SalesToCapitalResult] = {}
    for index, period in enumerate(annual.index):
        if index == 0:
            average = MetricCalculation(
                None, False, "missing_prior_invested_capital"
            )
        else:
            previous_period = annual.index[index - 1]
            average = calculate_average_invested_capital(
                optional(annual.loc[period], INVESTED_CAPITAL),
                optional(annual.loc[previous_period], INVESTED_CAPITAL),
                period,
                previous_period,
            )
        if average.available:
            annual.at[period, AVERAGE_INVESTED_CAPITAL] = average.value
        else:
            annual_reasons.at[period, AVERAGE_INVESTED_CAPITAL] = average.reason

        roic = calculate_roic(
            optional(annual.loc[period], NOPAT),
            optional(annual.loc[period], AVERAGE_INVESTED_CAPITAL),
        )
        if roic.available:
            annual.at[period, ROIC] = roic.value
        else:
            annual_reasons.at[period, ROIC] = roic.reason

        reinvestment_rate = calculate_reinvestment_rate(
            optional(annual.loc[period], NET_INVESTMENT),
            optional(annual.loc[period], NOPAT),
        )
        if reinvestment_rate.available:
            annual.at[period, REINVESTMENT_RATE] = reinvestment_rate.value
        else:
            annual_reasons.at[period, REINVESTMENT_RATE] = reinvestment_rate.reason

        growth_capacity = calculate_fundamental_growth_capacity(
            optional(annual.loc[period], ROIC),
            optional(annual.loc[period], REINVESTMENT_RATE),
        )
        if growth_capacity.available:
            annual.at[period, FUNDAMENTAL_GROWTH_CAPACITY] = growth_capacity.value
        else:
            annual_reasons.at[
                period, FUNDAMENTAL_GROWTH_CAPACITY
            ] = growth_capacity.reason

        if index == 0:
            sales_to_capital = SalesToCapitalResult(
                None, False, None, period, 1, "missing_prior_fiscal_year"
            )
        else:
            previous_period = annual.index[index - 1]
            sales_to_capital = calculate_sales_to_capital(
                optional(annual.loc[previous_period], REVENUE),
                optional(annual.loc[period], REVENUE),
                optional(annual.loc[previous_period], INVESTED_CAPITAL),
                optional(annual.loc[period], INVESTED_CAPITAL),
                previous_period,
                period,
            )
        annual_sales_to_capital[pd.Timestamp(period)] = sales_to_capital
        if sales_to_capital.delta_revenue is not None:
            annual.at[period, DELTA_REVENUE] = sales_to_capital.delta_revenue
        if sales_to_capital.delta_invested_capital is not None:
            annual.at[period, DELTA_INVESTED_CAPITAL] = (
                sales_to_capital.delta_invested_capital
            )
        if sales_to_capital.available:
            annual.at[period, SALES_TO_CAPITAL] = sales_to_capital.value
        else:
            annual_reasons.at[period, SALES_TO_CAPITAL] = sales_to_capital.reason

        approximate_roic = calculate_approximate_roic(
            optional(annual.loc[period], OPERATING_MARGIN),
            optional(annual.loc[period], OPERATING_TAX_RATE),
            sales_to_capital.value if sales_to_capital.available else None,
        )
        if approximate_roic.available:
            annual.at[period, APPROXIMATE_ROIC] = approximate_roic.value
        else:
            annual_reasons.at[period, APPROXIMATE_ROIC] = approximate_roic.reason

    # The active Yahoo-backed product exposes only 3Y anchors because Yahoo
    # normally supplies about five annual observations, while a true 5Y metric
    # needs six fiscal-year endpoints. The generic N-year helpers remain public.
    revenue_cagr = {3: calculate_revenue_cagr(annual[REVENUE], 3)}
    normalized_sales_to_capital = {
        3: calculate_normalized_sales_to_capital(
            annual[REVENUE], annual[INVESTED_CAPITAL], 3
        )
    }

    quarterly_income = build_period_fundamentals(
        revenue=quarterly_revenue,
        gross_profit=quarterly_gross_profit,
        operating_income=quarterly_operating_income,
        cfo=None,
        capex=None,
        periods=quarterly_income_periods,
        include_revenue_growth=False,
    )
    quarterly_cashflow = build_period_fundamentals(
        revenue=None,
        gross_profit=None,
        operating_income=None,
        cfo=quarterly_cfo,
        capex=quarterly_capex,
        periods=quarterly_cashflow_periods,
        include_revenue_growth=False,
    )

    ttm = {
        REVENUE: build_validated_ttm(
            quarterly_income[REVENUE], quarterly_income_periods
        ),
        GROSS_PROFIT: build_validated_ttm(
            quarterly_income[GROSS_PROFIT], quarterly_income_periods
        ),
        OPERATING_INCOME: build_validated_ttm(
            quarterly_income[OPERATING_INCOME], quarterly_income_periods
        ),
        CFO: build_validated_ttm(quarterly_cashflow[CFO], quarterly_cashflow_periods),
        CAPEX: build_validated_ttm(
            quarterly_cashflow[CAPEX], quarterly_cashflow_periods
        ),
        FCF: build_validated_ttm(quarterly_cashflow[FCF], quarterly_cashflow_periods),
    }
    ttm[GROSS_MARGIN] = _ttm_ratio(ttm[GROSS_PROFIT], ttm[REVENUE])
    ttm[OPERATING_MARGIN] = _ttm_ratio(ttm[OPERATING_INCOME], ttm[REVENUE])
    ttm[FCF_MARGIN] = _ttm_ratio(ttm[FCF], ttm[REVENUE])
    return FundamentalHistory(
        annual=annual,
        ttm=ttm,
        annual_reasons=annual_reasons,
        dcf_anchors=HistoricalDCFAnchors(
            revenue_cagr=revenue_cagr,
            annual_sales_to_capital=annual_sales_to_capital,
            normalized_sales_to_capital=normalized_sales_to_capital,
        ),
    )
