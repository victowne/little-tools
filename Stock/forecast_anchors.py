"""Auditable near-term Revenue forecast anchors, independent of valuation."""

from dataclasses import dataclass
import math
from typing import Literal

import pandas as pd


ISSUER_ANCHOR_TICKERS = {"GOOG": "GOOGL", "GOOGL": "GOOGL"}
STALE_ESTIMATE_DAYS = 120
NEAR_ALIGNED_MIN_OVERLAP = 0.90
NEAR_ALIGNED_MAX_BOUNDARY_SHIFT_DAYS = 45


@dataclass(frozen=True)
class ForwardRevenueEstimate:
    issuer_id: str
    source_ticker: str
    fiscal_period_end: pd.Timestamp | None
    fiscal_year_label: str | None
    revenue_estimate: float | None
    estimate_statistic: str
    analyst_count: int | None
    source: str
    source_as_of: pd.Timestamp | None
    retrieved_at: pd.Timestamp | None
    available: bool
    reason: str | None
    warnings: tuple[str, ...] = ()
    fiscal_period_explicit: bool = True
    implied_revenue_growth: float | None = None
    period_frequency: Literal["annual", "quarterly"] = "annual"
    revenue_estimate_median: float | None = None
    revenue_estimate_high: float | None = None
    revenue_estimate_low: float | None = None


@dataclass(frozen=True)
class ForwardRevenueEstimateSet:
    issuer_id: str
    source_ticker: str
    latest_actual_fiscal_period: pd.Timestamp
    latest_actual_revenue: float
    estimates: tuple[ForwardRevenueEstimate, ...]
    source: str
    source_as_of: pd.Timestamp | None
    retrieved_at: pd.Timestamp | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DCFRevenueForecastPeriod:
    year_index: int
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    revenue: float
    revenue_growth: float


@dataclass(frozen=True)
class ForecastPeriodAlignment:
    dcf_period_start: pd.Timestamp
    dcf_period_end: pd.Timestamp
    consensus_period_start: pd.Timestamp | None
    consensus_period_end: pd.Timestamp | None
    overlap_fraction: float | None
    alignment_status: Literal[
        "exact", "near_aligned", "partial_overlap", "mismatched", "unavailable"
    ]
    comparable: bool
    reason: str | None


@dataclass(frozen=True)
class ForecastAnchorPoint:
    forecast_year_index: int
    fiscal_period: pd.Timestamp | None
    revenue_estimate: float | None
    implied_revenue_growth: float | None
    source: str
    source_as_of: pd.Timestamp | None
    analyst_count: int | None
    available: bool
    reason: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevenueForecastAnchors:
    ticker: str
    issuer_ticker: str
    current_revenue_base: float
    base_period: pd.Timestamp | None
    base_kind: str
    latest_actual_fiscal_revenue: float | None
    latest_actual_fiscal_period: pd.Timestamp | None
    points: tuple[ForecastAnchorPoint, ForecastAnchorPoint, ForecastAnchorPoint]
    source: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevenueAnchorComparisonPoint:
    forecast_year_index: int
    fiscal_period: pd.Timestamp | None
    consensus_revenue: float | None
    consensus_fiscal_growth: float | None
    dcf_revenue: float
    dcf_growth: float
    assumption_minus_consensus_growth: float | None
    dcf_minus_consensus_revenue: float | None
    dcf_minus_consensus_revenue_percent: float | None
    period_aligned: bool
    reason: str | None


def issuer_anchor_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    return ISSUER_ANCHOR_TICKERS.get(normalized, normalized)


def _finite_positive(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _finite_optional(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _analyst_count(value) -> int | None:
    numeric = _finite_positive(value)
    if numeric is None or not float(numeric).is_integer():
        return None
    return int(numeric)


def _next_fiscal_period(period: pd.Timestamp, years: int) -> pd.Timestamp:
    return pd.Timestamp(period) + pd.DateOffset(years=years)


def _utc_timestamp(value) -> pd.Timestamp | None:
    """Normalize provider/retrieval timestamps for deterministic age checks."""
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp)


def build_forward_revenue_estimate_set(
    *,
    ticker: str,
    latest_actual_fiscal_period: pd.Timestamp,
    latest_actual_revenue: float,
    estimates: tuple[ForwardRevenueEstimate, ...],
    source: str,
    source_as_of: pd.Timestamp | None = None,
    retrieved_at: pd.Timestamp | None = None,
) -> ForwardRevenueEstimateSet:
    """Validate, order, and derive comparable FY-to-FY consensus growth."""
    actual_period = pd.Timestamp(latest_actual_fiscal_period)
    actual_revenue = _finite_positive(latest_actual_revenue)
    if pd.isna(actual_period) or actual_revenue is None:
        raise ValueError("latest actual fiscal period and Revenue must be valid")
    valid_periods = [
        pd.Timestamp(item.fiscal_period_end)
        for item in estimates
        if item.fiscal_period_end is not None
    ]
    duplicate_periods = {
        period for period in valid_periods if valid_periods.count(period) > 1
    }
    ordered = sorted(
        estimates,
        key=lambda item: (
            item.fiscal_period_end is None,
            pd.Timestamp.max if item.fiscal_period_end is None
            else pd.Timestamp(item.fiscal_period_end),
        ),
    )
    normalized = []
    prior_period = actual_period
    prior_revenue = actual_revenue
    for item in ordered:
        period = (
            pd.Timestamp(item.fiscal_period_end)
            if item.fiscal_period_end is not None else None
        )
        revenue = _finite_positive(item.revenue_estimate)
        warnings = list(item.warnings)
        reason = item.reason
        available = item.available and period is not None and revenue is not None
        if period in duplicate_periods:
            available = False
            reason = "duplicate_fiscal_period"
            revenue = None
        if period is None:
            available = False
            reason = reason or "fiscal_period_unavailable"
        if item.available and revenue is None:
            available = False
            reason = reason or "invalid_revenue_estimate"
        growth = None
        if available:
            expected_period = _next_fiscal_period(prior_period, 1)
            consecutive = (
                period.year == expected_period.year
                and period.month == expected_period.month
            )
            if consecutive and prior_revenue is not None:
                growth = revenue / prior_revenue - 1
            else:
                warnings.append("non_consecutive_fiscal_period")
            prior_period = period
            prior_revenue = revenue
        else:
            prior_revenue = None
            if period is not None:
                prior_period = period
        normalized.append(ForwardRevenueEstimate(
            item.issuer_id, item.source_ticker, period,
            item.fiscal_year_label, revenue, item.estimate_statistic,
            item.analyst_count, item.source, _utc_timestamp(item.source_as_of),
            _utc_timestamp(item.retrieved_at), available, reason,
            tuple(dict.fromkeys(warnings)), item.fiscal_period_explicit, growth,
            item.period_frequency,
            _finite_optional(item.revenue_estimate_median),
            _finite_optional(item.revenue_estimate_high),
            _finite_optional(item.revenue_estimate_low),
        ))
    warnings = []
    normalized_source_as_of = _utc_timestamp(source_as_of)
    normalized_retrieved_at = _utc_timestamp(retrieved_at)
    if normalized_source_as_of is None:
        warnings.append("source_as_of_unavailable")
    elif normalized_retrieved_at is not None:
        age_days = (normalized_retrieved_at - normalized_source_as_of).days
        if age_days > STALE_ESTIMATE_DAYS:
            warnings.append("stale_estimate_data")
    if any(not item.fiscal_period_explicit for item in normalized):
        warnings.append("contains_inferred_fiscal_periods")
    return ForwardRevenueEstimateSet(
        issuer_anchor_ticker(ticker), ticker.strip().upper(), actual_period,
        actual_revenue, tuple(normalized), source,
        normalized_source_as_of, normalized_retrieved_at,
        tuple(warnings),
    )


def build_dcf_revenue_forecast_periods(
    base_period: pd.Timestamp,
    operating_years,
) -> tuple[DCFRevenueForecastPeriod, ...]:
    """Model DCF Year t as the twelve-month interval after the prior endpoint."""
    base = pd.Timestamp(base_period)
    if pd.isna(base):
        raise ValueError("DCF base period must be a valid date")
    periods = []
    prior_end = base
    for expected_index, year in enumerate(tuple(operating_years), start=1):
        if year.year_index != expected_index:
            raise ValueError("operating forecast years must be consecutive")
        end = base + pd.DateOffset(years=expected_index)
        periods.append(DCFRevenueForecastPeriod(
            expected_index, prior_end + pd.Timedelta(days=1), end,
            float(year.revenue), float(year.revenue_growth),
        ))
        prior_end = end
    return tuple(periods)


def align_dcf_and_consensus_period(
    dcf_period: DCFRevenueForecastPeriod,
    estimate: ForwardRevenueEstimate,
    prior_consensus_period_end: pd.Timestamp | None,
) -> ForecastPeriodAlignment:
    """Classify date overlap using explicit, documented objective thresholds.

    ``exact`` requires identical inclusive boundaries. ``near_aligned`` requires
    at least 90% overlap relative to the longer interval and each boundary no
    more than 45 days apart. Only exact/near-aligned periods are comparable.
    Any positive lesser overlap is partial; zero overlap is mismatched.
    """
    if (
        estimate.fiscal_period_end is None
        or prior_consensus_period_end is None
        or not estimate.fiscal_period_explicit
    ):
        return ForecastPeriodAlignment(
            dcf_period.period_start, dcf_period.period_end, None,
            estimate.fiscal_period_end, None, "unavailable", False,
            "explicit_consensus_period_unavailable",
        )
    consensus_start = pd.Timestamp(prior_consensus_period_end) + pd.Timedelta(days=1)
    consensus_end = pd.Timestamp(estimate.fiscal_period_end)
    if consensus_end < consensus_start:
        return ForecastPeriodAlignment(
            dcf_period.period_start, dcf_period.period_end, consensus_start,
            consensus_end, 0.0, "unavailable", False,
            "invalid_consensus_period",
        )
    overlap_start = max(dcf_period.period_start, consensus_start)
    overlap_end = min(dcf_period.period_end, consensus_end)
    overlap_days = max(0, (overlap_end - overlap_start).days + 1)
    dcf_days = (dcf_period.period_end - dcf_period.period_start).days + 1
    consensus_days = (consensus_end - consensus_start).days + 1
    overlap_fraction = overlap_days / max(dcf_days, consensus_days)
    if (
        dcf_period.period_start == consensus_start
        and dcf_period.period_end == consensus_end
    ):
        status = "exact"
    elif (
        overlap_fraction >= NEAR_ALIGNED_MIN_OVERLAP
        and abs((dcf_period.period_start - consensus_start).days)
        <= NEAR_ALIGNED_MAX_BOUNDARY_SHIFT_DAYS
        and abs((dcf_period.period_end - consensus_end).days)
        <= NEAR_ALIGNED_MAX_BOUNDARY_SHIFT_DAYS
    ):
        status = "near_aligned"
    elif overlap_days > 0:
        status = "partial_overlap"
    else:
        status = "mismatched"
    comparable = status in {"exact", "near_aligned"}
    return ForecastPeriodAlignment(
        dcf_period.period_start, dcf_period.period_end, consensus_start,
        consensus_end, overlap_fraction, status, comparable,
        None if comparable else "forecast_periods_not_comparable",
    )


def compare_aligned_forward_estimate(
    dcf_period: DCFRevenueForecastPeriod,
    estimate: ForwardRevenueEstimate,
    alignment: ForecastPeriodAlignment,
) -> RevenueAnchorComparisonPoint:
    """Calculate deltas only after the period matcher declares comparability."""
    can_compare = (
        alignment.comparable
        and estimate.available
        and estimate.revenue_estimate is not None
    )
    if can_compare:
        revenue_delta = dcf_period.revenue - estimate.revenue_estimate
        growth_delta = (
            dcf_period.revenue_growth - estimate.implied_revenue_growth
            if estimate.implied_revenue_growth is not None else None
        )
        percent_delta = revenue_delta / estimate.revenue_estimate
        reason = None
    else:
        revenue_delta = None
        growth_delta = None
        percent_delta = None
        reason = alignment.reason or estimate.reason or "estimate_unavailable"
    return RevenueAnchorComparisonPoint(
        dcf_period.year_index, estimate.fiscal_period_end,
        estimate.revenue_estimate, estimate.implied_revenue_growth,
        dcf_period.revenue, dcf_period.revenue_growth, growth_delta,
        revenue_delta, percent_delta, can_compare, reason,
    )


def unavailable_revenue_forecast_anchors(
    ticker: str,
    current_revenue_base: float,
    base_period: pd.Timestamp | None,
    base_kind: str,
    reason: str,
    *,
    source: str = "unavailable",
) -> RevenueForecastAnchors:
    points = tuple(
        ForecastAnchorPoint(index, None, None, None, source, None, None, False, reason)
        for index in (1, 2, 3)
    )
    return RevenueForecastAnchors(
        ticker.strip().upper(), issuer_anchor_ticker(ticker),
        float(current_revenue_base), base_period, base_kind, None, None,
        points, source, (reason,),
    )


def build_revenue_forecast_anchors(
    *,
    ticker: str,
    current_revenue_base: float,
    base_period: pd.Timestamp | None,
    base_kind: str,
    latest_actual_fiscal_revenue: float | None,
    latest_actual_fiscal_period: pd.Timestamp | None,
    estimates: pd.DataFrame | None,
    source: str,
    source_as_of: pd.Timestamp | None = None,
) -> RevenueForecastAnchors:
    """Normalize explicit fiscal-year estimates into three non-filled anchors.

    ``estimates`` must contain ``fiscal_period`` and ``revenue_estimate`` and
    may contain ``analyst_count`` and ``prior_fiscal_revenue``. Duplicate
    periods are ambiguous. Missing years are never extrapolated or backfilled.
    Fiscal growth is derived only from comparable fiscal-year Revenue levels.
    """
    revenue_base = _finite_positive(current_revenue_base)
    if revenue_base is None:
        raise ValueError("current_revenue_base must be finite and positive")
    actual_revenue = _finite_positive(latest_actual_fiscal_revenue)
    actual_period = (
        pd.Timestamp(latest_actual_fiscal_period)
        if latest_actual_fiscal_period is not None else None
    )
    if estimates is None or estimates.empty or actual_period is None:
        return unavailable_revenue_forecast_anchors(
            ticker, revenue_base, base_period, base_kind,
            "forecast_consensus_unavailable", source=source,
        )
    required = {"fiscal_period", "revenue_estimate"}
    if not required.issubset(estimates.columns):
        return unavailable_revenue_forecast_anchors(
            ticker, revenue_base, base_period, base_kind,
            "invalid_provider_schema", source=source,
        )
    frame = estimates.copy()
    frame["fiscal_period"] = pd.to_datetime(frame["fiscal_period"], errors="coerce")
    frame = frame.dropna(subset=["fiscal_period"]).sort_values("fiscal_period")
    duplicates = set(frame.loc[frame["fiscal_period"].duplicated(False), "fiscal_period"])
    points = []
    prior_revenue = actual_revenue
    for year_index in (1, 2, 3):
        expected_period = _next_fiscal_period(actual_period, year_index)
        candidates = frame[
            (frame["fiscal_period"].dt.year == expected_period.year)
            & (frame["fiscal_period"].dt.month == expected_period.month)
        ]
        if len(candidates) > 1 or any(pd.Timestamp(p) in duplicates for p in candidates["fiscal_period"]):
            points.append(ForecastAnchorPoint(
                year_index, expected_period, None, None, source, source_as_of,
                None, False, "duplicate_fiscal_period",
            ))
            prior_revenue = None
            continue
        if candidates.empty:
            points.append(ForecastAnchorPoint(
                year_index, expected_period, None, None, source, source_as_of,
                None, False, "forecast_year_unavailable",
            ))
            prior_revenue = None
            continue
        row = candidates.iloc[0]
        estimate = _finite_positive(row["revenue_estimate"])
        if estimate is None:
            points.append(ForecastAnchorPoint(
                year_index, pd.Timestamp(row["fiscal_period"]), None, None,
                source, source_as_of, _analyst_count(row.get("analyst_count")),
                False, "invalid_revenue_estimate",
            ))
            prior_revenue = None
            continue
        explicit_prior = _finite_positive(row.get("prior_fiscal_revenue"))
        comparable_prior = explicit_prior if explicit_prior is not None else prior_revenue
        growth = (
            estimate / comparable_prior - 1
            if comparable_prior is not None else None
        )
        points.append(ForecastAnchorPoint(
            year_index, pd.Timestamp(row["fiscal_period"]), estimate, growth,
            source, source_as_of, _analyst_count(row.get("analyst_count")),
            True, None,
        ))
        prior_revenue = estimate
    warnings = []
    if base_kind == "ttm":
        warnings.append("ttm_base_not_directly_comparable_to_fiscal_consensus")
    if any(not point.available for point in points):
        warnings.append("incomplete_three_year_consensus")
    return RevenueForecastAnchors(
        ticker.strip().upper(), issuer_anchor_ticker(ticker), revenue_base,
        pd.Timestamp(base_period) if base_period is not None else None,
        base_kind, actual_revenue, actual_period, tuple(points), source,
        tuple(warnings),
    )


def yfinance_revenue_estimates_to_fiscal_frame(
    raw: pd.DataFrame | None,
    latest_actual_fiscal_period: pd.Timestamp | None,
) -> pd.DataFrame:
    """Adapt yfinance 0y/+1y rows; Yahoo exposes no third annual estimate."""
    columns = [
        "fiscal_period", "revenue_estimate", "prior_fiscal_revenue",
        "analyst_count",
    ]
    if raw is None or raw.empty or latest_actual_fiscal_period is None:
        return pd.DataFrame(columns=columns)
    rows = []
    for label, year_index in (("0y", 1), ("+1y", 2)):
        if label not in raw.index:
            continue
        row = raw.loc[label]
        if isinstance(row, pd.DataFrame):
            # Duplicate provider rows are retained so the normalizer rejects them.
            iterable = [entry for _, entry in row.iterrows()]
        else:
            iterable = [row]
        for entry in iterable:
            rows.append({
                "fiscal_period": _next_fiscal_period(
                    pd.Timestamp(latest_actual_fiscal_period), year_index
                ),
                "revenue_estimate": entry.get("avg"),
                "prior_fiscal_revenue": entry.get("yearAgoRevenue") if year_index == 1 else None,
                "analyst_count": entry.get("numberOfAnalysts"),
            })
    return pd.DataFrame(rows, columns=columns)


def load_revenue_forecast_anchors(
    *,
    ticker: str,
    current_revenue_base: float,
    base_period: pd.Timestamp | None,
    base_kind: str,
    latest_actual_fiscal_revenue: float | None,
    latest_actual_fiscal_period: pd.Timestamp | None,
    provider_data: pd.DataFrame | None,
    provider_as_of: pd.Timestamp | None,
    provider: str = "yfinance_analyst_consensus_mean",
) -> RevenueForecastAnchors:
    """Provider boundary for the current Yahoo adapter.

    The returned structures contain no yfinance objects. A later vendor or
    manually supplied research dataset can call ``build_revenue_forecast_anchors``
    directly without changing the DCF engine or UI comparison semantics.
    """
    estimates = yfinance_revenue_estimates_to_fiscal_frame(
        provider_data, latest_actual_fiscal_period
    )
    anchors = build_revenue_forecast_anchors(
        ticker=ticker,
        current_revenue_base=current_revenue_base,
        base_period=base_period,
        base_kind=base_kind,
        latest_actual_fiscal_revenue=latest_actual_fiscal_revenue,
        latest_actual_fiscal_period=latest_actual_fiscal_period,
        estimates=estimates,
        source=provider,
        source_as_of=provider_as_of,
    )
    warnings = list(anchors.warnings)
    if not estimates.empty:
        warnings.append("fiscal_period_inferred_from_latest_actual_year_end")
    if provider_as_of is not None:
        warnings.append("source_as_of_is_retrieval_time_not_vendor_update_time")
    return RevenueForecastAnchors(
        anchors.ticker, anchors.issuer_ticker, anchors.current_revenue_base,
        anchors.base_period, anchors.base_kind,
        anchors.latest_actual_fiscal_revenue,
        anchors.latest_actual_fiscal_period, anchors.points, anchors.source,
        tuple(dict.fromkeys(warnings)),
    )


def revenue_anchors_to_forward_estimate_set(
    anchors: RevenueForecastAnchors,
    *,
    retrieved_at: pd.Timestamp | None,
) -> ForwardRevenueEstimateSet:
    """Expose existing Yahoo anchors through the normalized provider boundary."""
    if (
        anchors.latest_actual_fiscal_period is None
        or anchors.latest_actual_fiscal_revenue is None
    ):
        raise ValueError("latest actual fiscal Revenue is required")
    estimates = tuple(
        ForwardRevenueEstimate(
            issuer_id=anchors.issuer_ticker,
            source_ticker=anchors.issuer_ticker,
            fiscal_period_end=point.fiscal_period,
            fiscal_year_label=(
                f"FY{point.fiscal_period.year}" if point.fiscal_period else None
            ),
            revenue_estimate=point.revenue_estimate,
            estimate_statistic="mean",
            analyst_count=point.analyst_count,
            source=anchors.source,
            source_as_of=None,
            retrieved_at=retrieved_at,
            available=point.available,
            reason=point.reason,
            warnings=point.warnings,
            fiscal_period_explicit=False,
        )
        for point in anchors.points
    )
    return build_forward_revenue_estimate_set(
        ticker=anchors.issuer_ticker,
        latest_actual_fiscal_period=anchors.latest_actual_fiscal_period,
        latest_actual_revenue=anchors.latest_actual_fiscal_revenue,
        estimates=estimates,
        source=anchors.source,
        source_as_of=None,
        retrieved_at=retrieved_at,
    )


def compare_revenue_anchors_to_forecast(
    anchors: RevenueForecastAnchors,
    forecast_years,
) -> tuple[RevenueAnchorComparisonPoint, ...]:
    """Compare only period-aligned fiscal anchors; never mutate assumptions."""
    operating_years = tuple(forecast_years)
    comparisons = []
    for point in anchors.points:
        if point.forecast_year_index > len(operating_years):
            break
        year = operating_years[point.forecast_year_index - 1]
        period_aligned = anchors.base_kind == "annual" and point.available
        if period_aligned:
            revenue_delta = year.revenue - point.revenue_estimate
            growth_delta = (
                year.revenue_growth - point.implied_revenue_growth
                if point.implied_revenue_growth is not None else None
            )
            percent_delta = revenue_delta / point.revenue_estimate
            reason = None
        else:
            revenue_delta = None
            growth_delta = None
            percent_delta = None
            reason = point.reason or "ttm_fiscal_period_mismatch"
        comparisons.append(RevenueAnchorComparisonPoint(
            point.forecast_year_index, point.fiscal_period,
            point.revenue_estimate, point.implied_revenue_growth,
            year.revenue, year.revenue_growth, growth_delta, revenue_delta,
            percent_delta, period_aligned, reason,
        ))
    return tuple(comparisons)
