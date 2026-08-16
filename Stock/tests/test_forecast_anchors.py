from types import SimpleNamespace

import pandas as pd
import pytest

from Stock.forecast_anchors import (
    build_revenue_forecast_anchors,
    compare_revenue_anchors_to_forecast,
    issuer_anchor_ticker,
    load_revenue_forecast_anchors,
    yfinance_revenue_estimates_to_fiscal_frame,
)


ACTUAL_PERIOD = pd.Timestamp("2025-12-31")


def estimates(values=(120.0, 138.0, 151.8), periods=None):
    periods = periods or ("2026-12-31", "2027-12-31", "2028-12-31")
    return pd.DataFrame({
        "fiscal_period": pd.to_datetime(periods),
        "revenue_estimate": values,
        "analyst_count": [20] * len(values),
    })


def anchors(frame=None, *, base_kind="annual", ticker="TEST"):
    return build_revenue_forecast_anchors(
        ticker=ticker,
        current_revenue_base=100.0,
        base_period=ACTUAL_PERIOD,
        base_kind=base_kind,
        latest_actual_fiscal_revenue=100.0,
        latest_actual_fiscal_period=ACTUAL_PERIOD,
        estimates=estimates() if frame is None else frame,
        source="fixture_consensus",
        source_as_of=pd.Timestamp("2026-01-15"),
    )


def forecast(growth=(0.20, 0.15, 0.10), revenues=(120.0, 138.0, 151.8)):
    return tuple(
        SimpleNamespace(year_index=i, revenue_growth=g, revenue=r)
        for i, (g, r) in enumerate(zip(growth, revenues), start=1)
    )


def test_three_valid_fiscal_year_estimates_derive_fy_growth():
    result = anchors()
    assert all(point.available for point in result.points)
    assert [point.revenue_estimate for point in result.points] == [120.0, 138.0, 151.8]
    assert [point.implied_revenue_growth for point in result.points] == pytest.approx([0.20, 0.15, 0.10])


@pytest.mark.parametrize("count", [1, 2])
def test_partial_consensus_does_not_fill_later_years(count):
    frame = estimates(values=(120.0, 138.0)[:count], periods=("2026-12-31", "2027-12-31")[:count])
    result = anchors(frame)
    assert sum(point.available for point in result.points) == count
    assert result.points[count].reason == "forecast_year_unavailable"
    assert "incomplete_three_year_consensus" in result.warnings


def test_missing_consensus_returns_three_unavailable_points():
    result = anchors(pd.DataFrame())
    assert all(not point.available for point in result.points)
    assert all(point.reason == "forecast_consensus_unavailable" for point in result.points)


def test_non_consecutive_period_does_not_shift_year_three_into_year_two():
    result = anchors(estimates(values=(120.0, 151.8), periods=("2026-12-31", "2028-12-31")))
    assert result.points[0].available is True
    assert result.points[1].reason == "forecast_year_unavailable"
    assert result.points[2].available is True
    assert result.points[2].implied_revenue_growth is None


def test_fiscal_year_month_mismatch_is_not_silently_aligned():
    result = anchors(estimates(values=(120.0,), periods=("2026-06-30",)))
    assert result.points[0].available is False
    assert result.points[0].reason == "forecast_year_unavailable"


def test_ttm_base_preserves_fiscal_anchor_but_suppresses_direct_deltas():
    result = anchors(base_kind="ttm")
    comparison = compare_revenue_anchors_to_forecast(result, forecast())
    assert result.points[0].implied_revenue_growth == pytest.approx(0.20)
    assert comparison[0].period_aligned is False
    assert comparison[0].assumption_minus_consensus_growth is None
    assert comparison[0].dcf_minus_consensus_revenue is None
    assert comparison[0].reason == "ttm_fiscal_period_mismatch"


@pytest.mark.parametrize(
    ("user_growth", "expected_delta"),
    [(0.25, 0.05), (0.15, -0.05), (0.20, 0.0)],
)
def test_aligned_user_assumption_delta_above_below_and_exact(user_growth, expected_delta):
    result = anchors(base_kind="annual")
    comparison = compare_revenue_anchors_to_forecast(
        result, forecast(growth=(user_growth, 0.15, 0.10))
    )
    assert comparison[0].assumption_minus_consensus_growth == pytest.approx(expected_delta)


def test_aligned_revenue_level_differences_are_exposed():
    comparison = compare_revenue_anchors_to_forecast(
        anchors(), forecast(revenues=(126.0, 140.0, 150.0))
    )
    assert comparison[0].dcf_minus_consensus_revenue == pytest.approx(6.0)
    assert comparison[0].dcf_minus_consensus_revenue_percent == pytest.approx(0.05)


def test_goog_and_googl_use_same_issuer_anchor_identity_and_economics():
    goog = anchors(ticker="GOOG")
    googl = anchors(ticker="GOOGL")
    assert issuer_anchor_ticker("GOOG") == issuer_anchor_ticker("GOOGL") == "GOOGL"
    assert goog.issuer_ticker == googl.issuer_ticker
    assert goog.points == googl.points


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_revenue_estimates_are_unavailable(bad):
    result = anchors(estimates(values=(bad,), periods=("2026-12-31",)))
    assert result.points[0].available is False
    assert result.points[0].reason == "invalid_revenue_estimate"


def test_duplicate_fiscal_period_is_rejected_not_arbitrarily_selected():
    frame = estimates(
        values=(120.0, 121.0),
        periods=("2026-12-31", "2026-12-31"),
    )
    result = anchors(frame)
    assert result.points[0].available is False
    assert result.points[0].reason == "duplicate_fiscal_period"


def test_unordered_estimate_rows_are_sorted_by_fiscal_period():
    frame = estimates(
        values=(151.8, 120.0, 138.0),
        periods=("2028-12-31", "2026-12-31", "2027-12-31"),
    )
    result = anchors(frame)
    assert [point.revenue_estimate for point in result.points] == [120.0, 138.0, 151.8]


def test_yfinance_adapter_exposes_only_zero_and_plus_one_year():
    raw = pd.DataFrame(
        {
            "avg": [120.0, 138.0],
            "yearAgoRevenue": [100.0, 120.0],
            "numberOfAnalysts": [25, 22],
        },
        index=["0y", "+1y"],
    )
    frame = yfinance_revenue_estimates_to_fiscal_frame(raw, ACTUAL_PERIOD)
    assert list(frame["fiscal_period"]) == [pd.Timestamp("2026-12-31"), pd.Timestamp("2027-12-31")]
    loaded = load_revenue_forecast_anchors(
        ticker="TEST", current_revenue_base=105.0,
        base_period=pd.Timestamp("2026-03-31"), base_kind="ttm",
        latest_actual_fiscal_revenue=100.0,
        latest_actual_fiscal_period=ACTUAL_PERIOD,
        provider_data=raw, provider_as_of=pd.Timestamp("2026-01-15"),
    )
    assert loaded.points[0].analyst_count == 25
    assert loaded.points[1].available is True
    assert loaded.points[2].reason == "forecast_year_unavailable"
    assert "fiscal_period_inferred_from_latest_actual_year_end" in loaded.warnings
