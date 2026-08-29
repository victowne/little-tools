from types import SimpleNamespace

import pandas as pd
import pytest

from Stock.forecast_anchors import (
    ForwardRevenueEstimate,
    align_dcf_and_consensus_period,
    build_dcf_revenue_forecast_periods,
    build_forward_revenue_estimate_set,
    compare_aligned_forward_estimate,
    issuer_anchor_ticker,
)


ACTUAL = pd.Timestamp("2025-12-31")
RETRIEVED = pd.Timestamp("2026-03-01", tz="UTC")


def estimate(period, revenue, *, source="primary", as_of=None, explicit=True):
    return ForwardRevenueEstimate(
        issuer_id="TEST", source_ticker="TEST",
        fiscal_period_end=pd.Timestamp(period) if period else None,
        fiscal_year_label=None, revenue_estimate=revenue,
        estimate_statistic="mean", analyst_count=20,
        source=source, source_as_of=as_of, retrieved_at=RETRIEVED,
        available=True, reason=None, fiscal_period_explicit=explicit,
    )


def estimate_set(items, *, ticker="TEST", source="primary", as_of=None):
    return build_forward_revenue_estimate_set(
        ticker=ticker, latest_actual_fiscal_period=ACTUAL,
        latest_actual_revenue=100.0, estimates=tuple(items), source=source,
        source_as_of=as_of, retrieved_at=RETRIEVED,
    )


def operating(count=3):
    return tuple(
        SimpleNamespace(year_index=i, revenue=100.0 + i * 20, revenue_growth=0.2)
        for i in range(1, count + 1)
    )


def test_explicit_fy1_fy2_fy3_dates_and_growth_chain():
    result = estimate_set([
        estimate("2026-12-31", 120.0),
        estimate("2027-12-31", 150.0),
        estimate("2028-12-31", 180.0),
    ])
    assert [x.fiscal_period_end for x in result.estimates] == list(
        pd.to_datetime(["2026-12-31", "2027-12-31", "2028-12-31"])
    )
    assert [x.implied_revenue_growth for x in result.estimates] == pytest.approx(
        [0.20, 0.25, 0.20]
    )


def test_normalized_estimate_set_supports_more_than_three_forward_years():
    result = estimate_set([
        estimate("2026-12-31", 120.0), estimate("2027-12-31", 150.0),
        estimate("2028-12-31", 180.0), estimate("2029-12-31", 198.0),
    ])
    assert len(result.estimates) == 4
    assert result.estimates[-1].implied_revenue_growth == pytest.approx(0.10)


def test_missing_fy2_suppresses_fy3_growth():
    result = estimate_set([
        estimate("2026-12-31", 120.0), estimate("2028-12-31", 180.0)
    ])
    assert result.estimates[0].implied_revenue_growth == pytest.approx(0.20)
    assert result.estimates[1].implied_revenue_growth is None
    assert "non_consecutive_fiscal_period" in result.estimates[1].warnings


def test_duplicate_period_is_unavailable():
    result = estimate_set([
        estimate("2026-12-31", 120.0), estimate("2026-12-31", 121.0)
    ])
    assert all(not item.available for item in result.estimates)
    assert all(item.reason == "duplicate_fiscal_period" for item in result.estimates)


def test_out_of_order_periods_are_sorted():
    result = estimate_set([
        estimate("2028-12-31", 180.0), estimate("2026-12-31", 120.0),
        estimate("2027-12-31", 150.0),
    ])
    assert [item.revenue_estimate for item in result.estimates] == [120.0, 150.0, 180.0]


def test_explicit_and_missing_as_of_and_staleness():
    recent = estimate_set(
        [estimate("2026-12-31", 120.0)],
        as_of=pd.Timestamp("2026-02-01", tz="UTC"),
    )
    missing = estimate_set([estimate("2026-12-31", 120.0)])
    stale = estimate_set(
        [estimate("2026-12-31", 120.0)],
        as_of=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    assert recent.source_as_of == pd.Timestamp("2026-02-01", tz="UTC")
    assert "source_as_of_unavailable" in missing.warnings
    assert "stale_estimate_data" in stale.warnings


def test_goog_googl_issuer_normalization():
    assert issuer_anchor_ticker("GOOG") == issuer_anchor_ticker("GOOGL") == "GOOGL"
    goog = estimate_set([estimate("2026-12-31", 120.0)], ticker="GOOG")
    googl = estimate_set([estimate("2026-12-31", 120.0)], ticker="GOOGL")
    assert goog.issuer_id == googl.issuer_id == "GOOGL"


def test_dcf_time_axis_is_twelve_month_intervals_after_base():
    periods = build_dcf_revenue_forecast_periods(ACTUAL, operating())
    assert periods[0].period_start == pd.Timestamp("2026-01-01")
    assert periods[0].period_end == pd.Timestamp("2026-12-31")
    assert periods[1].period_start == pd.Timestamp("2027-01-01")
    assert periods[1].period_end == pd.Timestamp("2027-12-31")


def alignment(base, fiscal_end, prior_end, *, explicit=True):
    dcf = build_dcf_revenue_forecast_periods(pd.Timestamp(base), operating(1))[0]
    item = estimate(fiscal_end, 120.0, explicit=explicit)
    return align_dcf_and_consensus_period(dcf, item, pd.Timestamp(prior_end) if prior_end else None)


def test_exact_fy_aligned_period_is_comparable():
    result = alignment("2025-12-31", "2026-12-31", "2025-12-31")
    assert result.alignment_status == "exact"
    assert result.overlap_fraction == 1.0
    assert result.comparable is True


def test_aligned_comparison_calculates_direct_deltas():
    dcf = build_dcf_revenue_forecast_periods(ACTUAL, operating(1))[0]
    item = estimate("2026-12-31", 110.0)
    aligned = align_dcf_and_consensus_period(dcf, item, ACTUAL)
    comparison = compare_aligned_forward_estimate(dcf, item, aligned)
    assert comparison.period_aligned is True
    assert comparison.dcf_minus_consensus_revenue == pytest.approx(10.0)
    assert comparison.assumption_minus_consensus_growth is None


def test_near_aligned_threshold_is_comparable():
    result = alignment("2025-12-15", "2026-12-31", "2025-12-31")
    assert result.alignment_status == "near_aligned"
    assert result.overlap_fraction >= 0.90
    assert result.comparable is True


@pytest.mark.parametrize(
    ("base", "fiscal_end", "prior_end"),
    [
        ("2026-04-30", "2027-01-31", "2026-01-31"),
        ("2026-06-30", "2026-12-31", "2025-12-31"),
    ],
)
def test_ttm_base_vs_january_or_december_fy_is_partial_and_suppressed(
    base, fiscal_end, prior_end
):
    result = alignment(base, fiscal_end, prior_end)
    assert result.alignment_status == "partial_overlap"
    assert result.comparable is False


def test_completely_mismatched_period_is_suppressed():
    result = alignment("2028-12-31", "2026-12-31", "2025-12-31")
    assert result.alignment_status == "mismatched"
    assert result.overlap_fraction == 0.0
    assert result.comparable is False


def test_noncomparable_period_suppresses_all_direct_deltas():
    dcf = build_dcf_revenue_forecast_periods(
        pd.Timestamp("2026-06-30"), operating(1)
    )[0]
    item = estimate("2026-12-31", 110.0)
    aligned = align_dcf_and_consensus_period(
        dcf, item, pd.Timestamp("2025-12-31")
    )
    comparison = compare_aligned_forward_estimate(dcf, item, aligned)
    assert comparison.period_aligned is False
    assert comparison.dcf_minus_consensus_revenue is None
    assert comparison.assumption_minus_consensus_growth is None


@pytest.mark.parametrize(
    ("fiscal_end", "prior_end", "explicit"),
    [(None, "2025-12-31", True), ("2026-12-31", None, True), ("2026-12-31", "2025-12-31", False)],
)
def test_unavailable_or_inferred_fiscal_date_suppresses_comparison(
    fiscal_end, prior_end, explicit
):
    result = alignment("2025-12-31", fiscal_end, prior_end, explicit=explicit)
    assert result.alignment_status == "unavailable"
    assert result.comparable is False
