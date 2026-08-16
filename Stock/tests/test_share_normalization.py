from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from Stock.share_normalization import (
    NormalizedShareCount,
    ShareCountComponent,
    normalize_share_count,
)


def statement(value, period="2025-12-31"):
    return pd.DataFrame(
        {pd.Timestamp(period): [value]}, index=["OrdinarySharesNumber"]
    )


def test_single_class_current_metadata_is_accepted(snapshot_factory):
    snapshot = snapshot_factory(
        ticker="NVDA",
        shares_outstanding=24.221e9,
        ticker_shares_outstanding=24.221e9,
    )

    result = normalize_share_count(snapshot)

    assert result.available is True
    assert result.shares_outstanding == 24.221e9
    assert result.scope == "consolidated_common"
    assert result.method == "confirmed_single_class_current_metadata"
    assert result.source == "info.sharesOutstanding"


def test_latest_quarterly_consolidated_total_has_highest_priority(snapshot_factory):
    snapshot = snapshot_factory(
        ticker="GOOGL",
        shares_outstanding=5.867e9,
        ticker_shares_outstanding=5.867e9,
        implied_shares_outstanding=12.23e9,
        quarterly_balance=statement(12.20e9, "2026-06-30"),
        annual_balance=statement(12.08e9, "2025-12-31"),
    )

    result = normalize_share_count(snapshot)

    assert result.shares_outstanding == 12.20e9
    assert result.source == "quarterly_balance.OrdinarySharesNumber"
    assert result.source_period == pd.Timestamp("2026-06-30")
    assert result.scope == "consolidated_common"
    assert result.method == "issuer_reported_consolidated_quarterly"
    assert "multi_class_issuer" in result.warnings


def test_alphabet_only_class_a_ticker_count_is_unavailable(snapshot_factory):
    result = normalize_share_count(
        snapshot_factory(
            ticker="GOOGL",
            shares_outstanding=5.867e9,
            ticker_shares_outstanding=5.867e9,
        )
    )

    assert result.available is False
    assert result.shares_outstanding is None
    assert result.reason == "consolidated_share_count_unavailable"


def test_class_a_and_c_counts_are_not_blindly_summed(snapshot_factory):
    # Even explicit A/C observations do not prove completeness because Class B
    # also participates in Alphabet's residual common equity.
    result = normalize_share_count(
        snapshot_factory(
            ticker="GOOGL",
            shares_outstanding=5.867e9,
            ticker_shares_outstanding=5.867e9,
        ),
        additional_class_counts=(
            ShareCountComponent(
                "peer_ticker.GOOG.info.sharesOutstanding",
                5.527e9,
                scope="single_class",
            ),
        ),
    )

    assert result.available is False
    assert result.shares_outstanding is None
    assert result.shares_outstanding != pytest.approx(11.394e9)
    assert any(component.source.startswith("peer_ticker.GOOG") for component in result.components)


def test_consolidated_total_prevents_double_counting_class_values(snapshot_factory):
    result = normalize_share_count(
        snapshot_factory(
            ticker="GOOG",
            ticker_shares_outstanding=5.527e9,
            implied_shares_outstanding=5.867e9,
            quarterly_balance=statement(12.229934831e9, "2026-06-30"),
        )
    )

    assert result.shares_outstanding == 12.229934831e9
    assert result.shares_outstanding != pytest.approx(
        12.229934831e9 + 5.527e9 + 5.867e9
    )


def test_annual_consolidated_total_is_used_when_quarterly_is_missing(
    snapshot_factory,
):
    result = normalize_share_count(
        snapshot_factory(
            ticker="GOOG",
            annual_balance=statement(12.088e9, "2025-12-31"),
        )
    )

    assert result.available is True
    assert result.shares_outstanding == 12.088e9
    assert result.method == "issuer_reported_consolidated_annual"


def test_unknown_structure_without_issuer_total_is_unavailable(snapshot_factory):
    result = normalize_share_count(
        snapshot_factory(
            ticker="UNKNOWN",
            shares_outstanding=100.0,
            ticker_shares_outstanding=100.0,
        )
    )

    assert result.available is False
    assert result.scope == "unknown"
    assert "share_structure_unknown" in result.warnings


def test_latest_missing_statement_value_does_not_use_older_period(snapshot_factory):
    balance = pd.DataFrame(
        [[12.0e9, float("nan")]],
        index=["Ordinary Shares Number"],
        columns=pd.to_datetime(["2025-12-31", "2026-06-30"]),
    )

    result = normalize_share_count(
        snapshot_factory(ticker="GOOGL", quarterly_balance=balance)
    )

    assert result.available is False


def test_units_remain_raw_shares_without_hidden_conversion(snapshot_factory):
    raw = 12_229_934_831.0
    result = normalize_share_count(
        snapshot_factory(
            ticker="GOOGL", quarterly_balance=statement(raw, "2026-06-30")
        )
    )

    assert result.shares_outstanding == raw
    assert result.shares_outstanding / 1e9 == pytest.approx(12.229934831)


def test_normalized_result_is_immutable(snapshot_factory):
    result = normalize_share_count(
        snapshot_factory(ticker="NVDA", ticker_shares_outstanding=10.0)
    )
    assert isinstance(result, NormalizedShareCount)

    with pytest.raises(FrozenInstanceError):
        result.shares_outstanding = 0.0
