"""Auditable current common-share normalization for consolidated valuation.

The resolver never adds ticker-level share classes. Issuer-reported statement
totals are treated as consolidated; ticker metadata is accepted only for an
issuer explicitly known to have a single participating common-share class.
"""

from dataclasses import dataclass
import math
from typing import Literal, Protocol

import pandas as pd


ShareScope = Literal["consolidated_common", "single_class", "unknown"]


@dataclass(frozen=True)
class ShareCountComponent:
    """One raw share-count observation retained for auditability."""

    source: str
    value: float | None
    period: pd.Timestamp | None = None
    scope: ShareScope = "unknown"


@dataclass(frozen=True)
class NormalizedShareCount:
    """Current shares aligned, when available, to consolidated Equity Value."""

    ticker: str
    shares_outstanding: float | None
    source: str | None
    source_period: pd.Timestamp | None
    scope: ShareScope
    method: str
    components: tuple[ShareCountComponent, ...]
    warnings: tuple[str, ...]
    available: bool
    reason: str | None


@dataclass(frozen=True)
class IssuerShareProfile:
    issuer: str
    share_structure: Literal["single_class", "multi_class"]
    related_tickers: tuple[str, ...]


# This is risk/scope metadata, not a hard-coded share count. It makes every
# company-specific judgment inspectable and allows later filing-backed profiles.
ISSUER_SHARE_PROFILES: dict[str, IssuerShareProfile] = {
    "NVDA": IssuerShareProfile("NVIDIA Corporation", "single_class", ("NVDA",)),
    "GOOG": IssuerShareProfile("Alphabet Inc.", "multi_class", ("GOOG", "GOOGL")),
    "GOOGL": IssuerShareProfile("Alphabet Inc.", "multi_class", ("GOOG", "GOOGL")),
}


class ShareSnapshotLike(Protocol):
    ticker: str
    shares_outstanding: float | None
    ticker_shares_outstanding: float | None
    implied_shares_outstanding: float | None
    fast_info_shares: float | None
    annual_balance: pd.DataFrame
    quarterly_balance: pd.DataFrame


def _finite_optional(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _normalized_label(value) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _latest_ordinary_shares(
    statement: pd.DataFrame,
) -> tuple[float | None, pd.Timestamp | None, str | None]:
    """Resolve only exact normalized Ordinary Shares Number at latest period."""
    if statement is None or statement.empty:
        return None, None, None
    matches = [
        row for row in statement.index
        if _normalized_label(row) == "ordinarysharesnumber"
    ]
    if len(matches) != 1:
        return None, None, "ambiguous_ordinary_shares_row" if matches else None
    row = statement.loc[matches[0]]
    if not isinstance(row, pd.Series):
        return None, None, "ambiguous_ordinary_shares_row"
    frame = pd.DataFrame(
        {
            "period": pd.to_datetime(pd.Index(row.index), errors="coerce"),
            "value": pd.to_numeric(pd.Series(row).reset_index(drop=True), errors="coerce"),
        }
    ).dropna(subset=["period"])
    if frame.empty:
        return None, None, None
    latest = frame.sort_values("period").iloc[-1]
    return (
        _finite_optional(latest["value"]),
        pd.Timestamp(latest["period"]),
        None,
    )


def normalize_share_count(
    snapshot: ShareSnapshotLike,
    additional_class_counts: tuple[ShareCountComponent, ...] = (),
) -> NormalizedShareCount:
    """Normalize current shares to consolidated common-equity ownership scope.

    Hierarchy: latest quarterly issuer-reported Ordinary Shares Number, latest
    annual issuer-reported Ordinary Shares Number, then ticker metadata only for
    an explicitly profiled single-class issuer. Multi-class ticker counts are
    retained as diagnostics but are never summed or substituted.
    """
    ticker = str(snapshot.ticker).strip().upper()
    profile = ISSUER_SHARE_PROFILES.get(ticker)
    ticker_raw = _finite_optional(
        getattr(snapshot, "ticker_shares_outstanding", None)
    )
    implied_raw = _finite_optional(
        getattr(snapshot, "implied_shares_outstanding", None)
    )
    fast_raw = _finite_optional(getattr(snapshot, "fast_info_shares", None))
    legacy_raw = _finite_optional(getattr(snapshot, "shares_outstanding", None))
    quarterly, quarterly_period, quarterly_warning = _latest_ordinary_shares(
        snapshot.quarterly_balance
    )
    annual, annual_period, annual_warning = _latest_ordinary_shares(
        snapshot.annual_balance
    )
    if not isinstance(additional_class_counts, tuple) or not all(
        isinstance(component, ShareCountComponent)
        for component in additional_class_counts
    ):
        raise TypeError("additional_class_counts must be ShareCountComponent tuple")
    if profile and profile.share_structure == "multi_class":
        ticker_scope: ShareScope = "single_class"
    elif profile and profile.share_structure == "single_class":
        ticker_scope = "consolidated_common"
    else:
        ticker_scope = "unknown"
    components = (
        ShareCountComponent(
            "quarterly_balance.OrdinarySharesNumber", quarterly,
            quarterly_period, "consolidated_common",
        ),
        ShareCountComponent(
            "annual_balance.OrdinarySharesNumber", annual,
            annual_period, "consolidated_common",
        ),
        ShareCountComponent(
            "info.sharesOutstanding", ticker_raw, None,
            ticker_scope,
        ),
        ShareCountComponent("info.impliedSharesOutstanding", implied_raw),
        ShareCountComponent("fast_info.shares", fast_raw),
    ) + additional_class_counts
    warnings = [warning for warning in (quarterly_warning, annual_warning) if warning]
    if profile and profile.share_structure == "multi_class":
        warnings.append("multi_class_issuer")

    if quarterly is not None:
        return NormalizedShareCount(
            ticker, quarterly, "quarterly_balance.OrdinarySharesNumber",
            quarterly_period, "consolidated_common",
            "issuer_reported_consolidated_quarterly", components,
            tuple(dict.fromkeys(warnings)), True, None,
        )
    if annual is not None:
        return NormalizedShareCount(
            ticker, annual, "annual_balance.OrdinarySharesNumber",
            annual_period, "consolidated_common",
            "issuer_reported_consolidated_annual", components,
            tuple(dict.fromkeys(warnings)), True, None,
        )
    if profile and profile.share_structure == "single_class":
        accepted = ticker_raw or implied_raw or fast_raw or legacy_raw
        if accepted is not None:
            source = (
                "info.sharesOutstanding" if ticker_raw is not None
                else "info.impliedSharesOutstanding" if implied_raw is not None
                else "fast_info.shares" if fast_raw is not None
                else "CompanySnapshot.shares_outstanding"
            )
            return NormalizedShareCount(
                ticker, accepted, source, None, "consolidated_common",
                "confirmed_single_class_current_metadata", components,
                tuple(dict.fromkeys(warnings)), True, None,
            )

    if profile is None:
        warnings.append("share_structure_unknown")
    return NormalizedShareCount(
        ticker, None, None, None, "unknown",
        "conservative_unavailable", components,
        tuple(dict.fromkeys(warnings)), False,
        "consolidated_share_count_unavailable",
    )
