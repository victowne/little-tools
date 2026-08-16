"""Pure support boundary for Phase 2 per-security valuation units.

Phase 2 does not perform foreign-exchange or ADR/share-unit conversion.  The
operating DCF may still be calculated in statement currency, but a numeric
per-security result is exposed only when the currency and security unit bridge
is directly reconcilable.
"""

from dataclasses import dataclass


FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED = (
    "foreign_listing_currency_or_unit_normalization_unsupported"
)
CURRENCY_METADATA_UNAVAILABLE = "valuation_currency_metadata_unavailable"


@dataclass(frozen=True)
class ListingUnitRequirement:
    """Known listing relationship that requires an unsupported conversion."""

    issuer_share_unit: str
    displayed_security_unit: str
    conversion_required: bool


@dataclass(frozen=True)
class PerSecurityValuationSupport:
    """Immutable result of the Phase 2 currency/security-unit support check."""

    supported: bool
    reason: str | None
    statement_currency: str | None
    security_currency: str | None
    requires_currency_conversion: bool
    requires_security_unit_conversion: bool


# This is listing metadata, not a UI ticker exception.  It documents a known
# issuer-share/security-unit relationship that Phase 2 intentionally does not
# normalize.  More listings may be added only when their units are verified.
LISTING_UNIT_REQUIREMENTS: dict[str, ListingUnitRequirement] = {
    "TSM": ListingUnitRequirement(
        issuer_share_unit="Taiwan-listed ordinary share",
        displayed_security_unit="NYSE ADR",
        conversion_required=True,
    ),
}


def _currency(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return normalized or None


def assess_per_security_valuation_support(
    *,
    ticker: str,
    statement_currency: str | None,
    security_currency: str | None,
) -> PerSecurityValuationSupport:
    """Fail closed unless the current per-security unit bridge is explicit.

    Equal, known currencies and no known ADR/share-unit conversion requirement
    form the supported Phase 2 path.  Missing currency metadata is not guessed.
    """
    normalized_ticker = str(ticker).strip().upper()
    statement = _currency(statement_currency)
    security = _currency(security_currency)
    requirement = LISTING_UNIT_REQUIREMENTS.get(normalized_ticker)
    unit_conversion = bool(
        requirement is not None and requirement.conversion_required
    )
    currency_conversion = bool(
        statement is not None
        and security is not None
        and statement != security
    )

    if currency_conversion or unit_conversion:
        return PerSecurityValuationSupport(
            supported=False,
            reason=FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED,
            statement_currency=statement,
            security_currency=security,
            requires_currency_conversion=currency_conversion,
            requires_security_unit_conversion=unit_conversion,
        )
    if statement is None or security is None:
        return PerSecurityValuationSupport(
            supported=False,
            reason=CURRENCY_METADATA_UNAVAILABLE,
            statement_currency=statement,
            security_currency=security,
            requires_currency_conversion=False,
            requires_security_unit_conversion=False,
        )
    return PerSecurityValuationSupport(
        supported=True,
        reason=None,
        statement_currency=statement,
        security_currency=security,
        requires_currency_conversion=False,
        requires_security_unit_conversion=False,
    )
