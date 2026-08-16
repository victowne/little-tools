from dataclasses import FrozenInstanceError

import pytest

from Stock.valuation_support import (
    CURRENCY_METADATA_UNAVAILABLE,
    FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED,
    assess_per_security_valuation_support,
)


def test_matching_known_currencies_are_supported():
    result = assess_per_security_valuation_support(
        ticker="TEST", statement_currency="usd", security_currency="USD"
    )

    assert result.supported
    assert result.reason is None
    assert result.statement_currency == "USD"
    assert result.security_currency == "USD"


def test_foreign_currency_listing_fails_closed():
    result = assess_per_security_valuation_support(
        ticker="FOREIGN", statement_currency="TWD", security_currency="USD"
    )

    assert not result.supported
    assert result.reason == FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED
    assert result.requires_currency_conversion


def test_known_adr_unit_requirement_fails_even_if_currency_matches():
    result = assess_per_security_valuation_support(
        ticker="TSM", statement_currency="USD", security_currency="USD"
    )

    assert not result.supported
    assert result.reason == FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED
    assert result.requires_security_unit_conversion


def test_missing_currency_metadata_is_not_guessed():
    result = assess_per_security_valuation_support(
        ticker="TEST", statement_currency=None, security_currency="USD"
    )

    assert not result.supported
    assert result.reason == CURRENCY_METADATA_UNAVAILABLE


def test_support_result_is_immutable():
    result = assess_per_security_valuation_support(
        ticker="TEST", statement_currency="USD", security_currency="USD"
    )

    with pytest.raises(FrozenInstanceError):
        result.supported = False
