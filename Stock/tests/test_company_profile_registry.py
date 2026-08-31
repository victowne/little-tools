import pytest

from Stock.company_profile_registry import (
    PRODUCTION_RESEARCH_TICKERS,
    CompanyProfileBuildContext,
    build_company_research_profile,
    get_company_research_registration,
    normalize_research_ticker,
)
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.tests.test_alphabet_research import current_assumptions, history


EXPECTED_TICKERS = (
    "NVDA",
    "GOOGL",
    "MSFT",
    "META",
    "AMZN",
    "MU",
    "AAPL",
    "AVGO",
    "AMD",
)


def _context() -> CompanyProfileBuildContext:
    return CompanyProfileBuildContext(
        current_assumptions=current_assumptions(),
        history=history(),
        retrieved_at="2026-08-31",
    )


def test_registry_contains_each_production_company_once() -> None:
    assert PRODUCTION_RESEARCH_TICKERS == EXPECTED_TICKERS
    assert len(set(PRODUCTION_RESEARCH_TICKERS)) == len(EXPECTED_TICKERS)


def test_alphabet_share_classes_use_one_issuer_registration() -> None:
    goog = get_company_research_registration("GOOG")
    googl = get_company_research_registration("googl")

    assert goog is googl
    assert normalize_research_ticker("GOOG") == "GOOGL"
    assert googl is not None
    assert googl.issuer_id == "ALPHABET_INC"


@pytest.mark.parametrize("ticker", EXPECTED_TICKERS)
def test_every_registered_builder_returns_the_common_profile_contract(
    ticker: str,
) -> None:
    current = current_assumptions()
    result = build_company_research_profile(
        ticker,
        CompanyProfileBuildContext(
            current_assumptions=current,
            history=history(),
            retrieved_at="2026-08-31",
        ),
    )
    profile = result.lookup.profile

    assert result.lookup.available
    assert profile is not None
    assert profile.profile_status == "research_in_progress"
    assert profile.last_reviewed_at is None
    assert result.current_assumptions == current
    assert isinstance(result.revenue_evidence, tuple)
    assert isinstance(result.growth_ranges, tuple)
    assert isinstance(result.period_reconciliation, tuple)
    assert isinstance(result.warnings, tuple)
    assert build_multistage_assumptions_from_profile(profile).available


def test_unregistered_ticker_is_explicitly_rejected() -> None:
    assert get_company_research_registration("UNKNOWN") is None

    with pytest.raises(
        ValueError,
        match="unsupported_company_research_profile",
    ):
        build_company_research_profile("UNKNOWN", _context())
