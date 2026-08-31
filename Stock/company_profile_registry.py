"""Canonical registry and build entry point for researched company profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from Stock.alphabet_research import build_alphabet_research_profile
from Stock.amazon_research import build_amazon_research_profile
from Stock.beta_audit import BetaRobustnessAudit
from Stock.bottom_up_beta import BottomUpBetaResult
from Stock.company_research_types import CompanyResearchResult
from Stock.forecast_anchors import RevenueForecastAnchors
from Stock.fundamentals import FundamentalHistory
from Stock.hyperscaler_research import (
    build_meta_research_profile,
    build_microsoft_research_profile,
)
from Stock.nvda_research import build_nvda_research_profile
from Stock.unified_company_research import (
    build_amd_research_profile,
    build_apple_research_profile,
    build_broadcom_research_profile,
    build_micron_research_profile,
)
from Stock.valuation import MultiStageDCFAssumptions
from Stock.wacc_audit import WACCAuditResult


@dataclass(frozen=True)
class CompanyProfileBuildContext:
    """Inputs shared by all researched Company Profile builders."""

    current_assumptions: MultiStageDCFAssumptions
    history: FundamentalHistory
    retrieved_at: str
    revenue_anchors: RevenueForecastAnchors | None = None
    wacc_audit: WACCAuditResult | None = None
    beta_audit: BetaRobustnessAudit | None = None
    bottom_up_beta: BottomUpBetaResult | None = None


ProfileBuilder = Callable[[CompanyProfileBuildContext], CompanyResearchResult]


@dataclass(frozen=True)
class CompanyResearchProfileRegistration:
    """Canonical issuer identity and adapter for one researched profile."""

    canonical_ticker: str
    issuer_id: str
    aliases: tuple[str, ...]
    builder: ProfileBuilder


def _build_nvda(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_nvda_research_profile(
        context.current_assumptions,
        context.history,
        revenue_anchors=context.revenue_anchors,
        wacc_audit=context.wacc_audit,
        beta_audit=context.beta_audit,
        bottom_up_beta=context.bottom_up_beta,
        retrieved_at=context.retrieved_at,
    )


def _build_alphabet(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_alphabet_research_profile(
        context.current_assumptions,
        context.history,
        revenue_anchors=context.revenue_anchors,
        wacc_audit=context.wacc_audit,
        beta_audit=context.beta_audit,
        bottom_up_beta=context.bottom_up_beta,
        retrieved_at=context.retrieved_at,
    )


def _build_microsoft(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_microsoft_research_profile(
        context.current_assumptions,
        context.history,
        revenue_anchors=context.revenue_anchors,
        wacc_audit=context.wacc_audit,
        retrieved_at=context.retrieved_at,
    )


def _build_meta(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_meta_research_profile(
        context.current_assumptions,
        context.history,
        revenue_anchors=context.revenue_anchors,
        wacc_audit=context.wacc_audit,
        retrieved_at=context.retrieved_at,
    )


def _build_amazon(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_amazon_research_profile(
        context.current_assumptions,
        context.history,
        wacc_audit=context.wacc_audit,
        retrieved_at=context.retrieved_at,
    )


def _build_micron(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_micron_research_profile(
        context.current_assumptions,
        context.history,
        revenue_anchors=context.revenue_anchors,
        wacc_audit=context.wacc_audit,
        retrieved_at=context.retrieved_at,
    )


def _build_apple(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_apple_research_profile(
        context.current_assumptions,
        context.history,
        revenue_anchors=context.revenue_anchors,
        wacc_audit=context.wacc_audit,
        retrieved_at=context.retrieved_at,
    )


def _build_broadcom(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_broadcom_research_profile(
        context.current_assumptions,
        context.history,
        revenue_anchors=context.revenue_anchors,
        wacc_audit=context.wacc_audit,
        retrieved_at=context.retrieved_at,
    )


def _build_amd(context: CompanyProfileBuildContext) -> CompanyResearchResult:
    return build_amd_research_profile(
        context.current_assumptions,
        context.history,
        revenue_anchors=context.revenue_anchors,
        wacc_audit=context.wacc_audit,
        retrieved_at=context.retrieved_at,
    )


COMPANY_RESEARCH_PROFILE_REGISTRATIONS = (
    CompanyResearchProfileRegistration("NVDA", "NVDA", (), _build_nvda),
    CompanyResearchProfileRegistration(
        "GOOGL", "ALPHABET_INC", ("GOOG",), _build_alphabet
    ),
    CompanyResearchProfileRegistration("MSFT", "MSFT", (), _build_microsoft),
    CompanyResearchProfileRegistration("META", "META", (), _build_meta),
    CompanyResearchProfileRegistration("AMZN", "AMZN", (), _build_amazon),
    CompanyResearchProfileRegistration("MU", "MU", (), _build_micron),
    CompanyResearchProfileRegistration("AAPL", "AAPL", (), _build_apple),
    CompanyResearchProfileRegistration("AVGO", "AVGO", (), _build_broadcom),
    CompanyResearchProfileRegistration("AMD", "AMD", (), _build_amd),
)

_REGISTRATION_BY_TICKER = {
    ticker: registration
    for registration in COMPANY_RESEARCH_PROFILE_REGISTRATIONS
    for ticker in (registration.canonical_ticker, *registration.aliases)
}

PRODUCTION_RESEARCH_TICKERS = tuple(
    registration.canonical_ticker
    for registration in COMPANY_RESEARCH_PROFILE_REGISTRATIONS
)


def get_company_research_registration(
    ticker: str,
) -> CompanyResearchProfileRegistration | None:
    """Return the canonical researched-profile registration for a ticker."""

    return _REGISTRATION_BY_TICKER.get(ticker.strip().upper())


def normalize_research_ticker(ticker: str) -> str:
    """Normalize a supported ticker alias to its canonical research ticker.

    Raises:
        ValueError: If no researched Company Profile is registered.
    """

    registration = get_company_research_registration(ticker)
    if registration is None:
        raise ValueError(f"unsupported_company_research_profile:{ticker}")
    return registration.canonical_ticker


def build_company_research_profile(
    ticker: str,
    context: CompanyProfileBuildContext,
) -> CompanyResearchResult:
    """Build one candidate through the canonical profile registry.

    Raises:
        ValueError: If no researched Company Profile is registered.
    """

    registration = get_company_research_registration(ticker)
    if registration is None:
        raise ValueError(f"unsupported_company_research_profile:{ticker}")
    return registration.builder(context)
