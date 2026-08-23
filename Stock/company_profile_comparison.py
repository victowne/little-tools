"""Pure compact comparison rows for unified-production company profiles."""

from dataclasses import dataclass
import math

from Stock.company_profiles import CompanyResearchProfile
from Stock.multistage_integration import MultiStageDCFRunResult


@dataclass(frozen=True)
class CompanyProfileComparisonRow:
    ticker: str
    year1_growth: float
    year2_growth: float
    year3_growth: float
    mature_operating_margin: float
    mature_sales_to_capital: float
    wacc: float
    terminal_growth: float
    terminal_roic: float
    intrinsic_value_per_share: float
    market_price: float | None
    dcf_to_price: float | None
    terminal_value_share: float
    research_status: str
    model_risk: str | None


def _required_value(item, name: str) -> float:
    if item is None or item.value is None:
        raise ValueError(f"{name}_unavailable")
    value = float(item.value)
    if not math.isfinite(value):
        raise ValueError(f"{name}_unavailable")
    return value


def build_company_profile_comparison_row(
    profile: CompanyResearchProfile,
    run: MultiStageDCFRunResult,
    *,
    market_price: float | None,
) -> CompanyProfileComparisonRow:
    """Build one diagnostic row without allowing price into assumptions."""
    revenue = profile.revenue_framework
    margin = profile.margin_framework
    capital = profile.capital_efficiency_framework
    wacc = profile.wacc_framework
    terminal = profile.terminal_framework
    price = None
    if market_price is not None:
        candidate = float(market_price)
        if math.isfinite(candidate) and candidate > 0:
            price = candidate
    intrinsic = run.per_share_value.intrinsic_value_per_share
    terminal_share = run.enterprise_value.terminal_value_share
    if terminal_share is None:
        raise ValueError("terminal_value_share_unavailable")
    return CompanyProfileComparisonRow(
        ticker=profile.ticker,
        year1_growth=_required_value(revenue.year1_growth, "year1_growth"),
        year2_growth=_required_value(revenue.year2_growth, "year2_growth"),
        year3_growth=_required_value(revenue.year3_growth, "year3_growth"),
        mature_operating_margin=_required_value(
            margin.mature_operating_margin, "mature_operating_margin"
        ),
        mature_sales_to_capital=_required_value(
            capital.mature_sales_to_capital, "mature_sales_to_capital"
        ),
        wacc=_required_value(wacc.research_wacc, "research_wacc"),
        terminal_growth=_required_value(terminal.terminal_growth, "terminal_growth"),
        terminal_roic=float(terminal.terminal_roic),
        intrinsic_value_per_share=intrinsic,
        market_price=price,
        dcf_to_price=(intrinsic / price if price is not None else None),
        terminal_value_share=terminal_share,
        research_status=profile.profile_status,
        model_risk=profile.model_risk,
    )


def build_company_profile_comparison(
    profiles_and_runs: tuple[
        tuple[CompanyResearchProfile, MultiStageDCFRunResult, float | None], ...
    ],
) -> tuple[CompanyProfileComparisonRow, ...]:
    return tuple(
        build_company_profile_comparison_row(
            profile, run, market_price=market_price
        )
        for profile, run, market_price in profiles_and_runs
    )
