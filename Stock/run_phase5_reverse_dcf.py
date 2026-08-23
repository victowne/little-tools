"""Live read-only Reverse DCF diagnostics for all eight target companies."""

import json
from pathlib import Path

import yfinance as yf

from Stock.alphabet_research import build_alphabet_research_profile
from Stock.amazon_research import build_amazon_research_profile, run_amazon_candidate_preview
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.hyperscaler_research import build_meta_research_profile, build_microsoft_research_profile
from Stock.multistage_integration import run_real_company_multistage_dcf
from Stock.nvda_research import build_nvda_research_profile
from Stock.reverse_dcf import research_ranges_from_profile, run_reverse_dcf
from Stock.stock_valuation_mvp import (
    build_company_fundamentals,
    build_company_revenue_forecast_anchors,
    build_multistage_assumptions_from_ui,
    load_company_snapshot,
    multistage_initial_defaults,
)
from Stock.unified_company_research import (
    build_apple_research_profile,
    build_broadcom_research_profile,
    build_micron_research_profile,
)


_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))

BUILDERS = {
    "NVDA": build_nvda_research_profile,
    "GOOGL": build_alphabet_research_profile,
    "META": build_meta_research_profile,
    "MSFT": build_microsoft_research_profile,
    "MU": build_micron_research_profile,
    "AAPL": build_apple_research_profile,
    "AVGO": build_broadcom_research_profile,
}


def _record(ticker: str) -> dict:
    snapshot = load_company_snapshot(ticker)
    history = build_company_fundamentals(snapshot)
    current = build_multistage_assumptions_from_ui(
        multistage_initial_defaults(ticker, history)
    )
    if ticker == "AMZN":
        research = build_amazon_research_profile(
            current, history, retrieved_at="2026-08-23"
        )
    else:
        anchors = build_company_revenue_forecast_anchors(ticker, snapshot, history)
        research = BUILDERS[ticker](
            current,
            history,
            revenue_anchors=anchors,
            retrieved_at="2026-08-23",
        )
    profile = research.lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    run = run_real_company_multistage_dcf(snapshot, history, assumptions)
    if ticker == "AMZN":
        # Standard production S/C candidate with SEC revenue-base override; no
        # Hybrid dispatch is used by either Base or Reverse DCF.
        run = run_amazon_candidate_preview(run.inputs, profile)
    analysis = run_reverse_dcf(
        run.inputs,
        run.assumptions,
        snapshot.price,
        ticker=ticker,
        base_source="Research Candidate",
        research_ranges=research_ranges_from_profile(profile),
    )
    dimensions = {}
    for result in analysis.results:
        dimensions[result.variable] = {
            "status": result.status,
            "research_value": result.research_value,
            "implied_value": result.implied_value,
            "implied_growth_path": result.implied_growth_path,
            "range_relation": result.range_relation,
            "expectation_gap": result.expectation_gap,
            "solved_dcf": result.implied_dcf_value,
            "tv_ev": result.terminal_value_share,
            "reason": result.reason,
        }
    return {
        "ticker": ticker,
        "production_model": "STANDARD_SALES_TO_CAPITAL",
        "base_source": analysis.base_source,
        "base_dcf_per_share": analysis.base_dcf_per_share,
        "market_price": analysis.market_price,
        "price_to_base_dcf": analysis.price_to_base_dcf,
        "dimensions": dimensions,
        "warnings": analysis.warnings,
        "model_risk": profile.model_risk,
        "research_status": profile.profile_status,
    }


def run_live_reverse_dcf() -> list[dict]:
    return [
        _record(ticker)
        for ticker in ("NVDA", "GOOGL", "META", "MSFT", "AMZN", "MU", "AAPL", "AVGO")
    ]


if __name__ == "__main__":
    print(json.dumps(run_live_reverse_dcf(), indent=2, default=str))
