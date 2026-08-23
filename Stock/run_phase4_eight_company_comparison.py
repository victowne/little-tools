"""Live read-only eight-company unified-production comparison."""

import json
from pathlib import Path

import yfinance as yf

from Stock.alphabet_research import build_alphabet_research_profile
from Stock.amazon_research import build_amazon_research_profile, run_amazon_candidate_preview
from Stock.company_profile_comparison import build_company_profile_comparison_row
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.hyperscaler_research import build_meta_research_profile, build_microsoft_research_profile
from Stock.multistage_integration import run_real_company_multistage_dcf
from Stock.nvda_research import build_nvda_research_profile
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
        research = build_amazon_research_profile(current, history)
    else:
        anchors = build_company_revenue_forecast_anchors(ticker, snapshot, history)
        research = BUILDERS[ticker](
            current, history, revenue_anchors=anchors, retrieved_at="2026-08-23"
        )
    profile = research.lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    run = run_real_company_multistage_dcf(snapshot, history, assumptions)
    if ticker == "AMZN":
        run = run_amazon_candidate_preview(run.inputs, profile)
    row = build_company_profile_comparison_row(
        profile, run, market_price=snapshot.price
    )
    return {
        "ticker": ticker,
        "y1": row.year1_growth,
        "y2": row.year2_growth,
        "y3": row.year3_growth,
        "mature_margin": row.mature_operating_margin,
        "mature_sc": row.mature_sales_to_capital,
        "wacc": row.wacc,
        "terminal_g": row.terminal_growth,
        "terminal_roic": row.terminal_roic,
        "dcf_per_share": row.intrinsic_value_per_share,
        "market_price": row.market_price,
        "dcf_to_price": row.dcf_to_price,
        "tv_ev": row.terminal_value_share,
        "research_status": row.research_status,
        "model_risk": row.model_risk,
    }


def run_live_comparison() -> list[dict]:
    return [
        _record(ticker)
        for ticker in ("NVDA", "GOOGL", "META", "MSFT", "AMZN", "MU", "AAPL", "AVGO")
    ]


if __name__ == "__main__":
    print(json.dumps(run_live_comparison(), indent=2, default=str))
