"""Live, read-only validation for the four Phase 4 unified profiles."""

import json
from pathlib import Path

import yfinance as yf

from Stock.amazon_research import build_amazon_research_profile, run_amazon_candidate_preview
from Stock.company_profile_comparison import build_company_profile_comparison_row
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_real_company_multistage_dcf
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
    anchors = build_company_revenue_forecast_anchors(ticker, snapshot, history)
    if ticker == "AMZN":
        research = build_amazon_research_profile(
            current, history, retrieved_at="2026-08-23"
        )
    else:
        research = BUILDERS[ticker](
            current, history, revenue_anchors=anchors,
            retrieved_at="2026-08-23",
        )
    profile = research.lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    if ticker == "AMZN":
        base_inputs_run = run_real_company_multistage_dcf(snapshot, history, assumptions)
        run = run_amazon_candidate_preview(base_inputs_run.inputs, profile)
    else:
        run = run_real_company_multistage_dcf(snapshot, history, assumptions)
    row = build_company_profile_comparison_row(
        profile, run, market_price=snapshot.price
    )
    years = run.operating_forecast.years
    explicit_fcff = sum(item.fcff for item in years)
    early_explicit_fcff = sum(item.fcff for item in years[:5])
    warnings = list(run.enterprise_value.warnings)
    diagnostics = []
    if early_explicit_fcff < 0:
        diagnostics.append("cumulative_y1_y5_fcff_negative")
    if run.enterprise_value.explicit_forecast_pv < 0:
        diagnostics.append("explicit_forecast_pv_negative")
    if run.enterprise_value.terminal_value_share is not None and run.enterprise_value.terminal_value_share > 1:
        diagnostics.append("terminal_value_share_above_100_percent")
    return {
        "ticker": ticker,
        "production_model": "STANDARD_SALES_TO_CAPITAL",
        "starting_revenue_b": run.inputs.starting_revenue / 1e9,
        "starting_revenue_source": run.inputs.starting_revenue_source,
        "growth": assumptions.near_term_revenue_growth,
        "implied_y4": years[3].revenue_growth,
        "implied_y5": years[4].revenue_growth,
        "starting_margin": assumptions.starting_operating_margin,
        "mature_margin": assumptions.mature_operating_margin,
        "starting_sc": assumptions.starting_sales_to_capital,
        "mature_sc": assumptions.mature_sales_to_capital,
        "tax": assumptions.operating_tax_rate,
        "wacc": assumptions.wacc,
        "terminal_growth": assumptions.terminal_growth,
        "terminal_roic": assumptions.derived_terminal_roic,
        "dcf_per_share": row.intrinsic_value_per_share,
        "market_price": row.market_price,
        "dcf_to_price": row.dcf_to_price,
        "enterprise_value_b": run.enterprise_value.enterprise_value / 1e9,
        "equity_value_b": run.equity_value.equity_value / 1e9,
        "explicit_fcff_pv_b": run.enterprise_value.explicit_forecast_pv / 1e9,
        "cumulative_explicit_fcff_b": explicit_fcff / 1e9,
        "cumulative_y1_y5_fcff_b": early_explicit_fcff / 1e9,
        "terminal_pv_b": run.enterprise_value.terminal_value_pv / 1e9,
        "tv_ev": row.terminal_value_share,
        "status": row.research_status,
        "model_risk": row.model_risk,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "reviewed": False,
        "applied": False,
    }


def run_live_validation() -> list[dict]:
    return [_record(ticker) for ticker in ("AMZN", "MU", "AAPL", "AVGO")]


if __name__ == "__main__":
    print(json.dumps(run_live_validation(), indent=2, default=str))
