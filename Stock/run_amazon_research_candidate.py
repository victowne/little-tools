"""Live, read-only orchestration for the Phase 3F.4 Amazon Candidate."""

import json
from pathlib import Path

import yfinance as yf

_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))

from Stock.amazon_research import build_amazon_research_profile, run_amazon_candidate_preview
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import extract_real_company_dcf_inputs
from Stock.stock_valuation_mvp import build_company_fundamentals, load_company_snapshot
from Stock.valuation import MultiStageDCFAssumptions


def _current_placeholder() -> MultiStageDCFAssumptions:
    return MultiStageDCFAssumptions(
        forecast_years=11, near_term_revenue_growth=(.15, .14, .12),
        revenue_fade_years=8, terminal_growth=.03,
        starting_operating_margin=.10, mature_operating_margin=.18,
        starting_sales_to_capital=.57, mature_sales_to_capital=.824,
        operating_tax_rate=.21, wacc=.105,
    )


def run_live_amazon_research_candidate():
    snapshot = load_company_snapshot("AMZN")
    history = build_company_fundamentals(snapshot)
    inputs = extract_real_company_dcf_inputs(snapshot, history)
    result = build_amazon_research_profile(
        _current_placeholder(), history, retrieved_at="2026-08-23"
    )
    profile = result.lookup.profile
    candidate = build_multistage_assumptions_from_profile(profile).assumptions
    preview = run_amazon_candidate_preview(inputs, profile)
    run = preview
    output = {
        "status": profile.profile_status,
        "reviewed": False,
        "applied": False,
        "starting_revenue_b": run.inputs.starting_revenue / 1e9,
        "starting_revenue_source": run.inputs.starting_revenue_source,
        "starting_revenue_periods": [str(item.date()) for item in run.inputs.starting_revenue_periods],
        "starting_operating_margin": candidate.starting_operating_margin,
        "growth_path": [row.revenue_growth for row in run.forecast_path.years],
        "mature_margin": candidate.mature_operating_margin,
        "mature_sales_to_capital": candidate.mature_sales_to_capital,
        "terminal_roic": candidate.derived_terminal_roic,
        "terminal_reinvestment_rate": candidate.terminal_reinvestment_rate,
        "terminal_fcff_conversion": 1 - candidate.terminal_reinvestment_rate,
        "strategy": "STANDARD_SALES_TO_CAPITAL",
        "dcf_per_share": run.per_share_value.intrinsic_value_per_share if run.per_share_value else None,
        "enterprise_value_b": run.enterprise_value.enterprise_value / 1e9,
        "equity_value_b": run.equity_value.equity_value / 1e9,
        "explicit_fcff_pv_b": run.enterprise_value.explicit_forecast_pv / 1e9,
        "terminal_pv_b": run.enterprise_value.terminal_value_pv / 1e9,
        "terminal_value_share": run.enterprise_value.terminal_value_share,
        "cumulative_explicit_reinvestment_b": sum(row.reinvestment for row in run.operating_forecast.years) / 1e9,
        "cumulative_explicit_fcff_b": sum(row.fcff for row in run.operating_forecast.years) / 1e9,
        "market_price_context": snapshot.price,
        "candidate_uses_market_price": False,
    }
    return output


if __name__ == "__main__":
    print(json.dumps(run_live_amazon_research_candidate(), indent=2, default=str))
