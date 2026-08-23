"""Live read-only validation for the three Phase 3E research candidates."""

import json
from pathlib import Path

import yfinance as yf

from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.hyperscaler_research import build_meta_research_profile, build_microsoft_research_profile
from Stock.alphabet_research import build_alphabet_research_profile
from Stock.multistage_integration import run_real_company_multistage_dcf
from Stock.stock_valuation_mvp import (
    build_company_fundamentals,
    build_company_revenue_forecast_anchors,
    build_multistage_assumptions_from_ui,
    load_company_snapshot,
    multistage_initial_defaults,
)


_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))


def _record(ticker, builder):
    snapshot = load_company_snapshot(ticker)
    history = build_company_fundamentals(snapshot)
    current = build_multistage_assumptions_from_ui(
        multistage_initial_defaults(ticker, history)
    )
    anchors = build_company_revenue_forecast_anchors(ticker, snapshot, history)
    result = builder(
        current, history, revenue_anchors=anchors, retrieved_at="2026-08-23"
    )
    profile = result.lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    run = run_real_company_multistage_dcf(snapshot, history, assumptions)
    years = run.operating_forecast.years
    per_share = run.per_share_value.intrinsic_value_per_share if run.per_share_value else None
    sc_points = []
    for delta in (-0.10, -0.05, 0.0, 0.05, 0.10):
        from dataclasses import replace
        point_run = run_real_company_multistage_dcf(
            snapshot, history,
            replace(assumptions, mature_sales_to_capital=assumptions.mature_sales_to_capital + delta),
        )
        sc_points.append({
            "mature_sc": assumptions.mature_sales_to_capital + delta,
            "terminal_roic": point_run.assumptions.derived_terminal_roic,
            "value_per_share": point_run.per_share_value.intrinsic_value_per_share if point_run.per_share_value else None,
            "tv_ev": point_run.enterprise_value.terminal_value_share,
        })
    y3_points = []
    for delta in (-0.02, 0.0, 0.02):
        from dataclasses import replace
        growth = assumptions.near_term_revenue_growth[:2] + (assumptions.near_term_revenue_growth[2] + delta,)
        point_run = run_real_company_multistage_dcf(snapshot, history, replace(assumptions, near_term_revenue_growth=growth))
        point_years = point_run.operating_forecast.years
        y3_points.append({
            "y3": growth[2], "implied_y4": point_years[3].revenue_growth,
            "implied_y5": point_years[4].revenue_growth,
            "value_per_share": point_run.per_share_value.intrinsic_value_per_share if point_run.per_share_value else None,
            "tv_ev": point_run.enterprise_value.terminal_value_share,
        })
    return {
        "ticker": ticker, "price": snapshot.price,
        "candidate": {
            "growth": assumptions.near_term_revenue_growth,
            "implied_y4": years[3].revenue_growth, "implied_y5": years[4].revenue_growth,
            "final_growth": years[-1].revenue_growth,
            "mature_margin": assumptions.mature_operating_margin,
            "starting_sc": assumptions.starting_sales_to_capital,
            "mature_sc": assumptions.mature_sales_to_capital,
            "tax": assumptions.operating_tax_rate, "wacc": assumptions.wacc,
            "terminal_growth": assumptions.terminal_growth,
            "terminal_roic": assumptions.derived_terminal_roic,
            "terminal_reinvestment": assumptions.terminal_reinvestment_rate,
        },
        "preview": {
            "value_per_share": per_share,
            "enterprise_value_b": run.enterprise_value.enterprise_value / 1e9,
            "equity_value_b": run.equity_value.equity_value / 1e9,
            "explicit_fcff_pv_b": run.enterprise_value.explicit_forecast_pv / 1e9,
            "terminal_pv_b": run.enterprise_value.terminal_value_pv / 1e9,
            "tv_ev": run.enterprise_value.terminal_value_share,
            "revenue_b": {"y1": years[0].revenue/1e9, "y3": years[2].revenue/1e9, "y5": years[4].revenue/1e9, "final": years[-1].revenue/1e9},
            "dcf_to_price": None if snapshot.price is None or per_share is None else per_share/snapshot.price,
        },
        "mature_sc_sensitivity": sc_points,
        "y3_sensitivity": y3_points,
        "confidence": [item.__dict__ for item in result.confidence_assessments],
        "reviewed": profile.last_reviewed_at is not None,
    }


def run_live_validation():
    return [
        _record("GOOGL", build_alphabet_research_profile),
        _record("MSFT", build_microsoft_research_profile),
        _record("META", build_meta_research_profile),
    ]


if __name__ == "__main__":
    print(json.dumps(run_live_validation(), indent=2))
