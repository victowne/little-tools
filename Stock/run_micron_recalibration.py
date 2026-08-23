"""Live read-only Phase 4.1 Micron old/new unified-DCF validation."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import yfinance as yf

from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.micron_recalibration import build_micron_period_alignment
from Stock.multistage_integration import run_real_company_multistage_dcf
from Stock.stock_valuation_mvp import (
    build_company_fundamentals,
    build_multistage_assumptions_from_ui,
    load_company_snapshot,
    multistage_initial_defaults,
)
from Stock.unified_company_research import build_micron_research_profile


_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))


def _run_summary(run):
    years = run.operating_forecast.years
    return {
        "growth_path": [row.revenue_growth for row in years],
        "revenue_b": {
            "y1": years[0].revenue / 1e9,
            "y2": years[1].revenue / 1e9,
            "y3": years[2].revenue / 1e9,
            "y5": years[4].revenue / 1e9,
            "final": years[-1].revenue / 1e9,
        },
        "dcf_per_share": run.per_share_value.intrinsic_value_per_share,
        "enterprise_value_b": run.enterprise_value.enterprise_value / 1e9,
        "equity_value_b": run.equity_value.equity_value / 1e9,
        "explicit_pv_b": run.enterprise_value.explicit_forecast_pv / 1e9,
        "terminal_pv_b": run.enterprise_value.terminal_value_pv / 1e9,
        "tv_ev": run.enterprise_value.terminal_value_share,
    }


def run_live_micron_recalibration():
    snapshot = load_company_snapshot("MU")
    history = build_company_fundamentals(snapshot)
    current = build_multistage_assumptions_from_ui(
        multistage_initial_defaults("MU", history)
    )
    result = build_micron_research_profile(
        current, history, retrieved_at="2026-08-23"
    )
    profile = result.lookup.profile
    revised = build_multistage_assumptions_from_profile(profile).assumptions
    old = replace(revised, near_term_revenue_growth=(.45, .12, -.08))
    old_run = run_real_company_multistage_dcf(snapshot, history, old)
    revised_run = run_real_company_multistage_dcf(snapshot, history, revised)
    alignment = build_micron_period_alignment()
    return {
        "ticker": "MU",
        "starting_revenue_b": revised_run.inputs.starting_revenue / 1e9,
        "starting_revenue_periods": [
            str(item.date()) for item in revised_run.inputs.starting_revenue_periods
        ],
        "alignment": asdict(alignment),
        "old": _run_summary(old_run),
        "new": _run_summary(revised_run),
        "mature_margin": revised.mature_operating_margin,
        "mature_sales_to_capital": revised.mature_sales_to_capital,
        "terminal_roic": revised.derived_terminal_roic,
        "wacc": revised.wacc,
        "terminal_growth": revised.terminal_growth,
        "market_price_after_selection": snapshot.price,
        "profile_status": profile.profile_status,
        "reviewed": False,
        "applied": False,
        "production_model": "STANDARD_SALES_TO_CAPITAL",
    }


if __name__ == "__main__":
    print(json.dumps(run_live_micron_recalibration(), indent=2, default=str))
