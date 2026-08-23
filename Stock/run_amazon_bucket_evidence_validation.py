"""Compact live Phase 3F.2 validation report; no production writes."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))

from Stock.amazon_bucket_evidence_validation import (
    aws_capital_diagnostics,
    capital_and_profit_allocation,
    phase3f1_change_attribution,
    run_validated_mature_valuations,
    shared_cost_sensitivity,
    validated_bucket_ranges,
    validated_mature_scenarios,
    validated_reverse_thirty_percent,
    validation_evidence,
)
from Stock.amazon_mature_economics_audit import (
    amazon_economic_mix_evidence,
    segment_summed_growth_diagnostic,
)
from Stock.fundamentals import OPERATING_MARGIN, build_validated_ttm
from Stock.multistage_integration import extract_real_company_dcf_inputs
from Stock.stock_valuation_mvp import (
    _reported_statement_series,
    build_company_fundamentals,
    load_company_snapshot,
)


def _latest(series):
    cleaned = series.dropna().sort_index()
    return None if cleaned.empty else float(cleaned.iloc[-1])


def _valuation(item, price):
    run = item.run
    value = None if run.per_share_value is None else run.per_share_value.intrinsic_value_per_share
    return {
        "case": item.case,
        "margin": item.scenario.consolidated_margin,
        "sales_to_capital": item.scenario.consolidated_sales_to_capital,
        "terminal_roic": item.scenario.terminal_roic,
        "per_share": value,
        "ev_b": run.enterprise_value.enterprise_value / 1e9,
        "equity_b": run.equity_value.equity_value / 1e9,
        "explicit_pv_b": run.enterprise_value.explicit_forecast_pv / 1e9,
        "terminal_pv_b": run.enterprise_value.terminal_value_pv / 1e9,
        "tv_ev": run.enterprise_value.terminal_value_share,
        "dcf_price": None if value is None or not price else value / price,
    }


def run_live_amazon_bucket_evidence_validation():
    snapshot = load_company_snapshot("AMZN")
    history = build_company_fundamentals(snapshot)
    extracted = extract_real_company_dcf_inputs(snapshot, history)
    inputs = replace(
        extracted, starting_revenue=775.680e9,
        starting_revenue_source="validated_ttm_sec_10q",
        starting_revenue_periods=tuple(pd.to_datetime(
            ("2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30")
        )),
    )
    margin_ttm = history.ttm.get(OPERATING_MARGIN)
    starting_margin = (
        float(margin_ttm.value) if margin_ttm and margin_ttm.available
        else _latest(history.annual[OPERATING_MARGIN])
    )
    quarterly_da = _reported_statement_series(snapshot.quarterly_cashflow, "depreciation_amortization")
    da_ttm = build_validated_ttm(quarterly_da, snapshot.quarterly_cashflow.columns)
    depreciation = float(da_ttm.value) if da_ttm.available else _latest(
        _reported_statement_series(snapshot.annual_cashflow, "depreciation_amortization")
    )
    if starting_margin is None or depreciation is None:
        raise ValueError("AMZN validation inputs unavailable")
    scenarios = validated_mature_scenarios()
    valuations = run_validated_mature_valuations(
        inputs, starting_operating_margin=starting_margin,
        starting_depreciation_to_revenue=depreciation / inputs.starting_revenue,
    )
    return {
        "retrieved_at": "2026-08-23",
        "price": snapshot.price,
        "revenue_base_b": inputs.starting_revenue / 1e9,
        "evidence": [asdict(x) for x in validation_evidence()],
        "historical_mix": [asdict(x) | {"shares": dict(x.shares)} for x in amazon_economic_mix_evidence()],
        "validated_ranges": [asdict(x) for x in validated_bucket_ranges()],
        "validated_scenarios": [asdict(x) for x in scenarios],
        "central_profit_capital_pool": [asdict(x) for x in capital_and_profit_allocation(scenarios[1])],
        "aws_capital_diagnostics": dict(aws_capital_diagnostics()),
        "shared_cost_sensitivity": [{
            "shared_cost": x.shared_cost_adjustment,
            "margin": x.consolidated_margin,
            "terminal_roic": x.terminal_roic,
        } for x in shared_cost_sensitivity(scenarios[1])],
        "reverse_30": asdict(validated_reverse_thirty_percent()),
        "changes": [asdict(x) for x in phase3f1_change_attribution()],
        "growth": [asdict(x) for x in segment_summed_growth_diagnostic()[:3]],
        "valuations": [_valuation(x, snapshot.price) for x in valuations],
        "profile_readiness": "NEED_MORE_RESEARCH",
        "warnings": (
            "advertising_operating_income_not_disclosed",
            "marketplace_operating_income_not_disclosed",
            "economic_bucket_assets_not_disclosed",
            "shared_logistics_and_AI_infrastructure",
            "hybrid_research_only",
            "market_price_excluded_from_assumption_validation",
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_live_amazon_bucket_evidence_validation(), indent=2, default=str))
