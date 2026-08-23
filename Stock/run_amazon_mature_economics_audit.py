"""Live evidence orchestration for Phase 3F.1; no production integration."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import yfinance as yf

_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))

from Stock.amazon_mature_economics_audit import (
    amazon_economic_mix_evidence,
    amazon_mature_scenarios,
    aws_mix_sensitivity,
    bucket_margin_sensitivity,
    economics_matrix,
    profit_pool,
    required_aws_share_for_profit_pool,
    reverse_bridge_for_margin,
    run_mature_economics_valuations,
    segment_summed_growth_diagnostic,
)
from Stock.fundamentals import OPERATING_MARGIN, REVENUE, build_validated_ttm
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
    per_share = None if run.per_share_value is None else run.per_share_value.intrinsic_value_per_share
    return {
        "case": item.case,
        "mature_margin": item.scenario.consolidated_margin,
        "mature_sales_to_capital": item.scenario.consolidated_sales_to_capital,
        "terminal_roic": item.scenario.terminal_roic,
        "dcf_per_share": per_share,
        "enterprise_value_b": run.enterprise_value.enterprise_value / 1e9,
        "equity_value_b": run.equity_value.equity_value / 1e9,
        "explicit_fcff_pv_b": run.enterprise_value.explicit_forecast_pv / 1e9,
        "terminal_value_pv_b": run.enterprise_value.terminal_value_pv / 1e9,
        "tv_ev": run.enterprise_value.terminal_value_share,
        "dcf_market_price": None if per_share is None or not price else per_share / price,
    }


def run_live_amazon_mature_economics_audit():
    snapshot = load_company_snapshot("AMZN")
    history = build_company_fundamentals(snapshot)
    live_inputs = extract_real_company_dcf_inputs(snapshot, history)
    annual_revenue = _latest(history.annual[REVENUE])
    annual_period = history.annual[REVENUE].dropna().sort_index().index[-1]
    inputs = replace(
        live_inputs, starting_revenue=annual_revenue,
        starting_revenue_source="annual_fallback",
        starting_revenue_periods=(annual_period,),
    )
    ttm_margin = history.ttm.get(OPERATING_MARGIN)
    starting_margin = (
        float(ttm_margin.value) if ttm_margin and ttm_margin.available
        else _latest(history.annual[OPERATING_MARGIN])
    )
    quarterly_da = _reported_statement_series(
        snapshot.quarterly_cashflow, "depreciation_amortization"
    )
    da_ttm = build_validated_ttm(
        quarterly_da, snapshot.quarterly_cashflow.columns
    )
    depreciation = (
        float(da_ttm.value) if da_ttm.available else
        _latest(_reported_statement_series(
            snapshot.annual_cashflow, "depreciation_amortization"
        ))
    )
    if annual_revenue is None or starting_margin is None or depreciation is None:
        raise ValueError("required AMZN live inputs unavailable")

    scenarios = amazon_mature_scenarios()
    central = scenarios[1]
    matrix = economics_matrix(
        tuple(x.consolidated_margin for x in scenarios) + (.30,),
        tuple(x.consolidated_sales_to_capital for x in scenarios),
    )
    valuations = run_mature_economics_valuations(
        inputs, starting_operating_margin=starting_margin,
        # Preserve the Phase 3F Hybrid normalization: the latest available
        # D&A amount is compared with the SEC-validated 775.680B Revenue TTM.
        starting_depreciation_to_revenue=depreciation / 775.680e9,
    )
    confidence = {
        "mature_aws_share": "Low", "aws_mature_margin": "Medium",
        "advertising_mature_margin": "Low", "retail_mature_margin": "Low",
        "mature_consolidated_margin": "Low", "mature_sales_to_capital": "Low",
        "terminal_roic": "Low", "year1_growth": "High",
        "year2_growth": "Medium", "year3_growth": "Low",
    }
    return {
        "retrieved_at": "2026-08-23",
        "current_price": snapshot.price,
        "current_baseline": {
            "mature_margin": .15683546885666055,
            "mature_sales_to_capital": .85,
            "terminal_roic": .15683546885666055 * .79 * .85,
            "wacc": .105, "terminal_growth": .03, "tax": .21,
        },
        "economic_mix_evidence": [
            asdict(x) | {"shares": dict(x.shares)}
            for x in amazon_economic_mix_evidence()
        ],
        "mature_scenarios": [asdict(x) for x in scenarios],
        "profit_pools": {
            x.name: [asdict(y) for y in profit_pool(x)] for x in scenarios
        },
        "reverse_30_margin": asdict(reverse_bridge_for_margin()),
        "aws_profit_pool_thresholds": {
            str(int(target * 100)): required_aws_share_for_profit_pool(target)
            for target in (.40, .50, .60)
        },
        "retail_margin_sensitivity": [asdict(x) for x in bucket_margin_sensitivity(
            central, "first_party_retail", (.03, .05, .07)
        )],
        "aws_margin_sensitivity": [asdict(x) for x in bucket_margin_sensitivity(
            central, "aws", (.28, .33, .38)
        )],
        "aws_mix_sensitivity": [asdict(x) for x in aws_mix_sensitivity(
            central, (.20, .25, .30)
        )],
        "economics_matrix": [asdict(x) for x in matrix],
        "growth_diagnostic": [asdict(x) for x in segment_summed_growth_diagnostic()],
        "valuations": [_valuation(x, snapshot.price) for x in valuations],
        "confidence": confidence,
        "warnings": (
            "research_only", "no_market_price_calibration",
            "bucket_margins_are_ranges_not_disclosures",
            "advertising_margin_not_disclosed",
            "hybrid_not_integrated_into_production",
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_live_amazon_mature_economics_audit(), indent=2, default=str))
