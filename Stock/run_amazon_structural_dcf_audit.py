"""Live orchestration for the read-only Phase 3F Amazon attribution audit."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))

from Stock.amazon_structural_dcf_audit import (
    amazon_segment_evidence,
    run_amazon_structural_audit,
)
from Stock.fundamentals import OPERATING_MARGIN, REVENUE, build_validated_ttm
from Stock.multistage_integration import extract_real_company_dcf_inputs
from Stock.stock_valuation_mvp import (
    _reported_statement_series,
    build_company_fundamentals,
    load_company_snapshot,
)


def _latest(series):
    values = series.dropna().sort_index()
    return None if values.empty else float(values.iloc[-1])


def _summary(item, market_price):
    if item is None:
        return None
    return {
        "model": item.model,
        "dcf_per_share": item.intrinsic_value_per_share,
        "enterprise_value_b": item.enterprise_value / 1e9,
        "equity_value_b": item.equity_value / 1e9,
        "explicit_fcff_pv_b": item.explicit_fcff_pv / 1e9,
        "terminal_value_pv_b": item.terminal_value_pv / 1e9,
        "tv_ev": item.terminal_value_share,
        "dcf_market_price": (
            None if item.intrinsic_value_per_share is None or not market_price
            else item.intrinsic_value_per_share / market_price
        ),
    }


def run_live_amazon_structural_audit():
    snapshot = load_company_snapshot("AMZN")
    history = build_company_fundamentals(snapshot)
    live_inputs = extract_real_company_dcf_inputs(snapshot, history)

    quarterly_revenue = _reported_statement_series(
        snapshot.quarterly_income, "revenue"
    )
    yahoo_revenue_ttm = build_validated_ttm(
        quarterly_revenue, snapshot.quarterly_income.columns
    )
    # Yahoo currently exposes the 2026-Q2 column but not AMZN Revenue in that
    # column. Validate the four SEC-reported values independently; do not fill
    # or mutate the Yahoo series.
    sec_quarterly_revenue = pd.Series(
        (180.169e9, 213.386e9, 181.519e9, 200.606e9),
        index=pd.to_datetime(
            ("2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30")
        ),
    )
    sec_revenue_ttm = build_validated_ttm(sec_quarterly_revenue)
    annual_revenue = _latest(history.annual[REVENUE])
    annual_period = history.annual[REVENUE].dropna().sort_index().index[-1]
    annual_inputs = replace(
        live_inputs,
        starting_revenue=annual_revenue,
        starting_revenue_source="annual_fallback",
        starting_revenue_periods=(annual_period,),
    )
    ttm_inputs = None
    if sec_revenue_ttm.available:
        ttm_inputs = replace(
            live_inputs,
            starting_revenue=float(sec_revenue_ttm.value),
            starting_revenue_source="validated_ttm_sec_10q",
            starting_revenue_periods=sec_revenue_ttm.periods_used,
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
    da = (
        float(da_ttm.value) if da_ttm.available
        else _latest(_reported_statement_series(
            snapshot.annual_cashflow, "depreciation_amortization"
        ))
    )
    if annual_revenue is None or starting_margin is None or da is None:
        raise ValueError("required Amazon live audit input unavailable")

    result = run_amazon_structural_audit(
        annual_inputs,
        starting_margin,
        validated_ttm_inputs=ttm_inputs,
        starting_depreciation_to_revenue=(
            da / (ttm_inputs.starting_revenue if ttm_inputs else annual_revenue)
        ),
    )
    models = (
        result.baseline, result.revenue_base_fix, result.margin_only,
        result.hybrid_only, result.margin_hybrid, result.segment_shadow,
    )
    baseline = result.baseline.run
    latest_quarter = sec_quarterly_revenue.iloc[-1]
    evidence = amazon_segment_evidence()
    return {
        "retrieved_at": "2026-08-23",
        "market_price": snapshot.price,
        "revenue_validation": {
            "latest_annual_b": annual_revenue / 1e9,
            "annual_period": str(annual_period.date()),
            "yahoo_ttm_available": yahoo_revenue_ttm.available,
            "yahoo_ttm_reason": yahoo_revenue_ttm.reason,
            "yahoo_candidate_periods": tuple(str(x.date()) for x in yahoo_revenue_ttm.periods_used),
            "sec_validated_ttm_available": sec_revenue_ttm.available,
            "validated_ttm_b": sec_revenue_ttm.value / 1e9,
            "ttm_periods": tuple(str(x.date()) for x in sec_revenue_ttm.periods_used),
            "ttm_source": "Amazon 2026 Q2 SEC 10-Q",
            "latest_quarter_b": float(latest_quarter) / 1e9,
            "latest_quarter_x4_b": float(latest_quarter) * 4 / 1e9,
            "quarterly_observations_b": tuple(
                (str(period.date()), float(value) / 1e9)
                for period, value in quarterly_revenue.dropna().sort_index().items()
            ),
        },
        "baseline_assumptions": {
            "revenue_base_b": annual_inputs.starting_revenue / 1e9,
            "source": annual_inputs.starting_revenue_source,
            "periods": tuple(str(x.date()) for x in annual_inputs.starting_revenue_periods),
            "growth": baseline.assumptions.near_term_revenue_growth,
            "implied_y4_y5": tuple(
                x.revenue_growth for x in baseline.forecast_path.years[3:5]
            ),
            "fade_years": baseline.assumptions.revenue_fade_years,
            "horizon": baseline.assumptions.forecast_years,
            "starting_margin": baseline.assumptions.starting_operating_margin,
            "mature_margin": baseline.assumptions.mature_operating_margin,
            "starting_sc": baseline.assumptions.starting_sales_to_capital,
            "mature_sc": baseline.assumptions.mature_sales_to_capital,
            "tax": baseline.assumptions.operating_tax_rate,
            "wacc": baseline.assumptions.wacc,
            "terminal_growth": baseline.assumptions.terminal_growth,
        },
        "valuation_waterfall": [_summary(x, snapshot.price) for x in models if x],
        "baseline_pathology": [vars(x) for x in result.baseline_pathology],
        "growth_monotonicity": [vars(x) for x in result.growth_monotonicity],
        "margin_sc_grid": [vars(x) for x in result.margin_sc_grid],
        "segment_evidence": [vars(x) | {"operating_margin": x.operating_margin} for x in evidence],
        "segment_forecast": [{
            "year": x.year,
            "revenue_b": x.revenue / 1e9,
            "operating_income_b": x.operating_income / 1e9,
            "operating_margin": x.operating_margin,
            "mix": {s.segment: s.revenue / x.revenue for s in x.segments},
            "segments": [vars(s) for s in x.segments],
        } for x in result.segment_forecast],
        "terminal_economics": {
            "roic": baseline.terminal_value.derived_terminal_roic,
            "wacc": baseline.assumptions.wacc,
            "reinvestment_rate": baseline.terminal_value.terminal_reinvestment_rate,
            "fcff_to_nopat": baseline.terminal_value.terminal_fcff_to_nopat,
        },
        "severity": dict(result.severity),
        "warnings": result.warnings,
    }


if __name__ == "__main__":
    print(json.dumps(run_live_amazon_structural_audit(), indent=2, default=str))
