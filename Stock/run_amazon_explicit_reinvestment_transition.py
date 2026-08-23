"""Live orchestration for the read-only Phase 3F.3 Amazon transition audit."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(_CACHE_DIR))

from Stock.amazon_explicit_reinvestment_transition import (
    apply_explicit_transition,
    build_explicit_transition_path,
    capex_definition_sensitivity,
    capex_taxonomy,
    frozen_mature_controls,
    historical_capital_evidence,
    historical_lead_lag_evidence,
    run_frozen_standard_model,
    run_transition_case,
    transition_case_specs,
    useful_life_evidence,
)
from Stock.fundamentals import OPERATING_MARGIN
from Stock.hybrid_reinvestment_prototype import compare_hybrid_reinvestment
from Stock.hyperscaler_hybrid_audit import (
    build_hyperscaler_hybrid_inputs,
    hyperscaler_hybrid_research_specs,
)
from Stock.multistage_integration import extract_real_company_dcf_inputs
from Stock.stock_valuation_mvp import build_company_fundamentals, load_company_snapshot


SEC_TTM_REVENUE = 775.680e9
SEC_TTM_TOTAL_DA = 75.200e9
SEC_TTM_PPE_DA = 49.741e9


def _latest(series):
    cleaned = series.dropna().sort_index()
    return None if cleaned.empty else float(cleaned.iloc[-1])


def _run_summary(model, run, cumulative_reinvestment, cumulative_fcff):
    value = None if run.per_share_value is None else run.per_share_value.intrinsic_value_per_share
    return {
        "model": model,
        "per_share": value,
        "enterprise_value_b": run.enterprise_value.enterprise_value / 1e9,
        "equity_value_b": run.equity_value.equity_value / 1e9,
        "explicit_fcff_pv_b": run.enterprise_value.explicit_forecast_pv / 1e9,
        "terminal_value_pv_b": run.enterprise_value.terminal_value_pv / 1e9,
        "terminal_value_share": run.enterprise_value.terminal_value_share,
        "cumulative_explicit_reinvestment_b": cumulative_reinvestment / 1e9,
        "cumulative_explicit_fcff_b": cumulative_fcff / 1e9,
    }


def _standard_summary(run):
    years = run.operating_forecast.years[:5]
    return _run_summary(
        "S_production_style_sales_to_capital", run,
        sum(row.reinvestment for row in years), sum(row.fcff for row in years),
    )


def _h0_summary(comparison):
    return _run_summary(
        "H0_existing_research_hybrid", comparison.hybrid_run,
        comparison.cumulative_hybrid_reinvestment,
        comparison.cumulative_hybrid_fcff,
    )


def _h1_summary(result):
    return _run_summary(
        result.model, result.run, result.cumulative_explicit_reinvestment,
        result.cumulative_explicit_fcff,
    )


def run_live_amazon_explicit_reinvestment_transition():
    snapshot = load_company_snapshot("AMZN")
    history = build_company_fundamentals(snapshot)
    extracted = extract_real_company_dcf_inputs(snapshot, history)
    inputs = replace(
        extracted,
        starting_revenue=SEC_TTM_REVENUE,
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
    if starting_margin is None:
        raise ValueError("AMZN starting operating margin unavailable")

    standard = run_frozen_standard_model(
        inputs, starting_operating_margin=starting_margin
    )
    h0_spec = next(x for x in hyperscaler_hybrid_research_specs() if x.ticker == "AMZN")
    h0_inputs = build_hyperscaler_hybrid_inputs(
        standard, h0_spec,
        starting_depreciation_to_revenue=SEC_TTM_TOTAL_DA / SEC_TTM_REVENUE,
    )
    h0 = compare_hybrid_reinvestment(
        standard, h0_inputs, classification=h0_spec.classification
    )

    h1 = tuple(run_transition_case(
        standard, spec, starting_ppe_depreciation=SEC_TTM_PPE_DA
    ) for spec in transition_case_specs())
    central_spec = transition_case_specs()[1]
    central_explicit = build_explicit_transition_path(
        standard, central_spec, starting_ppe_depreciation=SEC_TTM_PPE_DA
    )
    handoffs = tuple(apply_explicit_transition(
        standard, central_explicit, handoff_years=length,
        model_name=f"H1_central_handoff_{length}y",
    ) for length in (1, 2, 3))

    models = (_standard_summary(standard), _h0_summary(h0)) + tuple(
        _h1_summary(result) for result in h1
    )
    latest = historical_capital_evidence()[-1]
    result = {
        "retrieved_at": "2026-08-23",
        "revenue_base_b": SEC_TTM_REVENUE / 1e9,
        "starting_operating_margin": starting_margin,
        "frozen_controls": asdict(frozen_mature_controls()) | {
            "terminal_roic": frozen_mature_controls().terminal_roic,
            "terminal_reinvestment_rate": frozen_mature_controls().terminal_reinvestment_rate,
        },
        "growth_path": [row.revenue_growth for row in standard.operating_forecast.years],
        "capex_taxonomy": [asdict(row) for row in capex_taxonomy()],
        "historical_evidence": [asdict(row) | {
            "net_capex": row.net_capex,
            "cash_capex_to_revenue": row.cash_capex_to_revenue,
            "economic_capex_to_revenue": row.economic_capex_to_revenue,
            "depreciation_to_revenue": row.depreciation_to_revenue,
            "depreciation_to_capex": row.depreciation_to_capex,
        } for row in historical_capital_evidence()],
        "latest_capex_definition_sensitivity": dict(capex_definition_sensitivity(latest)),
        "lead_lag": historical_lead_lag_evidence(),
        "useful_lives": [asdict(row) for row in useful_life_evidence()],
        "h0_explicit_years": [asdict(row) for row in h0.hybrid_years],
        "h0_transition": asdict(h0.transition),
        "h1_specs": [asdict(row) for row in transition_case_specs()],
        "h1_explicit_years": {
            result.model: [asdict(row) for row in result.explicit_years]
            for result in h1
        },
        "handoff_comparison": {
            result.model: [asdict(row) for row in result.handoff]
            for result in handoffs
        },
        "valuation_models": models,
        "transition_warnings": {
            result.model: result.warnings for result in h1
        },
        "structural_conclusion": "B_explicit_transition_explains_part_gap_remains",
        "future_method": "C_hybrid_explicit_period_with_multi_year_handoff",
        "profile_readiness": "READY_FOR_PROFILE",
        "assumption_construction_excludes_market_price": True,
    }
    # Price is appended only after every assumption and model output is fixed.
    result["market_price_context"] = snapshot.price
    return result


if __name__ == "__main__":
    print(json.dumps(
        run_live_amazon_explicit_reinvestment_transition(),
        indent=2, default=str,
    ))
