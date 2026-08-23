"""Live evidence adapter for the pure Phase 3D.2 methodology audit."""

import json
from datetime import datetime, timezone

import pandas as pd

from Stock.five_year_growth_sc_audit import (
    hyperscaler_growth_sc_specs,
    run_five_year_growth_sc_audit,
)
from Stock.forecast_methodology_audit import build_audit_candidate, spec_for_ticker
from Stock.fundamentals import OPERATING_MARGIN, REVENUE, ROIC
from Stock.multistage_integration import extract_real_company_dcf_inputs
from Stock.stock_valuation_mvp import (
    _reported_statement_series,
    build_company_fundamentals,
    build_company_revenue_forecast_anchors,
    load_company_snapshot,
)


def _latest_annual(history, metric):
    if history.annual.empty or metric not in history.annual:
        return None
    series = history.annual[metric].dropna().sort_index()
    return None if series.empty else float(series.iloc[-1])


def _model(summary):
    run = summary.run
    years = run.operating_forecast.years
    return {
        "intrinsic_value_per_share": summary.intrinsic_value_per_share,
        "enterprise_value_b": summary.enterprise_value / 1e9,
        "equity_value_b": summary.equity_value / 1e9,
        "explicit_fcff_pv_b": summary.explicit_fcff_pv / 1e9,
        "terminal_value_pv_b": summary.terminal_value_pv / 1e9,
        "terminal_value_share": summary.terminal_value_share,
        "total_explicit_fcff_b": run.operating_forecast.total_fcff / 1e9,
        "revenue_b": {
            "starting": run.inputs.starting_revenue / 1e9,
            "year_1": years[0].revenue / 1e9,
            "year_3": years[2].revenue / 1e9,
            "year_5": years[4].revenue / 1e9,
            "final": years[-1].revenue / 1e9,
        },
    }


def _quarterly_revenue_evidence(snapshot):
    series = _reported_statement_series(snapshot.quarterly_income, "revenue")
    values = []
    for period, value in series.dropna().sort_index().items():
        # Yahoo reporting dates can shift by a day; use the nearest observation
        # 330-400 days earlier rather than claiming a fiscal-quarter calendar.
        prior = series[
            (series.index >= period - pd.Timedelta(days=400))
            & (series.index <= period - pd.Timedelta(days=330))
        ].dropna()
        yoy = None if prior.empty else float(value / prior.iloc[-1] - 1)
        values.append({"period": str(period.date()), "revenue_b": float(value) / 1e9, "yoy": yoy})
    return values[-8:]


def _forward_anchors(ticker, snapshot, history):
    anchors = build_company_revenue_forecast_anchors(ticker, snapshot, history)
    if anchors is None:
        return None
    return {
        "base_kind": anchors.base_kind,
        "base_period": None if anchors.base_period is None else str(anchors.base_period.date()),
        "source": anchors.source,
        "points": [{
            "year": point.forecast_year_index,
            "period": None if point.fiscal_period is None else str(point.fiscal_period.date()),
            "revenue_b": None if point.revenue_estimate is None else point.revenue_estimate / 1e9,
            "growth": point.implied_revenue_growth,
            "available": point.available,
            "reason": point.reason,
        } for point in anchors.points],
        "warnings": anchors.warnings,
    }


def run_live_audit():
    outputs = []
    for spec in hyperscaler_growth_sc_specs():
        snapshot = load_company_snapshot(spec.ticker)
        history = build_company_fundamentals(snapshot)
        inputs = extract_real_company_dcf_inputs(snapshot, history)
        ttm_margin = history.ttm.get(OPERATING_MARGIN)
        starting_margin = (
            float(ttm_margin.value)
            if ttm_margin is not None and ttm_margin.available
            else _latest_annual(history, OPERATING_MARGIN)
        )
        if starting_margin is None:
            raise ValueError(f"{spec.ticker}: starting margin unavailable")
        baseline_assumptions = build_audit_candidate(
            spec_for_ticker(spec.ticker), starting_margin
        )
        audit = run_five_year_growth_sc_audit(
            inputs, baseline_assumptions, spec
        )
        latest_sc = None
        if history.dcf_anchors.annual_sales_to_capital:
            latest_result = sorted(
                history.dcf_anchors.annual_sales_to_capital.items()
            )[-1][1]
            if latest_result.available:
                latest_sc = latest_result.value
        normalized_result = history.dcf_anchors.normalized_sales_to_capital.get(3)
        normalized_sc = (
            normalized_result.value
            if normalized_result is not None and normalized_result.available
            else None
        )
        baseline_path = audit.baseline.run.forecast_path.revenue_growth_path
        output = {
            "ticker": spec.ticker,
            "issuer": spec.issuer,
            "starting_revenue_b": inputs.starting_revenue / 1e9,
            "starting_revenue_source": inputs.starting_revenue_source,
            "baseline_assumptions": {
                "growth_y1_y5": baseline_path[:5],
                "fade_years": baseline_assumptions.revenue_fade_years,
                "forecast_years": baseline_assumptions.forecast_years,
                "starting_margin": baseline_assumptions.starting_operating_margin,
                "mature_margin": baseline_assumptions.mature_operating_margin,
                "starting_sc": baseline_assumptions.starting_sales_to_capital,
                "mature_sc": baseline_assumptions.mature_sales_to_capital,
                "tax": baseline_assumptions.operating_tax_rate,
                "wacc": baseline_assumptions.wacc,
                "terminal_growth": baseline_assumptions.terminal_growth,
            },
            "research_growth": [{
                "year": point.year,
                "current_growth": baseline_path[point.year - 1],
                "research_growth": point.growth,
                "confidence": point.confidence,
                "evidence": point.evidence,
                "rationale": point.rationale,
            } for point in spec.explicit_growth],
            "quarterly_revenue": _quarterly_revenue_evidence(snapshot),
            "forward_anchors": _forward_anchors(spec.ticker, snapshot, history),
            "sales_to_capital_evidence": {
                "latest_accounting": latest_sc,
                "normalized_historical_3y": normalized_sc,
                "research_mature_candidate": spec.research_mature_sales_to_capital,
                "range": spec.mature_sales_to_capital_values,
                "rationale": spec.mature_sales_to_capital_rationale,
                "current_accounting_roic": _latest_annual(history, ROIC),
            },
            "models": {
                "baseline": _model(audit.baseline),
                "growth_only": _model(audit.growth_only),
                "mature_sc_only": None if audit.mature_sc_only is None else _model(audit.mature_sc_only),
                "combined": None if audit.combined is None else _model(audit.combined),
            },
            "mature_sc_sensitivity": [{
                "mature_sc": point.mature_sales_to_capital,
                "terminal_roic": point.terminal_roic,
                "terminal_reinvestment_rate": point.terminal_reinvestment_rate,
                "terminal_fcff_to_nopat": point.terminal_fcff_to_nopat,
                "intrinsic_value_per_share": point.intrinsic_value_per_share,
                "terminal_value_share": point.terminal_value_share,
                "warnings": point.warnings,
            } for point in audit.mature_sc_sensitivity],
            "methodology": {
                "explicit_period": spec.explicit_period_classification,
                "mature_sc": spec.mature_sales_to_capital_classification,
            },
        }
        # Market data is accessed only after assumptions and all shadow models
        # above are complete; it has no path back into assumption generation.
        output["market"] = {
            "price": snapshot.price,
            "currency": snapshot.price_currency,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source": "yfinance CompanySnapshot",
            "dcf_to_price": {
                name: None if snapshot.price is None or model is None else model["intrinsic_value_per_share"] / snapshot.price
                for name, model in output["models"].items()
            },
        }
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    print(json.dumps(run_live_audit(), indent=2))
