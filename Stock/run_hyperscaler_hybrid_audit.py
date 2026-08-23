"""Run the Phase 3D.1 four-company live diagnostic and emit compact JSON.

This is an explicit diagnostic entry point, not an application data path.
"""

import json

from Stock.forecast_methodology_audit import audit_candidate_specs, build_audit_candidate
from Stock.fundamentals import OPERATING_MARGIN, build_validated_ttm
from Stock.hybrid_reinvestment_prototype import (
    compare_hybrid_reinvestment,
    scale_capex_path,
    scale_depreciation_path,
)
from Stock.hyperscaler_hybrid_audit import (
    build_hyperscaler_hybrid_inputs,
    hyperscaler_hybrid_research_specs,
)
from Stock.multistage_integration import extract_real_company_dcf_inputs, run_multistage_dcf
from Stock.stock_valuation_mvp import (
    _reported_statement_series,
    _statement_series,
    build_company_fundamentals,
    load_company_snapshot,
)


def _latest(series):
    cleaned = series.dropna().sort_index()
    return None if cleaned.empty else float(cleaned.iloc[-1])


def _ttm_or_annual(snapshot, concept: str):
    quarterly = _reported_statement_series(snapshot.quarterly_cashflow, concept)
    ttm = build_validated_ttm(quarterly, snapshot.quarterly_cashflow.columns)
    if ttm.available:
        return float(ttm.value), "ttm", tuple(str(item.date()) for item in ttm.periods_used)
    annual = _reported_statement_series(snapshot.annual_cashflow, concept)
    return _latest(annual), "latest_annual", ()


def _dated_amounts(series, *, limit: int):
    cleaned = series.dropna().sort_index().iloc[-limit:]
    return tuple(
        (str(period.date()), float(value) / 1e9)
        for period, value in cleaned.items()
    )


def _per_share(run):
    return (
        None if run.per_share_value is None
        else run.per_share_value.intrinsic_value_per_share
    )


def run_live_hyperscaler_hybrid_audit():
    candidate_specs = {item.ticker: item for item in audit_candidate_specs()}
    output = []
    for spec in hyperscaler_hybrid_research_specs():
        snapshot = load_company_snapshot(spec.ticker)
        fundamentals = build_company_fundamentals(snapshot)
        inputs = extract_real_company_dcf_inputs(snapshot, fundamentals)
        margin = fundamentals.ttm.get(OPERATING_MARGIN)
        if margin is not None and margin.available:
            starting_margin = float(margin.value)
        else:
            starting_margin = _latest(fundamentals.annual[OPERATING_MARGIN])
        if starting_margin is None:
            raise ValueError(f"{spec.ticker}: operating margin unavailable")
        assumptions = build_audit_candidate(
            candidate_specs[spec.ticker], starting_margin
        )
        existing = run_multistage_dcf(inputs, assumptions)
        depreciation, da_source, da_periods = _ttm_or_annual(
            snapshot, "depreciation_amortization"
        )
        capex, capex_source, capex_periods = _ttm_or_annual(
            snapshot, "capital_expenditure"
        )
        annual_capex_series = _reported_statement_series(
            snapshot.annual_cashflow, "capital_expenditure"
        )
        quarterly_capex_series = _reported_statement_series(
            snapshot.quarterly_cashflow, "capital_expenditure"
        )
        annual_da_series = _reported_statement_series(
            snapshot.annual_cashflow, "depreciation_amortization"
        )
        net_ppe = _statement_series(
            snapshot.annual_balance,
            ("Net PPE", "Property Plant And Equipment Net"),
        )
        construction_in_progress = _statement_series(
            snapshot.annual_balance,
            ("Construction In Progress",),
        )
        if depreciation is None:
            raise ValueError(f"{spec.ticker}: depreciation unavailable")
        hybrid_inputs = build_hyperscaler_hybrid_inputs(
            existing, spec,
            starting_depreciation_to_revenue=depreciation / inputs.starting_revenue,
        )
        comparison = compare_hybrid_reinvestment(
            existing, hybrid_inputs, classification=spec.classification
        )
        capex_low = compare_hybrid_reinvestment(
            existing, scale_capex_path(hybrid_inputs, 0.9)
        )
        capex_high = compare_hybrid_reinvestment(
            existing, scale_capex_path(hybrid_inputs, 1.1)
        )
        da_low = compare_hybrid_reinvestment(
            existing, scale_depreciation_path(hybrid_inputs, 0.9)
        )
        da_high = compare_hybrid_reinvestment(
            existing, scale_depreciation_path(hybrid_inputs, 1.1)
        )
        output.append({
            "ticker": spec.ticker,
            "classification": comparison.classification,
            "starting_revenue_b": inputs.starting_revenue / 1e9,
            "reported_capex_b": None if capex is None else abs(capex) / 1e9,
            "latest_annual_capex_b": None if _latest(annual_capex_series) is None else abs(_latest(annual_capex_series)) / 1e9,
            "recent_quarterly_capex_b": tuple(
                (period, abs(value))
                for period, value in _dated_amounts(quarterly_capex_series, limit=4)
            ),
            "capex_source": capex_source,
            "capex_periods": capex_periods,
            "reported_da_b": depreciation / 1e9,
            "latest_annual_da_b": None if _latest(annual_da_series) is None else _latest(annual_da_series) / 1e9,
            "da_source": da_source,
            "da_periods": da_periods,
            "latest_net_ppe_b": None if _latest(net_ppe) is None else _latest(net_ppe) / 1e9,
            "latest_construction_in_progress_b": None if _latest(construction_in_progress) is None else _latest(construction_in_progress) / 1e9,
            "management_capex_guidance_b": spec.year_one_capex_guidance / 1e9,
            "years": [{
                "year": year.year,
                "revenue_b": year.revenue / 1e9,
                "nopat_b": year.nopat / 1e9,
                "capex_b": year.gross_capex / 1e9,
                "da_b": year.depreciation_amortization / 1e9,
                "wc_b": year.change_in_working_capital / 1e9,
                "hybrid_reinvestment_b": year.total_reinvestment / 1e9,
                "hybrid_fcff_b": year.fcff / 1e9,
                "sales_to_capital_reinvestment_b": existing.operating_forecast.years[year.year - 1].reinvestment / 1e9,
                "sales_to_capital_fcff_b": existing.operating_forecast.years[year.year - 1].fcff / 1e9,
                "confidence": year.source_confidence,
            } for year in comparison.hybrid_years],
            "cumulative_reinvestment_b": {
                "sales_to_capital": comparison.cumulative_sales_to_capital_reinvestment / 1e9,
                "hybrid": comparison.cumulative_hybrid_reinvestment / 1e9,
            },
            "cumulative_fcff_b": {
                "sales_to_capital": comparison.cumulative_sales_to_capital_fcff / 1e9,
                "hybrid": comparison.cumulative_hybrid_fcff / 1e9,
            },
            "five_year_fcff_pv_b": {
                "sales_to_capital": comparison.five_year_sales_to_capital_fcff_pv / 1e9,
                "hybrid": comparison.five_year_hybrid_fcff_pv / 1e9,
            },
            "full_dcf": {
                "sales_to_capital_per_share": _per_share(existing),
                "hybrid_per_share": _per_share(comparison.hybrid_run),
                "sales_to_capital_terminal_share": existing.enterprise_value.terminal_value_share,
                "hybrid_terminal_share": comparison.hybrid_run.enterprise_value.terminal_value_share,
            },
            "transition": {
                "hybrid_y5_reinvestment_b": comparison.transition.final_explicit_total_reinvestment / 1e9,
                "sales_to_capital_y5_reinvestment_b": comparison.transition.final_year_sales_to_capital_reinvestment / 1e9,
                "sales_to_capital_y6_reinvestment_b": None if comparison.transition.first_normalized_year_sales_to_capital_reinvestment is None else comparison.transition.first_normalized_year_sales_to_capital_reinvestment / 1e9,
                "warnings": comparison.transition.warnings,
            },
            "sensitivity_per_share": {
                "capex_minus_10": _per_share(capex_low.hybrid_run),
                "capex_plus_10": _per_share(capex_high.hybrid_run),
                "da_minus_10": _per_share(da_low.hybrid_run),
                "da_plus_10": _per_share(da_high.hybrid_run),
            },
            "warnings": comparison.warnings,
        })
    return output


if __name__ == "__main__":
    print(json.dumps(run_live_hyperscaler_hybrid_audit(), indent=2))
