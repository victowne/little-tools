from dataclasses import replace

import pandas as pd
import pytest

from Stock.hyperscaler_hybrid_audit import (
    build_hyperscaler_hybrid_inputs,
    hyperscaler_hybrid_research_specs,
)
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions


def _run(ticker: str):
    assumptions = MultiStageDCFAssumptions(
        forecast_years=10,
        near_term_revenue_growth=(0.20, 0.15, 0.10),
        revenue_fade_years=5,
        starting_operating_margin=0.30,
        mature_operating_margin=0.25,
        starting_sales_to_capital=1.5,
        mature_sales_to_capital=1.2,
        operating_tax_rate=0.18,
        wacc=0.10,
        terminal_growth=0.04,
    )
    shares = NormalizedShareCount(
        ticker, 1e9, "fixture", pd.Timestamp("2025-12-31"),
        "consolidated_common", "fixture", (), (), True, None,
    )
    inputs = RealCompanyDCFInputs(
        ticker, 100e9, "ttm", (pd.Timestamp("2025-12-31"),),
        -10e9, "fixture", pd.Timestamp("2025-12-31"), 100.0, shares,
        0.7, 0.2, True, None, "USD", "USD",
    )
    return run_multistage_dcf(inputs, assumptions)


def test_fixed_audit_universe_is_exactly_four_required_hyperscalers():
    specs = hyperscaler_hybrid_research_specs()
    assert tuple(item.ticker for item in specs) == ("GOOGL", "META", "MSFT", "AMZN")
    assert all(item.evidence_as_of.startswith("2026-") for item in specs)


def test_research_spec_builds_exactly_five_years_and_hits_endpoints():
    spec = hyperscaler_hybrid_research_specs()[0]
    inputs = build_hyperscaler_hybrid_inputs(
        _run(spec.ticker), spec, starting_depreciation_to_revenue=0.05
    )
    assert len(inputs) == 5
    assert inputs[0].capex == pytest.approx(spec.year_one_capex_guidance)
    assert inputs[-1].capex / inputs[-1].revenue == pytest.approx(
        spec.normalized_capex_to_revenue
    )
    assert inputs[-1].depreciation_amortization / inputs[-1].revenue == pytest.approx(
        spec.normalized_capex_to_revenue
        * spec.normalized_depreciation_as_capex_share
    )


def test_research_spec_rejects_wrong_ticker_and_negative_depreciation():
    spec = hyperscaler_hybrid_research_specs()[0]
    with pytest.raises(ValueError, match="does not match"):
        build_hyperscaler_hybrid_inputs(
            _run("META"), spec, starting_depreciation_to_revenue=0.05
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        build_hyperscaler_hybrid_inputs(
            _run(spec.ticker), spec, starting_depreciation_to_revenue=-0.01
        )


def test_specs_are_immutable():
    with pytest.raises(Exception):
        replace(hyperscaler_hybrid_research_specs()[0], ticker="META").ticker = "NVDA"
