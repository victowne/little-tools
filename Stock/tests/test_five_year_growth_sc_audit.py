from dataclasses import FrozenInstanceError
import inspect

import pandas as pd
import pytest

from Stock.five_year_growth_sc_audit import (
    build_five_year_growth_assumptions,
    build_mature_sc_assumptions,
    build_mature_sc_sensitivity,
    hyperscaler_growth_sc_specs,
    run_five_year_growth_sc_audit,
)
from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions


def baseline():
    return MultiStageDCFAssumptions(
        forecast_years=11,
        near_term_revenue_growth=(.23, .20, .17),
        revenue_fade_years=8,
        terminal_growth=.0325,
        starting_operating_margin=.33,
        mature_operating_margin=.34,
        starting_sales_to_capital=.50,
        mature_sales_to_capital=.70,
        operating_tax_rate=.17,
        wacc=.0975,
    )


def inputs():
    shares = NormalizedShareCount(
        "GOOGL", 12e9, "fixture", pd.Timestamp("2026-06-30"),
        "consolidated_common", "fixture", (), (), True, None,
    )
    return RealCompanyDCFInputs(
        "GOOGL", 445e9, "ttm", (pd.Timestamp("2026-06-30"),),
        -50e9, "fixture", pd.Timestamp("2026-06-30"), 12e9, shares,
        .67, .28, True, None, "USD", "USD",
    )


def spec():
    return hyperscaler_growth_sc_specs()[0]


def test_audit_universe_is_exact_and_alphabet_is_one_issuer():
    specs = hyperscaler_growth_sc_specs()
    assert tuple(item.ticker for item in specs) == ("GOOGL", "META", "MSFT", "AMZN")
    assert len({item.issuer for item in specs}) == 4


def test_explicit_path_has_exact_y1_to_y5_research_values():
    result = build_five_year_growth_assumptions(baseline(), spec().explicit_growth)
    assert result.near_term_revenue_growth == (.23, .20, .20, .18, .16)
    assert result.near_term_years == 5


def test_fade_begins_after_y5_and_total_horizon_is_preserved():
    original = baseline()
    result = build_five_year_growth_assumptions(original, spec().explicit_growth)
    assert result.forecast_years == original.forecast_years == 11
    assert result.revenue_fade_years == 6
    audit = run_five_year_growth_sc_audit(inputs(), original, spec())
    assert audit.growth_only.run.forecast_path.years[4].stage == "near_term"
    assert audit.growth_only.run.forecast_path.years[5].stage == "fade"
    assert audit.growth_only.run.forecast_path.years[-1].revenue_growth == pytest.approx(.0325)


def test_growth_path_requires_five_consecutive_finite_values():
    with pytest.raises(ValueError, match="exactly five"):
        build_five_year_growth_assumptions(baseline(), spec().explicit_growth[:4])
    broken = tuple(
        point if point.year != 4 else type(point)(6, point.growth, point.confidence, point.evidence, point.rationale)
        for point in spec().explicit_growth
    )
    with pytest.raises(ValueError, match="consecutive"):
        build_five_year_growth_assumptions(baseline(), broken)


def test_growth_only_changes_growth_semantics_not_other_assumptions():
    base = baseline()
    audit = run_five_year_growth_sc_audit(inputs(), base, spec())
    shadow = audit.growth_only.assumptions
    assert shadow.starting_operating_margin == base.starting_operating_margin
    assert shadow.mature_operating_margin == base.mature_operating_margin
    assert shadow.starting_sales_to_capital == base.starting_sales_to_capital
    assert shadow.mature_sales_to_capital == base.mature_sales_to_capital
    assert shadow.operating_tax_rate == base.operating_tax_rate
    assert shadow.wacc == base.wacc
    assert shadow.terminal_growth == base.terminal_growth
    assert audit.growth_only.run.forecast_path.operating_margin_path == (
        audit.baseline.run.forecast_path.operating_margin_path
    )
    assert audit.growth_only.run.forecast_path.sales_to_capital_path == (
        audit.baseline.run.forecast_path.sales_to_capital_path
    )


def test_mature_sc_only_changes_one_assumption():
    base = baseline()
    changed = build_mature_sc_assumptions(base, .80)
    assert changed.mature_sales_to_capital == .80
    assert changed.near_term_revenue_growth == base.near_term_revenue_growth
    assert changed.starting_sales_to_capital == base.starting_sales_to_capital
    assert changed.mature_operating_margin == base.mature_operating_margin
    assert changed.wacc == base.wacc
    assert changed.terminal_growth == base.terminal_growth


def test_mature_sc_grid_requires_five_distinct_points():
    with pytest.raises(ValueError, match="at least five"):
        build_mature_sc_sensitivity(inputs(), baseline(), (.6, .7, .8, .9))
    with pytest.raises(ValueError, match="distinct"):
        build_mature_sc_sensitivity(inputs(), baseline(), (.6, .7, .8, .8, .9))


def test_terminal_roic_and_reinvestment_identities_hold_for_grid():
    points = build_mature_sc_sensitivity(inputs(), baseline(), (.6, .7, .8, .9, 1.0))
    for point in points:
        assert point.terminal_roic == pytest.approx(.34 * (1 - .17) * point.mature_sales_to_capital)
        assert point.terminal_reinvestment_rate == pytest.approx(.0325 / point.terminal_roic)
        assert point.terminal_fcff_to_nopat == pytest.approx(1 - point.terminal_reinvestment_rate)


def test_combined_model_uses_growth_path_and_research_mature_sc_only():
    audit = run_five_year_growth_sc_audit(inputs(), baseline(), spec())
    combined = audit.combined.assumptions
    assert combined.near_term_revenue_growth == (.23, .20, .20, .18, .16)
    assert combined.mature_sales_to_capital == .75
    assert combined.starting_sales_to_capital == .50
    assert combined.mature_operating_margin == .34
    assert combined.wacc == .0975
    assert combined.terminal_growth == .0325


def test_baseline_object_and_inputs_are_not_mutated():
    base = baseline()
    company_inputs = inputs()
    before = (base, company_inputs)
    run_five_year_growth_sc_audit(company_inputs, base, spec())
    assert before == (base, company_inputs)


def test_results_and_specs_are_immutable():
    audit = run_five_year_growth_sc_audit(inputs(), baseline(), spec())
    with pytest.raises(FrozenInstanceError):
        audit.ticker = "META"


def test_pure_module_excludes_market_network_ui_and_profile_dependencies():
    import Stock.five_year_growth_sc_audit as module

    source = inspect.getsource(module)
    forbidden = ("streamlit", "yfinance", "requests", "current_price", "CompanyProfile", "session_state")
    assert all(item not in source for item in forbidden)


def test_wrong_company_spec_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        run_five_year_growth_sc_audit(
            inputs(), baseline(), hyperscaler_growth_sc_specs()[1]
        )
