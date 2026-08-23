import inspect
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from Stock.amazon_explicit_reinvestment_transition import (
    FROZEN_MATURE_MARGIN,
    FROZEN_MATURE_SALES_TO_CAPITAL,
    FROZEN_NEAR_TERM_GROWTH,
    _cohort_states,
    apply_explicit_transition,
    assert_frozen_controls,
    build_explicit_transition_path,
    build_frozen_assumptions,
    capex_definition_sensitivity,
    capex_taxonomy,
    frozen_mature_controls,
    historical_capital_evidence,
    run_frozen_standard_model,
    run_transition_case,
    transition_case_specs,
    useful_life_evidence,
)
from Stock.company_profiles import get_company_profile
from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.share_normalization import NormalizedShareCount


def _inputs(ticker="AMZN"):
    period = pd.Timestamp("2026-06-30")
    shares = NormalizedShareCount(
        ticker, 10.8e9, "fixture", period, "consolidated_common", "fixture",
        (), (), True, None,
    )
    return RealCompanyDCFInputs(
        ticker, 775.680e9, "validated_ttm_sec_10q", (period,), 66e9,
        "fixture", period, 10.8e9, shares, .67, .18, True, None, "USD", "USD",
    )


@pytest.fixture
def standard():
    return run_frozen_standard_model(_inputs(), starting_operating_margin=.11155)


def test_frozen_controls_and_growth_are_exact_and_derived_mechanically():
    controls = frozen_mature_controls()
    assumptions = build_frozen_assumptions(.11)
    assert assumptions.near_term_revenue_growth == FROZEN_NEAR_TERM_GROWTH
    assert assumptions.mature_operating_margin == FROZEN_MATURE_MARGIN
    assert assumptions.mature_sales_to_capital == FROZEN_MATURE_SALES_TO_CAPITAL
    assert controls.terminal_roic == pytest.approx(.1834 * .79 * .824)
    assert controls.terminal_reinvestment_rate == pytest.approx(.03 / controls.terminal_roic)


@pytest.mark.parametrize("field,value", [
    ("mature_operating_margin", .20),
    ("mature_sales_to_capital", .90),
    ("operating_tax_rate", .20),
    ("wacc", .10),
    ("terminal_growth", .035),
])
def test_any_frozen_mature_control_change_is_rejected(field, value):
    assumptions = replace(build_frozen_assumptions(.11), **{field: value})
    with pytest.raises(ValueError, match="frozen control changed"):
        assert_frozen_controls(assumptions)


def test_capex_taxonomy_has_all_required_categories_and_one_hybrid_proxy():
    taxonomy = capex_taxonomy()
    assert tuple(row.code for row in taxonomy) == tuple("ABCDEFGHI")
    assert [row.code for row in taxonomy if row.included_in_hybrid_proxy] == ["B"]
    assert next(row for row in taxonomy if row.code == "C").double_counting_risk == "included in B"


def test_cash_and_economic_capex_are_reconciled_but_not_added_twice():
    latest = historical_capital_evidence()[-1]
    assert latest.cash_capex == pytest.approx(
        latest.cash_ppe_purchases - latest.ppe_sale_proceeds_and_incentives
    )
    definitions = dict(capex_definition_sensitivity(latest))
    assert definitions["total_economic_ppe_additions"] == latest.economic_ppe_additions
    assert definitions["cash_plus_finance_lease"] != definitions["total_economic_ppe_additions"]


def test_historical_da_net_capex_and_fcf_identities_hold():
    for row in historical_capital_evidence():
        assert row.net_capex == pytest.approx(row.economic_ppe_additions - row.ppe_depreciation)
        assert row.free_cash_flow == pytest.approx(row.operating_cash_flow - row.cash_capex)
        assert 0 < row.depreciation_to_capex < 1


def test_useful_life_evidence_separates_major_asset_classes():
    evidence = useful_life_evidence()
    assert {row.asset_class for row in evidence} == {
        "Servers and networking", "Heavy equipment", "Other equipment", "Buildings"
    }
    assert all(row.depreciation_method == "straight-line" for row in evidence)


@pytest.mark.parametrize("spec", transition_case_specs())
def test_utilization_and_placement_ramps_are_monotonic(spec):
    assert tuple(sorted(spec.utilization_ramp)) == spec.utilization_ramp
    assert tuple(sorted(spec.placed_in_service_ramp)) == spec.placed_in_service_ramp
    assert spec.utilization_ramp[-1] == 1
    assert spec.placed_in_service_ramp[-1] == 1


def test_capacity_cohort_activation_and_depreciation_are_linked():
    spec = transition_case_specs()[1]
    capex = (100.0,) * 5
    states = _cohort_states(capex, 5, spec)
    oldest, newest = states[0], states[-1]
    assert oldest.utilization_fraction == 1
    assert newest.utilization_fraction < oldest.utilization_fraction
    assert oldest.depreciation > newest.depreciation
    assert all(row.utilized_capital <= row.installed_capital for row in states)


@pytest.mark.parametrize("spec", transition_case_specs())
def test_transition_year_identities_and_operating_economics_are_preserved(standard, spec):
    rows = build_explicit_transition_path(
        standard, spec, starting_ppe_depreciation=49.741e9
    )
    for actual, original in zip(rows, standard.operating_forecast.years):
        assert actual.net_capex == pytest.approx(actual.capex - actual.depreciation_amortization)
        assert actual.total_reinvestment == pytest.approx(
            actual.net_capex + actual.change_in_working_capital + actual.other_reinvestment
        )
        assert actual.fcff == pytest.approx(actual.nopat - actual.total_reinvestment)
        assert actual.revenue == pytest.approx(original.revenue)
        assert actual.operating_margin == pytest.approx(original.operating_margin)
        assert actual.nopat == pytest.approx(original.nopat)


def test_central_da_catches_up_without_exceeding_capex(standard):
    rows = build_explicit_transition_path(
        standard, transition_case_specs()[1], starting_ppe_depreciation=49.741e9
    )
    assert rows[-1].depreciation_to_capex > rows[0].depreciation_to_capex
    assert all(row.depreciation_amortization < row.capex for row in rows)


@pytest.mark.parametrize("handoff_years", (1, 2, 3))
def test_handoff_replaces_not_adds_sc_reinvestment_and_converges(standard, handoff_years):
    explicit = build_explicit_transition_path(
        standard, transition_case_specs()[1], starting_ppe_depreciation=49.741e9
    )
    result = apply_explicit_transition(
        standard, explicit, handoff_years=handoff_years, model_name="fixture"
    )
    assert len(result.handoff) == handoff_years
    assert result.handoff[-1].handoff_weight_sales_to_capital == 1
    ending = result.handoff[-1]
    original = standard.operating_forecast.years[ending.year - 1]
    assert ending.reinvestment == pytest.approx(original.reinvestment)
    assert ending.implied_sales_to_capital == pytest.approx(original.sales_to_capital)
    assert result.run.terminal_value.mature_sales_to_capital == FROZEN_MATURE_SALES_TO_CAPITAL


def test_three_year_handoff_is_smoother_than_direct_switch(standard):
    spec = transition_case_specs()[1]
    direct = run_transition_case(
        standard, spec, starting_ppe_depreciation=49.741e9, handoff_years=1
    )
    smooth = run_transition_case(
        standard, spec, starting_ppe_depreciation=49.741e9, handoff_years=3
    )
    assert smooth.handoff[0].fcff_change_ratio < direct.handoff[0].fcff_change_ratio


def test_no_terminal_double_counting_and_years_after_handoff_revert_to_sc(standard):
    result = run_transition_case(
        standard, transition_case_specs()[1], starting_ppe_depreciation=49.741e9
    )
    first_after_handoff = 5 + transition_case_specs()[1].handoff_years
    for actual, original in zip(
        result.run.operating_forecast.years[first_after_handoff:],
        standard.operating_forecast.years[first_after_handoff:],
    ):
        assert actual.reinvestment == pytest.approx(original.reinvestment)
    assert result.run.terminal_value.terminal_reinvestment == pytest.approx(
        standard.terminal_value.terminal_reinvestment
    )
    final = result.run.operating_forecast.years[-1]
    assert final.sales_to_capital == FROZEN_MATURE_SALES_TO_CAPITAL
    assert abs(
        final.reinvestment / final.nopat
        - frozen_mature_controls().terminal_reinvestment_rate
    ) < .01


def test_case_order_changes_only_transition_not_fixed_forecast(standard):
    runs = tuple(run_transition_case(
        standard, spec, starting_ppe_depreciation=49.741e9
    ) for spec in transition_case_specs())
    for result in runs:
        assert result.run.assumptions == standard.assumptions
        for actual, original in zip(
            result.run.operating_forecast.years,
            standard.operating_forecast.years,
        ):
            assert actual.revenue == pytest.approx(original.revenue)
            assert actual.nopat == pytest.approx(original.nopat)
    assert runs[0].cumulative_explicit_reinvestment > runs[-1].cumulative_explicit_reinvestment


def test_market_price_is_excluded_and_modules_are_pure():
    signature = str(inspect.signature(run_transition_case))
    source = Path("Stock/amazon_explicit_reinvestment_transition.py").read_text("utf-8")
    assert "market_price" not in signature
    assert "import streamlit" not in source
    assert "import yfinance" not in source


def test_profiles_and_other_issuers_remain_unchanged():
    before = {ticker: get_company_profile(ticker) for ticker in ("GOOGL", "META", "MSFT")}
    assert get_company_profile("AMZN").available is True
    assert {ticker: get_company_profile(ticker) for ticker in before} == before


def test_wrong_issuer_is_rejected():
    other = run_frozen_standard_model(_inputs("GOOGL"), starting_operating_margin=.30)
    with pytest.raises(ValueError, match="AMZN"):
        build_explicit_transition_path(
            other, transition_case_specs()[1], starting_ppe_depreciation=49.741e9
        )
