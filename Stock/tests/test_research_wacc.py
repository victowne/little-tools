from dataclasses import FrozenInstanceError

import pytest

from Stock.beta_audit import BetaWACCContext, wacc_from_beta
from Stock.bottom_up_beta import (
    IndustryBetaReference,
    PeerBetaInput,
    build_bottom_up_beta_result,
)
from Stock.research_wacc import build_research_wacc_decision


def context():
    return BetaWACCContext(
        risk_free_rate=0.04, equity_risk_premium=0.05,
        after_tax_cost_of_debt=0.035, equity_weight=0.9, debt_weight=0.1,
    )


def bottom_up(ticker="TEST"):
    peers = tuple(
        PeerBetaInput(
            ticker=name, issuer=name, inclusion_rationale="fixture",
            levered_beta=beta, adjusted_beta=(2 / 3) * beta + 1 / 3,
            beta_method="fixture", market_cap=1000.0, gross_debt=0.0,
            tax_rate=0.20,
        )
        for name, beta in (("A", 0.9), ("B", 1.1), ("C", 1.3))
    )
    return build_bottom_up_beta_result(
        target_ticker=ticker, issuer="TEST_INC", peer_group_name="fixture",
        peer_inputs=peers, target_market_cap=950.0, target_gross_debt=50.0,
        target_tax_rate=0.20,
        industry_references=(IndustryBetaReference(
            industry="Fixture Industry", number_of_firms=10,
            levered_beta=1.2, unlevered_beta=1.05,
            debt_to_equity=0.10, source_date="January 2026",
            mapping_note="fixture",
        ),),
    )


def decision(**overrides):
    values = {
        "ticker": "TEST", "wacc_status": "user_reviewed",
        "research_wacc": 0.09, "formula_based_wacc": 0.095,
        "provisional_default_wacc": 0.085, "wacc_context": context(),
        "cost_of_equity_reference": 0.10, "historical_raw_beta": 1.2,
        "historical_adjusted_beta": 1.133333333333,
        "bottom_up_result": bottom_up(), "rationale": "User-authored reason",
        "created_at": "2026-08-13T12:00:00+00:00",
    }
    values.update(overrides)
    return build_research_wacc_decision(**values)


def test_provisional_decision_is_explicit_and_not_a_beta_selection():
    result = decision(wacc_status="provisional_default", research_wacc=0.085)
    assert result.wacc_status == "provisional_default"
    assert result.research_wacc == pytest.approx(0.085)
    assert result.selected_beta_reference is None
    assert result.selected_beta_value is None


def test_reviewed_decision_preserves_user_value_and_rationale():
    result = decision(research_wacc=0.091, rationale="  Long-horizon judgment  ")
    assert result.wacc_status == "user_reviewed"
    assert result.research_wacc == pytest.approx(0.091)
    assert result.rationale == "Long-horizon judgment"


def test_implied_beta_reconciles_research_wacc():
    result = decision(research_wacc=0.097)
    assert wacc_from_beta(result.research_wacc_implied_beta, context()) == pytest.approx(0.097)


def test_evidence_methods_include_raw_adjusted_bottom_up_and_industry():
    methods = {item.method for item in decision().evidence_methods}
    assert {"Historical Raw", "Historical Adjusted", "Bottom-Up Median", "Bottom-Up Mean"} <= methods
    assert "Damodaran: Fixture Industry" in methods


def test_missing_bottom_up_evidence_remains_available_as_missing():
    result = decision(bottom_up_result=None)
    methods = {item.method for item in result.evidence_methods}
    assert methods == {"Historical Raw", "Historical Adjusted"}
    assert result.bottom_up_beta_median is None
    assert result.bottom_up_beta_mean is None
    assert result.damodaran_beta_references == ()


def test_observed_evidence_range_is_mechanical_minimum_and_maximum():
    result = decision()
    values = [item.formula_based_wacc for item in result.evidence_methods]
    assert result.observed_wacc_minimum == pytest.approx(min(values))
    assert result.observed_wacc_maximum == pytest.approx(max(values))


def test_outside_range_flag_is_informational():
    result = decision(research_wacc=0.06)
    assert "research_wacc_outside_observed_evidence_range" in result.warnings
    assert result.research_wacc == pytest.approx(0.06)


def test_inside_range_has_no_outside_flag():
    baseline = decision()
    midpoint = (baseline.observed_wacc_minimum + baseline.observed_wacc_maximum) / 2
    result = decision(research_wacc=midpoint, formula_based_wacc=midpoint)
    assert "research_wacc_outside_observed_evidence_range" not in result.warnings


def test_material_formula_difference_threshold_is_one_percentage_point():
    flagged = decision(research_wacc=0.085, formula_based_wacc=0.095)
    quiet = decision(research_wacc=0.0851, formula_based_wacc=0.095)
    assert "research_wacc_materially_differs_from_formula_wacc" in flagged.warnings
    assert "research_wacc_materially_differs_from_formula_wacc" not in quiet.warnings


def test_formula_evidence_refresh_does_not_overwrite_research_wacc():
    before = decision(research_wacc=0.091, formula_based_wacc=0.095)
    after = decision(research_wacc=0.091, formula_based_wacc=0.12)
    assert before.research_wacc == after.research_wacc == pytest.approx(0.091)
    assert before.formula_based_wacc != after.formula_based_wacc


def test_difference_is_research_minus_formula_in_percentage_units():
    result = decision(research_wacc=0.09, formula_based_wacc=0.105)
    assert result.research_minus_formula_wacc == pytest.approx(-0.015)


def test_invalid_status_is_rejected():
    with pytest.raises(ValueError, match="invalid_wacc_status"):
        decision(wacc_status="rendered")


def test_decision_is_immutable():
    result = decision()
    with pytest.raises(FrozenInstanceError):
        result.research_wacc = 0.0
    with pytest.raises(FrozenInstanceError):
        result.evidence_methods[0].formula_based_wacc = 0.0


def test_no_recommendation_fields_exist():
    fields = result_fields = decision().__dataclass_fields__
    assert "recommended_wacc" not in fields
    assert "suggested_wacc" not in result_fields
    assert "optimal_wacc" not in fields
    assert "blended_wacc" not in fields
