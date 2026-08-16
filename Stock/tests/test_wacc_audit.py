from dataclasses import FrozenInstanceError

import pytest

from Stock.wacc_audit import (
    build_wacc_audit_result,
    issuer_normalization_metadata,
)


def production_reference(**overrides):
    risk_free = overrides.get("risk_free", 0.04)
    beta = overrides.get("beta", 1.2)
    erp = overrides.get("erp", 0.05)
    spread = overrides.get("spread", 0.01)
    tax_rate = overrides.get("tax_rate", 0.21)
    market_cap = overrides.get("market_cap", 900.0)
    debt = overrides.get("total_debt", 100.0)
    cost_equity = risk_free + beta * erp
    pretax_debt = risk_free + spread
    after_tax_debt = pretax_debt * (1 - tax_rate)
    equity_weight = market_cap / (market_cap + max(debt, 0))
    debt_weight = 1 - equity_weight
    values = {
        "wacc": equity_weight * cost_equity + debt_weight * after_tax_debt,
        "risk_free": risk_free,
        "risk_free_source": "US_Treasury_daily_10_year_par_yield",
        "risk_free_fallback_used": False,
        "treasury_date": "2026-08-12",
        "beta": beta,
        "beta_source": "five_year_monthly_regression_vs_sp500",
        "beta_months": 59,
        "beta_assumption_used": False,
        "erp": erp,
        "erp_source": "Damodaran_US_total_equity_risk_premium_minus_country_risk_premium",
        "erp_fallback_used": False,
        "erp_date": "Damodaran 2026",
        "cost_equity": cost_equity,
        "spread": spread,
        "pretax_cost_debt": pretax_debt,
        "after_tax_cost_debt": after_tax_debt,
        "tax_rate": tax_rate,
        "raw_tax_rate": tax_rate,
        "tax_assumption_used": False,
        "tax_rate_clipped": False,
        "tax_period": "2025-12-31",
        "market_cap": market_cap,
        "market_cap_source": "yfinance_info_market_cap",
        "market_cap_retrieved_at": "2026-08-13T12:00:00+00:00",
        "total_debt": debt,
        "total_debt_source": "annual_balance_total_debt",
        "total_debt_period": "2025-12-31",
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "ebit": 50.0,
        "interest_expense": 2.0,
        "interest_expense_period": "2025-12-31",
        "interest_assumption_used": False,
        "coverage": 25.0,
        "rating": "Aaa/AAA",
        "error": None,
    }
    values.update(overrides)
    return values


def test_capm_reconciles_from_existing_inputs():
    audit = build_wacc_audit_result("TEST", production_reference())

    assert audit.cost_of_equity == pytest.approx(0.04 + 1.2 * 0.05)


def test_debt_cost_and_tax_shield_reconcile():
    audit = build_wacc_audit_result("TEST", production_reference())

    assert audit.pre_tax_cost_of_debt == pytest.approx(0.04 + 0.01)
    assert audit.after_tax_cost_of_debt == pytest.approx(0.05 * (1 - 0.21))


def test_capital_weights_and_wacc_reconcile():
    audit = build_wacc_audit_result("TEST", production_reference())

    assert audit.equity_weight + audit.debt_weight == pytest.approx(1.0)
    assert audit.calculated_wacc == pytest.approx(
        audit.equity_weight * audit.cost_of_equity
        + audit.debt_weight * audit.after_tax_cost_of_debt
    )


def test_zero_debt_is_visible_and_produces_all_equity_weight():
    reference = production_reference(total_debt=0.0)
    audit = build_wacc_audit_result("TEST", reference)

    assert audit.debt_value == 0.0
    assert audit.equity_weight == 1.0
    assert audit.debt_weight == 0.0
    assert "zero_gross_debt_and_zero_debt_weight" in audit.warnings


def test_missing_debt_production_failure_remains_unavailable():
    audit = build_wacc_audit_result(
        "TEST", {"wacc": None, "error": "缺少总债务数据"}
    )

    assert not audit.available
    assert audit.reason == "缺少总债务数据"
    assert audit.calculated_wacc is None


def test_beta_fallback_is_explicit():
    reference = production_reference(
        beta_source="yfinance_metadata_beta_fallback"
    )
    audit = build_wacc_audit_result("TEST", reference)

    assert "beta:yfinance_metadata_beta_fallback" in audit.fallbacks_used


def test_static_beta_fallback_is_explicit():
    reference = production_reference(
        beta=1.0,
        beta_source="static_beta_1.0_fallback",
        cost_equity=0.09,
    )
    reference["wacc"] = (
        reference["equity_weight"] * reference["cost_equity"]
        + reference["debt_weight"] * reference["after_tax_cost_debt"]
    )
    audit = build_wacc_audit_result("TEST", reference)

    assert "beta:static_beta_1.0_fallback" in audit.fallbacks_used


def test_tax_and_interest_fallbacks_are_explicit():
    audit = build_wacc_audit_result(
        "TEST",
        production_reference(
            tax_assumption_used=True,
            interest_expense=None,
            interest_assumption_used=True,
            coverage=float("inf"),
        ),
    )

    assert "tax_rate_assumed_21_percent" in audit.fallbacks_used
    assert "interest_expense_assumed_zero" in audit.fallbacks_used


def test_alphabet_issuer_metadata_keeps_security_classes_distinct():
    assert issuer_normalization_metadata("GOOGL") == (
        "ALPHABET_INC", "Class A common stock"
    )
    assert issuer_normalization_metadata("GOOG") == (
        "ALPHABET_INC", "Class C capital stock"
    )
    googl = build_wacc_audit_result("GOOGL", production_reference())
    goog = build_wacc_audit_result("GOOG", production_reference())

    assert googl.issuer_key == goog.issuer_key
    assert googl.security_class != goog.security_class


def test_equity_and_debt_contributions_sum_to_wacc():
    audit = build_wacc_audit_result("TEST", production_reference())

    assert audit.equity_contribution == pytest.approx(0.9 * 0.10)
    assert audit.debt_contribution == pytest.approx(0.1 * 0.0395)
    assert audit.equity_contribution + audit.debt_contribution == pytest.approx(
        audit.calculated_wacc
    )


def test_beta_plus_minus_point_one_sensitivity():
    sensitivity = build_wacc_audit_result(
        "TEST", production_reference()
    ).beta_sensitivity

    expected_change = 0.9 * 0.05 * 0.1
    assert sensitivity.lower_wacc == pytest.approx(sensitivity.base_wacc - expected_change)
    assert sensitivity.upper_wacc == pytest.approx(sensitivity.base_wacc + expected_change)
    assert sensitivity.base_pre_tax_cost_of_debt == sensitivity.lower_pre_tax_cost_of_debt


def test_erp_plus_minus_fifty_basis_points_sensitivity():
    sensitivity = build_wacc_audit_result(
        "TEST", production_reference()
    ).erp_sensitivity

    expected_change = 0.9 * 1.2 * 0.005
    assert sensitivity.lower_wacc == pytest.approx(sensitivity.base_wacc - expected_change)
    assert sensitivity.upper_wacc == pytest.approx(sensitivity.base_wacc + expected_change)


def test_risk_free_plus_minus_fifty_basis_points_affects_both_costs():
    sensitivity = build_wacc_audit_result(
        "TEST", production_reference()
    ).risk_free_sensitivity

    expected_change = 0.005 * (0.9 + 0.1 * (1 - 0.21))
    assert sensitivity.lower_wacc == pytest.approx(sensitivity.base_wacc - expected_change)
    assert sensitivity.upper_wacc == pytest.approx(sensitivity.base_wacc + expected_change)
    assert sensitivity.upper_pre_tax_cost_of_debt == pytest.approx(0.055)


def test_audit_result_is_immutable():
    audit = build_wacc_audit_result("TEST", production_reference())

    with pytest.raises(FrozenInstanceError):
        audit.calculated_wacc = 0.0
