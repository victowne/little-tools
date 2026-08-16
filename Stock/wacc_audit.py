"""Read-only observability and reconciliation for the existing WACC output.

This module does not fetch data and does not choose WACC inputs.  It consumes
the values already selected by the production WACC path, reconciles its
arithmetic, and exposes deterministic driver diagnostics.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math


RECONCILIATION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class WACCDriverSensitivity:
    """Mechanical WACC response to one input moving down/up by a fixed step."""

    driver: str
    lower_input: float
    base_input: float
    upper_input: float
    lower_cost_of_equity: float
    base_cost_of_equity: float
    upper_cost_of_equity: float
    lower_pre_tax_cost_of_debt: float
    base_pre_tax_cost_of_debt: float
    upper_pre_tax_cost_of_debt: float
    lower_wacc: float
    base_wacc: float
    upper_wacc: float


@dataclass(frozen=True)
class WACCAuditResult:
    """Immutable trace of the current production WACC methodology."""

    ticker: str
    issuer_key: str
    security_class: str | None
    available: bool
    reason: str | None
    risk_free_rate: float | None
    risk_free_source: str | None
    risk_free_period: str | None
    beta: float | None
    beta_source: str | None
    beta_observations: int | None
    equity_risk_premium: float | None
    erp_source: str | None
    erp_period: str | None
    cost_of_equity: float | None
    market_cap: float | None
    market_cap_source: str | None
    market_cap_retrieved_at: str | None
    debt_value: float | None
    debt_source: str | None
    debt_period: str | None
    ebit: float | None
    interest_expense: float | None
    interest_period: str | None
    interest_coverage: float | None
    synthetic_spread: float | None
    synthetic_rating: str | None
    pre_tax_cost_of_debt: float | None
    raw_tax_rate: float | None
    tax_rate: float | None
    tax_period: str | None
    after_tax_cost_of_debt: float | None
    equity_weight: float | None
    debt_weight: float | None
    equity_contribution: float | None
    debt_contribution: float | None
    calculated_wacc: float | None
    fallbacks_used: tuple[str, ...]
    warnings: tuple[str, ...]
    beta_sensitivity: WACCDriverSensitivity | None
    erp_sensitivity: WACCDriverSensitivity | None
    risk_free_sensitivity: WACCDriverSensitivity | None


def issuer_normalization_metadata(ticker: str) -> tuple[str, str | None]:
    """Identify security-class metadata without forcing issuer inputs to match."""
    normalized = ticker.strip().upper()
    if normalized == "GOOGL":
        return "ALPHABET_INC", "Class A common stock"
    if normalized == "GOOG":
        return "ALPHABET_INC", "Class C capital stock"
    return normalized, None


def _optional_finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _period_text(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=RECONCILIATION_TOLERANCE,
        abs_tol=RECONCILIATION_TOLERANCE,
    )


def _driver_sensitivity(
    *,
    driver: str,
    lower_input: float,
    base_input: float,
    upper_input: float,
    risk_free: float,
    beta: float,
    erp: float,
    spread: float,
    tax_rate: float,
    equity_weight: float,
    debt_weight: float,
) -> WACCDriverSensitivity:
    def calculate(input_value: float) -> tuple[float, float, float]:
        local_risk_free = input_value if driver == "risk_free_rate" else risk_free
        local_beta = input_value if driver == "beta" else beta
        local_erp = input_value if driver == "equity_risk_premium" else erp
        cost_equity = local_risk_free + local_beta * local_erp
        pre_tax_debt = local_risk_free + spread
        after_tax_debt = pre_tax_debt * (1 - tax_rate)
        wacc = equity_weight * cost_equity + debt_weight * after_tax_debt
        return cost_equity, pre_tax_debt, wacc

    lower_equity, lower_debt, lower_wacc = calculate(lower_input)
    base_equity, base_debt, base_wacc = calculate(base_input)
    upper_equity, upper_debt, upper_wacc = calculate(upper_input)
    return WACCDriverSensitivity(
        driver=driver,
        lower_input=lower_input,
        base_input=base_input,
        upper_input=upper_input,
        lower_cost_of_equity=lower_equity,
        base_cost_of_equity=base_equity,
        upper_cost_of_equity=upper_equity,
        lower_pre_tax_cost_of_debt=lower_debt,
        base_pre_tax_cost_of_debt=base_debt,
        upper_pre_tax_cost_of_debt=upper_debt,
        lower_wacc=lower_wacc,
        base_wacc=base_wacc,
        upper_wacc=upper_wacc,
    )


def build_wacc_audit_result(
    ticker: str,
    production_reference: Mapping,
) -> WACCAuditResult:
    """Reconcile a production WACC result without selecting new inputs."""
    normalized_ticker = ticker.strip().upper()
    issuer_key, security_class = issuer_normalization_metadata(normalized_ticker)
    production_wacc = _optional_finite(production_reference.get("wacc"))
    if production_wacc is None:
        return WACCAuditResult(
            ticker=normalized_ticker,
            issuer_key=issuer_key,
            security_class=security_class,
            available=False,
            reason=str(production_reference.get("error") or "wacc_unavailable"),
            risk_free_rate=None,
            risk_free_source=None,
            risk_free_period=None,
            beta=None,
            beta_source=None,
            beta_observations=None,
            equity_risk_premium=None,
            erp_source=None,
            erp_period=None,
            cost_of_equity=None,
            market_cap=None,
            market_cap_source=None,
            market_cap_retrieved_at=None,
            debt_value=None,
            debt_source=None,
            debt_period=None,
            ebit=None,
            interest_expense=None,
            interest_period=None,
            interest_coverage=None,
            synthetic_spread=None,
            synthetic_rating=None,
            pre_tax_cost_of_debt=None,
            raw_tax_rate=None,
            tax_rate=None,
            tax_period=None,
            after_tax_cost_of_debt=None,
            equity_weight=None,
            debt_weight=None,
            equity_contribution=None,
            debt_contribution=None,
            calculated_wacc=None,
            fallbacks_used=(),
            warnings=("production_wacc_unavailable",),
            beta_sensitivity=None,
            erp_sensitivity=None,
            risk_free_sensitivity=None,
        )

    required_names = (
        "risk_free",
        "beta",
        "erp",
        "cost_equity",
        "pretax_cost_debt",
        "after_tax_cost_debt",
        "tax_rate",
        "equity_weight",
        "debt_weight",
        "market_cap",
        "total_debt",
    )
    values = {name: _optional_finite(production_reference.get(name)) for name in required_names}
    missing = tuple(name for name, value in values.items() if value is None)
    if missing:
        raise ValueError("WACC audit missing production fields: " + ", ".join(missing))

    risk_free = values["risk_free"]
    beta = values["beta"]
    erp = values["erp"]
    tax_rate = values["tax_rate"]
    equity_weight = values["equity_weight"]
    debt_weight = values["debt_weight"]
    cost_equity = risk_free + beta * erp
    spread = _optional_finite(production_reference.get("spread"))
    if spread is None:
        spread = values["pretax_cost_debt"] - risk_free
    pre_tax_debt = risk_free + spread
    after_tax_debt = pre_tax_debt * (1 - tax_rate)
    if not _close(cost_equity, values["cost_equity"]):
        raise ValueError("production cost of equity does not reconcile")
    if not _close(pre_tax_debt, values["pretax_cost_debt"]):
        raise ValueError("production pre-tax cost of debt does not reconcile")
    if not _close(after_tax_debt, values["after_tax_cost_debt"]):
        raise ValueError("production after-tax cost of debt does not reconcile")
    if not _close(equity_weight + debt_weight, 1.0):
        raise ValueError("production capital weights do not sum to one")

    equity_contribution = equity_weight * cost_equity
    debt_contribution = debt_weight * after_tax_debt
    calculated_wacc = equity_contribution + debt_contribution
    if not _close(calculated_wacc, production_wacc):
        raise ValueError("production WACC does not reconcile")

    fallbacks = []
    if production_reference.get("risk_free_fallback_used"):
        fallbacks.append("risk_free_rate_fallback")
    if production_reference.get("erp_fallback_used"):
        fallbacks.append("equity_risk_premium_fallback")
    beta_source = str(production_reference.get("beta_source") or "unknown")
    if beta_source != "five_year_monthly_regression_vs_sp500":
        fallbacks.append(f"beta:{beta_source}")
    if production_reference.get("interest_assumption_used"):
        fallbacks.append("interest_expense_assumed_zero")
    if production_reference.get("tax_assumption_used"):
        fallbacks.append("tax_rate_assumed_21_percent")
    if production_reference.get("tax_rate_clipped"):
        fallbacks.append("effective_tax_rate_clipped_to_0_35")
    if production_reference.get("market_cap_source") == "derived_current_price_times_shares":
        fallbacks.append("market_cap_derived_from_price_and_shares")
    if production_reference.get("total_debt_source") == "yfinance_info_total_debt":
        fallbacks.append("total_debt_metadata_fallback")

    warnings = []
    if values["total_debt"] == 0:
        warnings.append("zero_gross_debt_and_zero_debt_weight")
    if issuer_key == "ALPHABET_INC":
        warnings.append("alphabet_security_class_market_cap_semantics_require_comparison")
    warnings.extend(fallbacks)

    sensitivity_inputs = dict(
        risk_free=risk_free,
        beta=beta,
        erp=erp,
        spread=spread,
        tax_rate=tax_rate,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
    )
    beta_sensitivity = _driver_sensitivity(
        driver="beta",
        lower_input=beta - 0.1,
        base_input=beta,
        upper_input=beta + 0.1,
        **sensitivity_inputs,
    )
    erp_sensitivity = _driver_sensitivity(
        driver="equity_risk_premium",
        lower_input=erp - 0.005,
        base_input=erp,
        upper_input=erp + 0.005,
        **sensitivity_inputs,
    )
    risk_free_sensitivity = _driver_sensitivity(
        driver="risk_free_rate",
        lower_input=risk_free - 0.005,
        base_input=risk_free,
        upper_input=risk_free + 0.005,
        **sensitivity_inputs,
    )

    return WACCAuditResult(
        ticker=normalized_ticker,
        issuer_key=issuer_key,
        security_class=security_class,
        available=True,
        reason=None,
        risk_free_rate=risk_free,
        risk_free_source=str(production_reference.get("risk_free_source") or "unknown"),
        risk_free_period=_period_text(production_reference.get("treasury_date")),
        beta=beta,
        beta_source=beta_source,
        beta_observations=int(production_reference.get("beta_months", 0)),
        equity_risk_premium=erp,
        erp_source=str(production_reference.get("erp_source") or "unknown"),
        erp_period=_period_text(production_reference.get("erp_date")),
        cost_of_equity=cost_equity,
        market_cap=values["market_cap"],
        market_cap_source=production_reference.get("market_cap_source"),
        market_cap_retrieved_at=_period_text(production_reference.get("market_cap_retrieved_at")),
        debt_value=values["total_debt"],
        debt_source=production_reference.get("total_debt_source"),
        debt_period=_period_text(production_reference.get("total_debt_period")),
        ebit=_optional_finite(production_reference.get("ebit")),
        interest_expense=_optional_finite(production_reference.get("interest_expense")),
        interest_period=_period_text(production_reference.get("interest_expense_period")),
        interest_coverage=_optional_finite(production_reference.get("coverage")),
        synthetic_spread=spread,
        synthetic_rating=production_reference.get("rating"),
        pre_tax_cost_of_debt=pre_tax_debt,
        raw_tax_rate=_optional_finite(production_reference.get("raw_tax_rate")),
        tax_rate=tax_rate,
        tax_period=_period_text(production_reference.get("tax_period")),
        after_tax_cost_of_debt=after_tax_debt,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        equity_contribution=equity_contribution,
        debt_contribution=debt_contribution,
        calculated_wacc=calculated_wacc,
        fallbacks_used=tuple(fallbacks),
        warnings=tuple(dict.fromkeys(warnings)),
        beta_sensitivity=beta_sensitivity,
        erp_sensitivity=erp_sensitivity,
        risk_free_sensitivity=risk_free_sensitivity,
    )
