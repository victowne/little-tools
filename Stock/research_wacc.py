"""Pure, auditable Research WACC decision evidence.

Research WACC is supplied by the user.  This module never recommends, blends,
optimizes, initializes, or mutates that value; it only reconciles it with
already-calculated evidence.
"""

from dataclasses import dataclass
import math
from typing import Literal

from Stock.beta_audit import (
    BetaWACCContext,
    implied_beta_from_target_wacc,
    wacc_from_beta,
)
from Stock.bottom_up_beta import BottomUpBetaResult, IndustryBetaReference


WACCStatus = Literal["provisional_default", "user_reviewed"]
OUTSIDE_TOLERANCE = 1e-12
MATERIAL_DIFFERENCE_THRESHOLD = 0.01


@dataclass(frozen=True)
class WACCEvidenceMethod:
    method: str
    beta: float
    formula_based_wacc: float
    source: str


@dataclass(frozen=True)
class ResearchWACCDecision:
    ticker: str
    wacc_status: WACCStatus
    research_wacc: float
    formula_based_wacc: float
    provisional_default_wacc: float
    selected_beta_reference: str | None
    selected_beta_value: float | None
    risk_free_rate: float
    equity_risk_premium: float
    cost_of_equity_reference: float
    bottom_up_beta_median: float | None
    bottom_up_beta_mean: float | None
    historical_raw_beta: float
    historical_adjusted_beta: float
    damodaran_beta_references: tuple[IndustryBetaReference, ...]
    evidence_methods: tuple[WACCEvidenceMethod, ...]
    observed_wacc_minimum: float
    observed_wacc_maximum: float
    research_minus_formula_wacc: float
    research_wacc_implied_beta: float | None
    rationale: str
    created_at: str | None
    warnings: tuple[str, ...]


def _required_finite(name: str, value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}_must_be_finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name}_must_be_finite")
    return numeric


def _optional_finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _target_relevered_industry_beta(
    reference: IndustryBetaReference,
    bottom_up: BottomUpBetaResult,
) -> float | None:
    beta = _optional_finite(reference.unlevered_beta)
    debt_to_equity = _optional_finite(bottom_up.target_debt_to_equity)
    tax = _optional_finite(bottom_up.target_tax_rate)
    if beta is None or debt_to_equity is None or tax is None:
        return None
    return beta * (1 + (1 - tax) * debt_to_equity)


def build_research_wacc_decision(
    *,
    ticker: str,
    wacc_status: WACCStatus,
    research_wacc: float,
    formula_based_wacc: float,
    provisional_default_wacc: float,
    wacc_context: BetaWACCContext,
    cost_of_equity_reference: float,
    historical_raw_beta: float,
    historical_adjusted_beta: float,
    bottom_up_result: BottomUpBetaResult | None,
    rationale: str = "",
    created_at: str | None = None,
) -> ResearchWACCDecision:
    """Reconcile a user-controlled WACC with mechanical evidence only."""
    if wacc_status not in {"provisional_default", "user_reviewed"}:
        raise ValueError("invalid_wacc_status")
    research = _required_finite("research_wacc", research_wacc)
    formula = _required_finite("formula_based_wacc", formula_based_wacc)
    provisional = _required_finite("provisional_default_wacc", provisional_default_wacc)
    raw = _required_finite("historical_raw_beta", historical_raw_beta)
    adjusted = _required_finite("historical_adjusted_beta", historical_adjusted_beta)
    if research <= 0 or formula <= 0 or provisional <= 0:
        raise ValueError("wacc_values_must_be_positive")

    candidates = [
        WACCEvidenceMethod(
            "Historical Raw", raw, wacc_from_beta(raw, wacc_context),
            "5Y monthly raw regression beta",
        ),
        WACCEvidenceMethod(
            "Historical Adjusted", adjusted,
            wacc_from_beta(adjusted, wacc_context),
            "Blume-adjusted historical beta",
        ),
    ]
    if bottom_up_result is not None:
        for method, beta in (
            ("Bottom-Up Median", bottom_up_result.relevered_beta_median),
            ("Bottom-Up Mean", bottom_up_result.relevered_beta_mean),
        ):
            numeric = _optional_finite(beta)
            if numeric is not None:
                candidates.append(WACCEvidenceMethod(
                    method, numeric, wacc_from_beta(numeric, wacc_context),
                    "peer unlevered beta relevered to target capital structure",
                ))
        for reference in bottom_up_result.industry_references:
            beta = _target_relevered_industry_beta(reference, bottom_up_result)
            if beta is not None:
                candidates.append(WACCEvidenceMethod(
                    f"Damodaran: {reference.industry}", beta,
                    wacc_from_beta(beta, wacc_context),
                    "industry unlevered beta relevered to target; independent reference",
                ))
    if not candidates:
        raise ValueError("wacc_evidence_unavailable")

    evidence_values = tuple(item.formula_based_wacc for item in candidates)
    minimum, maximum = min(evidence_values), max(evidence_values)
    warnings = []
    if research < minimum - OUTSIDE_TOLERANCE or research > maximum + OUTSIDE_TOLERANCE:
        warnings.append("research_wacc_outside_observed_evidence_range")
    if abs(research - formula) + OUTSIDE_TOLERANCE >= MATERIAL_DIFFERENCE_THRESHOLD:
        warnings.append("research_wacc_materially_differs_from_formula_wacc")

    return ResearchWACCDecision(
        ticker=ticker.strip().upper(), wacc_status=wacc_status,
        research_wacc=research, formula_based_wacc=formula,
        provisional_default_wacc=provisional,
        # Research WACC is not treated as selection of one beta method.
        selected_beta_reference=None, selected_beta_value=None,
        risk_free_rate=wacc_context.risk_free_rate,
        equity_risk_premium=wacc_context.equity_risk_premium,
        cost_of_equity_reference=_required_finite(
            "cost_of_equity_reference", cost_of_equity_reference
        ),
        bottom_up_beta_median=(
            bottom_up_result.relevered_beta_median if bottom_up_result else None
        ),
        bottom_up_beta_mean=(
            bottom_up_result.relevered_beta_mean if bottom_up_result else None
        ),
        historical_raw_beta=raw, historical_adjusted_beta=adjusted,
        damodaran_beta_references=(
            bottom_up_result.industry_references if bottom_up_result else ()
        ),
        evidence_methods=tuple(candidates), observed_wacc_minimum=minimum,
        observed_wacc_maximum=maximum,
        research_minus_formula_wacc=research - formula,
        research_wacc_implied_beta=implied_beta_from_target_wacc(
            research, wacc_context
        ),
        rationale=str(rationale).strip(), created_at=created_at,
        warnings=tuple(warnings),
    )
