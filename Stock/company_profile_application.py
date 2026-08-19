"""Pure application planning for reviewed Company Research Profiles.

This module validates and compares immutable reviewed snapshots.  It never
imports Streamlit, mutates session state, fetches evidence, or duplicates DCF
calculation logic.
"""

from dataclasses import dataclass
import hashlib
import json
import math

from Stock.company_profile_review import ReviewedCompanyProfileSnapshot
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.valuation import MultiStageDCFAssumptions


APPLICATION_SOURCE = "reviewed_company_profile"
ASSUMPTION_COMPARISON_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ProfileAssumptionChange:
    field: str
    current_value: float | int
    reviewed_value: float | int


@dataclass(frozen=True)
class ReviewedProfileApplication:
    """Immutable provenance for the snapshot explicitly applied by a user."""

    source: str
    issuer: str
    reviewed_at: str
    applied_at: str
    snapshot_fingerprint: str
    assumptions: MultiStageDCFAssumptions


@dataclass(frozen=True)
class ProfileApplyPlan:
    available: bool
    assumptions: MultiStageDCFAssumptions | None
    reviewed_at: str | None
    source: str
    issuer: str | None
    snapshot_fingerprint: str | None
    changed_fields: tuple[ProfileAssumptionChange, ...]
    already_applied: bool
    base_diverged: bool
    newer_review_available: bool
    reason: str | None
    warnings: tuple[str, ...] = ()


def _assumption_items(
    assumptions: MultiStageDCFAssumptions,
) -> tuple[tuple[str, float | int], ...]:
    growth = assumptions.near_term_revenue_growth
    return (
        ("year_1_growth", growth[0]),
        ("year_2_growth", growth[1]),
        ("year_3_growth", growth[2]),
        ("revenue_fade_years", assumptions.revenue_fade_years),
        ("forecast_years", assumptions.forecast_years),
        ("starting_operating_margin", assumptions.starting_operating_margin),
        ("mature_operating_margin", assumptions.mature_operating_margin),
        ("starting_sales_to_capital", assumptions.starting_sales_to_capital),
        ("mature_sales_to_capital", assumptions.mature_sales_to_capital),
        ("operating_tax_rate", assumptions.operating_tax_rate),
        ("research_wacc", assumptions.wacc),
        ("terminal_growth", assumptions.terminal_growth),
    )


def assumptions_fingerprint(assumptions: MultiStageDCFAssumptions) -> str:
    """Hash only the economically relevant, validated DCF assumptions."""
    payload = json.dumps(
        _assumption_items(assumptions),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assumptions_match(
    left: MultiStageDCFAssumptions,
    right: MultiStageDCFAssumptions,
    *,
    tolerance: float = ASSUMPTION_COMPARISON_TOLERANCE,
) -> bool:
    left_items = _assumption_items(left)
    right_items = _assumption_items(right)
    for (left_name, left_value), (right_name, right_value) in zip(
        left_items, right_items
    ):
        if left_name != right_name:
            return False
        if isinstance(left_value, int) and isinstance(right_value, int):
            if left_value != right_value:
                return False
        elif not math.isclose(
            float(left_value), float(right_value), rel_tol=0.0, abs_tol=tolerance
        ):
            return False
    return True


def _changes(
    current: MultiStageDCFAssumptions,
    reviewed: MultiStageDCFAssumptions,
) -> tuple[ProfileAssumptionChange, ...]:
    changes = []
    for (name, current_value), (_, reviewed_value) in zip(
        _assumption_items(current), _assumption_items(reviewed)
    ):
        if isinstance(current_value, int) and isinstance(reviewed_value, int):
            equal = current_value == reviewed_value
        else:
            equal = math.isclose(
                float(current_value), float(reviewed_value), rel_tol=0.0,
                abs_tol=ASSUMPTION_COMPARISON_TOLERANCE,
            )
        if not equal:
            changes.append(ProfileAssumptionChange(
                name, current_value, reviewed_value
            ))
    return tuple(changes)


def build_profile_apply_plan(
    snapshot: ReviewedCompanyProfileSnapshot | None,
    current_base: MultiStageDCFAssumptions,
    *,
    previous_application: ReviewedProfileApplication | None = None,
) -> ProfileApplyPlan:
    """Validate one reviewed snapshot and compare it with the current Base."""
    if snapshot is None:
        return ProfileApplyPlan(
            False, None, None, APPLICATION_SOURCE, None, None, (), False,
            False, False, "reviewed_snapshot_unavailable",
        )
    if snapshot.profile.profile_status != "reviewed":
        return ProfileApplyPlan(
            False, None, snapshot.reviewed_at, APPLICATION_SOURCE,
            snapshot.profile.issuer_id, None, (), False, False, False,
            "profile_not_reviewed",
        )
    translation = build_multistage_assumptions_from_profile(snapshot.profile)
    if not translation.available or translation.assumptions is None:
        return ProfileApplyPlan(
            False, None, snapshot.reviewed_at, APPLICATION_SOURCE,
            snapshot.profile.issuer_id, None, (), False, False, False,
            translation.reason or "reviewed_profile_incomplete_or_invalid",
            tuple(translation.warnings),
        )

    reviewed = translation.assumptions
    fingerprint = assumptions_fingerprint(reviewed)
    current_matches = assumptions_match(current_base, reviewed)
    same_application = (
        previous_application is not None
        and previous_application.issuer == snapshot.profile.issuer_id
        and previous_application.snapshot_fingerprint == fingerprint
    )
    already_applied = same_application and current_matches
    base_diverged = same_application and not current_matches
    newer_review = (
        previous_application is not None
        and previous_application.issuer == snapshot.profile.issuer_id
        and previous_application.snapshot_fingerprint != fingerprint
    )
    warnings = []
    if base_diverged:
        warnings.append("base_diverged_from_reviewed_profile")
    if newer_review:
        warnings.append("newer_reviewed_profile_available")
    return ProfileApplyPlan(
        True, reviewed, snapshot.reviewed_at, APPLICATION_SOURCE,
        snapshot.profile.issuer_id, fingerprint,
        _changes(current_base, reviewed), already_applied, base_diverged,
        newer_review, None, tuple(warnings),
    )


def create_reviewed_profile_application(
    plan: ProfileApplyPlan,
    *,
    applied_at: str,
) -> ReviewedProfileApplication:
    if not plan.available or plan.assumptions is None:
        raise ValueError(plan.reason or "reviewed_profile_not_applicable")
    timestamp = str(applied_at).strip()
    if not timestamp:
        raise ValueError("applied_at_required")
    return ReviewedProfileApplication(
        source=plan.source,
        issuer=plan.issuer or "",
        reviewed_at=plan.reviewed_at or "",
        applied_at=timestamp,
        snapshot_fingerprint=plan.snapshot_fingerprint or "",
        assumptions=plan.assumptions,
    )
