"""Pure transaction plan for one-click Company Profile Review & Apply."""

from dataclasses import dataclass

from Stock.company_profile_application import (
    ReviewedProfileApplication,
    build_profile_apply_plan,
    create_reviewed_profile_application,
)
from Stock.company_profile_review import (
    CompanyProfileReviewState,
    ReviewedCompanyProfileSnapshot,
    review_complete_profile_one_click,
)
from Stock.company_profiles import CompanyResearchProfile
from Stock.valuation import MultiStageDCFAssumptions


@dataclass(frozen=True)
class OneClickReviewApplyResult:
    review_state: CompanyProfileReviewState
    reviewed_snapshot: ReviewedCompanyProfileSnapshot
    application: ReviewedProfileApplication
    assumptions: MultiStageDCFAssumptions
    reused_existing_snapshot: bool


def build_one_click_review_apply(
    candidate: CompanyResearchProfile,
    current_base: MultiStageDCFAssumptions,
    *,
    reviewed_at: str,
    applied_at: str,
    previous_review_state: CompanyProfileReviewState | None = None,
    previous_application: ReviewedProfileApplication | None = None,
    preview_validated: bool,
) -> OneClickReviewApplyResult:
    """Validate, snapshot, then apply the exact snapshot as one pure plan."""
    if not preview_validated:
        raise ValueError("candidate_dcf_preview_unavailable")
    previous_snapshot = (
        previous_review_state.reviewed_snapshot
        if previous_review_state is not None else None
    )
    review_state = review_complete_profile_one_click(
        candidate,
        reviewed_at=reviewed_at,
        previous_state=previous_review_state,
    )
    snapshot = review_state.reviewed_snapshot
    if snapshot is None:
        raise RuntimeError("one_click_review_did_not_create_snapshot")
    plan = build_profile_apply_plan(
        snapshot, current_base, previous_application=previous_application
    )
    application = create_reviewed_profile_application(
        plan, applied_at=applied_at
    )
    return OneClickReviewApplyResult(
        review_state, snapshot, application, application.assumptions,
        previous_snapshot is snapshot,
    )
