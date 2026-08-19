"""Pure, immutable human-review workflow for Company Research Profiles.

The workflow only creates a reviewed snapshot.  It never mutates Streamlit
state, fetches evidence, runs a valuation, or applies assumptions to a DCF.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Literal

from Stock.company_profiles import (
    CompanyResearchProfile,
    ResearchAssumption,
    build_multistage_assumptions_from_profile,
)


ReviewGroup = Literal["revenue", "margin", "capital", "tax", "wacc", "terminal"]
REQUIRED_REVIEW_GROUPS: tuple[ReviewGroup, ...] = (
    "revenue", "margin", "capital", "tax", "wacc", "terminal",
)


@dataclass(frozen=True)
class ReviewGroupState:
    group: ReviewGroup
    reviewed: bool
    user_note: str
    candidate_signature: str
    reviewed_at: str | None = None


@dataclass(frozen=True)
class ReviewedCompanyProfileSnapshot:
    profile: CompanyResearchProfile
    reviewed_at: str
    group_reviews: tuple[ReviewGroupState, ...]
    overall_review_note: str
    assumption_signature: str
    evidence_signature: str


@dataclass(frozen=True)
class CompanyProfileReviewState:
    issuer_id: str
    profile_status: Literal["research_in_progress", "reviewed"]
    group_reviews: tuple[ReviewGroupState, ...]
    overall_review_note: str = ""
    reviewed_snapshot: ReviewedCompanyProfileSnapshot | None = None
    reopened_at: str | None = None
    warnings: tuple[str, ...] = ()

    def group(self, name: ReviewGroup) -> ReviewGroupState:
        return next(item for item in self.group_reviews if item.group == name)

    @property
    def incomplete_groups(self) -> tuple[ReviewGroup, ...]:
        return tuple(item.group for item in self.group_reviews if not item.reviewed)

    @property
    def eligible_for_full_review(self) -> bool:
        return not self.incomplete_groups


def _assumption_payload(assumption: ResearchAssumption | None):
    if assumption is None:
        return None
    return {
        "id": assumption.assumption_id,
        "value": assumption.value,
        "rationale": assumption.rationale,
        "evidence_references": assumption.evidence_references,
    }


def _group_payload(profile: CompanyResearchProfile, group: ReviewGroup):
    if group == "revenue":
        revenue = profile.revenue_framework
        return tuple(_assumption_payload(item) for item in (
            revenue.year1_growth, revenue.year2_growth, revenue.year3_growth,
            revenue.revenue_fade_years, profile.forecast_years,
        ))
    if group == "margin":
        return (_assumption_payload(profile.margin_framework.mature_operating_margin),)
    if group == "capital":
        capital = profile.capital_efficiency_framework
        return tuple(_assumption_payload(item) for item in (
            capital.starting_sales_to_capital, capital.mature_sales_to_capital,
        ))
    if group == "tax":
        return (_assumption_payload(profile.operating_tax_rate),)
    if group == "wacc":
        return (_assumption_payload(profile.wacc_framework.research_wacc),)
    if group == "terminal":
        return (_assumption_payload(profile.terminal_framework.terminal_growth),)
    raise ValueError("invalid_review_group")


def _digest(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_group_signature(
    profile: CompanyResearchProfile,
    group: ReviewGroup,
) -> str:
    return _digest(_group_payload(profile, group))


def candidate_assumption_signature(profile: CompanyResearchProfile) -> str:
    return _digest(tuple(
        _group_payload(profile, group) for group in REQUIRED_REVIEW_GROUPS
    ))


def profile_evidence_signature(profile: CompanyResearchProfile) -> str:
    """Ignore retrieval time so a normal rerun does not create false staleness."""
    return _digest(tuple(
        (
            item.evidence_id, item.value, item.unit, item.period, item.source,
            item.source_date, item.analyst_count, item.notes, item.available,
        )
        for item in profile.evidence_items
    ))


def initialize_profile_review(
    profile: CompanyResearchProfile,
) -> CompanyProfileReviewState:
    if profile.profile_status != "research_in_progress":
        raise ValueError("review_requires_research_in_progress_profile")
    groups = tuple(
        ReviewGroupState(
            group, False, "", candidate_group_signature(profile, group), None
        )
        for group in REQUIRED_REVIEW_GROUPS
    )
    return CompanyProfileReviewState(
        profile.issuer_id, "research_in_progress", groups
    )


def set_review_group(
    state: CompanyProfileReviewState,
    profile: CompanyResearchProfile,
    group: ReviewGroup,
    *,
    reviewed: bool,
    user_note: str = "",
    reviewed_at: str | None = None,
) -> CompanyProfileReviewState:
    if state.profile_status == "reviewed":
        raise ValueError("reopen_review_before_editing_review_groups")
    if group not in REQUIRED_REVIEW_GROUPS:
        raise ValueError("invalid_review_group")
    updated = []
    for item in state.group_reviews:
        if item.group != group:
            updated.append(item)
            continue
        updated.append(ReviewGroupState(
            group=group,
            reviewed=bool(reviewed),
            user_note=str(user_note),
            candidate_signature=candidate_group_signature(profile, group),
            reviewed_at=reviewed_at if reviewed else None,
        ))
    return replace(state, group_reviews=tuple(updated))


def set_overall_review_note(
    state: CompanyProfileReviewState,
    note: str,
) -> CompanyProfileReviewState:
    if state.profile_status == "reviewed":
        raise ValueError("reviewed_snapshot_note_is_immutable")
    return replace(state, overall_review_note=str(note))


def reconcile_review_state(
    state: CompanyProfileReviewState,
    current_candidate: CompanyResearchProfile,
) -> CompanyProfileReviewState:
    """Reset changed candidate groups or flag evidence drift after review."""
    if state.issuer_id != current_candidate.issuer_id:
        raise ValueError("review_state_issuer_mismatch")
    warnings = list(state.warnings)
    if state.profile_status == "reviewed":
        snapshot = state.reviewed_snapshot
        if snapshot is None:
            raise ValueError("reviewed_state_requires_snapshot")
        if (
            candidate_assumption_signature(current_candidate)
            != snapshot.assumption_signature
        ):
            warnings.append("review_refresh_recommended")
        if profile_evidence_signature(current_candidate) != snapshot.evidence_signature:
            warnings.append("reviewed_profile_evidence_changed")
        return replace(state, warnings=tuple(dict.fromkeys(warnings)))

    groups = []
    for item in state.group_reviews:
        current_signature = candidate_group_signature(
            current_candidate, item.group
        )
        if item.reviewed and item.candidate_signature != current_signature:
            groups.append(ReviewGroupState(
                item.group, False, item.user_note, current_signature, None
            ))
            warnings.append(f"{item.group}_review_reset_after_candidate_change")
        else:
            groups.append(replace(item, candidate_signature=current_signature))
    return replace(
        state,
        group_reviews=tuple(groups),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _reviewed_assumption(
    assumption: ResearchAssumption | None,
    reviewed_at: str,
) -> ResearchAssumption | None:
    if assumption is None:
        return None
    return replace(
        assumption, status="reviewed", last_reviewed_at=reviewed_at
    )


def _reviewed_profile(
    candidate: CompanyResearchProfile,
    reviewed_at: str,
) -> CompanyResearchProfile:
    revenue = candidate.revenue_framework
    margin = candidate.margin_framework
    capital = candidate.capital_efficiency_framework
    terminal = candidate.terminal_framework
    return replace(
        candidate,
        profile_status="reviewed",
        revenue_framework=replace(
            revenue,
            year1_growth=_reviewed_assumption(revenue.year1_growth, reviewed_at),
            year2_growth=_reviewed_assumption(revenue.year2_growth, reviewed_at),
            year3_growth=_reviewed_assumption(revenue.year3_growth, reviewed_at),
            revenue_fade_years=_reviewed_assumption(
                revenue.revenue_fade_years, reviewed_at
            ),
            terminal_growth=_reviewed_assumption(
                revenue.terminal_growth, reviewed_at
            ),
        ),
        margin_framework=replace(
            margin,
            starting_operating_margin=_reviewed_assumption(
                margin.starting_operating_margin, reviewed_at
            ),
            mature_operating_margin=_reviewed_assumption(
                margin.mature_operating_margin, reviewed_at
            ),
        ),
        capital_efficiency_framework=replace(
            capital,
            starting_sales_to_capital=_reviewed_assumption(
                capital.starting_sales_to_capital, reviewed_at
            ),
            mature_sales_to_capital=_reviewed_assumption(
                capital.mature_sales_to_capital, reviewed_at
            ),
        ),
        wacc_framework=replace(
            candidate.wacc_framework,
            research_wacc=_reviewed_assumption(
                candidate.wacc_framework.research_wacc, reviewed_at
            ),
        ),
        terminal_framework=replace(
            terminal,
            terminal_growth=_reviewed_assumption(
                terminal.terminal_growth, reviewed_at
            ),
            mature_operating_margin=_reviewed_assumption(
                terminal.mature_operating_margin, reviewed_at
            ),
            mature_sales_to_capital=_reviewed_assumption(
                terminal.mature_sales_to_capital, reviewed_at
            ),
        ),
        operating_tax_rate=_reviewed_assumption(
            candidate.operating_tax_rate, reviewed_at
        ),
        forecast_years=_reviewed_assumption(
            candidate.forecast_years, reviewed_at
        ),
        last_reviewed_at=reviewed_at,
    )


def mark_profile_reviewed(
    state: CompanyProfileReviewState,
    candidate: CompanyResearchProfile,
    *,
    reviewed_at: str,
) -> CompanyProfileReviewState:
    if state.profile_status == "reviewed":
        return state
    reconciled = reconcile_review_state(state, candidate)
    if not reconciled.eligible_for_full_review:
        raise ValueError(
            "review_groups_incomplete:" + ",".join(reconciled.incomplete_groups)
        )
    translation = build_multistage_assumptions_from_profile(candidate)
    if not translation.available:
        raise ValueError("research_profile_incomplete_or_invalid")
    timestamp = str(reviewed_at).strip()
    if not timestamp:
        raise ValueError("reviewed_at_required")
    reviewed_profile = _reviewed_profile(candidate, timestamp)
    snapshot = ReviewedCompanyProfileSnapshot(
        profile=reviewed_profile,
        reviewed_at=timestamp,
        group_reviews=reconciled.group_reviews,
        overall_review_note=reconciled.overall_review_note,
        assumption_signature=candidate_assumption_signature(candidate),
        evidence_signature=profile_evidence_signature(candidate),
    )
    return replace(
        reconciled,
        profile_status="reviewed",
        reviewed_snapshot=snapshot,
        reopened_at=None,
    )


def reopen_profile_review(
    state: CompanyProfileReviewState,
    current_candidate: CompanyResearchProfile,
    *,
    reopened_at: str,
) -> CompanyProfileReviewState:
    if state.profile_status != "reviewed" or state.reviewed_snapshot is None:
        raise ValueError("only_reviewed_profile_can_be_reopened")
    groups = tuple(
        ReviewGroupState(
            group, False, state.group(group).user_note,
            candidate_group_signature(current_candidate, group), None,
        )
        for group in REQUIRED_REVIEW_GROUPS
    )
    return CompanyProfileReviewState(
        issuer_id=state.issuer_id,
        profile_status="research_in_progress",
        group_reviews=groups,
        overall_review_note=state.overall_review_note,
        # Preserve the prior immutable reviewed snapshot; it is no longer the
        # active profile but remains auditable until a later review replaces it.
        reviewed_snapshot=state.reviewed_snapshot,
        reopened_at=str(reopened_at),
        warnings=tuple(dict.fromkeys(state.warnings + ("review_reopened",))),
    )
