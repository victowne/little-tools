from dataclasses import FrozenInstanceError, replace

import pytest

from Stock.company_profiles import (
    BusinessContext,
    CompanyResearchProfile,
    ResearchAssumption,
    ResearchEvidenceItem,
    build_multistage_assumptions_from_profile,
    build_provisional_company_profile,
    compare_research_assumption_to_evidence,
    get_company_profile,
    normalize_profile_issuer,
)
from Stock.valuation import MultiStageDCFAssumptions


def assumptions(**overrides):
    values = {
        "forecast_years": 10,
        "near_term_revenue_growth": (0.30, 0.25, 0.20),
        "revenue_fade_years": 7,
        "terminal_growth": 0.035,
        "starting_operating_margin": 0.64,
        "mature_operating_margin": 0.40,
        "starting_sales_to_capital": 1.5,
        "mature_sales_to_capital": 1.2,
        "operating_tax_rate": 0.16,
        "wacc": 0.09,
    }
    values.update(overrides)
    return MultiStageDCFAssumptions(**values)


def complete_profile(ticker="NVDA"):
    result = build_provisional_company_profile(ticker, assumptions())
    assert result.available
    return result.profile


def test_evidence_item_is_immutable():
    item = ResearchEvidenceItem(
        "consensus", "forward_consensus", "Consensus growth", 0.40,
        "ratio", "2027-01-31", "fixture",
    )
    with pytest.raises(FrozenInstanceError):
        item.value = 0.50


def test_company_profile_is_immutable():
    profile = complete_profile()
    with pytest.raises(FrozenInstanceError):
        profile.profile_status = "reviewed"


def test_provisional_skeleton_is_explicit_and_not_reviewed():
    profile = complete_profile()
    assert profile.profile_status == "provisional"
    assert "profile_contains_provisional_assumptions" in profile.warnings
    assert profile.revenue_framework.year1_growth.status == "provisional"


def test_reviewed_status_requires_explicit_profile_construction():
    provisional = complete_profile()
    reviewed = replace(
        provisional, profile_status="reviewed",
        last_reviewed_at="2026-08-16T12:00:00Z",
    )
    assert provisional.profile_status == "provisional"
    assert reviewed.profile_status == "reviewed"
    assert reviewed.last_reviewed_at == "2026-08-16T12:00:00Z"


def test_invalid_profile_and_assumption_statuses_are_rejected():
    with pytest.raises(ValueError, match="invalid_research_assumption_status"):
        ResearchAssumption("growth", 0.2, "automatic")
    with pytest.raises(ValueError, match="invalid_company_profile_status"):
        replace(complete_profile(), profile_status="automatic")


def test_incomplete_skeleton_returns_explicit_translation_reason():
    profile = get_company_profile("NVDA").profile
    result = build_multistage_assumptions_from_profile(profile)

    assert not result.available
    assert result.assumptions is None
    assert result.reason == "research_profile_incomplete"
    assert "revenue_framework.year1_growth" in result.missing_fields
    assert "operating_tax_rate" in result.missing_fields


def test_complete_profile_translates_exact_research_assumptions():
    profile = complete_profile()
    result = build_multistage_assumptions_from_profile(profile)

    assert result.available
    assert result.profile_status == "provisional"
    assert result.assumptions == assumptions()


def test_evidence_does_not_overwrite_research_growth():
    profile = complete_profile()
    evidence = ResearchEvidenceItem(
        "consensus_y1", "forward_consensus", "Consensus Y1", 0.80,
        "ratio", "2027-01-31", "fixture",
    )
    changed_revenue = replace(profile.revenue_framework, latest_annual_growth=evidence)
    changed_profile = replace(profile, revenue_framework=changed_revenue)

    translated = build_multistage_assumptions_from_profile(changed_profile)
    assert translated.assumptions.near_term_revenue_growth[0] == pytest.approx(0.30)


def test_research_wacc_tax_and_horizon_map_without_inference():
    profile = complete_profile()
    profile = replace(
        profile,
        wacc_framework=replace(
            profile.wacc_framework,
            research_wacc=ResearchAssumption("research_wacc", 0.105, "reviewed"),
        ),
        operating_tax_rate=ResearchAssumption(
            "operating_tax_rate", 0.19, "reviewed"
        ),
        forecast_years=ResearchAssumption("forecast_years", 12, "reviewed"),
        revenue_framework=replace(
            profile.revenue_framework,
            revenue_fade_years=ResearchAssumption(
                "revenue_fade_years", 9, "reviewed"
            ),
        ),
    )

    translated = build_multistage_assumptions_from_profile(profile).assumptions
    assert translated.wacc == pytest.approx(0.105)
    assert translated.operating_tax_rate == pytest.approx(0.19)
    assert translated.forecast_years == 12


def test_existing_dcf_validation_rejects_invalid_profile_assumptions():
    profile = complete_profile()
    bad_wacc = ResearchAssumption("research_wacc", 0.02, "reviewed")
    profile = replace(
        profile,
        wacc_framework=replace(profile.wacc_framework, research_wacc=bad_wacc),
    )
    result = build_multistage_assumptions_from_profile(profile)

    assert not result.available
    assert result.reason == "invalid_research_assumptions"
    assert any("wacc must be greater" in warning for warning in result.warnings)


def test_profile_warnings_propagate_through_translation():
    profile = replace(
        complete_profile(), warnings=("historical_anchor_unstable",)
    )
    result = build_multistage_assumptions_from_profile(profile)
    assert "historical_anchor_unstable" in result.warnings


def test_comparison_helper_returns_delta_without_judgment_language():
    comparison = compare_research_assumption_to_evidence(
        "year1_growth",
        ResearchAssumption("year1_growth", 0.30, "research_in_progress"),
        ResearchEvidenceItem(
            "consensus", "forward_consensus", "Consensus Y1", 0.40,
            "ratio", "2027-12-31", "fixture",
        ),
        unit="percentage_points",
    )
    assert comparison.available
    assert comparison.delta == pytest.approx(-0.10)
    rendered = repr(comparison).lower()
    assert "wrong" not in rendered
    assert "too low" not in rendered
    assert "bearish" not in rendered


def test_comparison_helper_preserves_unavailable_state():
    comparison = compare_research_assumption_to_evidence(
        "research_wacc", None, None, unit="percentage_points"
    )
    assert not comparison.available
    assert comparison.delta is None
    assert comparison.reason == "comparison_inputs_unavailable"


@pytest.mark.parametrize(
    ("metric", "research", "evidence", "expected", "unit"),
    [
        ("starting_margin", 0.30, 0.35, -0.05, "percentage_points"),
        ("starting_sales_to_capital", 1.2, 1.5, -0.3, "multiple"),
        ("research_wacc", 0.09, 0.085, 0.005, "percentage_points"),
    ],
)
def test_margin_capital_and_wacc_comparisons_are_descriptive(
    metric, research, evidence, expected, unit
):
    result = compare_research_assumption_to_evidence(
        metric,
        ResearchAssumption(metric, research, "reviewed"),
        ResearchEvidenceItem(
            f"{metric}_evidence", "historical_financial", metric,
            evidence, "ratio", "2025-12-31", "fixture",
        ),
        unit=unit,
    )
    assert result.delta == pytest.approx(expected)
    assert result.unit == unit


def test_goog_and_googl_resolve_to_same_issuer_profile():
    goog = get_company_profile("GOOG")
    googl = get_company_profile("GOOGL")
    assert normalize_profile_issuer("GOOG") == "ALPHABET_INC"
    assert normalize_profile_issuer("GOOGL") == "ALPHABET_INC"
    assert goog == googl
    assert goog.profile.issuer_id == "ALPHABET_INC"


def test_nvda_and_new_hyperscalers_resolve_and_unknown_is_explicit():
    nvda = get_company_profile("NVDA")
    microsoft = get_company_profile("MSFT")
    meta = get_company_profile("META")
    amazon = get_company_profile("AMZN")
    unknown = get_company_profile("TSLA")
    assert nvda.available
    assert nvda.profile.issuer_id == "NVDA"
    assert microsoft.available and microsoft.profile.issuer_id == "MSFT"
    assert meta.available and meta.profile.issuer_id == "META"
    assert amazon.available and amazon.profile.issuer_id == "AMZN"
    assert not unknown.available
    assert unknown.profile is None
    assert unknown.reason == "profile_unavailable"


def test_business_context_supports_structural_distortion_notes():
    context = BusinessContext(
        cyclicality_notes=("Memory-cycle margins may be economically unstable.",),
        capital_intensity_notes=("Acquisition goodwill affects capital anchors.",),
    )
    assert context.cyclicality_notes
    assert context.capital_intensity_notes
