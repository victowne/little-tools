from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


FIXTURE_APP = Path(__file__).with_name("scenario_ui_fixture_app.py")


def element_with_key(elements, key):
    return next(element for element in elements if element.key == key)


def summary_frame(app_test):
    return next(
        item.value for item in app_test.dataframe
        if {"Bear", "Base", "Bull"}.issubset(item.value.columns)
    )


def displayed_number(value):
    return float(str(value).replace("$", "").replace(",", ""))


def test_nvda_scenario_editor_changes_bear_only_and_reconciles_base():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    assert not at.exception
    assert any(
        item.value == "Bear / Base / Bull Scenario Analysis"
        for item in at.header
    )
    before = summary_frame(at).copy()
    main_base = float(at.metric[0].value.replace("$", ""))
    assert displayed_number(
        before.loc["Intrinsic Value / Share", "Base"]
    ) == pytest.approx(
        main_base, abs=0.005
    )

    element_with_key(
        at.number_input, "scenario_NVDA_bear_mature_margin"
    ).set_value(20.0).run(timeout=30)
    assert not at.exception
    after = summary_frame(at)
    assert displayed_number(
        after.loc["Intrinsic Value / Share", "Bear"]
    ) != pytest.approx(
        displayed_number(before.loc["Intrinsic Value / Share", "Bear"])
    )
    assert displayed_number(
        after.loc["Intrinsic Value / Share", "Base"]
    ) == pytest.approx(
        displayed_number(before.loc["Intrinsic Value / Share", "Base"])
    )
    assert displayed_number(
        after.loc["Intrinsic Value / Share", "Bull"]
    ) == pytest.approx(
        displayed_number(before.loc["Intrinsic Value / Share", "Bull"])
    )


def test_alphabet_scenario_state_is_shared_between_googl_and_goog():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    at.selectbox[0].set_value("GOOGL").run(timeout=30)
    assert not at.exception
    bull_key = "scenario_ALPHABET_INC_bull_year_1_growth"
    rationale_key = "scenario_ALPHABET_INC_bull_rationale"
    element_with_key(at.number_input, bull_key).set_value(27.0).run(timeout=30)
    element_with_key(at.text_area, rationale_key).set_value(
        "Shared issuer-level scenario rationale"
    ).run(timeout=30)
    googl_summary = summary_frame(at).copy()

    at.selectbox[0].set_value("GOOG").run(timeout=30)
    assert not at.exception
    assert element_with_key(at.number_input, bull_key).value == 27.0
    assert element_with_key(at.text_area, rationale_key).value == (
        "Shared issuer-level scenario rationale"
    )
    assert summary_frame(at).equals(googl_summary)


def test_scenario_editor_has_no_probability_or_market_price_controls():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    labels = " ".join(
        str(element.label)
        for collection in (at.number_input, at.text_input, at.text_area)
        for element in collection
    ).lower()
    assert "probability" not in labels
    assert "market price" not in labels


def test_tsm_fixture_fails_closed_across_main_scenarios_and_sensitivity():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    at.selectbox[0].set_value("TSM").run(timeout=30)

    assert not at.exception
    assert at.metric[0].value == "N/A"
    warnings = " ".join(str(item.value) for item in at.warning)
    assert "foreign-listing currency / ADR normalization" in warnings

    scenario_summary = summary_frame(at)
    assert all(
        value == "N/A"
        for value in scenario_summary.loc["Intrinsic Value / Share"]
    )
    sensitivity_frame = at.dataframe[-1].value
    assert sensitivity_frame.isna().all().all()
    per_share_metrics = [
        item.value
        for item in at.metric
        if item.label in {
            "Main Multi-Stage DCF Base",
            "Base Value",
            "Minimum Valid Value",
            "Maximum Valid Value",
        }
    ]
    assert per_share_metrics
    assert all(not str(value).startswith("$") for value in per_share_metrics)
    assert "N/A" in per_share_metrics


def test_supported_fixture_still_renders_numeric_main_value():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)

    assert not at.exception
    assert at.metric[0].value.startswith("$")


def test_nvda_company_profile_shows_read_only_research_candidate():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)

    assert not at.exception
    assert any(
        item.value == "Company Research Profile 公司研究档案"
        for item in at.subheader
    )
    captions = " ".join(str(item.value) for item in at.caption)
    infos = " ".join(str(item.value) for item in at.info)
    assert "Issuer：NVDA · Status：Research in progress 研究中" in captions
    assert "read-only and unreviewed" in infos
    assert any(
        "Research Candidate DCF Preview" in item.value
        for item in at.markdown
    )
    assert any(
        item.label == "Intrinsic Value / Share" for item in at.metric
    )
    assert not at.button or all("Apply Profile" not in str(item.label) for item in at.button)


def test_nvda_candidate_does_not_change_main_dcf_value_or_add_recommendations():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    assert not at.exception
    assert at.metric[0].label == "Main Multi-Stage DCF Base"
    assert at.metric[0].value == "$150.672201"
    rendered = " ".join(
        str(item.value) for collection in (
            at.markdown, at.caption, at.info, at.warning
        ) for item in collection
    ).lower()
    assert "buy" not in rendered
    assert "sell" not in rendered


def test_nvda_review_is_one_explicit_action_without_group_checkboxes():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    assert not at.exception
    assert not [
        item for item in at.checkbox
        if str(item.key).startswith("nvda_review_")
    ]
    action = element_with_key(at.button, "one_click_review_apply_NVDA")
    assert action.label == "Review & Apply Research Profile"
    assert not action.disabled
    assert any(
        "Current Base vs Research Candidate" in str(item.value)
        for item in at.markdown
    )


def test_nvda_one_click_creates_snapshot_and_applies_exact_candidate():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    candidate_preview = next(
        displayed_number(item.value)
        for item in at.metric[1:]
        if item.label == "Intrinsic Value / Share"
    )
    element_with_key(at.button, "one_click_review_apply_NVDA").click().run(timeout=30)
    assert not at.exception
    assert displayed_number(at.metric[0].value) == pytest.approx(
        candidate_preview, abs=0.005
    )
    captions = " ".join(str(item.value) for item in at.caption)
    success = " ".join(str(item.value) for item in at.success)
    assert "Reviewed at:" in captions
    assert "Reviewed profile already applied" in success
    snapshot = at.session_state["company_profile_review_NVDA"].reviewed_snapshot
    application = at.session_state["reviewed_profile_application_NVDA"]
    assert snapshot is not None
    assert application.reviewed_at == snapshot.reviewed_at
    assert application.issuer == snapshot.profile.issuer_id


def test_nvda_explicit_apply_updates_base_sensitivity_and_scenario_then_reapplies():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    reviewed_preview = next(
        displayed_number(item.value)
        for item in at.metric[1:]
        if item.label == "Intrinsic Value / Share"
    )
    element_with_key(at.button, "one_click_review_apply_NVDA").click().run(timeout=30)
    assert not at.exception
    applied_base = displayed_number(at.metric[0].value)
    assert applied_base == pytest.approx(reviewed_preview, abs=0.005)
    success = " ".join(str(item.value) for item in at.success)
    captions = " ".join(str(item.value) for item in at.caption)
    assert "Reviewed profile already applied" in success
    assert "Applied at:" in captions

    scenario = summary_frame(at)
    assert displayed_number(
        scenario.loc["Intrinsic Value / Share", "Base"]
    ) == pytest.approx(applied_base, abs=0.005)
    sensitivity_caption = " ".join(str(item.value) for item in at.caption)
    assert "WACC 11.5%" in sensitivity_caption
    assert "Terminal Growth 3.2%" in sensitivity_caption

    element_with_key(
        at.number_input, "multistage_NVDA_year_1_growth"
    ).set_value(48.0).run(timeout=30)
    assert displayed_number(at.metric[0].value) != pytest.approx(applied_base)
    warnings = " ".join(str(item.value) for item in at.warning)
    assert "diverged from the applied Reviewed Profile" in warnings
    reapply = element_with_key(at.button, "one_click_reapply_NVDA")
    assert reapply.label == "Reapply Reviewed Profile"

    reapply.click().run(timeout=30)
    assert not at.exception
    assert displayed_number(at.metric[0].value) == pytest.approx(
        applied_base, abs=1e-5
    )


def test_nvda_applied_state_is_idempotent_on_rerun():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    element_with_key(at.button, "one_click_review_apply_NVDA").click().run(timeout=30)
    application = at.session_state["reviewed_profile_application_NVDA"]
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["reviewed_profile_application_NVDA"].applied_at == (
        application.applied_at
    )
    assert any(
        "Reviewed profile already applied" in str(item.value)
        for item in at.success
    )


def test_alphabet_share_classes_display_one_research_candidate_profile():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    at.selectbox[0].set_value("GOOGL").run(timeout=30)
    googl_captions = " ".join(str(item.value) for item in at.caption)
    googl_markdown = " ".join(str(item.value) for item in at.markdown)
    googl_base = at.metric[0].value
    googl_preview = next(
        item.value for item in at.metric[1:]
        if item.label == "Intrinsic Value / Share"
    )
    assert "Issuer：ALPHABET_INC · Status：Research in progress 研究中" in googl_captions
    assert "Alphabet Research Candidate DCF Preview" in googl_markdown
    assert any(
        item.label == "Alphabet Revenue Evidence and Period Reconciliation"
        for item in at.expander
    )
    assert any(
        item.label == "Alphabet Segment, AI Infrastructure and Capital Context"
        for item in at.expander
    )
    assert any(
        item.label == "Alphabet Growth & Mature Economics Reassessment"
        for item in at.expander
    )
    assert any("Review & Apply" in str(item.label) for item in at.button)

    at.selectbox[0].set_value("GOOG").run(timeout=30)
    goog_captions = " ".join(str(item.value) for item in at.caption)
    goog_preview = next(
        item.value for item in at.metric[1:]
        if item.label == "Intrinsic Value / Share"
    )
    assert "Issuer：ALPHABET_INC · Status：Research in progress 研究中" in goog_captions
    assert at.metric[0].value == googl_base
    assert goog_preview == googl_preview
    assert any("Apply" in str(item.label) for item in at.button)


def test_main_ui_source_contains_no_legacy_simple_dcf_controls():
    import inspect

    from Stock import stock_valuation_mvp as app

    source = inspect.getsource(app.main)
    forbidden = (
        "Simple FCFF DCF",
        "未来N年增长率 Future Growth",
        "🚀 运行估值 Run Valuation",
        "FCFF 投影与折现 Projection & Discount",
        "参数敏感性热力图",
    )
    assert all(label not in source for label in forbidden)
    assert "render_multistage_dcf_panel" in source
