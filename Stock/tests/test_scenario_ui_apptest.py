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
    assert "apply profile" not in rendered
    assert "buy" not in rendered
    assert "sell" not in rendered


def test_nvda_review_checklist_defaults_incomplete_and_groups_are_independent():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    assert not at.exception
    review_checks = {
        item.key: item for item in at.checkbox
        if str(item.key).startswith("nvda_review_")
    }
    assert set(review_checks) == {
        "nvda_review_revenue_checked",
        "nvda_review_margin_checked",
        "nvda_review_capital_checked",
        "nvda_review_tax_checked",
        "nvda_review_wacc_checked",
        "nvda_review_terminal_checked",
    }
    assert all(not item.value for item in review_checks.values())
    finalize = element_with_key(at.button, "nvda_review_finalize")
    assert finalize.disabled

    review_checks["nvda_review_revenue_checked"].set_value(True).run(timeout=30)
    assert not at.exception
    assert element_with_key(
        at.checkbox, "nvda_review_revenue_checked"
    ).value
    assert not element_with_key(
        at.checkbox, "nvda_review_margin_checked"
    ).value
    assert element_with_key(at.button, "nvda_review_finalize").disabled


def test_nvda_full_review_creates_snapshot_without_changing_base():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    main_before = at.metric[0].value
    element_with_key(
        at.text_area, "nvda_review_revenue_note"
    ).set_value("Revenue evidence and period alignment reviewed.")
    element_with_key(
        at.text_area, "nvda_review_overall_note"
    ).set_value("Accepted the current research profile snapshot.")
    for item in at.checkbox:
        if str(item.key).startswith("nvda_review_"):
            item.set_value(True)
    at.run(timeout=30)
    assert not at.exception
    finalize = element_with_key(at.button, "nvda_review_finalize")
    assert not finalize.disabled

    finalize.click().run(timeout=30)
    assert not at.exception
    assert at.metric[0].value == main_before
    captions = " ".join(str(item.value) for item in at.caption)
    success = " ".join(str(item.value) for item in at.success)
    infos = " ".join(str(item.value) for item in at.info)
    markdown = " ".join(str(item.value) for item in at.markdown)
    assert "Issuer：NVDA · Status：Reviewed 已复核" in captions
    assert "Reviewed at:" in captions
    assert "Status: Reviewed Research Profile" in success
    assert "Review and application are separate actions" in infos
    assert "Reviewed Research DCF Preview" in markdown
    assert any(
        item.label == "Reviewed notes" for item in at.expander
    )
    assert element_with_key(at.button, "nvda_review_reopen")
    assert element_with_key(at.button, "nvda_reviewed_profile_apply").label == (
        "Apply Reviewed NVDA Profile to Base DCF"
    )


def test_nvda_explicit_apply_updates_base_sensitivity_and_scenario_then_reapplies():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    original_base = displayed_number(at.metric[0].value)
    for item in at.checkbox:
        if str(item.key).startswith("nvda_review_"):
            item.set_value(True)
    at.run(timeout=30)
    element_with_key(at.button, "nvda_review_finalize").click().run(timeout=30)

    reviewed_preview = next(
        displayed_number(item.value)
        for item in at.metric[1:]
        if item.label == "Intrinsic Value / Share"
    )
    assert displayed_number(at.metric[0].value) == pytest.approx(original_base)
    apply_button = element_with_key(at.button, "nvda_reviewed_profile_apply")
    assert apply_button.label == "Apply Reviewed NVDA Profile to Base DCF"

    apply_button.click().run(timeout=30)
    assert not at.exception
    applied_base = displayed_number(at.metric[0].value)
    assert applied_base == pytest.approx(reviewed_preview, abs=0.005)
    success = " ".join(str(item.value) for item in at.success)
    captions = " ".join(str(item.value) for item in at.caption)
    assert "Reviewed NVDA Research Profile applied to Current Base DCF" in success
    assert "Applied at:" in captions
    already = element_with_key(at.button, "nvda_reviewed_profile_apply")
    assert already.disabled
    assert already.label == "Reviewed profile already applied"

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
    assert "modified since the reviewed profile was applied" in warnings
    reapply = element_with_key(at.button, "nvda_reviewed_profile_apply")
    assert reapply.label == "Reapply Reviewed NVDA Profile"

    reapply.click().run(timeout=30)
    assert not at.exception
    assert displayed_number(at.metric[0].value) == pytest.approx(
        applied_base, abs=1e-5
    )


def test_nvda_reopen_returns_to_research_without_applying_profile():
    at = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)
    main_before = at.metric[0].value
    for item in at.checkbox:
        if str(item.key).startswith("nvda_review_"):
            item.set_value(True)
    at.run(timeout=30)
    element_with_key(at.button, "nvda_review_finalize").click().run(timeout=30)
    element_with_key(at.button, "nvda_review_reopen").click().run(timeout=30)

    assert not at.exception
    assert at.metric[0].value == main_before
    captions = " ".join(str(item.value) for item in at.caption)
    assert "Issuer：NVDA · Status：Research in progress 研究中" in captions
    assert all(
        not item.value for item in at.checkbox
        if str(item.key).startswith("nvda_review_")
    )
    assert element_with_key(at.button, "nvda_review_finalize").disabled


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
    assert all("Apply" not in str(item.label) for item in at.button)

    at.selectbox[0].set_value("GOOG").run(timeout=30)
    goog_captions = " ".join(str(item.value) for item in at.caption)
    goog_preview = next(
        item.value for item in at.metric[1:]
        if item.label == "Intrinsic Value / Share"
    )
    assert "Issuer：ALPHABET_INC · Status：Research in progress 研究中" in goog_captions
    assert at.metric[0].value == googl_base
    assert goog_preview == googl_preview
    assert all("Apply" not in str(item.label) for item in at.button)


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
