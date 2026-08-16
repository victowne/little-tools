from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


FIXTURE_APP = Path(__file__).with_name("scenario_ui_fixture_app.py")


def element_with_key(elements, key):
    return next(element for element in elements if element.key == key)


def summary_frame(app_test):
    return app_test.dataframe[0].value


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
