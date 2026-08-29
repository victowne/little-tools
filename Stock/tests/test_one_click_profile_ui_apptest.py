from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


FIXTURE_APP = Path(__file__).with_name("one_click_profile_ui_fixture_app.py")
ALPHABET_FIXTURE_APP = Path(__file__).with_name(
    "one_click_alphabet_ui_fixture_app.py"
)


def _button(app, label):
    return next(item for item in app.button if item.label == label)


def test_one_click_review_apply_and_reapply_end_to_end():
    app = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)

    assert not app.exception
    assert not app.checkbox
    assert any("Current Base vs Research Candidate" in item.value for item in app.markdown)
    _button(app, "Review & Apply Research Profile").click().run(timeout=30)

    assert not app.exception
    assert any("Reviewed profile already applied" in item.value for item in app.success)
    assert any("Scenario Base Y1: 60.00%" in item.value for item in app.caption)
    assert any("Sensitivity center WACC: 11.50%" in item.value for item in app.caption)
    review = app.session_state["company_profile_review_NVDA"]
    application = app.session_state["reviewed_profile_application_NVDA"]
    assert review.reviewed_snapshot is not None
    assert review.reviewed_snapshot.reviewed_at
    assert application.applied_at
    assert application.reviewed_at == review.reviewed_snapshot.reviewed_at

    app.session_state["multistage_NVDA_year_1_growth"] = 50.0
    app.run(timeout=30)
    assert any("diverged" in item.value for item in app.warning)
    _button(app, "Reapply Reviewed Profile").click().run(timeout=30)

    assert not app.exception
    assert app.session_state["multistage_NVDA_year_1_growth"] == pytest.approx(60.0)
    assert any("Reviewed profile already applied" in item.value for item in app.success)


def test_alphabet_one_click_applies_to_current_security_base_and_shared_state():
    app = AppTest.from_file(str(ALPHABET_FIXTURE_APP)).run(timeout=30)

    assert not app.exception
    assert not app.checkbox
    _button(app, "Review & Apply Research Profile").click().run(timeout=30)

    assert not app.exception
    assert app.session_state["multistage_GOOG_year_1_growth"] == pytest.approx(23.0)
    assert app.session_state["research_wacc_ALPHABET_INC_value"] == pytest.approx(9.75)
    assert "company_profile_review_ALPHABET_INC" in app.session_state
    assert "reviewed_profile_application_ALPHABET_INC" in app.session_state
    assert any("GOOG Base Y1: 23.00%" in item.value for item in app.caption)
    assert any(
        "GOOG sensitivity center WACC: 9.75%" in item.value
        for item in app.caption
    )
