from pathlib import Path

from streamlit.testing.v1 import AppTest


FIXTURE_APP = Path(__file__).with_name("nvda_growth_ui_fixture_app.py")


def test_nvda_growth_reassessment_panel_is_read_only_and_renders():
    app = AppTest.from_file(str(FIXTURE_APP)).run(timeout=30)

    assert not app.exception
    assert any(
        item.label == "NVDA Growth Duration & Product-Cycle Reassessment"
        for item in app.expander
    )
    assert any("INSUFFICIENT EVIDENCE" in item.value for item in app.warning)
    assert not any(
        token in button.label.lower()
        for button in app.button
        for token in ("apply", "review", "reapply")
    )
