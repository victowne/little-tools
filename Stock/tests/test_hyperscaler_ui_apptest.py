from pathlib import Path

from streamlit.testing.v1 import AppTest


FIXTURE = Path(__file__).with_name("one_click_hyperscaler_ui_fixture_app.py")


def test_msft_and_meta_use_same_profile_panel_without_auto_apply():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)
    assert not app.exception
    markdown = " ".join(str(item.value) for item in app.markdown)
    captions = " ".join(str(item.value) for item in app.caption)
    assert any("Microsoft Corporation Revenue Evidence" in item.label for item in app.expander)
    assert "Mature Sales-to-Capital" in markdown
    assert "Y3 duration" in markdown
    assert "Implied fade growth" in captions
    assert "Post-assumption market diagnostic" in captions
    assert any(item.label == "Review & Apply Research Profile" for item in app.button)
    assert "reviewed_profile_application_MSFT" not in app.session_state

    app.selectbox[0].select("META").run(timeout=30)
    assert not app.exception
    markdown = " ".join(str(item.value) for item in app.markdown)
    assert any("Meta Platforms, Inc. Revenue Evidence" in item.label for item in app.expander)
    assert "reviewed_profile_application_META" not in app.session_state
