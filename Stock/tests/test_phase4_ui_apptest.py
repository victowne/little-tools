from pathlib import Path

from streamlit.testing.v1 import AppTest


FIXTURE = Path(__file__).with_name("one_click_phase4_ui_fixture_app.py")


def _button(app):
    return next(
        item for item in app.button
        if item.label == "Review & Apply Research Profile"
    )


def test_all_phase4_profiles_use_same_panel_without_auto_apply():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)
    for index, ticker in enumerate(("AMZN", "MU", "AAPL", "AVGO", "AMD")):
        if index:
            app.selectbox[0].select(ticker).run(timeout=30)
        assert not app.exception
        assert any(
            ticker in item.value or ticker in str(item.label)
            for item in (*app.caption, *app.expander)
        )
        assert any(
            item.label == "Review & Apply Research Profile"
            for item in app.button
        )
        assert f"reviewed_profile_application_{ticker}" not in app.session_state
        text = " ".join(str(item.value) for item in (*app.info, *app.caption))
        assert "HYBRID_EXPLICIT_WITH_HANDOFF" not in text


def test_each_phase4_profile_can_use_one_click_review_and_apply():
    for ticker in ("AMZN", "MU", "AAPL", "AVGO", "AMD"):
        app = AppTest.from_file(str(FIXTURE)).run(timeout=30)
        if ticker != "AMZN":
            app.selectbox[0].select(ticker).run(timeout=30)
        _button(app).click().run(timeout=30)
        assert not app.exception
        assert f"reviewed_profile_application_{ticker}" in app.session_state
        application = app.session_state[f"reviewed_profile_application_{ticker}"]
        assert application.issuer == ticker


def test_micron_ui_exposes_rolling_period_alignment_without_auto_apply():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)
    app.selectbox[0].select("MU").run(timeout=30)
    assert not app.exception
    assert any(
        item.label == "Micron Forecast-Period Alignment Audit"
        for item in app.expander
    )
    warnings = " ".join(str(item.value) for item in app.warning)
    assert "Old 45% Y1 implied Revenue" in warnings
    assert "Period-alignment error: Yes" in warnings
    assert "reviewed_profile_application_MU" not in app.session_state
