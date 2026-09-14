from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_applied_profile_and_manual_edits_survive_ticker_round_trip():
    app = AppTest.from_file(
        str(Path(__file__).with_name("ticker_switch_fixture_app.py"))
    ).run()

    def apply():
        next(b for b in app.button if b.label == "Review & Apply Research Profile").click().run()
        assert not app.exception

    apply()
    app.button(key="research_wacc_NVDA_status_confirm").click().run()
    assert not app.exception
    app.run()
    assert not app.exception
    assert not app.session_state["research_wacc_NVDA_status_confirm"]
    nvda = app.session_state["observed_assumptions"]
    nvda_value = app.session_state["observed_value"]
    application = app.session_state["reviewed_profile_application_NVDA"]
    app.selectbox[0].select("AVGO").run()
    apply()
    avgo = app.session_state["observed_assumptions"]
    app.selectbox[0].select("NVDA").run()
    assert not app.exception
    assert app.session_state["observed_assumptions"] == nvda
    assert app.session_state["observed_value"] == pytest.approx(nvda_value)
    assert app.session_state["reviewed_profile_application_NVDA"] == application

    app.number_input(key="multistage_NVDA_year_1_growth").set_value(48.0).run()
    app.number_input(key="research_wacc_NVDA_value").set_value(12.0).run()
    app.text_area[0].set_value("My edited WACC rationale").run()
    edited = app.session_state["observed_assumptions"]
    app.selectbox[0].select("AVGO").run()
    assert app.session_state["observed_assumptions"] == avgo
    app.selectbox[0].select("NVDA").run()
    assert not app.exception
    assert app.session_state["observed_assumptions"] == edited
    assert app.text_area[0].value == "My edited WACC rationale"
    assert app.session_state["reviewed_profile_application_NVDA"] == application
