from pathlib import Path

from streamlit.testing.v1 import AppTest


FIXTURE = Path(__file__).with_name("reverse_dcf_ui_fixture_app.py")


def test_reverse_dcf_panel_is_read_only_and_explains_single_variable_semantics():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)

    assert not app.exception
    assert any(
        item.value == "Reverse DCF — Market-Implied Expectations"
        for item in app.header
    )
    captions = " ".join(str(item.value) for item in app.caption)
    assert "Holding all other Research Base assumptions constant" in captions
    assert app.session_state["protected_base"].near_term_revenue_growth == (
        0.20, 0.15, 0.10
    )
    assert not app.button


def test_reverse_dcf_panel_identifies_base_source_and_market_context():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)

    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Reverse DCF Base"] == "Research Candidate"
    assert metrics["Research Base DCF"].startswith("$")
    assert metrics["Market Price"].startswith("$")
    assert metrics["Price / Base DCF"] == "1.15x"
