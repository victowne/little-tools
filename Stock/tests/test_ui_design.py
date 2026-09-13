from pathlib import Path

from streamlit.testing.v1 import AppTest

FIXTURE = Path(__file__).with_name("ui_design_fixture_app.py")


def test_research_theme_and_navigation_render() -> None:
    """Verify the custom theme and research navigation render without errors."""
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)

    assert not app.exception
    markup = " ".join(str(item.value) for item in app.markdown)
    assert "st-key-company_hero" in markup
    assert "--ui-canvas: #eef2f8" in markup
    assert 'class="ui-nav"' in markup
    assert "Reverse DCF" in markup


def test_health_checks_render_as_three_vertical_cards() -> None:
    """Verify health checks use three full-width stacked cards."""
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)

    assert not app.exception
    markup = " ".join(str(item.value) for item in app.markdown)
    assert markup.count('class="health-check-card"') == 3
    assert "grid-template-columns" in markup
    assert "health-check-status--pass" in markup
    assert "health-check-status--review" in markup
    assert "health-check-status--unknown" in markup
