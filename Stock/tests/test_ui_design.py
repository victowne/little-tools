from pathlib import Path

import pytest
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
    sidebar_markup = " ".join(str(item.value) for item in app.sidebar.markdown)
    assert 'href="#overview"' in sidebar_markup
    assert 'href="#research-profile"' in sidebar_markup
    assert 'href="#operating-health-checks"' in sidebar_markup


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


@pytest.mark.parametrize(
    ("price", "value", "expected"),
    [
        (0.0, 0.0, ("$0.00", "$0.00", "N/A")),
        (None, None, ("N/A", "N/A", "N/A")),
        (200.0, 150.0, ("$200.00", "$150.00", "0.75x")),
    ],
)
def test_compact_header_preserves_values_and_escapes_context(
    price: float | None, value: float | None, expected: tuple[str, str, str]
) -> None:
    app = AppTest.from_string(
        f"""
from types import SimpleNamespace
from Stock.stock_valuation_mvp import render_final_company_header
render_final_company_header(
    "TEST",
    SimpleNamespace(price={price!r}),
    SimpleNamespace(
        company_name="A long company name with multiple business divisions",
        model_risk="High",
    ),
    SimpleNamespace(per_share_value=SimpleNamespace(intrinsic_value_per_share={value!r})),
    "Candidate <unreviewed>",
    "Research Candidate",
)
"""
    ).run(timeout=30)

    assert not app.exception
    assert tuple(item.value for item in app.metric) == expected
    markup = " ".join(str(item.value) for item in app.markdown)
    assert "Candidate &lt;unreviewed&gt;" in markup
    assert "Model risk <strong>High</strong>" in markup
