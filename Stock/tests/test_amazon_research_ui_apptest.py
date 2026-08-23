from pathlib import Path

from streamlit.testing.v1 import AppTest


FIXTURE = Path(__file__).with_name("amazon_research_ui_fixture_app.py")


def test_amazon_candidate_ui_is_unified_standard_sc_and_not_auto_applied():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)
    assert not app.exception
    markdown = " ".join(str(item.value) for item in app.markdown)
    captions = " ".join(str(item.value) for item in app.caption)
    infos = " ".join(str(item.value) for item in app.info)
    assert "Amazon.com, Inc. Research Candidate DCF Preview" in markdown
    assert "HYBRID_EXPLICIT_WITH_HANDOFF" not in infos
    assert "Applied execution strategy: STANDARD_SALES_TO_CAPITAL" in captions
    assert any(item.label == "Review & Apply Research Profile" for item in app.button)
    assert "reviewed_profile_application_AMZN" not in app.session_state


def test_explicit_click_applies_reviewed_standard_sc_profile():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)
    button = next(
        item for item in app.button
        if item.label == "Review & Apply Research Profile"
    )
    button.click().run(timeout=30)
    assert not app.exception
    application = app.session_state["reviewed_profile_application_AMZN"]
    review = app.session_state["company_profile_review_AMZN"]
    assert review.reviewed_snapshot is not None
    assert application.reviewed_at == review.reviewed_snapshot.reviewed_at
    assert application.issuer == "AMZN"
    assert any("Reviewed profile already applied" in item.value for item in app.success)
    captions = " ".join(str(item.value) for item in app.caption)
    assert "Applied execution strategy: STANDARD_SALES_TO_CAPITAL" in captions
