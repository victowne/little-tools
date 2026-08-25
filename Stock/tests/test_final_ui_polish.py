from pathlib import Path

from streamlit.testing.v1 import AppTest


FIXTURE = Path(__file__).with_name("final_ui_fixture_app.py")
AMD_FIXTURE = Path(__file__).with_name("amd_final_ui_fixture_app.py")


def test_final_information_hierarchy_and_neutral_header():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)

    assert not app.exception
    headings = [item.value for item in (*app.title, *app.header, *app.subheader)]
    for required in (
        "Stock Valuation Research Workstation",
        "Apple Inc. · AAPL",
        "Research Profile",
        "Key Fundamentals",
        "Research Base DCF",
        "Base Valuation",
        "Reverse DCF — Market-Implied Expectations",
        "Evidence & Research Interpretation",
        "Model Limitations",
    ):
        assert required in headings
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Market Price"].startswith("$")
    assert metrics["Research Base DCF"].startswith("$")
    assert metrics["DCF / Market Price"].endswith("x")
    assert "BUY" not in " ".join(headings).upper()
    assert "SELL" not in " ".join(headings).upper()


def test_final_profile_has_one_review_apply_and_no_obsolete_research_ui():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)

    assert not app.exception
    labels = [item.label for item in app.button]
    assert labels.count("Review & Apply Research Profile") == 1
    visible = " ".join(
        str(item.value)
        for item in (*app.info, *app.warning, *app.caption, *app.markdown)
    )
    assert "Research Candidate DCF Preview" not in visible
    assert "Hybrid" not in visible
    assert "Phase 3" not in visible
    assert "Phase 4" not in visible
    assert "Phase 5" not in visible


def test_reverse_caveat_evidence_and_limitations_are_visible():
    app = AppTest.from_file(str(FIXTURE)).run(timeout=30)

    assert not app.exception
    info = " ".join(str(item.value) for item in app.info)
    assert "Each Reverse DCF result is independent" in info
    expanders = [item.label for item in app.expander]
    assert any("Revenue / Growth" in label for label in expanders)
    text = " ".join(str(item.value) for item in (*app.caption, *app.markdown))
    assert "outsourced production" in text


def test_amd_final_workstation_exposes_candidate_reverse_evidence_and_limitations():
    app = AppTest.from_file(str(AMD_FIXTURE)).run(timeout=30)

    assert not app.exception
    headings = [item.value for item in (*app.title, *app.header, *app.subheader)]
    assert "Advanced Micro Devices, Inc. · AMD" in headings
    assert "Research Profile" in headings
    assert "Research Base DCF" in headings
    assert "Reverse DCF — Market-Implied Expectations" in headings
    assert "Evidence & Research Interpretation" in headings
    assert "Model Limitations" in headings
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Profile State"] == "Research Candidate"
    assert metrics["Base Source"] == "Research Candidate"
    assert metrics["Model Risk"] == "High"
    assert sum(
        item.label == "Review & Apply Research Profile" for item in app.button
    ) == 1
    assert "reviewed_profile_application_AMD" not in app.session_state
    visible = " ".join(
        str(item.value)
        for item in (*app.info, *app.warning, *app.caption, *app.markdown)
    )
    assert "GPU deployment timing" in visible
