"""Deterministic Streamlit fixture for one-click Review & Apply."""

import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_nvda_research import history, inputs, research


profile = research().lookup.profile
values = app.initialize_multistage_session_state(st.session_state, "NVDA", history())
current = app.build_multistage_assumptions_from_ui(values)
candidate = build_multistage_assumptions_from_profile(profile).assumptions
candidate_run = run_multistage_dcf(inputs(), candidate)
review_state = app.initialize_profile_review_session_state(
    st.session_state, "NVDA", profile
)

st.markdown("Research Candidate")
app.render_one_click_profile_workflow(
    "NVDA", profile, review_state, current, candidate_run
)
st.caption(f"Scenario Base Y1: {current.near_term_revenue_growth[0]:.2%}")
st.caption(f"Sensitivity center WACC: {current.wacc:.2%}")
