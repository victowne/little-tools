"""Deterministic Alphabet fixture for the generic one-click workflow."""

import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_alphabet_research import history, inputs, research


profile = research("GOOG").lookup.profile
values = app.initialize_multistage_session_state(st.session_state, "GOOG", history())
current = app.build_multistage_assumptions_from_ui(values)
candidate = build_multistage_assumptions_from_profile(profile).assumptions
candidate_run = run_multistage_dcf(inputs("GOOG"), candidate)
review_state = app.initialize_profile_review_session_state(
    st.session_state, "GOOG", profile
)

app.render_one_click_profile_workflow(
    "GOOG", profile, review_state, current, candidate_run
)
st.caption(f"GOOG Base Y1: {current.near_term_revenue_growth[0]:.2%}")
st.caption(f"GOOG sensitivity center WACC: {current.wacc:.2%}")
