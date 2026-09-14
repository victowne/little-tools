"""Offline real-widget regression for per-ticker Base persistence."""

from dataclasses import replace

import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_nvda_research import history, inputs, research
from Stock.unified_company_research import build_broadcom_research_profile


ticker = st.selectbox("Ticker", ["NVDA", "AVGO"])
values = app.initialize_multistage_session_state(st.session_state, ticker, history())
current = app.build_multistage_assumptions_from_ui(values)
profile = (
    research().lookup.profile if ticker == "NVDA"
    else build_broadcom_research_profile(current, history()).lookup.profile
)
candidate = build_multistage_assumptions_from_profile(profile).assumptions
company_inputs = replace(inputs(), ticker=ticker)
review = app.initialize_profile_review_session_state(st.session_state, ticker, profile)
app.render_one_click_profile_workflow(
    ticker, profile, review, current, run_multistage_dcf(company_inputs, candidate)
)
for name in values:
    key = (
        app.research_wacc_session_keys(ticker)["value"] if name == "wacc"
        else f"multistage_{ticker}_{name}"
    )
    st.number_input(name, key=key)
st.text_area("WACC note", key=app.research_wacc_session_keys(ticker)["rationale"])
st.button(
    "Confirm current Research WACC as reviewed",
    key=app.research_wacc_session_keys(ticker)["status"] + "_confirm",
    on_click=app.mark_research_wacc_reviewed,
    args=(st.session_state, ticker),
)
st.session_state["observed_assumptions"] = current
st.session_state["observed_value"] = run_multistage_dcf(
    company_inputs, current
).per_share_value.intrinsic_value_per_share
