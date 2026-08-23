"""Deterministic MSFT/META generic Research Profile UI fixture."""

import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.hyperscaler_research import build_meta_research_profile, build_microsoft_research_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_alphabet_research import current_assumptions, history, inputs


ticker = st.selectbox("Ticker", ("MSFT", "META"))
builder = build_microsoft_research_profile if ticker == "MSFT" else build_meta_research_profile
profile_result = builder(current_assumptions(), history(), retrieved_at="2026-08-23")
profile = profile_result.lookup.profile
candidate = build_multistage_assumptions_from_profile(profile).assumptions
base_run = run_multistage_dcf(inputs(ticker), current_assumptions())
candidate_run = run_multistage_dcf(inputs(ticker), candidate)
app.render_company_research_profile(
    profile_result.lookup,
    statement_currency="USD",
    current_assumptions=current_assumptions(),
    current_run=base_run,
    candidate_run=candidate_run,
    hyperscaler_research=profile_result,
    current_price=400.0,
)
