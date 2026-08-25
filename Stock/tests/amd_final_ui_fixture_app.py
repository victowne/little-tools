"""Deterministic AMD final-workstation fixture without live network data."""

from dataclasses import replace
from types import SimpleNamespace

import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.reverse_dcf import run_reverse_dcf
from Stock.tests.test_alphabet_research import current_assumptions
from Stock.tests.test_amazon_research import amazon_history, amazon_inputs
from Stock.unified_company_research import build_amd_research_profile


ticker = "AMD"
history = amazon_history()
current = current_assumptions()
research = build_amd_research_profile(current, history)
profile = research.lookup.profile
candidate = build_multistage_assumptions_from_profile(profile).assumptions
inputs = replace(
    amazon_inputs(),
    ticker=ticker,
    starting_revenue=float(profile.revenue_framework.starting_revenue.value),
)
candidate_run = run_multistage_dcf(inputs, candidate)
market_price = candidate_run.per_share_value.intrinsic_value_per_share * 1.5
snapshot = SimpleNamespace(price=market_price)

st.title("Stock Valuation Research Workstation")
app.render_final_company_header(
    ticker,
    snapshot,
    profile,
    candidate_run,
    "Research Candidate",
    "Research Candidate",
)
app.render_final_research_profile(
    research.lookup,
    ticker=ticker,
    current_assumptions=current,
    candidate_run=candidate_run,
    research_details=research,
)
st.header("Research Base DCF")
st.subheader("Base Valuation")
st.metric(
    "Research Base DCF / Share",
    f"${candidate_run.per_share_value.intrinsic_value_per_share:.2f}",
)
analysis = run_reverse_dcf(
    inputs,
    candidate,
    market_price,
    ticker=ticker,
    base_source="Research Candidate",
)
app.render_reverse_dcf(
    analysis,
    model_risk=profile.model_risk,
    limitations=app.FINAL_MODEL_LIMITATIONS[ticker],
)
app.render_final_evidence(profile, research)
app.render_final_model_limitations(profile)
