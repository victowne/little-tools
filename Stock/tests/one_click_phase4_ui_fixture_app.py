"""Deterministic unified-production UI fixture for Phase 4 profiles."""

from dataclasses import replace

import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.amazon_research import build_amazon_research_profile, run_amazon_candidate_preview
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_alphabet_research import current_assumptions
from Stock.tests.test_amazon_research import amazon_history, amazon_inputs
from Stock.unified_company_research import (
    build_amd_research_profile,
    build_apple_research_profile,
    build_broadcom_research_profile,
    build_micron_research_profile,
)


ticker = st.selectbox("Ticker", ("AMZN", "MU", "AAPL", "AVGO", "AMD"))
builders = {
    "MU": build_micron_research_profile,
    "AAPL": build_apple_research_profile,
    "AVGO": build_broadcom_research_profile,
    "AMD": build_amd_research_profile,
}
history = amazon_history()
current = current_assumptions()
inputs = replace(amazon_inputs(), ticker=ticker)
if ticker == "AMZN":
    research = build_amazon_research_profile(current, history)
    profile = research.lookup.profile
    candidate_run = run_amazon_candidate_preview(inputs, profile)
    amazon_research = research
    unified_research = None
else:
    research = builders[ticker](current, history)
    profile = research.lookup.profile
    candidate = build_multistage_assumptions_from_profile(profile).assumptions
    inputs = replace(
        inputs, starting_revenue=float(profile.revenue_framework.starting_revenue.value)
    )
    candidate_run = run_multistage_dcf(inputs, candidate)
    amazon_research = None
    unified_research = research

base_run = run_multistage_dcf(inputs, current)
app.render_company_research_profile(
    research.lookup,
    statement_currency="USD",
    current_assumptions=current,
    current_run=base_run,
    candidate_run=candidate_run,
    amazon_research=amazon_research,
    unified_research=unified_research,
    current_price=200.0,
)
