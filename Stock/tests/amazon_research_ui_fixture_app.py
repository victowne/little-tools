"""Deterministic read-only Amazon Research Candidate UI fixture."""

from dataclasses import replace

import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.amazon_research import build_amazon_research_profile, run_amazon_candidate_preview
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_amazon_research import amazon_history, amazon_inputs


values = app.initialize_multistage_session_state(
    st.session_state, "AMZN", amazon_history()
)
current = app.build_multistage_assumptions_from_ui(values)
result = build_amazon_research_profile(
    current, amazon_history(), retrieved_at="2026-08-23"
)
profile = result.lookup.profile
preview = run_amazon_candidate_preview(amazon_inputs(), profile)
result = replace(result, candidate_preview=preview)
application = st.session_state.get(app.base_profile_application_key("AMZN"))
applied_assumptions = application.assumptions if application is not None else current
base_run = run_multistage_dcf(amazon_inputs(), applied_assumptions)
app.render_company_research_profile(
    result.lookup,
    statement_currency="USD",
    current_assumptions=current,
    current_run=base_run,
    candidate_run=preview,
    amazon_research=result,
    current_price=259.0,
)
st.caption("Applied execution strategy: STANDARD_SALES_TO_CAPITAL")
st.caption(
    "Applied intrinsic value: "
    f"${base_run.per_share_value.intrinsic_value_per_share:.6f}"
)
