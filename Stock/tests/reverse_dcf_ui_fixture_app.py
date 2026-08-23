"""Deterministic read-only Reverse DCF UI fixture."""

import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.multistage_integration import run_multistage_dcf
from Stock.reverse_dcf import run_reverse_dcf
from Stock.tests.test_reverse_dcf import assumptions, company_inputs


inputs = company_inputs("NVDA")
base = assumptions()
base_value = run_multistage_dcf(
    inputs, base
).per_share_value.intrinsic_value_per_share
st.session_state.setdefault("protected_base", base)
analysis = run_reverse_dcf(
    inputs,
    base,
    base_value * 1.15,
    ticker="NVDA",
    base_source="Research Candidate",
)
app.render_reverse_dcf(analysis)

