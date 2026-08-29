"""Deterministic Streamlit fixture for scenario editor AppTest coverage."""

import pandas as pd
import streamlit as st

from Stock import stock_valuation_mvp as app
from Stock.fundamentals import (
    OPERATING_MARGIN,
    OPERATING_TAX_RATE,
    FundamentalHistory,
    HistoricalDCFAnchors,
    TTMResult,
)
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation_sensitivity import build_wacc_terminal_growth_sensitivity
from Stock.valuation_support import FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED


def fixture_history(ticker: str) -> FundamentalHistory:
    margin = 0.64 if ticker == "NVDA" else 0.3311
    tax = 0.16 if ticker == "NVDA" else 0.17
    annual = pd.DataFrame(
        {OPERATING_MARGIN: [margin], OPERATING_TAX_RATE: [tax]},
        index=[pd.Timestamp("2025-12-31")],
    )
    return FundamentalHistory(
        annual=annual,
        ttm={
            OPERATING_MARGIN: TTMResult(
                margin, True,
                tuple(pd.to_datetime([
                    "2025-03-31", "2025-06-30",
                    "2025-09-30", "2025-12-31",
                ])),
                None,
            )
        },
        annual_reasons=pd.DataFrame(index=annual.index),
        dcf_anchors=HistoricalDCFAnchors(),
    )


def fixture_inputs(ticker: str) -> RealCompanyDCFInputs:
    shares = 10.0
    normalized = NormalizedShareCount(
        ticker=ticker, shares_outstanding=shares, source="fixture",
        source_period=pd.Timestamp("2025-12-31"),
        scope="consolidated_common", method="fixture", components=(),
        warnings=("multi_class_issuer",) if ticker in {"GOOG", "GOOGL"} else (),
        available=True, reason=None,
    )
    return RealCompanyDCFInputs(
        ticker=ticker, starting_revenue=100.0,
        starting_revenue_source="ttm", starting_revenue_periods=(),
        net_debt=5.0, net_debt_source="fixture", net_debt_period=None,
        shares_outstanding=shares, normalized_share_count=normalized,
        historical_sales_to_capital_3y=1.0,
        current_accounting_roic=0.30,
        per_security_valuation_supported=ticker != "TSM",
        per_security_valuation_unsupported_reason=(
            FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED
            if ticker == "TSM" else None
        ),
        statement_currency="TWD" if ticker == "TSM" else "USD",
        security_currency="USD",
    )


ticker = st.selectbox("Fixture Ticker", ("NVDA", "GOOGL", "GOOG", "TSM"))
history = fixture_history(ticker)
base_values = app.initialize_multistage_session_state(
    st.session_state, ticker, history
)
base_prefix = f"multistage_{ticker}_"
base_values["year_1_growth"] = st.number_input(
    "Fixture Base Y1 Growth (%)",
    key=base_prefix + "year_1_growth",
)
base = app.build_multistage_assumptions_from_ui(
    base_values
)
base_run = run_multistage_dcf(fixture_inputs(ticker), base)
st.metric(
    "Main Multi-Stage DCF Base",
    (
        f"${base_run.per_share_value.intrinsic_value_per_share:.6f}"
        if base_run.per_share_value is not None else "N/A"
    ),
)
if base_run.per_share_value is None:
    st.warning(
        app._per_security_unavailable_message(
            base_run.per_share_unavailable_reason
        )
    )
app.render_scenario_analysis(
    ticker, history, base, base_run, base_run.inputs.statement_currency
)
app.render_multistage_sensitivity(
    base_run,
    base,
    build_wacc_terminal_growth_sensitivity(base_run.inputs, base),
)
