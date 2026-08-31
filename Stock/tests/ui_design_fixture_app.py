import streamlit as st

from Stock.stock_valuation_mvp import (
    inject_research_workstation_theme,
    render_health_checks,
    render_section_navigation,
)

st.set_page_config(page_title="UI Design Fixture", layout="wide")
inject_research_workstation_theme()
st.title("UI Design Fixture")
render_section_navigation()
render_health_checks(
    "TEST",
    [
        {
            "title": "Asset Coverage",
            "rule": "Total Assets > Total Liabilities",
            "status": True,
            "detail": "Assets 10.00B · Liabilities 4.00B",
            "basis": "As of 2026-06-30",
        },
        {
            "title": "Long-Term Debt Burden",
            "rule": "Long-term Debt / Net Income < 4",
            "status": False,
            "detail": "Long-term debt 8.00B · Net income 1.00B · Ratio 8.00x",
            "basis": "TTM as of 2026-06-30",
        },
        {
            "title": "Operating Cash Flow Coverage",
            "rule": "OCF > |Investing CF| and OCF > |Financing CF|",
            "status": None,
            "detail": "Operating, investing or financing cash flow is unavailable",
            "basis": "No comparable period",
        },
    ],
)
