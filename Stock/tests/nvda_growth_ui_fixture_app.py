"""Deterministic Streamlit fixture for the read-only NVDA reassessment panel."""

import pandas as pd

from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.nvda_growth_reassessment import (
    ConsensusRevenuePoint,
    build_nvda_growth_reassessment,
    compare_growth_duration_dcf,
)
from Stock.share_normalization import NormalizedShareCount
from Stock.stock_valuation_mvp import render_nvda_growth_duration_reassessment
from Stock.valuation import MultiStageDCFAssumptions


assumptions = MultiStageDCFAssumptions(
    forecast_years=12,
    near_term_revenue_growth=(0.55, 0.40, 0.25),
    revenue_fade_years=9,
    terminal_growth=0.0325,
    starting_operating_margin=0.6402,
    mature_operating_margin=0.45,
    starting_sales_to_capital=1.35,
    mature_sales_to_capital=1.00,
    operating_tax_rate=0.17,
    wacc=0.115,
)
shares = NormalizedShareCount(
    "NVDA", 24.3e9, "fixture", pd.Timestamp("2026-04-30"),
    "consolidated_common", "fixture", (), (), True, None,
)
inputs = RealCompanyDCFInputs(
    "NVDA", 253.491e9, "ttm",
    tuple(pd.to_datetime(["2025-07-31", "2025-10-31", "2026-01-31", "2026-04-30"])),
    -50e9, "fixture", pd.Timestamp("2026-04-30"), 24.3e9, shares,
    1.49, 0.9283, True, None, "USD", "USD",
)
consensus = (
    ConsensusRevenuePoint(
        "FY2027", "2027-01-31", 395.213e9, 0.8302, 53,
        "fixture", "2026-08-22",
    ),
    ConsensusRevenuePoint(
        "FY2028", "2028-01-31", 568.184e9, 0.4377, 55,
        "fixture", "2026-08-22",
    ),
)
result = build_nvda_growth_reassessment(
    assumptions, ttm_revenue=253.491e9,
    ttm_period_end="2026-04-30", consensus=consensus,
)
comparison = compare_growth_duration_dcf(inputs, result)
render_nvda_growth_duration_reassessment(result, comparison)
