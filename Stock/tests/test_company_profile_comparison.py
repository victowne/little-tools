from dataclasses import replace

import pytest

from Stock.company_profile_comparison import (
    build_company_profile_comparison,
    build_company_profile_comparison_row,
)
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.multistage_integration import run_multistage_dcf
from Stock.tests.test_amazon_research import amazon_history, amazon_inputs
from Stock.tests.test_alphabet_research import current_assumptions
from Stock.unified_company_research import build_apple_research_profile


def apple_profile_and_run():
    profile = build_apple_research_profile(
        current_assumptions(), amazon_history()
    ).lookup.profile
    assumptions = build_multistage_assumptions_from_profile(profile).assumptions
    inputs = replace(amazon_inputs(), ticker="AAPL", starting_revenue=775.680e9)
    return profile, run_multistage_dcf(inputs, assumptions)


def test_comparison_row_is_read_only_market_diagnostic():
    profile, run = apple_profile_and_run()
    row = build_company_profile_comparison_row(profile, run, market_price=200.0)
    assert row.ticker == "AAPL"
    assert row.year1_growth == pytest.approx(.12)
    assert row.mature_operating_margin == pytest.approx(.32)
    assert row.mature_sales_to_capital == pytest.approx(1.8)
    assert row.terminal_roic == pytest.approx(.32 * .84 * 1.8)
    assert row.dcf_to_price == pytest.approx(row.intrinsic_value_per_share / 200)
    assert row.research_status == "research_in_progress"
    assert row.model_risk == "Medium"


def test_missing_or_invalid_market_price_stays_missing():
    profile, run = apple_profile_and_run()
    for price in (None, 0.0, float("nan")):
        row = build_company_profile_comparison_row(
            profile, run, market_price=price
        )
        assert row.market_price is None
        assert row.dcf_to_price is None


def test_compact_comparison_preserves_input_order():
    profile, run = apple_profile_and_run()
    rows = build_company_profile_comparison(((profile, run, 200.0),) * 2)
    assert len(rows) == 2
    assert tuple(row.ticker for row in rows) == ("AAPL", "AAPL")
