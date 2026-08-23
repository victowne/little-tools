from dataclasses import replace
import inspect
from pathlib import Path

import pandas as pd
import pytest

from Stock.amazon_mature_economics_audit import (
    amazon_economic_mix_evidence,
    amazon_mature_scenarios,
    aws_mix_sensitivity,
    bucket_margin_sensitivity,
    build_mature_scenario,
    economics_matrix,
    profit_pool,
    required_aws_share_for_profit_pool,
    reverse_bridge_for_margin,
    run_mature_economics_valuations,
    segment_summed_growth_diagnostic,
)
from Stock.company_profiles import get_company_profile
from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.share_normalization import NormalizedShareCount


def _inputs():
    period = pd.Timestamp("2025-12-31")
    shares = NormalizedShareCount(
        "AMZN", 10.8e9, "fixture", period, "consolidated_common",
        "fixture", (), (), True, None,
    )
    return RealCompanyDCFInputs(
        "AMZN", 716.924e9, "annual_fallback", (period,), 66e9,
        "fixture", period, 10.8e9, shares, .67, .18,
        True, None, "USD", "USD",
    )


def test_each_mature_mix_sums_to_one_and_margin_identity_includes_shared_cost():
    for scenario in amazon_mature_scenarios():
        assert sum(x.revenue_share for x in scenario.buckets) == pytest.approx(1)
        weighted = sum(
            x.revenue_share * x.operating_margin for x in scenario.buckets
        )
        assert scenario.consolidated_margin == pytest.approx(
            weighted - scenario.shared_cost_adjustment
        )


def test_exclusive_sec_mix_has_no_advertising_double_counting():
    for period in amazon_economic_mix_evidence():
        values = dict(period.bucket_revenue)
        assert sum(values.values()) == pytest.approx(period.total_revenue)
        assert set(values) == {
            "first_party_retail", "marketplace", "advertising",
            "subscriptions", "aws", "other",
        }
    assert amazon_economic_mix_evidence()[-2].total_revenue == pytest.approx(775.680e9)


def test_mature_sales_to_capital_uses_weighted_incremental_capital():
    central = amazon_mature_scenarios()[1]
    expected = 1 / sum(x.revenue_share / x.sales_to_capital for x in central.buckets)
    assert central.consolidated_sales_to_capital == pytest.approx(expected)
    arithmetic = sum(x.revenue_share * x.sales_to_capital for x in central.buckets)
    assert central.consolidated_sales_to_capital != pytest.approx(arithmetic)


def test_terminal_roic_and_reinvestment_identities():
    for scenario in amazon_mature_scenarios():
        assert scenario.terminal_roic == pytest.approx(
            scenario.consolidated_margin * .79 * scenario.consolidated_sales_to_capital
        )
        assert scenario.terminal_reinvestment_rate == pytest.approx(
            .03 / scenario.terminal_roic
        )
        assert scenario.terminal_fcff_to_nopat == pytest.approx(
            1 - scenario.terminal_reinvestment_rate
        )


def test_reverse_thirty_percent_bridge_requires_aws_to_displace_retail():
    result = reverse_bridge_for_margin(.30)
    assert result.required_aws_revenue_share == pytest.approx(.46516129)
    assert result.resulting_first_party_share == pytest.approx(.06483871)
    assert result.assessment == "aggressive"


def test_profit_pool_and_required_aws_share_diagnostics():
    central = amazon_mature_scenarios()[1]
    aws = next(x for x in profit_pool(central) if x.bucket == "aws")
    assert aws.share_of_consolidated_operating_income == pytest.approx(.0825 / .189)
    shares = tuple(required_aws_share_for_profit_pool(x) for x in (.4, .5, .6))
    assert shares[0] < shares[1] < shares[2]


def test_margin_and_mix_sensitivity_is_deterministic():
    low, central, high = amazon_mature_scenarios()
    assert low.consolidated_margin < central.consolidated_margin < high.consolidated_margin
    assert low.consolidated_sales_to_capital < central.consolidated_sales_to_capital < high.consolidated_sales_to_capital
    matrix = economics_matrix(
        (low.consolidated_margin, central.consolidated_margin),
        (low.consolidated_sales_to_capital, central.consolidated_sales_to_capital),
    )
    assert len(matrix) == 4
    retail = bucket_margin_sensitivity(central, "first_party_retail", (.03, .05, .07))
    aws = bucket_margin_sensitivity(central, "aws", (.28, .33, .38))
    mix = aws_mix_sensitivity(central, (.20, .25, .30))
    assert retail[0].consolidated_margin < retail[-1].consolidated_margin
    assert aws[0].consolidated_margin < aws[-1].consolidated_margin
    assert mix[0].consolidated_margin < mix[-1].consolidated_margin


def test_segment_summed_growth_is_exclusive_and_slows_deterministically():
    years = segment_summed_growth_diagnostic()
    assert len(years) == 5
    assert tuple(x.year for x in years) == (1, 2, 3, 4, 5)
    assert all(sum(dict(x.bucket_mix).values()) == pytest.approx(1) for x in years)
    assert years[0].consolidated_growth > years[-1].consolidated_growth


def test_invalid_mix_and_zero_sales_to_capital_fail_clearly():
    central = amazon_mature_scenarios()[1]
    with pytest.raises(ValueError, match="sum to 100"):
        build_mature_scenario(
            "bad", (replace(central.buckets[0], revenue_share=.01),),
            shared_cost_adjustment=.01,
        )
    broken = (replace(central.buckets[0], sales_to_capital=0),) + central.buckets[1:]
    with pytest.raises(ValueError, match="must be positive"):
        build_mature_scenario("bad", broken, shared_cost_adjustment=.01)


def test_valuation_cases_change_only_mature_economics_and_remain_hybrid_research():
    results = run_mature_economics_valuations(
        _inputs(), starting_operating_margin=.11155,
        starting_depreciation_to_revenue=.075,
    )
    assert tuple(x.case for x in results) == (
        "phase3f_bridge", "conservative", "central",
        "high_platform_cloud", "thirty_percent_margin_diagnostic",
    )
    first = results[0].run.assumptions
    for result in results[1:]:
        assumptions = result.run.assumptions
        assert assumptions.near_term_revenue_growth == first.near_term_revenue_growth
        assert assumptions.wacc == first.wacc
        assert assumptions.terminal_growth == first.terminal_growth
        assert assumptions.operating_tax_rate == first.operating_tax_rate


def test_no_price_input_profile_mutation_or_other_company_mutation():
    signature = str(inspect.signature(run_mature_economics_valuations))
    assert "market_price" not in signature
    before = get_company_profile("GOOGL")
    amazon = get_company_profile("AMZN")
    assert amazon.available is True
    assert get_company_profile("GOOGL") == before
    source = Path("Stock/amazon_mature_economics_audit.py").read_text("utf-8")
    assert "import streamlit" not in source
    assert "import yfinance" not in source
