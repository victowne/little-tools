import inspect
from pathlib import Path

import pandas as pd
import pytest

from Stock.amazon_bucket_evidence_validation import (
    aws_capital_diagnostics,
    capital_and_profit_allocation,
    phase3f1_change_attribution,
    run_validated_mature_valuations,
    shared_cost_sensitivity,
    validated_bucket_ranges,
    validated_mature_scenarios,
    validated_reverse_thirty_percent,
    validation_evidence,
)
from Stock.amazon_mature_economics_audit import amazon_economic_mix_evidence
from Stock.company_profiles import get_company_profile
from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.share_normalization import NormalizedShareCount


def _inputs():
    period = pd.Timestamp("2026-06-30")
    shares = NormalizedShareCount("AMZN", 10.8e9, "fixture", period, "consolidated_common", "fixture", (), (), True, None)
    return RealCompanyDCFInputs("AMZN", 775.68e9, "validated_ttm_sec_10q", (period,), 66e9, "fixture", period, 10.8e9, shares, .67, .18, True, None, "USD", "USD")


def test_revenue_taxonomy_is_exclusive_and_historical_mix_is_complete():
    for period in amazon_economic_mix_evidence():
        assert sum(value for _, value in period.bucket_revenue) == pytest.approx(period.total_revenue)
        assert len({name for name, _ in period.bucket_revenue}) == 6


def test_evidence_registry_preserves_tier_quality_source_and_period():
    evidence = validation_evidence()
    assert {x.tier for x in evidence} <= {1, 2, 3}
    assert all(x.source and x.period and x.retrieved_at for x in evidence)
    assert any(x.quality == "Direct disclosure" for x in evidence)
    assert any(x.quality == "Comparable-supported" for x in evidence)


def test_validated_ranges_are_ordered_and_positive():
    for item in validated_bucket_ranges():
        assert item.margin_low <= item.margin_central <= item.margin_high
        assert 0 < item.sales_to_capital_low <= item.sales_to_capital_central <= item.sales_to_capital_high


def test_validated_scenarios_obey_margin_and_harmonic_sc_identities():
    for scenario in validated_mature_scenarios():
        assert sum(x.revenue_share for x in scenario.buckets) == pytest.approx(1)
        assert scenario.consolidated_margin == pytest.approx(sum(x.revenue_share * x.operating_margin for x in scenario.buckets) - scenario.shared_cost_adjustment)
        assert scenario.consolidated_sales_to_capital == pytest.approx(1 / sum(x.revenue_share / x.sales_to_capital for x in scenario.buckets))
        assert scenario.terminal_roic == pytest.approx(scenario.consolidated_margin * .79 * scenario.consolidated_sales_to_capital)


def test_incremental_capital_and_profit_pool_shares_are_observable():
    rows = capital_and_profit_allocation(validated_mature_scenarios()[1])
    assert sum(x.incremental_capital_share for x in rows) == pytest.approx(1)
    assert next(x for x in rows if x.bucket == "aws").incremental_capital_share > next(x for x in rows if x.bucket == "aws").revenue_share
    assert next(x for x in rows if x.bucket == "advertising").operating_income_share > next(x for x in rows if x.bucket == "advertising").revenue_share


def test_shared_cost_sensitivity_changes_margin_and_roic_only_as_expected():
    rows = shared_cost_sensitivity(validated_mature_scenarios()[1])
    assert tuple(x.shared_cost_adjustment for x in rows) == (.01, .015, .02, .025)
    assert rows[0].consolidated_margin > rows[-1].consolidated_margin
    assert rows[0].consolidated_sales_to_capital == pytest.approx(rows[-1].consolidated_sales_to_capital)


def test_aws_capital_diagnostics_document_current_buildout():
    diagnostics = dict(aws_capital_diagnostics())
    assert diagnostics["TTM_Revenue_to_net_PPE"] == pytest.approx(148.404 / 263.750)
    assert diagnostics["H1_net_additions_to_H1_Revenue"] > 1


def test_change_attribution_preserves_old_and_validated_values():
    changes = phase3f1_change_attribution()
    assert {x.assumption for x in changes} == {"mature_margin", "mature_sales_to_capital", "terminal_roic"}
    assert all(x.previous != x.validated for x in changes)
    reverse = validated_reverse_thirty_percent()
    assert reverse.required_aws_revenue_share == pytest.approx(.49193548)
    assert reverse.resulting_first_party_share == pytest.approx(.03806452)


def test_valuation_isolates_mature_economics_and_excludes_price():
    results = run_validated_mature_valuations(_inputs(), starting_operating_margin=.11155, starting_depreciation_to_revenue=.07)
    assert tuple(x.case for x in results) == ("central", "validated_conservative", "validated_central", "validated_high", "validated_thirty_percent_diagnostic")
    first = results[0].run.assumptions
    for result in results[1:]:
        assert result.run.assumptions.near_term_revenue_growth == first.near_term_revenue_growth
        assert result.run.assumptions.wacc == first.wacc
        assert result.run.assumptions.terminal_growth == first.terminal_growth
    assert "market_price" not in str(inspect.signature(run_validated_mature_valuations))


def test_no_profile_or_production_engine_mutation():
    before = get_company_profile("GOOGL")
    assert get_company_profile("AMZN").available is True
    assert get_company_profile("GOOGL") == before
    source = Path("Stock/amazon_bucket_evidence_validation.py").read_text("utf-8")
    assert "import streamlit" not in source
    assert "import yfinance" not in source
