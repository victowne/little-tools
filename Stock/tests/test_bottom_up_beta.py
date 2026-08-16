from dataclasses import FrozenInstanceError

import pytest
import pandas as pd

from Stock import stock_valuation_mvp as app
from Stock.beta_audit import BetaWACCContext
from Stock.bottom_up_beta import (
    PeerBetaInput,
    build_beta_evidence_comparison,
    build_bottom_up_beta_result,
    build_peer_observation,
    relever_beta,
    unlever_beta,
)
from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation import MultiStageDCFAssumptions


def peer(ticker="PEER", beta=1.2, adjusted=1.13, equity=900.0, debt=100.0,
         tax=0.21):
    return PeerBetaInput(
        ticker=ticker, issuer=ticker, inclusion_rationale="fixture peer",
        levered_beta=beta, adjusted_beta=adjusted,
        beta_method="5y_monthly_raw_regression_vs_sp500",
        market_cap=equity, gross_debt=debt, tax_rate=tax,
    )


def result(peers=None, **overrides):
    values = {
        "target_ticker": "TEST", "issuer": "TEST_INC",
        "peer_group_name": "Test peers",
        "peer_inputs": tuple(peers or [
            peer("A", 0.9, 0.93), peer("B", 1.2, 1.13), peer("C", 1.5, 1.33)
        ]),
        "target_market_cap": 950.0, "target_gross_debt": 50.0,
        "target_tax_rate": 0.20, "historical_raw_beta": 1.1,
    }
    values.update(overrides)
    return build_bottom_up_beta_result(**values)


def test_unlevering_formula_uses_gross_debt():
    assert unlever_beta(1.2, 100.0, 900.0, 0.21) == pytest.approx(
        1.2 / (1 + (1 - 0.21) * 100 / 900)
    )


def test_relevering_formula():
    assert relever_beta(1.1, 50.0, 950.0, 0.20) == pytest.approx(
        1.1 * (1 + (1 - 0.20) * 50 / 950)
    )


def test_zero_debt_leaves_beta_unchanged():
    assert unlever_beta(1.4, 0.0, 500.0, 0.21) == pytest.approx(1.4)
    assert relever_beta(1.4, 0.0, 500.0, 0.21) == pytest.approx(1.4)


def test_high_debt_has_larger_unlevering_effect():
    low = unlever_beta(1.5, 50.0, 500.0, 0.21)
    high = unlever_beta(1.5, 500.0, 500.0, 0.21)
    assert high < low < 1.5


def test_higher_tax_rate_reduces_leverage_adjustment():
    low_tax = unlever_beta(1.5, 200.0, 500.0, 0.10)
    high_tax = unlever_beta(1.5, 200.0, 500.0, 0.40)
    assert high_tax > low_tax


def test_latest_valid_peer_tax_accepts_direct_yahoo_tax_rate():
    statement = pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [0.24, 100.0]},
        index=["Tax Rate For Calcs", "Pretax Income"],
    )
    assert app._latest_valid_effective_tax_rate(statement) == pytest.approx(0.24)


def test_latest_valid_peer_tax_uses_reported_tax_and_pretax_without_default():
    statement = pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [20.0, 100.0]},
        index=["Tax Provision", "Pretax Income"],
    )
    assert app._latest_valid_effective_tax_rate(statement) == pytest.approx(0.20)


def test_distribution_reports_median_mean_and_standard_deviation():
    audit = result([
        peer("A", 0.8, 0.87, debt=0),
        peer("B", 1.0, 1.0, debt=0),
        peer("C", 1.8, 1.53, debt=0),
    ])
    distribution = audit.raw_unlevered_distribution
    assert distribution.median == pytest.approx(1.0)
    assert distribution.mean == pytest.approx(1.2)
    assert distribution.minimum == pytest.approx(0.8)
    assert distribution.maximum == pytest.approx(1.8)
    assert distribution.standard_deviation > 0


def test_invalid_peer_is_excluded_without_beta_one_substitution():
    audit = result([peer("A"), peer("BAD", beta=None), peer("C", 1.4, 1.27)])
    bad = next(item for item in audit.peer_observations if item.ticker == "BAD")
    assert not bad.valid
    assert bad.unlevered_beta is None
    assert audit.valid_peer_count == 2
    assert "insufficient_valid_peers" in audit.warnings
    assert "invalid_peers_excluded" in audit.warnings


def test_real_zero_debt_is_valid_but_missing_debt_is_invalid():
    zero = build_peer_observation(peer(debt=0.0))
    missing = build_peer_observation(peer(debt=None))
    assert zero.valid and zero.debt_to_equity == 0.0
    assert not missing.valid and missing.reason == "invalid_gross_debt"


def test_mean_median_divergence_warning_uses_threshold():
    audit = result([
        peer("A", 0.8, 0.87, debt=0), peer("B", 0.9, 0.93, debt=0),
        peer("C", 1.0, 1.0, debt=0), peer("OUTLIER", 2.5, 2.0, debt=0),
    ])
    assert "peer_mean_median_materially_different" in audit.warnings
    assert audit.relevered_beta_mean > audit.relevered_beta_median


def test_leave_one_out_reports_median_and_mean_ranges():
    audit = result([
        peer("A", 0.7, 0.8, debt=0), peer("B", 1.0, 1.0, debt=0),
        peer("C", 1.8, 1.53, debt=0),
    ], target_gross_debt=0.0)
    assert audit.raw_leave_one_out.median_minimum == pytest.approx(0.85)
    assert audit.raw_leave_one_out.median_maximum == pytest.approx(1.4)
    assert audit.raw_leave_one_out.mean_minimum == pytest.approx(0.85)
    assert audit.raw_leave_one_out.mean_maximum == pytest.approx(1.4)
    assert "peer_result_sensitive_to_single_company" in audit.warnings


def test_warning_thresholds_include_dispersion_and_historical_difference():
    audit = result([
        peer("A", 0.7, 0.8, debt=0), peer("B", 1.0, 1.0, debt=0),
        peer("C", 1.3, 1.2, debt=0),
    ], target_gross_debt=0.0, historical_raw_beta=1.8)
    assert "peer_beta_dispersion_high" in audit.warnings
    assert "bottom_up_beta_far_from_historical_beta" in audit.warnings


def test_invalid_target_capital_structure_does_not_create_relevered_beta():
    audit = result(target_market_cap=None)
    assert audit.target_debt_to_equity is None
    assert audit.relevered_beta_median is None
    assert "invalid_target_capital_structure_or_tax" in audit.warnings


def dcf_inputs():
    shares = NormalizedShareCount(
        ticker="TEST", shares_outstanding=10.0, source="fixture",
        source_period=None, scope="consolidated_common", method="fixture",
        components=(), warnings=(), available=True, reason=None,
    )
    return RealCompanyDCFInputs(
        ticker="TEST", starting_revenue=100.0, starting_revenue_source="ttm",
        starting_revenue_periods=(), net_debt=5.0, net_debt_source="fixture",
        net_debt_period=None, shares_outstanding=10.0,
        normalized_share_count=shares, historical_sales_to_capital_3y=1.2,
        current_accounting_roic=0.25,
    )


def assumptions():
    return MultiStageDCFAssumptions(
        forecast_years=6, near_term_revenue_growth=(0.20, 0.15),
        revenue_fade_years=2, terminal_growth=0.03,
        starting_operating_margin=0.30, mature_operating_margin=0.25,
        starting_sales_to_capital=1.5, mature_sales_to_capital=1.2,
        operating_tax_rate=0.20, wacc=0.09,
    )


def test_wacc_and_full_dcf_translation_change_only_beta_selected_wacc():
    context = BetaWACCContext(0.04, 0.05, 0.035, 0.9, 0.1)
    comparison = build_beta_evidence_comparison(
        inputs=dcf_inputs(), base_assumptions=assumptions(),
        wacc_context=context, historical_raw_beta=1.5,
        historical_adjusted_beta=4 / 3, bottom_up_result=result(),
    )
    provisional, historical_raw, historical_adjusted, bottom_median, bottom_mean = comparison.points
    assert provisional.evidence_method == "Provisional DCF Default"
    assert provisional.formula_based_wacc == pytest.approx(0.09)
    assert historical_raw.formula_based_wacc == pytest.approx(
        0.9 * (0.04 + 1.5 * 0.05) + 0.1 * 0.035
    )
    assert historical_raw.intrinsic_value_per_share != pytest.approx(
        historical_adjusted.intrinsic_value_per_share
    )
    assert bottom_median.intrinsic_value_per_share is not None
    assert bottom_mean.intrinsic_value_per_share is not None


def test_goog_and_googl_share_issuer_level_bottom_up_values():
    common = dict(
        issuer="Alphabet Inc.", peer_group_name="Platforms", peer_inputs=(
            peer("META", 1.1, 1.07), peer("MSFT", 0.9, 0.93),
            peer("AMZN", 1.3, 1.2),
        ), target_market_cap=2_000.0, target_gross_debt=100.0,
        target_tax_rate=0.18,
    )
    goog = build_bottom_up_beta_result(target_ticker="GOOG", **common)
    googl = build_bottom_up_beta_result(target_ticker="GOOGL", **common)
    assert goog.issuer == googl.issuer == "ALPHABET_INC"
    assert goog.relevered_beta_median == pytest.approx(googl.relevered_beta_median)
    assert goog.relevered_beta_mean == pytest.approx(googl.relevered_beta_mean)


def test_industry_mapping_ambiguity_is_explicit():
    audit = result(industry_mapping_ambiguous=True)
    assert "target_industry_mapping_ambiguous" in audit.warnings


def test_result_structures_are_immutable():
    audit = result()
    with pytest.raises(FrozenInstanceError):
        audit.classification = "stable"
    with pytest.raises(FrozenInstanceError):
        audit.peer_observations[0].unlevered_beta = 0.0
