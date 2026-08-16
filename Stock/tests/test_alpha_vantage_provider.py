import requests
import pandas as pd
import pytest

from Stock.alpha_vantage_provider import (
    alpha_vantage_api_key,
    clear_alpha_vantage_success_cache,
    fetch_alpha_vantage_earnings_estimates,
)
from Stock.forecast_anchors import (
    alpha_vantage_response_to_provider_result,
    assess_forward_ttm_feasibility,
    audit_forward_estimate_provider,
    compare_forward_estimate_sources,
)


ACTUAL = pd.Timestamp("2025-12-31")
RETRIEVED = pd.Timestamp("2026-03-01", tz="UTC")


def row(horizon, period, revenue, analysts=20, **extra):
    return {
        "horizon": horizon,
        "fiscalDateEnding": period,
        "revenue_estimate_average": revenue,
        "revenue_estimate_analyst_count": analysts,
        "revenue_estimate_high": extra.get("high"),
        "revenue_estimate_low": extra.get("low"),
    }


def normalize(payload, ticker="TEST"):
    return alpha_vantage_response_to_provider_result(
        ticker=ticker, payload=payload,
        latest_actual_fiscal_period=ACTUAL,
        latest_actual_revenue=100.0, retrieved_at=RETRIEVED,
        latest_actual_quarterly_period=pd.Timestamp("2025-12-31"),
    )


@pytest.fixture(autouse=True)
def empty_live_cache():
    clear_alpha_vantage_success_cache()
    yield
    clear_alpha_vantage_success_cache()


def test_valid_fy1_fy3_documented_shape_and_explicit_dates():
    result = normalize({"estimates": [
        row("next fiscal year", "2026-12-31", "120", 25, high="130", low="110"),
        row("next fiscal year", "2027-12-31", "150", 22),
        row("next fiscal year", "2028-12-31", "180", 18),
    ]})
    assert [item.revenue_estimate for item in result.annual.estimates] == [120, 150, 180]
    assert [item.implied_revenue_growth for item in result.annual.estimates] == pytest.approx([.2, .25, .2])
    assert all(item.fiscal_period_explicit for item in result.annual.estimates)
    assert result.annual.estimates[0].revenue_estimate_high == 130
    assert result.provider_as_of is None


def test_live_schema_date_field_and_historical_quarters_are_normalized():
    result = normalize({"symbol": "TEST", "estimates": [
        {
            "date": "2028-12-31", "horizon": "fiscal year",
            "revenue_estimate_average": "180.00",
            "revenue_estimate_high": "190.00",
            "revenue_estimate_low": "170.00",
            "revenue_estimate_analyst_count": "18.00",
        },
        {
            "date": "2026-03-31", "horizon": "fiscal quarter",
            "revenue_estimate_average": "30.00",
            "revenue_estimate_analyst_count": "12.00",
        },
        {
            "date": "2025-09-30", "horizon": "fiscal quarter",
            "revenue_estimate_average": "25.00",
            "revenue_estimate_analyst_count": "11.00",
        },
    ]})
    assert result.annual.estimates[0].fiscal_period_end == pd.Timestamp("2028-12-31")
    assert [item.fiscal_period_end for item in result.quarterly] == [pd.Timestamp("2026-03-31")]


def test_quarterly_response_and_forward_ttm_feasibility():
    result = normalize({"estimates": [
        row("next fiscal quarter", period, 30 + index, 12)
        for index, period in enumerate(
            ["2026-03-31", "2026-06-30", "2026-09-30", "2026-12-31"]
        )
    ]})
    assert len(result.quarterly) == 4
    assert assess_forward_ttm_feasibility(result.quarterly)[0] == "feasible"


def test_missing_fy3_fails_primary_eligibility():
    result = normalize({"estimates": [
        row("next fiscal year", "2026-12-31", 120),
        row("next fiscal year", "2027-12-31", 150),
    ]})
    audit = audit_forward_estimate_provider(result)
    assert audit.annual_years_available == 2
    assert audit.suitable_for_primary_use is False


def test_complete_consistent_series_meets_per_company_primary_criteria():
    result = normalize({"estimates": [
        row("next fiscal year", "2026-12-31", 120, 25),
        row("next fiscal year", "2027-12-31", 150, 22),
        row("next fiscal year", "2028-12-31", 180, 18),
    ]})
    audit = audit_forward_estimate_provider(result)
    assert audit.explicit_fiscal_dates is True
    assert audit.analyst_counts_available is True
    assert audit.suitable_for_primary_use is True


def test_fiscal_calendar_mismatch_is_preserved_and_rejected():
    result = normalize({"estimates": [
        row("next fiscal year", "2026-06-30", 120, 25),
        row("next fiscal year", "2027-06-30", 150, 22),
        row("next fiscal year", "2028-06-30", 180, 18),
    ]})
    audit = audit_forward_estimate_provider(result)
    assert "provider_fiscal_calendar_mismatch" in audit.warnings
    assert audit.suitable_for_primary_use is False


def test_missing_date_and_analyst_count_remain_missing():
    result = normalize({"estimates": [
        row("next fiscal year", None, 120, None),
        row("next fiscal year", "2027-12-31", 150, None),
    ]})
    assert result.annual.estimates[-1].analyst_count is None
    assert any(not item.fiscal_period_explicit for item in result.annual.estimates)
    assert audit_forward_estimate_provider(result).suitable_for_primary_use is False


def test_zero_analyst_count_is_unavailable_not_real_zero():
    result = normalize({"estimates": [
        row("next fiscal year", "2026-12-31", 120, 0),
    ]})
    assert result.annual.estimates[0].analyst_count is None


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"Error Message": "bad ticker"}, "ticker_unavailable"),
        ({"Note": "rate limit"}, "rate_limit_reached"),
        ({"Information": "premium endpoint"}, "provider_information_response"),
        ({"Information": "standard API rate limit reached"}, "rate_limit_reached"),
        ("not-json-object", "malformed_json"),
    ],
)
def test_provider_error_payloads_are_explicit(payload, reason):
    result = normalize(payload)
    assert result.available is False
    assert result.reason == reason


def test_malformed_values_and_duplicate_periods_are_unavailable():
    malformed = normalize({"estimates": [
        row("next fiscal year", "2026-12-31", "not-a-number"),
    ]})
    duplicate = normalize({"estimates": [
        row("next fiscal year", "2026-12-31", 120),
        row("next fiscal year", "2026-12-31", 121),
    ]})
    assert malformed.annual.estimates[0].available is False
    assert all(not item.available for item in duplicate.annual.estimates)


def test_goog_and_googl_share_issuer_normalization():
    payload = {"estimates": [row("next fiscal year", "2026-12-31", 120)]}
    assert normalize(payload, "GOOG").issuer_id == normalize(payload, "GOOGL").issuer_id == "GOOGL"


def test_source_disagreement_over_five_percent_is_flagged():
    primary = normalize({"estimates": [
        row("next fiscal year", "2026-12-31", 120),
    ]}).annual
    reference = normalize({"estimates": [
        row("next fiscal year", "2026-12-31", 100),
    ]}).annual
    comparison = compare_forward_estimate_sources(primary, reference)[0]
    assert comparison.percentage_difference == pytest.approx(.2)
    assert "forward_revenue_sources_materially_disagree" in comparison.warnings


def test_environment_key_missing_reports_not_configured_without_http_call():
    assert alpha_vantage_api_key({}) is None
    result = fetch_alpha_vantage_earnings_estimates(
        "NVDA", environ={}, http_get=lambda *args, **kwargs: pytest.fail()
    )
    assert result.reason == "not_configured"


def test_goog_live_request_uses_issuer_level_googl_symbol():
    captured = {}

    def get(*args, **kwargs):
        captured.update(kwargs["params"])
        return Response({"estimates": []})

    result = fetch_alpha_vantage_earnings_estimates(
        "GOOG", environ={"ALPHAVANTAGE_API_KEY": "secret"}, http_get=get
    )
    assert result.ticker == "GOOGL"
    assert captured["symbol"] == "GOOGL"


class Response:
    def __init__(self, payload=None, json_error=False):
        self.payload = payload
        self.json_error = json_error

    def raise_for_status(self):
        return None

    def json(self):
        if self.json_error:
            raise ValueError("malformed")
        return self.payload


def test_success_is_cached_but_rate_limit_and_malformed_json_are_not():
    calls = []
    payload = {"estimates": [row("next fiscal year", "2026-12-31", 120)]}

    def successful(*args, **kwargs):
        calls.append(kwargs)
        return Response(payload)

    first = fetch_alpha_vantage_earnings_estimates(
        "NVDA", environ={"ALPHAVANTAGE_API_KEY": "secret"},
        http_get=successful, now=lambda: 100,
    )
    second = fetch_alpha_vantage_earnings_estimates(
        "NVDA", environ={"ALPHAVANTAGE_API_KEY": "secret"},
        http_get=successful, now=lambda: 101,
    )
    assert first.available and second.from_cache
    assert len(calls) == 1
    assert "secret" not in repr(first)

    clear_alpha_vantage_success_cache()
    limited = fetch_alpha_vantage_earnings_estimates(
        "NVDA", environ={"ALPHAVANTAGE_API_KEY": "secret"},
        http_get=lambda *a, **k: Response({"Note": "limit"}),
    )
    malformed = fetch_alpha_vantage_earnings_estimates(
        "NVDA", environ={"ALPHAVANTAGE_API_KEY": "secret"},
        http_get=lambda *a, **k: Response(json_error=True),
    )
    assert limited.reason == "rate_limit_reached"
    assert malformed.reason == "malformed_json"


def test_information_rate_limit_payload_is_classified_as_rate_limit():
    result = fetch_alpha_vantage_earnings_estimates(
        "NVDA", environ={"ALPHAVANTAGE_API_KEY": "secret"},
        http_get=lambda *a, **k: Response(
            {"Information": "API call frequency and rate limit exceeded"}
        ),
    )
    assert result.reason == "rate_limit_reached"


def test_request_timeout_is_graceful():
    def timeout(*args, **kwargs):
        raise requests.Timeout()

    result = fetch_alpha_vantage_earnings_estimates(
        "NVDA", environ={"ALPHAVANTAGE_API_KEY": "secret"},
        http_get=timeout,
    )
    assert result.reason == "timeout"
