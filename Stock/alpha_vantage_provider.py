"""Live Alpha Vantage acquisition boundary for development/audit use only."""

from copy import deepcopy
from dataclasses import dataclass
import os
import time

import requests

from Stock.forecast_anchors import issuer_anchor_ticker


ALPHA_VANTAGE_API_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_CACHE_TTL_SECONDS = 6 * 60 * 60
ALPHA_VANTAGE_ERROR_CACHE_TTL_SECONDS = 0
ALPHA_VANTAGE_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class AlphaVantagePayloadResult:
    ticker: str
    payload: dict | None
    available: bool
    reason: str | None
    from_cache: bool = False


_SUCCESS_CACHE: dict[str, tuple[float, dict]] = {}


def alpha_vantage_api_key(environ=None) -> str | None:
    """Read the secret only from the process environment."""
    source = os.environ if environ is None else environ
    value = source.get("ALPHAVANTAGE_API_KEY")
    return value.strip() if isinstance(value, str) and value.strip() else None


def clear_alpha_vantage_success_cache() -> None:
    """Testing/development hook; production callers normally rely on TTL."""
    _SUCCESS_CACHE.clear()


def fetch_alpha_vantage_earnings_estimates(
    ticker: str,
    *,
    environ=None,
    http_get=requests.get,
    now=time.time,
) -> AlphaVantagePayloadResult:
    """Fetch estimates with success-only six-hour caching.

    Failures and rate-limit payloads are intentionally not cached, so recovery
    is visible on the next request. The API key is never returned or logged.
    """
    normalized = issuer_anchor_ticker(ticker)
    key = alpha_vantage_api_key(environ)
    if key is None:
        return AlphaVantagePayloadResult(
            normalized, None, False, "not_configured"
        )
    cached = _SUCCESS_CACHE.get(normalized)
    current_time = float(now())
    if cached is not None and current_time - cached[0] < ALPHA_VANTAGE_CACHE_TTL_SECONDS:
        return AlphaVantagePayloadResult(
            normalized, deepcopy(cached[1]), True, None, True
        )
    try:
        response = http_get(
            ALPHA_VANTAGE_API_URL,
            params={
                "function": "EARNINGS_ESTIMATES",
                "symbol": normalized,
                "apikey": key,
            },
            timeout=ALPHA_VANTAGE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return AlphaVantagePayloadResult(normalized, None, False, "timeout")
    except requests.RequestException:
        return AlphaVantagePayloadResult(normalized, None, False, "request_failed")
    except (TypeError, ValueError):
        return AlphaVantagePayloadResult(normalized, None, False, "malformed_json")
    if not isinstance(payload, dict):
        return AlphaVantagePayloadResult(normalized, None, False, "malformed_json")
    if "Note" in payload:
        return AlphaVantagePayloadResult(
            normalized, payload, False, "rate_limit_reached"
        )
    if "Information" in payload:
        information = str(payload.get("Information", "")).lower()
        reason = (
            "rate_limit_reached"
            if "rate limit" in information or "call frequency" in information
            else "provider_information_response"
        )
        return AlphaVantagePayloadResult(
            normalized, payload, False, reason
        )
    if "Error Message" in payload:
        return AlphaVantagePayloadResult(
            normalized, payload, False, "ticker_unavailable"
        )
    if not isinstance(payload.get("estimates"), list) and not any(
        isinstance(payload.get(key_name), list)
        for key_name in ("annualEstimates", "quarterlyEstimates")
    ):
        return AlphaVantagePayloadResult(
            normalized, payload, False, "unexpected_provider_schema"
        )
    _SUCCESS_CACHE[normalized] = (current_time, deepcopy(payload))
    return AlphaVantagePayloadResult(normalized, payload, True, None)
