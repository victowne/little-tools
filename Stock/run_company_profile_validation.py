"""Generic live validation CLI for all production Company Profiles.

Run from the repository root, for example:

    python -m Stock.run_company_profile_validation --ticker AMD
    python -m Stock.run_company_profile_validation --all
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from Stock.alphabet_research import build_alphabet_research_profile
from Stock.amazon_research import (
    build_amazon_research_profile,
    run_amazon_candidate_preview,
)
from Stock.company_profiles import build_multistage_assumptions_from_profile
from Stock.hyperscaler_research import (
    build_meta_research_profile,
    build_microsoft_research_profile,
)
from Stock.nvda_research import build_nvda_research_profile
from Stock.stock_valuation_mvp import (
    build_company_fundamentals,
    build_multistage_assumptions_from_ui,
    load_company_snapshot,
    multistage_initial_defaults,
)
from Stock.multistage_integration import run_real_company_multistage_dcf
from Stock.unified_company_research import (
    build_amd_research_profile,
    build_apple_research_profile,
    build_broadcom_research_profile,
    build_micron_research_profile,
)


PRODUCTION_TICKERS = (
    "NVDA", "GOOGL", "META", "MSFT", "AMZN", "MU", "AAPL", "AVGO", "AMD",
)


def _builder(ticker: str):
    return {
        "NVDA": build_nvda_research_profile,
        "GOOGL": build_alphabet_research_profile,
        "META": build_meta_research_profile,
        "MSFT": build_microsoft_research_profile,
        "AMZN": build_amazon_research_profile,
        "MU": build_micron_research_profile,
        "AAPL": build_apple_research_profile,
        "AVGO": build_broadcom_research_profile,
        "AMD": build_amd_research_profile,
    }[ticker]


def validate_ticker(ticker: str) -> dict:
    normalized = "GOOGL" if ticker.strip().upper() == "GOOG" else ticker.strip().upper()
    if normalized not in PRODUCTION_TICKERS:
        raise ValueError(f"unsupported_profile:{normalized}")
    snapshot = load_company_snapshot(normalized)
    history = build_company_fundamentals(snapshot)
    current = build_multistage_assumptions_from_ui(
        multistage_initial_defaults(normalized, history)
    )
    base_run = run_real_company_multistage_dcf(snapshot, history, current)
    research = _builder(normalized)(
        current,
        history,
        retrieved_at=date.today().isoformat(),
    )
    profile = research.lookup.profile
    translation = build_multistage_assumptions_from_profile(profile)
    if not translation.available or translation.assumptions is None:
        raise ValueError(translation.reason or "profile_translation_unavailable")
    candidate = (
        run_amazon_candidate_preview(base_run.inputs, profile)
        if normalized == "AMZN"
        else run_real_company_multistage_dcf(
            snapshot, history, translation.assumptions
        )
    )
    per_share = candidate.per_share_value
    return {
        "ticker": normalized,
        "profile_status": profile.profile_status,
        "model_risk": profile.model_risk,
        "market_price": snapshot.price,
        "intrinsic_value_per_share": (
            per_share.intrinsic_value_per_share if per_share else None
        ),
        "enterprise_value": candidate.enterprise_value.enterprise_value,
        "equity_value": candidate.equity_value.equity_value,
        "terminal_value_share": candidate.enterprise_value.terminal_value_share,
        "terminal_roic": candidate.terminal_value.derived_terminal_roic,
        "warnings": candidate.warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ticker")
    selection.add_argument("--all", action="store_true")
    args = parser.parse_args()
    tickers = PRODUCTION_TICKERS if args.all else (args.ticker,)
    results = []
    for ticker in tickers:
        try:
            results.append(validate_ticker(ticker))
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
