import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from io import StringIO

import requests

from Stock.fundamentals import (
    CAPEX,
    CFO,
    FCF,
    FCF_MARGIN,
    FUNDAMENTAL_GROWTH_CAPACITY,
    GROSS_MARGIN,
    GROSS_PROFIT,
    NET_INVESTMENT,
    NOPAT,
    OPERATING_INCOME,
    OPERATING_MARGIN,
    OPERATING_TAX_RATE,
    REVENUE,
    REVENUE_GROWTH,
    REINVESTMENT_RATE,
    ROIC,
    FundamentalHistory,
    TTMResult,
    build_fundamental_history,
    build_period_fundamentals,
    build_validated_ttm,
)
from Stock.assumption_diagnostics import build_assumption_diagnostics
from Stock.forecast_anchors import (
    align_dcf_and_consensus_period,
    build_dcf_revenue_forecast_periods,
    compare_aligned_forward_estimate,
    load_revenue_forecast_anchors,
    revenue_anchors_to_forward_estimate_set,
)
from Stock.multistage_integration import (
    MultiStageDCFRunResult,
    run_multistage_dcf,
    run_real_company_multistage_dcf,
)
from Stock.valuation import MultiStageDCFAssumptions
from Stock.valuation_sensitivity import (
    WACCTerminalGrowthSensitivity,
    build_wacc_terminal_growth_sensitivity,
)
from Stock.reverse_dcf import (
    GROWTH_UPLIFT,
    MATURE_MARGIN,
    MATURE_SALES_TO_CAPITAL,
    SOLVED,
    WACC,
    ReverseDCFAnalysis,
    ReverseResearchRange,
    research_ranges_from_profile,
    run_reverse_dcf,
)
from Stock.valuation_scenarios import (
    MultiScenarioDCFResult,
    ScenarioRunResult,
    create_scenario_from_base,
    run_multi_scenario_dcf,
)
from Stock.wacc_audit import (
    WACCAuditResult,
    build_wacc_audit_result,
    issuer_normalization_metadata,
)
from Stock.beta_audit import (
    BetaRobustnessAudit,
    BetaWACCContext,
    build_beta_robustness_audit,
    calculate_beta_estimate,
    resample_adjusted_prices,
)
from Stock.bottom_up_beta import (
    BottomUpBetaResult,
    IndustryBetaReference,
    PeerBetaInput,
    build_beta_evidence_comparison,
    build_bottom_up_beta_result,
    peer_group_for_target,
)
from Stock.research_wacc import (
    ResearchWACCDecision,
    build_research_wacc_decision,
)
from Stock.valuation_support import (
    FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED,
)
from Stock.company_profiles import (
    CompanyProfileLookupResult,
    ResearchAssumption,
    ResearchEvidenceItem,
    build_multistage_assumptions_from_profile,
    build_provisional_company_profile,
)
from Stock.nvda_research import (
    NVDAResearchProfileResult,
    build_nvda_research_profile,
)
from Stock.nvda_growth_reassessment import (
    ConsensusRevenuePoint,
    GrowthDurationDCFComparison,
    GrowthDurationReassessment,
    build_nvda_growth_reassessment,
    compare_growth_duration_dcf,
)
from Stock.alphabet_research import (
    AlphabetResearchProfileResult,
    build_alphabet_research_profile,
)
from Stock.hyperscaler_research import (
    HyperscalerResearchProfileResult,
    build_meta_research_profile,
    build_microsoft_research_profile,
)
from Stock.amazon_research import (
    AmazonResearchProfileResult,
    build_amazon_research_profile,
    run_amazon_candidate_preview,
)
from Stock.unified_company_research import (
    UnifiedCompanyResearchResult,
    build_amd_research_profile,
    build_apple_research_profile,
    build_broadcom_research_profile,
    build_micron_research_profile,
)
from Stock.company_profile_review import (
    REQUIRED_REVIEW_GROUPS,
    CompanyProfileReviewState,
    candidate_assumption_signature,
    initialize_profile_review,
    mark_profile_reviewed,
    reconcile_review_state,
    reopen_profile_review,
    set_overall_review_note,
    set_review_group,
)
from Stock.company_profile_application import (
    ProfileApplyPlan,
    ReviewedProfileApplication,
    assumptions_match,
    build_profile_apply_plan,
    create_reviewed_profile_application,
)
from Stock.company_profile_one_click import build_one_click_review_apply

warnings.filterwarnings("ignore")

# ================= 1. 数据获取层 =================
@dataclass(frozen=True)
class CompanySnapshot:
    """一次页面运行所需的公司原始数据快照；金额保留报表原始币种。"""
    ticker: str
    price: float | None
    market_cap: float | None
    shares_outstanding: float | None
    cash: float | None
    total_debt: float | None
    net_debt: float | None
    sector: str | None
    industry: str | None
    beta: float | None
    annual_income: pd.DataFrame
    quarterly_income: pd.DataFrame
    annual_balance: pd.DataFrame
    quarterly_balance: pd.DataFrame
    annual_cashflow: pd.DataFrame
    quarterly_cashflow: pd.DataFrame
    net_debt_source: str | None = None
    net_debt_period: pd.Timestamp | None = None
    ticker_shares_outstanding: float | None = None
    implied_shares_outstanding: float | None = None
    fast_info_shares: float | None = None
    revenue_estimates: pd.DataFrame | None = None
    revenue_estimates_as_of: pd.Timestamp | None = None
    market_cap_source: str | None = None
    market_cap_retrieved_at: pd.Timestamp | None = None
    total_debt_source: str | None = None
    total_debt_period: pd.Timestamp | None = None
    financial_currency: str | None = None
    price_currency: str | None = None


@dataclass(frozen=True)
class FinancialFieldMatch:
    """Explain which Yahoo row, if any, resolved a financial concept."""
    concept: str
    row: pd.Series | None
    row_name: str | None
    tier: int | None
    reason: str | None


FINANCIAL_FIELD_ALIASES = {
    "revenue": {
        "canonical": "Total Revenue",
        "aliases": ("Operating Revenue", "Revenue"),
    },
    "gross_profit": {"canonical": "Gross Profit", "aliases": ()},
    "operating_income": {
        "canonical": "Operating Income",
        "aliases": ("Operating Profit", "Total Operating Income As Reported"),
    },
    "ebit": {
        "canonical": "EBIT",
        "aliases": ("Operating Income", "Total Operating Income As Reported"),
    },
    "pretax_income": {
        "canonical": "Pretax Income",
        "aliases": ("Income Before Tax",),
    },
    "tax_provision": {
        "canonical": "Tax Provision",
        "aliases": ("Income Tax Expense",),
    },
    "effective_tax_rate": {
        "canonical": "Tax Rate For Calcs",
        "aliases": (),
    },
    "net_income": {
        "canonical": "Net Income",
        "aliases": ("Net Income Common Stockholders",),
    },
    "operating_cash_flow": {
        "canonical": "Operating Cash Flow",
        "aliases": ("Total Cash From Operating Activities",),
    },
    "investing_cash_flow": {
        "canonical": "Investing Cash Flow",
        "aliases": ("Total Cashflows From Investing Activities",),
    },
    "financing_cash_flow": {
        "canonical": "Financing Cash Flow",
        "aliases": ("Total Cash From Financing Activities",),
    },
    "capital_expenditure": {
        "canonical": "Capital Expenditure",
        "aliases": ("Capital Expenditures",),
    },
    "free_cash_flow": {"canonical": "Free Cash Flow", "aliases": ()},
    "interest_expense": {
        "canonical": "Interest Expense",
        "aliases": ("Interest Expense Non Operating",),
    },
    "cash": {
        "canonical": "Cash And Cash Equivalents",
        "aliases": ("Cash Cash Equivalents And Short Term Investments",),
    },
    "roic_cash": {
        "canonical": "Cash And Cash Equivalents",
        "aliases": (),
    },
    "total_equity": {
        "canonical": "Stockholders Equity",
        "aliases": ("Common Stock Equity",),
    },
    "total_debt": {"canonical": "Total Debt", "aliases": ()},
    "long_term_debt": {
        "canonical": "Long Term Debt And Capital Lease Obligation",
        "aliases": ("Long Term Debt",),
    },
    "net_debt": {"canonical": "Net Debt", "aliases": ()},
    "total_assets": {"canonical": "Total Assets", "aliases": ()},
    "total_liabilities": {
        "canonical": "Total Liabilities Net Minority Interest",
        "aliases": ("Total Liabilities",),
    },
    "shares_outstanding": {
        "canonical": "Ordinary Shares Number",
        "aliases": (),
    },
    "retained_earnings": {"canonical": "Retained Earnings", "aliases": ()},
    "depreciation_amortization": {
        "canonical": "Depreciation And Amortization",
        "aliases": ("Depreciation Amortization Depletion",),
    },
}


def _normalize_financial_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_financial_field(
    statement: pd.DataFrame,
    concept: str | tuple[str, ...],
) -> FinancialFieldMatch:
    """Resolve only normalized canonical names and explicit ordered aliases.

    Tier 1 is the concept's canonical name. Tier 2 contains concept-specific,
    ordered aliases. There is deliberately no generic substring/fuzzy tier.
    Duplicate rows with the same normalized accepted name are ambiguous.
    """
    concept_name = concept if isinstance(concept, str) else "explicit_candidates"
    if statement is None or statement.empty:
        return FinancialFieldMatch(str(concept_name), None, None, None, "empty_statement")

    if isinstance(concept, str):
        specification = FINANCIAL_FIELD_ALIASES.get(concept)
        if specification is None:
            return FinancialFieldMatch(concept, None, None, None, "unknown_concept")
        canonical = specification["canonical"]
        aliases = specification["aliases"]
    else:
        if not concept:
            return FinancialFieldMatch(
                concept_name, None, None, None, "no_approved_names"
            )
        canonical, *aliases = concept

    normalized_rows: dict[str, list] = {}
    for label in statement.index:
        normalized_rows.setdefault(_normalize_financial_field(label), []).append(label)

    for tier, approved_names in ((1, (canonical,)), (2, tuple(aliases))):
        for approved_name in approved_names:
            matches = normalized_rows.get(_normalize_financial_field(approved_name), [])
            if len(matches) > 1:
                return FinancialFieldMatch(
                    str(concept_name), None, None, None, "ambiguous_normalized_match"
                )
            if len(matches) == 1:
                row_name = matches[0]
                row = statement.loc[row_name]
                if not isinstance(row, pd.Series):
                    return FinancialFieldMatch(
                        str(concept_name), None, None, None, "ambiguous_row"
                    )
                return FinancialFieldMatch(
                    str(concept_name), row, str(row_name), tier, None
                )

    return FinancialFieldMatch(str(concept_name), None, None, None, "not_found")


def _find_statement_row(statement: pd.DataFrame,
                        concept: str | tuple[str, ...]):
    """Compatibility wrapper returning only the conservatively resolved row."""
    return resolve_financial_field(statement, concept).row


def _statement_series(statement: pd.DataFrame,
                      concept: str | tuple[str, ...]) -> pd.Series:
    """读取财务报表科目并按日期升序返回数值序列。"""
    if statement is None or statement.empty:
        return pd.Series(dtype=float)
    row = _find_statement_row(statement, concept)
    if row is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(row, errors="coerce").dropna().sort_index()


def _reported_statement_series(statement: pd.DataFrame,
                               concept: str) -> pd.Series:
    """Return the resolved reported row while preserving period-level NaN."""
    if statement is None or statement.empty:
        return pd.Series(dtype=float)
    row = _find_statement_row(statement, concept)
    if row is None:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(pd.Index(row.index), errors="coerce")
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": pd.to_numeric(
                pd.Series(row).reset_index(drop=True), errors="coerce"
            ),
            "order": np.arange(len(row)),
        }
    ).dropna(subset=["date"])
    frame = frame.sort_values(["date", "order"]).drop_duplicates(
        "date", keep="last"
    )
    return pd.Series(
        frame["value"].to_numpy(), index=pd.DatetimeIndex(frame["date"])
    )


def _optional_float(value) -> float | None:
    """将元数据数值转成 float，同时保留缺失状态。"""
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _latest_statement_optional(statement: pd.DataFrame,
                               concept: str | tuple[str, ...]) -> float | None:
    """读取最近一期报表数值；字段缺失时返回 None 而不是零。"""
    if statement is None or statement.empty:
        return None
    row = _find_statement_row(statement, concept)
    if row is None:
        return None
    dates = pd.to_datetime(pd.Index(row.index), errors="coerce")
    values = pd.to_numeric(pd.Series(row).reset_index(drop=True), errors="coerce")
    dated_values = pd.DataFrame({"date": dates, "value": values}).dropna(
        subset=["date"]
    )
    if dated_values.empty:
        return None
    latest = dated_values.sort_values("date").iloc[-1]["value"]
    return _optional_float(latest)


def _derive_net_debt(reported_net_debt: float | None,
                     statement_debt: float | None,
                     statement_cash: float | None,
                     info_debt: float | None,
                     info_cash: float | None) -> float | None:
    """Preserve reported zero and derive net debt only from a complete pair."""
    if reported_net_debt is not None:
        return reported_net_debt
    if statement_debt is not None and statement_cash is not None:
        return statement_debt - statement_cash
    if info_debt is not None and info_cash is not None:
        return info_debt - info_cash
    return None


def _latest_statement_period(statement: pd.DataFrame) -> pd.Timestamp | None:
    """Return the latest parseable statement column without inventing a date."""
    if statement is None or statement.empty:
        return None
    periods = pd.to_datetime(pd.Index(statement.columns), errors="coerce")
    periods = periods[~pd.isna(periods)]
    return pd.Timestamp(periods.max()) if len(periods) else None


@st.cache_data(ttl=3600, show_spinner=False)
def load_company_snapshot(ticker: str) -> CompanySnapshot:
    """集中获取公司元数据以及年度/季度三张财务报表。"""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("股票代码不能为空")

    ticker_obj = yf.Ticker(ticker)

    try:
        fast_info = ticker_obj.fast_info
    except Exception:
        fast_info = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        info = {}

    def statement(method_name: str, frequency: str) -> pd.DataFrame:
        try:
            value = getattr(ticker_obj, method_name)(freq=frequency)
            return value if value is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    annual_income = statement("get_income_stmt", "yearly")
    quarterly_income = statement("get_income_stmt", "quarterly")
    annual_balance = statement("get_balance_sheet", "yearly")
    quarterly_balance = statement("get_balance_sheet", "quarterly")
    annual_cashflow = statement("get_cash_flow", "yearly")
    quarterly_cashflow = statement("get_cash_flow", "quarterly")
    try:
        revenue_estimates = ticker_obj.get_revenue_estimate()
        if revenue_estimates is None:
            revenue_estimates = pd.DataFrame()
        revenue_estimates_as_of = pd.Timestamp.now(tz="UTC")
    except Exception:
        revenue_estimates = pd.DataFrame()
        revenue_estimates_as_of = None

    price = _optional_float(fast_info.get("last_price"))
    if price is None:
        price = _optional_float(fast_info.get("lastPrice"))
    if price is None:
        price = _optional_float(info.get("currentPrice"))
    if price is None:
        price = _optional_float(info.get("regularMarketPrice"))
    if price is None:
        try:
            closes = ticker_obj.history(period="5d")["Close"].dropna()
            price = float(closes.iloc[-1]) if not closes.empty else None
        except Exception:
            price = None

    market_cap = _optional_float(info.get("marketCap"))
    market_cap_source = "yfinance_info_market_cap" if market_cap is not None else None
    ticker_shares = _optional_float(info.get("sharesOutstanding"))
    implied_shares = _optional_float(info.get("impliedSharesOutstanding"))
    fast_info_shares = _optional_float(fast_info.get("shares"))
    shares = ticker_shares
    if shares is None:
        shares = implied_shares
    if shares is None:
        shares = fast_info_shares
    if shares is None:
        shares = _latest_statement_optional(
            annual_balance, "shares_outstanding"
        )
    if shares is None and market_cap is not None and price is not None and price > 0:
        shares = market_cap / price
    if market_cap is None and price is not None and shares is not None:
        market_cap = price * shares
        market_cap_source = "derived_current_price_times_shares"
    market_cap_retrieved_at = (
        pd.Timestamp.now(tz="UTC") if market_cap is not None else None
    )

    statement_debt = _latest_statement_optional(annual_balance, "total_debt")
    info_debt = _optional_float(info.get("totalDebt"))
    total_debt = statement_debt if statement_debt is not None else info_debt
    if statement_debt is not None:
        total_debt_source = "annual_balance_total_debt"
        total_debt_period = _latest_statement_period(annual_balance)
    elif info_debt is not None:
        total_debt_source = "yfinance_info_total_debt"
        total_debt_period = None
    else:
        total_debt_source = None
        total_debt_period = None

    statement_cash = _latest_statement_optional(
        annual_balance, "cash"
    )
    info_cash = _optional_float(info.get("totalCash"))
    cash = statement_cash if statement_cash is not None else info_cash

    reported_net_debt = _latest_statement_optional(annual_balance, "net_debt")
    net_debt = _derive_net_debt(
        reported_net_debt,
        statement_debt,
        statement_cash,
        info_debt,
        info_cash,
    )
    if reported_net_debt is not None:
        net_debt_source = "annual_balance_reported_net_debt"
        net_debt_period = _latest_statement_period(annual_balance)
    elif statement_debt is not None and statement_cash is not None:
        net_debt_source = "annual_balance_debt_minus_cash"
        net_debt_period = _latest_statement_period(annual_balance)
    elif info_debt is not None and info_cash is not None:
        net_debt_source = "company_metadata_debt_minus_cash"
        net_debt_period = None
    else:
        net_debt_source = None
        net_debt_period = None

    return CompanySnapshot(
        ticker=ticker,
        price=price,
        market_cap=market_cap,
        shares_outstanding=shares,
        cash=cash,
        total_debt=total_debt,
        net_debt=net_debt,
        sector=str(info.get("sector")) if info.get("sector") else None,
        industry=str(info.get("industry")) if info.get("industry") else None,
        beta=_optional_float(info.get("beta")),
        annual_income=annual_income,
        quarterly_income=quarterly_income,
        annual_balance=annual_balance,
        quarterly_balance=quarterly_balance,
        annual_cashflow=annual_cashflow,
        quarterly_cashflow=quarterly_cashflow,
        net_debt_source=net_debt_source,
        net_debt_period=net_debt_period,
        ticker_shares_outstanding=ticker_shares,
        implied_shares_outstanding=implied_shares,
        fast_info_shares=fast_info_shares,
        revenue_estimates=revenue_estimates,
        revenue_estimates_as_of=revenue_estimates_as_of,
        market_cap_source=market_cap_source,
        market_cap_retrieved_at=market_cap_retrieved_at,
        total_debt_source=total_debt_source,
        total_debt_period=total_debt_period,
        financial_currency=(
            str(info.get("financialCurrency")).strip().upper()
            if info.get("financialCurrency") else None
        ),
        price_currency=(
            str(info.get("currency") or fast_info.get("currency")).strip().upper()
            if (info.get("currency") or fast_info.get("currency")) else None
        ),
    )


def build_company_fundamentals(snapshot: CompanySnapshot) -> FundamentalHistory:
    """Adapt one normalized company snapshot to the pure fundamentals engine."""
    annual_periods = list(snapshot.annual_income.columns) + list(
        snapshot.annual_cashflow.columns
    ) + list(snapshot.annual_balance.columns)
    return build_fundamental_history(
        annual_revenue=_reported_statement_series(
            snapshot.annual_income, "revenue"
        ),
        annual_gross_profit=_reported_statement_series(
            snapshot.annual_income, "gross_profit"
        ),
        annual_operating_income=_reported_statement_series(
            snapshot.annual_income, "operating_income"
        ),
        annual_cfo=_reported_statement_series(
            snapshot.annual_cashflow, "operating_cash_flow"
        ),
        annual_capex=_reported_statement_series(
            snapshot.annual_cashflow, "capital_expenditure"
        ),
        annual_pretax_income=_reported_statement_series(
            snapshot.annual_income, "pretax_income"
        ),
        annual_tax_provision=_reported_statement_series(
            snapshot.annual_income, "tax_provision"
        ),
        annual_total_equity=_reported_statement_series(
            snapshot.annual_balance, "total_equity"
        ),
        annual_total_debt=_reported_statement_series(
            snapshot.annual_balance, "total_debt"
        ),
        annual_cash=_reported_statement_series(
            snapshot.annual_balance, "roic_cash"
        ),
        annual_depreciation_amortization=_reported_statement_series(
            snapshot.annual_cashflow, "depreciation_amortization"
        ),
        quarterly_revenue=_reported_statement_series(
            snapshot.quarterly_income, "revenue"
        ),
        quarterly_gross_profit=_reported_statement_series(
            snapshot.quarterly_income, "gross_profit"
        ),
        quarterly_operating_income=_reported_statement_series(
            snapshot.quarterly_income, "operating_income"
        ),
        quarterly_cfo=_reported_statement_series(
            snapshot.quarterly_cashflow, "operating_cash_flow"
        ),
        quarterly_capex=_reported_statement_series(
            snapshot.quarterly_cashflow, "capital_expenditure"
        ),
        annual_periods=annual_periods,
        quarterly_income_periods=snapshot.quarterly_income.columns,
        quarterly_cashflow_periods=snapshot.quarterly_cashflow.columns,
    )


def _calculate_fcff_series(income: pd.DataFrame,
                           cashflow: pd.DataFrame) -> tuple[pd.Series, str]:
    """用 CFO 口径从年度或季度报表计算 FCFF 原始美元序列。"""
    operating_cf = _statement_series(cashflow, "operating_cash_flow")
    interest_expense = _statement_series(income, "interest_expense")
    pretax = _statement_series(income, "pretax_income")
    tax = _statement_series(income, "tax_provision")
    capex = _statement_series(cashflow, "capital_expenditure")

    if not operating_cf.empty and not capex.empty:
        frame = pd.concat(
            {
                "operating_cf": operating_cf,
                "interest_expense": interest_expense,
                "pretax": pretax,
                "tax": tax,
                "capex": capex,
            },
            axis=1,
        ).sort_index()
        effective_tax = (frame["tax"] / frame["pretax"]).replace(
            [np.inf, -np.inf], np.nan
        )
        loss_period = frame["pretax"].notna() & (frame["pretax"] <= 0)
        default_tax_assumption = (~loss_period) & (
            frame["pretax"].isna()
            | frame["tax"].isna()
            | (frame["pretax"] == 0)
        )
        effective_tax = effective_tax.where(frame["pretax"] > 0)
        effective_tax.loc[loss_period] = 0.0
        effective_tax = effective_tax.clip(lower=0.0, upper=0.35).fillna(0.21)
        valid_period = frame["operating_cf"].notna() & frame["capex"].notna()
        # Compatibility assumptions are explicit here rather than changing the
        # normalized statement values: absent interest is treated as zero and
        # absent/undefined tax inputs use the existing 21% default.
        interest_assumption_used = bool(
            (frame["interest_expense"].isna() & valid_period).any()
        )
        tax_assumption_used = bool((default_tax_assumption & valid_period).any())
        interest_for_fcff = frame["interest_expense"].abs().fillna(0.0)
        fcff = (
            frame["operating_cf"]
            + frame["capex"]
            + interest_for_fcff * (1 - effective_tax)
        ).loc[valid_period].dropna()
        if not fcff.empty:
            assumptions = []
            if interest_assumption_used:
                assumptions.append("缺失利息按 0")
            if tax_assumption_used:
                assumptions.append("缺失税率按 21%")
            suffix = f"（假设：{'；'.join(assumptions)}）" if assumptions else ""
            return fcff, f"FCFF = CFO + CapEx + 税后利息{suffix}"

    fallback = _statement_series(cashflow, "free_cash_flow")
    if fallback.empty:
        if not operating_cf.empty and not capex.empty:
            complete = pd.concat(
                {"operating_cf": operating_cf, "capex": capex}, axis=1
            ).dropna()
            fallback = complete["operating_cf"] + complete["capex"]
    return fallback, "yfinance FCF 回退口径" if not fallback.empty else "无数据"


def fetch_fcff_data(ticker: str,
                    snapshot: CompanySnapshot | None = None) -> tuple[pd.Series, str]:
    """返回年度 FCFF，并用最近四个季度追加最新 TTM FCFF。"""
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.Series(dtype=float), "无数据"

    try:
        snapshot = snapshot or load_company_snapshot(ticker)
        annual, annual_source = _calculate_fcff_series(
            snapshot.annual_income,
            snapshot.annual_cashflow,
        )
        quarterly, quarterly_source = _calculate_fcff_series(
            snapshot.quarterly_income,
            snapshot.quarterly_cashflow,
        )

        result = annual.iloc[-5:] / 1_000_000_000
        source = annual_source
        ttm = build_validated_ttm(
            quarterly,
            expected_periods=snapshot.quarterly_cashflow.columns,
        )
        if ttm.available:
            ttm_end = ttm.periods_used[-1]
            ttm_fcff = ttm.value / 1_000_000_000
            if result.empty or ttm_end > pd.Timestamp(result.index[-1]):
                result.loc[ttm_end] = ttm_fcff
                source = (
                    f"{annual_source}；最新值为截至 {ttm_end.date()} 的 TTM "
                    f"（季度口径：{quarterly_source}）"
                )
        else:
            source = f"{annual_source}；TTM 不可用：{ttm.reason}"
        return result.sort_index(), source
    except Exception:
        return pd.Series(dtype=float), "无数据"


def fetch_fcf_data(ticker: str,
                   debug: bool = False,
                   snapshot: CompanySnapshot | None = None) -> pd.Series:
    """兼容旧调用；返回 FCFF/FCF 序列，单位为十亿美元。"""
    data, source = fetch_fcff_data(ticker, snapshot)
    if debug and data.empty:
        st.warning(f"⚠️ 无法获取现金流数据（{source}）。")
    return data


def _free_cash_flow_series(cashflow: pd.DataFrame) -> pd.Series:
    """Return operating FCF = CFO + CapEx, distinct from DCF FCFF."""
    frame = build_period_fundamentals(
        revenue=None,
        gross_profit=None,
        operating_income=None,
        cfo=_reported_statement_series(cashflow, "operating_cash_flow"),
        capex=_reported_statement_series(cashflow, "capital_expenditure"),
        periods=cashflow.columns if cashflow is not None else None,
        include_revenue_growth=False,
    )
    return frame[FCF].dropna()


def _financial_trend_frame(income: pd.DataFrame,
                           cashflow: pd.DataFrame,
                           balance: pd.DataFrame) -> pd.DataFrame:
    """整理一组可直接绘图的财报指标，金额和股数统一为 billion。"""
    periods = list(income.columns if income is not None else []) + list(
        cashflow.columns if cashflow is not None else []
    )
    fundamentals = build_period_fundamentals(
        revenue=_reported_statement_series(income, "revenue"),
        gross_profit=_reported_statement_series(income, "gross_profit"),
        operating_income=_reported_statement_series(income, "operating_income"),
        cfo=_reported_statement_series(cashflow, "operating_cash_flow"),
        capex=_reported_statement_series(cashflow, "capital_expenditure"),
        periods=periods,
        include_revenue_growth=False,
    )
    amount_series = {
        "Revenue": fundamentals[REVENUE].dropna(),
        "Gross Profit": fundamentals[GROSS_PROFIT].dropna(),
        "Operating Income": fundamentals[OPERATING_INCOME].dropna(),
        "Net Income": _statement_series(income, "net_income"),
        "Free Cash Flow": fundamentals[FCF].dropna(),
        "Retained Earnings": _statement_series(balance, "retained_earnings"),
        "Shares Outstanding": _statement_series(balance, "shares_outstanding"),
    }
    ratio_series = {
        "Gross Margin": fundamentals[GROSS_MARGIN].dropna(),
        "Operating Margin": fundamentals[OPERATING_MARGIN].dropna(),
    }
    available_amounts = {
        name: values for name, values in amount_series.items() if not values.empty
    }
    available_ratios = {
        name: values for name, values in ratio_series.items() if not values.empty
    }
    if not available_amounts and not available_ratios:
        return pd.DataFrame()
    frames = []
    if available_amounts:
        frames.append(
            pd.concat(available_amounts, axis=1) / 1_000_000_000
        )
    if available_ratios:
        frames.append(pd.concat(available_ratios, axis=1))
    return pd.concat(frames, axis=1).sort_index()


def _latest_flow_value(quarterly: pd.Series,
                       annual: pd.Series,
                       expected_periods=None,
                       annual_expected_periods=None) -> tuple[float | None, str]:
    """流量指标优先取最新四季 TTM，否则回退到最近财年。"""
    ttm = build_validated_ttm(quarterly, expected_periods)
    if ttm.available:
        return ttm.value, f"TTM 截至 {ttm.periods_used[-1].date()}"
    if not annual.empty:
        if annual_expected_periods is not None:
            annual_dates = pd.to_datetime(
                pd.Index(annual_expected_periods), errors="coerce"
            ).dropna()
            if len(annual_dates) and pd.Timestamp(annual.index[-1]) < max(annual_dates):
                return None, "最新财年数据缺失"
        return float(annual.iloc[-1]), f"财年截至 {annual.index[-1].date()}"
    return None, "无可用数据"


def _build_health_checks(annual_income: pd.DataFrame,
                         annual_cashflow: pd.DataFrame,
                         annual_balance: pd.DataFrame,
                         quarterly_income: pd.DataFrame,
                         quarterly_cashflow: pd.DataFrame,
                         quarterly_balance: pd.DataFrame) -> list[dict]:
    """基于最新资产负债表和 TTM/年度流量数据生成基础运营检查。"""
    balance = quarterly_balance if quarterly_balance is not None and not quarterly_balance.empty else annual_balance
    balance_basis = "无可用数据"
    if balance is not None and not balance.empty:
        balance_dates = pd.to_datetime(balance.columns, errors="coerce").dropna()
        if len(balance_dates):
            balance_basis = f"截至 {max(balance_dates).date()}"

    assets = _latest_statement_optional(balance, "total_assets")
    liabilities = _latest_statement_optional(
        balance, "total_liabilities"
    )
    asset_status = (
        assets > liabilities
        if assets is not None and liabilities is not None else None
    )

    long_term_debt = _latest_statement_optional(
        balance, "long_term_debt"
    )
    annual_net_income = _statement_series(annual_income, "net_income")
    quarterly_net_income = _statement_series(quarterly_income, "net_income")
    net_income, income_basis = _latest_flow_value(
        quarterly_net_income,
        annual_net_income,
        expected_periods=quarterly_income.columns,
        annual_expected_periods=annual_income.columns,
    )
    debt_ratio = (
        long_term_debt / net_income
        if long_term_debt is not None and net_income is not None and net_income > 0
        else None
    )
    debt_status = (
        debt_ratio < 4
        if debt_ratio is not None
        else False if long_term_debt is not None and net_income is not None
        else None
    )

    def flow_pair(concept: str) -> tuple[float | None, str]:
        return _latest_flow_value(
            _statement_series(quarterly_cashflow, concept),
            _statement_series(annual_cashflow, concept),
            expected_periods=quarterly_cashflow.columns,
            annual_expected_periods=annual_cashflow.columns,
        )

    operating_cf, cash_basis = flow_pair("operating_cash_flow")
    investing_cf, _ = flow_pair("investing_cash_flow")
    financing_cf, _ = flow_pair("financing_cash_flow")
    cash_available = all(value is not None for value in (operating_cf, investing_cf, financing_cf))
    cash_status = (
        operating_cf > abs(investing_cf) and operating_cf > abs(financing_cf)
        if cash_available else None
    )

    billion = 1_000_000_000
    return [
        {
            "title": "资产覆盖负债",
            "rule": "Total Assets > Total Liabilities",
            "status": asset_status,
            "detail": (
                f"资产 {assets / billion:.2f}B · 负债 {liabilities / billion:.2f}B"
                if assets is not None and liabilities is not None else "资产或负债数据缺失"
            ),
            "basis": balance_basis,
        },
        {
            "title": "长期债务负担",
            "rule": "Long-term Debt / Net Income < 4",
            "status": debt_status,
            "detail": (
                f"长期债务 {long_term_debt / billion:.2f}B · 净利润 {net_income / billion:.2f}B · 比率 {debt_ratio:.2f}x"
                if debt_ratio is not None else
                "长期债务或净利润数据缺失"
                if long_term_debt is None or net_income is None else
                "净利润为负或为零，无法形成有效覆盖"
            ),
            "basis": f"债务{balance_basis} · 净利润{income_basis}",
        },
        {
            "title": "经营现金流覆盖",
            "rule": "OCF > |Investing CF| 且 OCF > |Financing CF|",
            "status": cash_status,
            "detail": (
                f"OCF {operating_cf / billion:.2f}B · ICF {investing_cf / billion:.2f}B · Financing CF {financing_cf / billion:.2f}B"
                if cash_available else "经营、投资或融资现金流数据缺失"
            ),
            "basis": cash_basis,
        },
    ]


def fetch_financial_overview(
    ticker: str,
    snapshot: CompanySnapshot | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """获取年度/季度财报趋势和基础运营检查。"""
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame(), pd.DataFrame(), []
    try:
        snapshot = snapshot or load_company_snapshot(ticker)
        annual_income = snapshot.annual_income
        annual_cashflow = snapshot.annual_cashflow
        annual_balance = snapshot.annual_balance
        quarterly_income = snapshot.quarterly_income
        quarterly_cashflow = snapshot.quarterly_cashflow
        quarterly_balance = snapshot.quarterly_balance
        annual = _financial_trend_frame(
            annual_income, annual_cashflow, annual_balance
        )
        quarterly = _financial_trend_frame(
            quarterly_income, quarterly_cashflow, quarterly_balance
        )
        checks = _build_health_checks(
            annual_income,
            annual_cashflow,
            annual_balance,
            quarterly_income,
            quarterly_cashflow,
            quarterly_balance,
        )
        return annual, quarterly, checks
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), []


def _latest_statement_value(statement: pd.DataFrame,
                            concept: str | tuple[str, ...]) -> float | None:
    """兼容旧调用名；缺失字段和 NaN 均保留为 None。"""
    return _latest_statement_optional(statement, concept)


def fetch_market_data(ticker: str,
                      snapshot: CompanySnapshot | None = None
                      ) -> tuple[float | None, float | None, float | None]:
    """获取股价、净债务（十亿美元）和总股本（十亿股）。"""
    ticker = ticker.strip().upper()
    if not ticker:
        return None, None, None

    snapshot = snapshot or load_company_snapshot(ticker)
    net_debt = (
        snapshot.net_debt / 1_000_000_000
        if snapshot.net_debt is not None else None
    )
    shares = (
        snapshot.shares_outstanding / 1_000_000_000
        if snapshot.shares_outstanding is not None else None
    )
    return snapshot.price, net_debt, shares


def _percent_value(value) -> float:
    """将网页中的百分数字符串转换为小数。"""
    if pd.isna(value):
        return np.nan
    return float(str(value).replace("%", "").strip()) / 100


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_macro_assumptions() -> dict:
    """从美国财政部和 Damodaran 获取无风险利率与成熟市场 ERP。"""
    risk_free = 0.045
    erp = 0.045
    treasury_date = "回退值"
    erp_date = "回退值"
    risk_free_source = "static_fallback_4.5_percent"
    erp_source = "static_fallback_4.5_percent"
    risk_free_fallback_used = True
    erp_fallback_used = True

    try:
        year = pd.Timestamp.now().year
        response = requests.get(
            f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{year}/all",
            params={
                "type": "daily_treasury_yield_curve",
                "field_tdr_date_value": year,
                "page": "",
                "_format": "csv",
            },
            timeout=15,
        )
        response.raise_for_status()
        treasury = pd.read_csv(StringIO(response.text))
        treasury["Date"] = pd.to_datetime(treasury["Date"], errors="coerce")
        latest = treasury.dropna(subset=["Date", "10 Yr"]).sort_values("Date").iloc[-1]
        risk_free = float(latest["10 Yr"]) / 100
        treasury_date = latest["Date"].strftime("%Y-%m-%d")
        risk_free_source = "US_Treasury_daily_10_year_par_yield"
        risk_free_fallback_used = False
    except Exception:
        try:
            treasury_yield = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
            if not treasury_yield.empty:
                risk_free = float(treasury_yield.iloc[-1]) / 100
                treasury_date = "^TNX 回退"
                risk_free_source = "yfinance_^TNX_5d_close_fallback"
        except Exception:
            pass

    try:
        response = requests.get(
            "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ctryprem.html",
            timeout=15,
        )
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        country_table = next(
            table for table in tables
            if any(table.iloc[:, 0].astype(str).eq("United States"))
        )
        us_row = country_table[country_table.iloc[:, 0].astype(str).eq("United States")].iloc[0]
        # 使用美国 ERP 减美国主权违约利差，得到与完整美债利率匹配的成熟市场 ERP。
        erp = _percent_value(us_row.iloc[4]) - _percent_value(us_row.iloc[2])
        erp_date = "Damodaran 2026"
        erp_source = (
            "Damodaran_US_total_equity_risk_premium_minus_country_risk_premium"
        )
        erp_fallback_used = False
    except Exception:
        pass

    return {
        "risk_free": risk_free,
        "erp": erp,
        "treasury_date": treasury_date,
        "erp_date": erp_date,
        "risk_free_source": risk_free_source,
        "erp_source": erp_source,
        "risk_free_fallback_used": risk_free_fallback_used,
        "erp_fallback_used": erp_fallback_used,
    }


def _default_spread(coverage: float, financial: bool = False) -> tuple[float, str]:
    """按 Damodaran 2026 利息保障倍数表返回违约利差和合成评级。"""
    non_financial = [
        (0.20, 0.1900, "D2/D"), (0.65, 0.1600, "C2/C"),
        (0.80, 0.1261, "Ca2/CC"), (1.25, 0.0885, "Caa/CCC"),
        (1.50, 0.0509, "B3/B-"), (1.75, 0.0321, "B2/B"),
        (2.00, 0.0275, "B1/B+"), (2.25, 0.0184, "Ba2/BB"),
        (2.50, 0.0138, "Ba1/BB+"), (3.00, 0.0111, "Baa2/BBB"),
        (4.25, 0.0089, "A3/A-"), (5.50, 0.0078, "A2/A"),
        (6.50, 0.0070, "A1/A+"), (8.50, 0.0055, "Aa2/AA"),
        (np.inf, 0.0040, "Aaa/AAA"),
    ]
    financial_table = [
        (0.05, 0.1900, "D2/D"), (0.10, 0.1600, "C2/C"),
        (0.20, 0.1261, "Ca2/CC"), (0.30, 0.0885, "Caa/CCC"),
        (0.40, 0.0509, "B3/B-"), (0.50, 0.0321, "B2/B"),
        (0.60, 0.0275, "B1/B+"), (0.75, 0.0184, "Ba2/BB"),
        (0.90, 0.0138, "Ba1/BB+"), (1.20, 0.0111, "Baa2/BBB"),
        (1.50, 0.0089, "A3/A-"), (2.00, 0.0078, "A2/A"),
        (2.50, 0.0070, "A1/A+"), (3.00, 0.0055, "Aa2/AA"),
        (np.inf, 0.0040, "Aaa/AAA"),
    ]
    rating_table = financial_table if financial else non_financial
    for upper, spread, rating in rating_table:
        if coverage <= upper:
            return spread, rating
    return 0.0040, "Aaa/AAA"


@st.cache_data(ttl=3600, show_spinner=False)
def _regression_beta(ticker: str) -> tuple[float | None, int]:
    """用五年月度收益率相对标普500计算回归 Beta。"""
    try:
        stock = yf.Ticker(ticker).history(period="5y", interval="1mo", auto_adjust=True)["Close"]
        market = yf.Ticker("^GSPC").history(period="5y", interval="1mo", auto_adjust=True)["Close"]
        returns = pd.concat(
            {"stock": stock.pct_change(), "market": market.pct_change()}, axis=1
        ).dropna()
        if len(returns) < 24 or returns["market"].var() <= 0:
            return None, len(returns)
        beta = returns["stock"].cov(returns["market"]) / returns["market"].var()
        return float(beta), len(returns)
    except Exception:
        return None, 0


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_industry_wacc(industry: str) -> dict:
    """从 Damodaran 行业表中寻找最接近的行业 WACC。"""
    if not industry:
        return {"wacc": None, "matched_industry": None}
    aliases = {
        "Consumer Electronics": "Electronics (Consumer & Office)",
        "Software - Infrastructure": "Software (System & Application)",
        "Software - Application": "Software (System & Application)",
        "Semiconductors": "Semiconductor",
        "Internet Content & Information": "Software (Internet)",
        "Auto Manufacturers": "Auto & Truck",
        "Banks - Diversified": "Bank (Money Center)",
    }
    target = aliases.get(industry, industry)
    try:
        response = requests.get(
            "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html",
            timeout=15,
        )
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
        table.columns = table.iloc[0]
        table = table.iloc[1:].copy()
        names = table["Industry Name"].astype(str).tolist()
        matched = max(
            names,
            key=lambda name: SequenceMatcher(None, target.lower(), name.lower()).ratio(),
        )
        score = SequenceMatcher(None, target.lower(), matched.lower()).ratio()
        if score < 0.35:
            return {"wacc": None, "matched_industry": None}
        row = table[table["Industry Name"].astype(str).eq(matched)].iloc[0]
        return {"wacc": _percent_value(row["Cost  of Capital"]), "matched_industry": matched}
    except Exception:
        return {"wacc": None, "matched_industry": None}


def fetch_wacc_reference(ticker: str,
                         snapshot: CompanySnapshot | None = None) -> dict:
    """计算公司级 WACC，并返回可解释的各项输入。"""
    empty = {"wacc": None, "error": "数据不足"}
    ticker = ticker.strip().upper()
    if not ticker:
        return empty
    try:
        snapshot = snapshot or load_company_snapshot(ticker)
        income = snapshot.annual_income
        price, _, shares_b = fetch_market_data(ticker, snapshot)

        beta, beta_months = _regression_beta(ticker)
        beta_assumption_used = beta is None and snapshot.beta is None
        beta_source = "five_year_monthly_regression_vs_sp500"
        if beta is None:
            if snapshot.beta is not None:
                beta = snapshot.beta
                beta_source = "yfinance_metadata_beta_fallback"
            else:
                beta = 1.0
                beta_source = "static_beta_1.0_fallback"
        macro = fetch_macro_assumptions()

        total_debt = snapshot.total_debt
        if total_debt is None:
            return {"wacc": None, "error": "缺少总债务数据"}
        market_cap = snapshot.market_cap
        if (
            market_cap is None
            and price is not None and price > 0
            and shares_b is not None and shares_b > 0
        ):
            market_cap = price * shares_b * 1_000_000_000
        if market_cap is None or market_cap <= 0:
            return empty

        ebit = _latest_statement_optional(income, "ebit")
        interest_reported = _latest_statement_optional(income, "interest_expense")
        interest_assumption_used = interest_reported is None
        interest = abs(interest_reported) if interest_reported is not None else 0.0
        pretax = _latest_statement_optional(income, "pretax_income")
        tax = _latest_statement_optional(income, "tax_provision")
        tax_inputs_reported = (
            pretax is not None and pretax > 0 and tax is not None and tax >= 0
        )
        tax_assumption_used = not tax_inputs_reported
        raw_tax_rate = tax / pretax if tax_inputs_reported else 0.21
        tax_rate = raw_tax_rate
        tax_rate = float(np.clip(tax_rate, 0.0, 0.35))
        tax_rate_clipped = not np.isclose(tax_rate, raw_tax_rate)
        if interest > 0 and ebit is None:
            return {"wacc": None, "error": "缺少 EBIT，无法计算利息覆盖率"}
        coverage = ebit / interest if interest > 0 else np.inf
        financial = snapshot.sector == "Financial Services"
        spread, rating = _default_spread(coverage, financial)

        cost_equity = macro["risk_free"] + beta * macro["erp"]
        pretax_cost_debt = macro["risk_free"] + spread
        after_tax_cost_debt = pretax_cost_debt * (1 - tax_rate)
        equity_weight = market_cap / (market_cap + max(total_debt, 0))
        debt_weight = 1 - equity_weight
        wacc = cost_equity * equity_weight + after_tax_cost_debt * debt_weight
        industry_ref = fetch_industry_wacc(snapshot.industry or "")
        return {
            "wacc": float(wacc),
            "cost_equity": float(cost_equity),
            "pretax_cost_debt": float(pretax_cost_debt),
            "after_tax_cost_debt": float(after_tax_cost_debt),
            "risk_free": macro["risk_free"],
            "erp": macro["erp"],
            "beta": float(beta),
            "beta_months": beta_months,
            "beta_assumption_used": beta_assumption_used,
            "beta_source": beta_source,
            "tax_rate": tax_rate,
            "raw_tax_rate": float(raw_tax_rate),
            "tax_rate_clipped": tax_rate_clipped,
            "tax_provision": tax,
            "tax_assumption_used": tax_assumption_used,
            "interest_expense": interest_reported,
            "interest_assumption_used": interest_assumption_used,
            "coverage": float(coverage),
            "rating": rating,
            "equity_weight": float(equity_weight),
            "debt_weight": float(debt_weight),
            "market_cap": float(market_cap),
            "market_cap_source": snapshot.market_cap_source,
            "market_cap_retrieved_at": snapshot.market_cap_retrieved_at,
            "total_debt": float(total_debt),
            "total_debt_source": snapshot.total_debt_source,
            "total_debt_period": snapshot.total_debt_period,
            "ebit": ebit,
            "ebit_period": _latest_statement_period(income),
            "interest_expense_period": _latest_statement_period(income),
            "pretax_income": pretax,
            "tax_period": _latest_statement_period(income),
            "spread": float(spread),
            "industry_wacc": industry_ref["wacc"],
            "matched_industry": industry_ref["matched_industry"],
            "treasury_date": macro["treasury_date"],
            "erp_date": macro["erp_date"],
            "risk_free_source": macro.get("risk_free_source", "unknown"),
            "erp_source": macro.get("erp_source", "unknown"),
            "risk_free_fallback_used": macro.get(
                "risk_free_fallback_used", False
            ),
            "erp_fallback_used": macro.get("erp_fallback_used", False),
            "error": None,
        }
    except Exception as exc:
        return {"wacc": None, "error": str(exc)}


@st.cache_data(ttl=3600, show_spinner=False)
def load_beta_robustness_audit(
    ticker: str,
    risk_free_rate: float,
    equity_risk_premium: float,
    after_tax_cost_of_debt: float,
    equity_weight: float,
    debt_weight: float,
    current_dcf_wacc: float,
) -> BetaRobustnessAudit:
    """Load adjusted prices once, then delegate all statistics to the pure audit."""
    ticker = ticker.strip().upper()

    def history(symbol: str, interval: str) -> pd.Series:
        try:
            result = yf.Ticker(symbol).history(
                period="5y", interval=interval, auto_adjust=True
            )["Close"]
            return result.dropna()
        except Exception:
            return pd.Series(dtype=float)

    monthly_stock = history(ticker, "1mo")
    monthly_market = history("^GSPC", "1mo")
    monthly_total_market = history("VTI", "1mo")
    daily_stock = history(ticker, "1d")
    daily_market = history("^GSPC", "1d")
    weekly_stock = resample_adjusted_prices(daily_stock, "weekly")
    weekly_market = resample_adjusted_prices(daily_market, "weekly")
    return build_beta_robustness_audit(
        ticker,
        monthly_stock,
        monthly_market,
        weekly_stock,
        weekly_market,
        wacc_context=BetaWACCContext(
            risk_free_rate=risk_free_rate,
            equity_risk_premium=equity_risk_premium,
            after_tax_cost_of_debt=after_tax_cost_of_debt,
            equity_weight=equity_weight,
            debt_weight=debt_weight,
        ),
        current_dcf_wacc=current_dcf_wacc,
        alternative_benchmark_prices=(
            monthly_total_market if not monthly_total_market.empty else None
        ),
        alternative_benchmark="VTI",
    )


def _latest_valid_effective_tax_rate(
    income_statement: pd.DataFrame,
) -> float | None:
    """Return the latest reported, usable tax/pretax pair without a fallback."""
    direct_rates = _statement_series(income_statement, "effective_tax_rate")
    for value in direct_rates.sort_index(ascending=False):
        rate = _optional_float(value)
        if rate is not None and 0 <= rate <= 1:
            return rate
    pretax = _statement_series(income_statement, "pretax_income")
    tax = _statement_series(income_statement, "tax_provision")
    frame = pd.concat({"pretax": pretax, "tax": tax}, axis=1).sort_index(
        ascending=False
    )
    for _, row in frame.iterrows():
        if pd.notna(row["pretax"]) and row["pretax"] > 0 and pd.notna(row["tax"]):
            rate = float(row["tax"] / row["pretax"])
            if np.isfinite(rate) and 0 <= rate <= 1:
                return rate
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def load_damodaran_industry_beta_references(
    ticker: str,
) -> tuple[IndustryBetaReference, ...]:
    """Load only explicitly mapped US industry rows; never fuzzy-match them."""
    group = peer_group_for_target(ticker)
    if group is None:
        return ()
    try:
        response = requests.get(
            "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html",
            timeout=15,
        )
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
        if not any(
            _normalize_financial_field(column) == "industryname"
            for column in table.columns
        ):
            table.columns = table.iloc[0]
            table = table.iloc[1:].copy()
        columns = {
            _normalize_financial_field(column): column for column in table.columns
        }
        industry_column = columns["industryname"]
        firms_column = columns["numberoffirms"]
        beta_column = columns["beta"]
        de_column = columns["deratio"]
        unlevered_column = columns["unleveredbeta"]
        date_match = re.search(
            r"Date of Analysis.*?(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
            response.text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        source_date = (
            re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", date_match.group(0), re.IGNORECASE).group(0)
            if date_match else None
        )
        references = []
        for industry, note in group.damodaran_industries:
            normalized_industry = re.sub(r"\s+", " ", industry).strip()
            normalized_rows = table[industry_column].astype(str).map(
                lambda value: re.sub(r"\s+", " ", value).strip()
            )
            rows = table[normalized_rows.eq(normalized_industry)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            references.append(IndustryBetaReference(
                industry=industry,
                number_of_firms=int(float(row[firms_column])) if pd.notna(row[firms_column]) else None,
                levered_beta=_optional_float(row[beta_column]),
                unlevered_beta=_optional_float(row[unlevered_column]),
                debt_to_equity=_percent_value(row[de_column]),
                source_date=source_date,
                mapping_note=note,
            ))
        return tuple(references)
    except Exception:
        return ()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_bottom_up_beta_audit_for_issuer(
    target_ticker: str,
) -> BottomUpBetaResult | None:
    """Fetch explicit peer inputs, then delegate all finance math to the pure module."""
    target_ticker = target_ticker.strip().upper()
    group = peer_group_for_target(target_ticker)
    if group is None:
        return None
    target = load_company_snapshot(target_ticker)
    market_prices = yf.Ticker("^GSPC").history(
        period="5y", interval="1mo", auto_adjust=True
    )["Close"].dropna()
    peer_inputs = []
    for definition in group.peers:
        snapshot = load_company_snapshot(definition.ticker)
        try:
            peer_prices = yf.Ticker(definition.ticker).history(
                period="5y", interval="1mo", auto_adjust=True
            )["Close"].dropna()
        except Exception:
            peer_prices = pd.Series(dtype=float)
        estimate = calculate_beta_estimate(
            definition.ticker,
            "^GSPC",
            peer_prices,
            market_prices,
            lookback_years=5,
            frequency="monthly",
            minimum_observations=24,
        )
        peer_inputs.append(PeerBetaInput(
            ticker=definition.ticker,
            issuer=definition.issuer,
            inclusion_rationale=definition.inclusion_rationale,
            levered_beta=estimate.raw_beta if estimate.available else None,
            adjusted_beta=estimate.adjusted_beta if estimate.available else None,
            beta_method="5y_monthly_raw_regression_vs_sp500_adjusted_prices",
            market_cap=snapshot.market_cap,
            gross_debt=snapshot.total_debt,
            tax_rate=_latest_valid_effective_tax_rate(snapshot.annual_income),
            warnings=estimate.warnings if estimate.available else (estimate.reason,),
        ))

    canonical_beta, _ = _regression_beta(target_ticker)
    return build_bottom_up_beta_result(
        target_ticker=target_ticker,
        issuer=group.issuer,
        peer_group_name=group.name,
        peer_inputs=tuple(peer_inputs),
        target_market_cap=target.market_cap,
        target_gross_debt=target.total_debt,
        target_tax_rate=_latest_valid_effective_tax_rate(target.annual_income),
        historical_raw_beta=canonical_beta,
        exclusion_rationales=group.exclusions,
        industry_references=load_damodaran_industry_beta_references(target_ticker),
        industry_mapping_ambiguous=True,
    )


def load_bottom_up_beta_audit(ticker: str) -> BottomUpBetaResult | None:
    """Normalize share classes before cache lookup for issuer-consistent results."""
    normalized = ticker.strip().upper()
    issuer_ticker = "GOOGL" if normalized in {"GOOG", "GOOGL"} else normalized
    return _load_bottom_up_beta_audit_for_issuer(issuer_ticker)


# ================= 2. 估值计算引擎 =================
def calculate_dcf(historical_fcf: pd.Series,
                  growth_rate: float,
                  wacc: float,
                  terminal_growth: float,
                  forecast_years: int,
                  net_debt: float | None,
                  shares_outstanding: float | None) -> dict:
    """
    计算DCF内在价值。现金流、净债务使用十亿美元，股本使用十亿股。
    返回：字典包含当前价、内在价值、安全边际、投影数据等
    """
    if (
        len(historical_fcf) == 0
        or net_debt is None
        or shares_outstanding is None
        or shares_outstanding <= 0
    ):
        return {"error": "数据不足或股本无效"}

    last_fcf = historical_fcf.iloc[-1]

    # 边界条件检查（物理类比：避免发散/奇点）
    if wacc <= terminal_growth:
        return {"error": "WACC 必须大于终值增长率，否则模型发散"}
    if wacc <= 0:
        return {"error": "WACC 必须为正数"}

    # 1. 投影未来FCF
    years = np.arange(1, forecast_years + 1)
    projected_fcf = last_fcf * (1 + growth_rate) ** years

    # 2. 折现
    pv_fcf = projected_fcf / (1 + wacc) ** years

    # 3. 终值 (Gordon Growth Model)
    terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** forecast_years

    # 4. 企业价值 & 股权价值
    enterprise_value = np.sum(pv_fcf) + pv_terminal
    equity_value = enterprise_value - net_debt
    intrinsic_value = equity_value / shares_outstanding

    return {
        "last_fcf": last_fcf,
        "projected_fcf": projected_fcf,
        "pv_fcf": pv_fcf,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value": intrinsic_value,
        "shares": shares_outstanding
    }

# ================= 3. 敏感性分析（网格搜索） =================
def sensitivity_grid(historical_fcf, wacc_range, growth_range, terminal_growth, years, net_debt, shares):
    """计算不同WACC和增长率组合下的内在价值矩阵"""
    grid = np.zeros((len(wacc_range), len(growth_range)))
    for i, w in enumerate(wacc_range):
        for j, g in enumerate(growth_range):
            res = calculate_dcf(historical_fcf, g, w, terminal_growth, years, net_debt, shares)
            grid[i, j] = res.get("intrinsic_value", np.nan)
    return grid


def _margin_of_safety(intrinsic_value: float,
                      current_price: float | None) -> float | None:
    """Return margin of safety only when a real positive market price exists."""
    if current_price is None or current_price <= 0:
        return None
    return (intrinsic_value - current_price) / current_price * 100


def _render_financial_trends_content(
    ticker: str,
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
    statement_currency: str | None = "USD",
) -> None:
    """绘制已展开的年度/季度财报趋势内容。"""
    currency = statement_currency or "报表原始币种"
    st.caption(
        f"损益、现金流和留存收益均以 {currency} 十亿为单位；"
        "Margin 使用百分比；股本为十亿股。"
    )
    period = st.radio(
        "报告周期",
        ("季度", "年度"),
        horizontal=True,
        key=f"financial_period_{ticker}",
    )
    frame = quarterly if period == "季度" else annual
    limit = 8 if period == "季度" else 5
    frame = frame.tail(limit)
    if frame.empty:
        st.warning(f"yfinance 暂无 {ticker} 的{period}财报趋势数据。")
        return

    operating_metrics = [
        "Revenue",
        "Gross Profit",
        "Gross Margin",
        "Operating Income",
        "Operating Margin",
        "Net Income",
        "Free Cash Flow",
    ]
    metric_labels = {
        "Revenue": "营业收入 Revenue",
        "Gross Profit": "毛利润 Gross Profit",
        "Gross Margin": "毛利率 Gross Margin",
        "Operating Income": "营业利润 Operating Income",
        "Operating Margin": "营业利润率 Operating Margin",
        "Net Income": "净利润 Net Income",
        "Free Cash Flow": "自由现金流 Free Cash Flow",
    }
    colors = {
        "Revenue": "#4C78A8",
        "Gross Profit": "#72B7B2",
        "Gross Margin": "#76B7B2",
        "Operating Income": "#F2CF5B",
        "Operating Margin": "#EDC948",
        "Net Income": "#59A14F",
        "Free Cash Flow": "#E45756",
    }
    percentage_metrics = {"Gross Margin", "Operating Margin"}
    available_metrics = [metric for metric in operating_metrics if metric in frame]
    if available_metrics:
        rows = (len(available_metrics) + 1) // 2
        fig = make_subplots(
            rows=rows,
            cols=2,
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            subplot_titles=[metric_labels[metric] for metric in available_metrics],
        )
        for index, metric in enumerate(available_metrics):
            row, col = divmod(index, 2)
            values = frame[metric].dropna()
            is_percentage = metric in percentage_metrics
            plotted_values = values * 100 if is_percentage else values
            fig.add_trace(
                go.Bar(
                    x=values.index,
                    y=plotted_values.values,
                    name=metric_labels[metric],
                    marker_color=colors[metric],
                    text=[
                        f"{value:.1f}%" if is_percentage else f"{value:.1f}"
                        for value in plotted_values.values
                    ],
                    textposition="outside",
                ),
                row=row + 1,
                col=col + 1,
            )
            fig.update_yaxes(
                title_text="%" if is_percentage else "B",
                row=row + 1,
                col=col + 1,
            )
        fig.update_layout(
            height=max(520, rows * 260),
            showlegend=False,
            margin=dict(t=70, b=30),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("收入、利润和自由现金流指标暂不可用。")

    balance_metrics = [
        metric for metric in ("Retained Earnings", "Shares Outstanding")
        if metric in frame
    ]
    if balance_metrics:
        st.subheader("🏦 留存收益与股本变化")
        balance_labels = {
            "Retained Earnings": "留存收益 Retained Earnings",
            "Shares Outstanding": "流通股本 Shares Outstanding",
        }
        fig_balance = make_subplots(
            rows=1,
            cols=len(balance_metrics),
            subplot_titles=[balance_labels[metric] for metric in balance_metrics],
        )
        for index, metric in enumerate(balance_metrics):
            values = frame[metric].dropna()
            fig_balance.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values.values,
                    mode="lines+markers",
                    name=balance_labels[metric],
                    line=dict(width=3),
                ),
                row=1,
                col=index + 1,
            )
            fig_balance.update_yaxes(title_text="B", row=1, col=index + 1)
        fig_balance.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_balance, width="stretch")

    with st.expander("查看财报数据表", expanded=False):
        display = frame.copy()
        display.index = pd.to_datetime(display.index).strftime("%Y-%m-%d")
        formatters = {
            column: ("{:.1%}" if column in percentage_metrics else "{:.2f}")
            for column in display.columns
        }
        st.dataframe(
            display.style.format(formatters, na_rep="—"), width="stretch"
        )


def render_financial_trends(
    ticker: str,
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
    statement_currency: str | None = "USD",
) -> None:
    """以默认折叠状态展示年度/季度财报趋势面板。"""
    with st.expander("📚 财报趋势（年度 / 季度）", expanded=False):
        _render_financial_trends_content(
            ticker, annual, quarterly, statement_currency
        )


def render_fundamental_quality(ticker: str,
                               history: FundamentalHistory | None,
                               statement_currency: str | None = "USD") -> None:
    """Render existing fundamental-engine evidence without recalculation."""
    st.header("Key Fundamentals")
    if history is None or history.annual.empty:
        st.warning(f"{ticker} 基本面质量指标数据不足。")
        return

    latest_period = pd.Timestamp(history.annual.index[-1])
    latest = history.annual.iloc[-1]
    st.caption(
        f"金额指标以 {statement_currency or '报表原始币种'} 十亿为单位；"
        "百分比和倍数不受币种显示影响。"
    )

    def display_value(metric: str, kind: str) -> str:
        value = latest.get(metric, np.nan)
        if pd.isna(value):
            return "数据不足"
        if kind == "amount":
            return _diagnostic_display(
                float(value), "amount", statement_currency
            )
        return f"{float(value) * 100:.1f}%"

    st.subheader("最新年度 Latest Annual")
    st.caption(f"财年截至 {latest_period.date()}")
    annual_metrics = [
        ("Revenue", REVENUE, "amount"),
        ("Revenue Growth", REVENUE_GROWTH, "percent"),
        ("Operating Margin", OPERATING_MARGIN, "percent"),
        ("FCF", FCF, "amount"),
        ("FCF Margin", FCF_MARGIN, "percent"),
        ("NOPAT", NOPAT, "amount"),
        ("ROIC", ROIC, "percent"),
    ]
    for start in range(0, len(annual_metrics), 4):
        batch = annual_metrics[start:start + 4]
        columns = st.columns(len(batch))
        for column, (label, metric, kind) in zip(columns, batch):
            column.metric(label, display_value(metric, kind))

    st.subheader("DCF 历史锚点")
    st.caption(
        "3Y 指标是历史会计锚点，Latest Annual Sales-to-Capital 反映最近年度的"
        "资本效率；两者都不是增长预测或建议参数。"
    )
    anchors = history.dcf_anchors
    latest_sales_to_capital = anchors.annual_sales_to_capital.get(latest_period)

    def anchor_value(result, kind: str) -> str:
        if result is None or not result.available or result.value is None:
            return "数据不足"
        if kind == "percent":
            return f"{result.value * 100:.2f}%"
        return f"{result.value:.2f}x"

    anchor_metrics = [
        ("Revenue CAGR 3Y", anchors.revenue_cagr.get(3), "percent"),
        (
            "Sales-to-Capital (S/C) 3Y",
            anchors.normalized_sales_to_capital.get(3),
            "multiple",
        ),
        ("Latest Annual Sales-to-Capital", latest_sales_to_capital, "multiple"),
    ]
    anchor_columns = st.columns(len(anchor_metrics))
    for column, (label, result, kind) in zip(anchor_columns, anchor_metrics):
        column.metric(label, anchor_value(result, kind))

    observable_results = [
        ("Latest Annual", latest_sales_to_capital),
        ("3Y Normalized", anchors.normalized_sales_to_capital.get(3)),
    ]
    with st.expander("查看 Sales-to-Capital 计算明细", expanded=False):
        for label, result in observable_results:
            st.markdown(f"**{label}**")
            if result is None or not result.available:
                reason = result.reason if result is not None else "unavailable"
                st.write(f"数据不足（{reason}）")
                continue
            period_start = (
                result.start_period.date()
                if result.start_period is not None else "N/A"
            )
            period_end = (
                result.end_period.date()
                if result.end_period is not None else "N/A"
            )
            st.write(f"期间：{period_start} → {period_end}")
            def optional_amount(value) -> str:
                return (
                    _diagnostic_display(value, "amount", statement_currency)
                    if value is not None else "数据不足"
                )
            st.write(
                f"Revenue：{optional_amount(result.start_revenue)} → "
                f"{optional_amount(result.end_revenue)}；"
                f"ΔRevenue：{optional_amount(result.delta_revenue)}"
            )
            st.write(
                "Invested Capital："
                f"{optional_amount(result.start_invested_capital)} → "
                f"{optional_amount(result.end_invested_capital)}；"
                "ΔInvested Capital："
                f"{optional_amount(result.delta_invested_capital)}"
            )
            st.write(f"Sales-to-Capital：{result.value:.2f}x")
    st.caption(
        "Sales-to-Capital 是历史会计资本效率锚点，不是精确因果指标；全额扣除会计现金，"
        "且可能受未资本化研发、收购与商誉、回购及营运资本时点影响。"
    )

    st.subheader("最近十二个月 TTM")
    ttm_metrics = [
        ("Revenue", REVENUE, "amount"),
        ("Operating Margin", OPERATING_MARGIN, "percent"),
        ("FCF", FCF, "amount"),
        ("FCF Margin", FCF_MARGIN, "percent"),
    ]
    ttm_columns = st.columns(len(ttm_metrics))
    period_groups: dict[tuple[pd.Timestamp, ...], list[str]] = {}
    for column, (label, metric, kind) in zip(ttm_columns, ttm_metrics):
        result = history.ttm.get(metric)
        if result is None or not result.available or result.value is None:
            formatted = "数据不足"
        elif kind == "amount":
            formatted = _diagnostic_display(
                result.value, "amount", statement_currency
            )
        else:
            formatted = f"{result.value * 100:.1f}%"
        column.metric(label, formatted)
        if result is not None and result.available and result.periods_used:
            period_groups.setdefault(result.periods_used, []).append(label)

    if period_groups:
        with st.expander("查看 TTM 报告期间", expanded=False):
            for periods, labels in period_groups.items():
                st.write(
                    f"{', '.join(labels)}：{periods[0].date()} → "
                    f"{periods[-1].date()}（{len(periods)} 个季度）"
                )

    chart_specs = [
        (REVENUE_GROWTH, "Revenue Growth"),
        (OPERATING_MARGIN, "Operating Margin"),
        (FCF_MARGIN, "FCF Margin"),
        (ROIC, "ROIC"),
    ]
    chart_data = []
    for metric, label in chart_specs:
        if metric in history.annual:
            values = history.annual[metric].dropna()
            if len(values) >= 2:
                chart_data.append((metric, label, values))
    if chart_data:
        st.subheader("年度质量趋势 Annual Quality Trends")
        rows = (len(chart_data) + 1) // 2
        figure = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=[label for _, label, _ in chart_data],
            vertical_spacing=0.16,
            horizontal_spacing=0.10,
        )
        for index, (_, label, values) in enumerate(chart_data):
            row, col = divmod(index, 2)
            figure.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values.values * 100,
                    mode="lines+markers",
                    name=label,
                    line=dict(width=3),
                ),
                row=row + 1,
                col=col + 1,
            )
            figure.update_yaxes(title_text="%", row=row + 1, col=col + 1)
        figure.update_layout(
            height=max(340, rows * 280),
            showlegend=False,
            margin=dict(t=70, b=30),
        )
        st.plotly_chart(figure, width="stretch")

    st.caption(
        "ROIC 为简化会计口径：全额扣除会计现金，未资本化研发，亦未单独估算超额现金；"
        "资产轻型科技公司的 ROIC 可能显得异常高。"
    )
    st.caption(
        "简化再投资率基于资本开支现金流出减 D&A，尚未包含营运资本变化或收购；"
        "Fundamental Growth Capacity 是历史结构关系，不是增长预测。"
    )


def render_health_checks(ticker: str, checks: list[dict]) -> None:
    """展示基础财务规则的通过、未通过或数据不足状态。"""
    st.header("🩺 运营体检")
    st.caption("这是快速筛查，不构成投资结论；行业特性、周期性和一次性项目仍需人工判断。")
    if not checks:
        st.warning(f"yfinance 暂无足够的 {ticker} 财务数据用于体检。")
        return

    columns = st.columns(len(checks))
    for column, check in zip(columns, checks):
        with column.container(border=True):
            st.subheader(check["title"])
            status = check["status"]
            if status is True:
                st.success("✅ 通过")
            elif status is False:
                st.warning("⚠️ 未通过")
            else:
                st.info("➖ 数据不足")
            st.code(check["rule"], language=None)
            st.write(check["detail"])
            st.caption(check["basis"])


MULTISTAGE_GENERIC_DEFAULTS = {
    "year_1_growth": 10.0,
    "year_2_growth": 8.0,
    "year_3_growth": 6.0,
    "forecast_years": 10,
    "fade_years": 7,
    "terminal_growth": 3.0,
    "mature_margin": 20.0,
    "starting_sales_to_capital": 1.0,
    "mature_sales_to_capital": 1.0,
    "tax_rate": 21.0,
    "wacc": 9.0,
}


def _valid_history_value(history: FundamentalHistory | None,
                         metric: str,
                         *,
                         ttm: bool = False) -> float | None:
    """Read one existing fundamental metric without calculating a replacement."""
    if history is None:
        return None
    if ttm:
        result = history.ttm.get(metric)
        if result is None or not result.available or result.value is None:
            return None
        value = float(result.value)
    else:
        if history.annual.empty or metric not in history.annual:
            return None
        value = history.annual.iloc[-1].get(metric)
        if pd.isna(value):
            return None
        value = float(value)
    return value if np.isfinite(value) else None


def multistage_initial_defaults(ticker: str,
                                history: FundamentalHistory | None) -> dict:
    """Return editable initial assumptions; never mutate history or forecasts."""
    ticker = ticker.strip().upper()
    values = dict(MULTISTAGE_GENERIC_DEFAULTS)
    ttm_margin = _valid_history_value(history, OPERATING_MARGIN, ttm=True)
    if ttm_margin is not None:
        values["starting_margin"] = ttm_margin * 100
    else:
        annual_margin = _valid_history_value(history, OPERATING_MARGIN)
        values["starting_margin"] = (
            annual_margin * 100 if annual_margin is not None else 20.0
        )
    annual_tax = _valid_history_value(history, OPERATING_TAX_RATE)
    if annual_tax is not None and 0 <= annual_tax <= 1:
        values["tax_rate"] = annual_tax * 100

    if ticker == "NVDA":
        values.update({
            "year_1_growth": 30.0, "year_2_growth": 25.0,
            "year_3_growth": 20.0, "forecast_years": 10,
            "fade_years": 7, "terminal_growth": 3.5,
            "mature_margin": 40.0, "starting_sales_to_capital": 1.5,
            "mature_sales_to_capital": 1.2, "tax_rate": 16.0,
            "wacc": 9.0,
        })
    elif ticker in {"GOOG", "GOOGL"}:
        values.update({
            "year_1_growth": 15.0, "year_2_growth": 13.0,
            "year_3_growth": 11.0, "forecast_years": 10,
            "fade_years": 7, "terminal_growth": 3.5,
            "mature_margin": 30.0, "starting_sales_to_capital": 0.8,
            "mature_sales_to_capital": 0.7, "tax_rate": 17.0,
            "wacc": 8.5,
        })
    return values


def research_wacc_session_keys(ticker: str) -> dict[str, str]:
    """Return minimal issuer-level keys for one user-controlled WACC decision."""
    issuer_key, _ = issuer_normalization_metadata(ticker)
    prefix = f"research_wacc_{issuer_key}_"
    return {
        "value": prefix + "value",
        "status": prefix + "status",
        "rationale": prefix + "rationale",
        "created_at": prefix + "created_at",
    }


def base_profile_application_key(ticker: str) -> str:
    issuer_key, _ = issuer_normalization_metadata(ticker)
    return f"reviewed_profile_application_{issuer_key}"


def _multistage_base_state_values(
    assumptions: MultiStageDCFAssumptions,
) -> dict[str, float | int]:
    """Translate validated engine units to existing UI/session-state units."""
    growth = assumptions.near_term_revenue_growth
    return {
        "year_1_growth": growth[0] * 100,
        "year_2_growth": growth[1] * 100,
        "year_3_growth": growth[2] * 100,
        "fade_years": assumptions.revenue_fade_years,
        "forecast_years": assumptions.forecast_years,
        "terminal_growth": assumptions.terminal_growth * 100,
        "starting_margin": assumptions.starting_operating_margin * 100,
        "mature_margin": assumptions.mature_operating_margin * 100,
        "starting_sales_to_capital": assumptions.starting_sales_to_capital,
        "mature_sales_to_capital": assumptions.mature_sales_to_capital,
        "tax_rate": assumptions.operating_tax_rate * 100,
        "wacc": assumptions.wacc * 100,
    }


def apply_reviewed_profile_to_base_session_state(
    state,
    ticker: str,
    snapshot,
    current_base: MultiStageDCFAssumptions,
    *,
    applied_at: str,
) -> ReviewedProfileApplication:
    """Perform the explicit UI-boundary mutation for one complete snapshot."""
    application_key = base_profile_application_key(ticker)
    previous = state.get(application_key)
    if not isinstance(previous, ReviewedProfileApplication):
        previous = None
    plan = build_profile_apply_plan(
        snapshot, current_base, previous_application=previous
    )
    application = create_reviewed_profile_application(
        plan, applied_at=applied_at
    )
    normalized_ticker = ticker.strip().upper()
    prefix = f"multistage_{normalized_ticker}_"
    values = _multistage_base_state_values(application.assumptions)
    for name, value in values.items():
        if name != "wacc":
            state[prefix + name] = value

    wacc_keys = research_wacc_session_keys(ticker)
    state[wacc_keys["value"]] = values["wacc"]
    state[wacc_keys["status"]] = "user_reviewed"
    state[wacc_keys["created_at"]] = applied_at
    wacc_review_note = next(
        (
            item.user_note.strip()
            for item in snapshot.group_reviews
            if item.group == "wacc" and item.user_note.strip()
        ),
        "",
    )
    if wacc_review_note:
        state[wacc_keys["rationale"]] = wacc_review_note
    elif not str(state.get(wacc_keys["rationale"], "")).strip():
        research_wacc = snapshot.profile.wacc_framework.research_wacc
        state[wacc_keys["rationale"]] = (
            research_wacc.rationale if research_wacc is not None else ""
        )
    state[application_key] = application
    return application


def review_and_apply_profile_to_base_session_state(
    state,
    ticker: str,
    candidate_profile,
    current_base: MultiStageDCFAssumptions,
    *,
    reviewed_at: str,
    applied_at: str,
    preview_validated: bool,
):
    """Atomically commit one complete Candidate snapshot and exact Base apply."""
    review_key = profile_review_state_key(ticker)
    application_key = base_profile_application_key(ticker)
    previous_review = state.get(review_key)
    if not isinstance(previous_review, CompanyProfileReviewState):
        previous_review = None
    previous_application = state.get(application_key)
    if not isinstance(previous_application, ReviewedProfileApplication):
        previous_application = None

    # All validation and object creation happen before any session mutation.
    result = build_one_click_review_apply(
        candidate_profile,
        current_base,
        reviewed_at=reviewed_at,
        applied_at=applied_at,
        previous_review_state=previous_review,
        previous_application=previous_application,
        preview_validated=preview_validated,
    )
    values = _multistage_base_state_values(result.assumptions)
    normalized_ticker = ticker.strip().upper()
    prefix = f"multistage_{normalized_ticker}_"
    updates = {
        prefix + name: value
        for name, value in values.items()
        if name != "wacc"
    }
    wacc_keys = research_wacc_session_keys(ticker)
    updates.update({
        wacc_keys["value"]: values["wacc"],
        wacc_keys["status"]: "user_reviewed",
        wacc_keys["created_at"]: applied_at,
        review_key: result.review_state,
        application_key: result.application,
    })
    wacc_note = next((
        item.user_note.strip()
        for item in result.reviewed_snapshot.group_reviews
        if item.group == "wacc" and item.user_note.strip()
    ), "")
    research_wacc = result.reviewed_snapshot.profile.wacc_framework.research_wacc
    updates[wacc_keys["rationale"]] = (
        wacc_note
        or str(state.get(wacc_keys["rationale"], "")).strip()
        or (research_wacc.rationale if research_wacc is not None else "")
    )

    for key, value in updates.items():
        state[key] = value
    return result


def mark_research_wacc_reviewed(
    state,
    ticker: str,
    created_at: str | None = None,
) -> None:
    """Mark review only after an explicit widget change or confirmation action."""
    keys = research_wacc_session_keys(ticker)
    state[keys["status"]] = "user_reviewed"
    state[keys["created_at"]] = created_at or pd.Timestamp.now(tz="UTC").isoformat()


def initialize_multistage_session_state(state,
                                        ticker: str,
                                        history: FundamentalHistory | None) -> dict:
    """Initialize ticker operating inputs and issuer-level Research WACC state."""
    normalized_ticker = ticker.strip().upper()
    defaults = multistage_initial_defaults(normalized_ticker, history)
    prefix = f"multistage_{normalized_ticker}_"
    for name, value in defaults.items():
        if name != "wacc":
            state.setdefault(prefix + name, value)

    keys = research_wacc_session_keys(normalized_ticker)
    legacy_wacc_key = prefix + "wacc"
    if keys["value"] not in state:
        legacy_value = state.get(legacy_wacc_key, defaults["wacc"])
        state[keys["value"]] = legacy_value
        # A non-default legacy value can only have arisen from an earlier edit.
        legacy_was_edited = (
            legacy_wacc_key in state
            and not np.isclose(float(legacy_value), float(defaults["wacc"]))
        )
        state[keys["status"]] = (
            "user_reviewed" if legacy_was_edited else "provisional_default"
        )
    else:
        state.setdefault(keys["status"], "provisional_default")
    state.setdefault(keys["rationale"], "")
    state.setdefault(keys["created_at"], None)

    values = {
        name: state[prefix + name]
        for name in defaults
        if name != "wacc"
    }
    values["wacc"] = state[keys["value"]]
    return values


def build_multistage_assumptions_from_ui(values: dict) -> MultiStageDCFAssumptions:
    """Adapt percent-form UI values to the existing pure assumptions model."""
    return MultiStageDCFAssumptions(
        forecast_years=int(values["forecast_years"]),
        near_term_revenue_growth=(
            values["year_1_growth"] / 100,
            values["year_2_growth"] / 100,
            values["year_3_growth"] / 100,
        ),
        revenue_fade_years=int(values["fade_years"]),
        terminal_growth=values["terminal_growth"] / 100,
        starting_operating_margin=values["starting_margin"] / 100,
        mature_operating_margin=values["mature_margin"] / 100,
        starting_sales_to_capital=values["starting_sales_to_capital"],
        mature_sales_to_capital=values["mature_sales_to_capital"],
        operating_tax_rate=values["tax_rate"] / 100,
        wacc=values["wacc"] / 100,
    )


def build_sensitivity_display_frame(
    sensitivity: WACCTerminalGrowthSensitivity,
) -> pd.DataFrame:
    """Build the read-only WACC-row/g-column per-share display matrix."""
    rows = []
    for wacc in sensitivity.wacc_values:
        row = {}
        for growth in sensitivity.terminal_growth_values:
            point = sensitivity.point_at(wacc, growth)
            row[growth] = (
                point.intrinsic_value_per_share
                if point is not None and point.valid
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows, index=sensitivity.wacc_values)


def _sensitivity_value_label(value: float | None) -> str:
    return f"${value:.2f}" if value is not None and np.isfinite(value) else "N/A"


def _sensitivity_delta_label(change: float | None) -> str | None:
    return f"{change:+.1%} vs base" if change is not None else None


def render_multistage_sensitivity(
    run,
    assumptions: MultiStageDCFAssumptions,
    sensitivity: WACCTerminalGrowthSensitivity | None = None,
) -> None:
    """Render a current-assumption sensitivity grid without independent state."""
    sensitivity = sensitivity or build_wacc_terminal_growth_sensitivity(
        run.inputs, assumptions
    )
    base = sensitivity.base_case_point
    st.subheader("WACC × Terminal Growth Sensitivity")
    st.caption(
        "每个格点均使用相同公司输入与经营假设，完整重跑多阶段 DCF；"
        "表中范围只是所示参数网格的敏感性范围，不是概率区间。"
    )
    if not run.per_security_valuation_supported:
        st.warning(
            _per_security_unavailable_message(
                run.per_share_unavailable_reason
            )
        )

    frame = build_sensitivity_display_frame(sensitivity)
    row_labels = {value: f"{value:.1%}" for value in frame.index}
    column_labels = {value: f"{value:.1%}" for value in frame.columns}
    display_frame = frame.rename(index=row_labels, columns=column_labels)
    base_row = row_labels[sensitivity.base_wacc]
    base_column = column_labels[sensitivity.base_terminal_growth]
    styles = pd.DataFrame("", index=display_frame.index, columns=display_frame.columns)
    styles.loc[base_row, base_column] = "background-color: #fff3b0; font-weight: bold"
    styled = display_frame.style.apply(lambda _: styles, axis=None).format(
        "${:.2f}", na_rep="N/A"
    )
    st.dataframe(styled, width="stretch")
    st.caption(
        f"Base case：WACC {sensitivity.base_wacc:.1%} · Terminal Growth "
        f"{sensitivity.base_terminal_growth:.1%} · Intrinsic Value "
        f"{_sensitivity_value_label(base.intrinsic_value_per_share)}"
    )

    coordinates = (
        ("Base Value", sensitivity.base_wacc, sensitivity.base_terminal_growth),
        ("WACC -50bp", sensitivity.base_wacc - 0.005, sensitivity.base_terminal_growth),
        ("WACC +50bp", sensitivity.base_wacc + 0.005, sensitivity.base_terminal_growth),
        ("Terminal g -50bp", sensitivity.base_wacc, sensitivity.base_terminal_growth - 0.005),
        ("Terminal g +50bp", sensitivity.base_wacc, sensitivity.base_terminal_growth + 0.005),
    )
    summary_columns = st.columns(5)
    for column, (label, wacc, growth) in zip(summary_columns, coordinates):
        impact = sensitivity.impact_at(wacc, growth)
        point = impact.point
        value = point.intrinsic_value_per_share if point and point.valid else None
        column.metric(
            label,
            _sensitivity_value_label(value),
            _sensitivity_delta_label(impact.percentage_change),
        )

    range_columns = st.columns(3)
    range_columns[0].metric("Minimum Valid Value", _sensitivity_value_label(sensitivity.min_value_per_share))
    range_columns[1].metric("Base Value", _sensitivity_value_label(base.intrinsic_value_per_share))
    range_columns[2].metric("Maximum Valid Value", _sensitivity_value_label(sensitivity.max_value_per_share))
    st.caption("Sensitivity range under displayed WACC / terminal-growth grid.")

    st.markdown("**Terminal Value / Enterprise Value context**")
    tv_columns = st.columns(5)
    for column, (label, wacc, growth) in zip(tv_columns, coordinates):
        point = sensitivity.point_at(wacc, growth)
        tv_share = point.terminal_value_share if point and point.valid else None
        column.metric(label, _diagnostic_display(tv_share))

    if sensitivity.invalid_point_count:
        st.caption(
            f"{sensitivity.invalid_point_count} grid cells are N/A because existing "
            "assumption validation rejected those combinations."
        )
        with st.expander("Unavailable sensitivity cells", expanded=False):
            for point in sensitivity.points:
                if not point.valid:
                    st.write(
                        f"WACC {point.wacc:.2%} · g {point.terminal_growth:.2%}："
                        f"{point.reason}"
                    )


@st.cache_data(show_spinner=False)
def calculate_reverse_dcf_cached(
    inputs,
    assumptions: MultiStageDCFAssumptions,
    market_price: float | None,
    ticker: str,
    base_source: str,
    range_items: tuple[tuple[str, float, float], ...] = (),
) -> ReverseDCFAnalysis:
    """Cache the expensive full-engine scan without caching mutable UI state."""
    ranges = {
        variable: ReverseResearchRange(lower, upper)
        for variable, lower, upper in range_items
    }
    return run_reverse_dcf(
        inputs,
        assumptions,
        market_price,
        ticker=ticker,
        base_source=base_source,
        research_ranges=ranges,
    )


def _reverse_value_label(variable: str, value) -> str:
    if value is None:
        return "N/A"
    if variable == GROWTH_UPLIFT:
        if isinstance(value, tuple):
            return " / ".join(f"{item:.1%}" for item in value)
        return f"{value * 100:+.2f} pp"
    if variable in {MATURE_MARGIN, WACC}:
        return f"{float(value):.2%}"
    return f"{float(value):.3f}x"


def _reverse_gap_label(result) -> str:
    if result.status != SOLVED or result.implied_value is None:
        return "N/A"
    if result.variable == GROWTH_UPLIFT:
        return f"{result.implied_value * 100:+.2f} pp each year"
    base = float(result.research_value)
    difference = result.implied_value - base
    if result.variable == MATURE_MARGIN:
        return f"{difference * 100:+.2f} pp"
    if result.variable == WACC:
        return f"{difference * 10_000:+.0f} bp"
    return f"{difference:+.3f}x"


def render_reverse_dcf(
    analysis: ReverseDCFAnalysis,
    *,
    model_risk: str | None = None,
    limitations: tuple[str, ...] = (),
) -> None:
    """Render read-only market-implied expectations from a pure result."""
    st.header("Reverse DCF — Market-Implied Expectations")
    st.caption(
        "Holding all other Research Base assumptions constant, the following "
        "values would individually reconcile the DCF with the current market price."
    )
    summary = st.columns(4)
    summary[0].metric(
        "Research Base DCF",
        f"${analysis.base_dcf_per_share:.2f}"
        if analysis.base_dcf_per_share is not None else "N/A",
    )
    summary[1].metric(
        "Market Price",
        f"${analysis.market_price:.2f}" if analysis.market_price is not None else "N/A",
    )
    summary[2].metric(
        "Price / Base DCF",
        f"{analysis.price_to_base_dcf:.2f}x"
        if analysis.price_to_base_dcf is not None else "N/A",
    )
    summary[3].metric("Reverse DCF Base", analysis.base_source)

    labels = {
        GROWTH_UPLIFT: "Y1/Y2/Y3 equal growth uplift",
        MATURE_MARGIN: "Mature Operating Margin",
        MATURE_SALES_TO_CAPITAL: "Mature Sales-to-Capital",
        WACC: "Research WACC",
    }
    rows = []
    for result in analysis.results:
        implied = result.implied_value
        if result.variable == GROWTH_UPLIFT and result.status == SOLVED:
            implied_label = (
                f"{_reverse_value_label(result.variable, implied)} → "
                f"{_reverse_value_label(result.variable, result.implied_growth_path)}"
            )
        else:
            implied_label = _reverse_value_label(result.variable, implied)
        research_range = (
            f"{_reverse_value_label(result.variable, result.research_range.lower)} – "
            f"{_reverse_value_label(result.variable, result.research_range.upper)}"
            if result.research_range is not None else "N/A"
        )
        range_labels = {
            "within_research_range": "Within Research Range",
            "above_research_range": "Above Research Range",
            "below_research_range": "Below Research Range",
            "not_available": "",
        }
        relation = range_labels.get(result.range_relation, result.range_relation)
        if relation:
            research_range = f"{research_range} · {relation}"
        status_labels = {
            "SOLVED": "Solved",
            "NO_BRACKET": "No single solution in tested range",
            "OUTSIDE_REASONABLE_RANGE": "Outside reasonable search range",
            "NON_MONOTONIC": "Non-monotonic relationship",
            "AMBIGUOUS": "Multiple possible solutions",
            "VALUATION_FAILED": "Valuation unavailable",
            "INVALID_BASE_ASSUMPTIONS": "Invalid Base assumptions",
            "MARKET_PRICE_UNAVAILABLE": "Market price unavailable",
        }
        rows.append({
            "Variable": labels[result.variable],
            "Research Base": _reverse_value_label(
                result.variable, result.research_value
            ),
            "Market-Implied": implied_label,
            "Gap": _reverse_gap_label(result),
            "Research Range": research_range,
            "Status": status_labels.get(result.status, result.status),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Search bounds：Growth Δ −30pp to +50pp · Mature Margin 0–80% · "
        "Mature S/C 0.1–5.0x · WACC terminal g + 0.5pp to 20%."
    )
    st.info(
        "Each Reverse DCF result is independent. The market does not need to "
        "satisfy all implied assumptions simultaneously."
    )
    if any(
        warning != "single_variable_results_are_not_joint_requirements"
        for warning in analysis.warnings
    ):
        st.warning(
            "The current market price requires material changes to more than one "
            "plausible economic assumption, or cannot be reconciled by a modest "
            "single-variable change."
        )
    if model_risk or limitations:
        with st.expander("Reverse DCF model context", expanded=False):
            st.write(f"Company Profile model risk：{model_risk or 'N/A'}")
            for limitation in limitations:
                st.write(f"• {limitation}")


def render_beta_robustness(
    ticker: str,
    audit: WACCAuditResult,
    current_dcf_wacc: float,
) -> BetaRobustnessAudit | None:
    """Render compact read-only beta diagnostics without beta controls."""
    try:
        beta_audit = load_beta_robustness_audit(
            ticker,
            audit.risk_free_rate,
            audit.equity_risk_premium,
            audit.after_tax_cost_of_debt,
            audit.equity_weight,
            audit.debt_weight,
            current_dcf_wacc,
        )
    except (TypeError, ValueError) as exc:
        st.caption(f"Beta robustness unavailable：{exc}")
        return None

    with st.expander("Beta Robustness", expanded=False):
        rows = []
        for estimate in beta_audit.estimates:
            rows.append({
                "Window": f"{estimate.lookback_years}Y",
                "Frequency": estimate.frequency.title(),
                "Raw Beta": estimate.raw_beta,
                "Adjusted Beta": estimate.adjusted_beta,
                "R²": estimate.r_squared,
                "Correlation": estimate.correlation,
                "Observations": estimate.observation_count,
                "95% CI Low": estimate.confidence_interval_low,
                "95% CI High": estimate.confidence_interval_high,
                "Raw-beta WACC": estimate.implied_wacc_raw,
            })
        frame = pd.DataFrame(rows).set_index(["Window", "Frequency"])
        st.dataframe(
            frame.style.format({
                "Raw Beta": "{:.3f}",
                "Adjusted Beta": "{:.3f}",
                "R²": "{:.3f}",
                "Correlation": "{:.3f}",
                "95% CI Low": "{:.3f}",
                "95% CI High": "{:.3f}",
                "Raw-beta WACC": "{:.2%}",
            }, na_rep="N/A"),
            width="stretch",
        )
        production = beta_audit.production_estimate
        st.caption(
            f"Production method reproduction：{production.raw_beta:.6f} · "
            f"current production beta：{audit.beta:.6f} · "
            f"difference：{production.raw_beta - audit.beta:+.6f}"
        )
        summary_columns = st.columns(5)
        summary_columns[0].metric(
            "Minimum Raw Beta", _sensitivity_value_label(beta_audit.minimum_raw_beta).replace("$", "")
        )
        summary_columns[1].metric(
            "Median Raw Beta", _sensitivity_value_label(beta_audit.median_raw_beta).replace("$", "")
        )
        summary_columns[2].metric(
            "Maximum Raw Beta", _sensitivity_value_label(beta_audit.maximum_raw_beta).replace("$", "")
        )
        summary_columns[3].metric(
            "Current DCF WACC implied beta",
            _sensitivity_value_label(
                beta_audit.implied_beta_for_current_dcf_wacc
            ).replace("$", ""),
        )
        summary_columns[4].metric("Classification", beta_audit.classification)
        rolling = beta_audit.rolling_beta
        if rolling.points:
            rolling_frame = pd.DataFrame(
                {
                    "Period": [point.period_end for point in rolling.points],
                    "36M Rolling Raw Beta": [point.raw_beta for point in rolling.points],
                }
            ).set_index("Period")
            st.line_chart(rolling_frame)
            st.caption(
                f"36M rolling beta · latest {rolling.latest:.3f} · "
                f"median {rolling.median:.3f} · min {rolling.minimum:.3f} · "
                f"max {rolling.maximum:.3f} · std {rolling.standard_deviation:.3f}"
            )
        alternative = beta_audit.alternative_benchmark_estimate
        if alternative is not None and alternative.available:
            st.caption(
                f"Benchmark check：5Y Monthly vs VTI raw beta "
                f"{alternative.raw_beta:.3f} (R² {alternative.r_squared:.3f})."
            )
        if beta_audit.flags:
            st.caption("Objective flags：" + "；".join(beta_audit.flags))
        st.caption(
            "Adjusted beta = 2/3 × raw beta + 1/3 × 1. It is a diagnostic "
            "shrinkage assumption and does not replace the production beta."
        )
    return beta_audit


def render_bottom_up_beta(
    ticker: str,
    wacc_audit: WACCAuditResult,
    assumptions: MultiStageDCFAssumptions,
    run,
    historical_audit: BetaRobustnessAudit | None,
) -> BottomUpBetaResult | None:
    """Render peer-derived beta evidence without changing any input state."""
    if historical_audit is None or not historical_audit.production_estimate.available:
        return None
    try:
        bottom_up = load_bottom_up_beta_audit(ticker)
    except Exception as exc:
        st.caption(f"Bottom-up beta unavailable: {exc}")
        return None
    if bottom_up is None:
        return None
    production = historical_audit.production_estimate
    context = BetaWACCContext(
        risk_free_rate=wacc_audit.risk_free_rate,
        equity_risk_premium=wacc_audit.equity_risk_premium,
        after_tax_cost_of_debt=wacc_audit.after_tax_cost_of_debt,
        equity_weight=wacc_audit.equity_weight,
        debt_weight=wacc_audit.debt_weight,
    )
    comparison = build_beta_evidence_comparison(
        inputs=run.inputs,
        base_assumptions=assumptions,
        wacc_context=context,
        historical_raw_beta=production.raw_beta,
        historical_adjusted_beta=production.adjusted_beta,
        bottom_up_result=bottom_up,
    )

    with st.expander("Bottom-Up Beta", expanded=False):
        st.caption(
            "Independent evidence for a future Research WACC decision. It does not "
            "change production beta, Formula-Based WACC, or the provisional DCF default."
        )
        st.markdown(f"**{bottom_up.peer_group_name}**")
        peer_rows = []
        for peer in bottom_up.peer_observations:
            peer_rows.append({
                "Peer": peer.ticker,
                "Levered Beta": peer.levered_beta,
                "Adjusted Beta": peer.adjusted_beta,
                "Debt / Equity": peer.debt_to_equity,
                "Tax": peer.tax_rate,
                "Unlevered Beta": peer.unlevered_beta,
                "Adjusted Unlevered": peer.adjusted_unlevered_beta,
                "Status": "valid" if peer.valid else peer.reason,
            })
        st.dataframe(
            pd.DataFrame(peer_rows).set_index("Peer").style.format({
                "Levered Beta": "{:.3f}", "Adjusted Beta": "{:.3f}",
                "Debt / Equity": "{:.2%}", "Tax": "{:.2%}",
                "Unlevered Beta": "{:.3f}", "Adjusted Unlevered": "{:.3f}",
            }, na_rep="N/A"),
            width="stretch",
        )
        for peer in bottom_up.peer_observations:
            st.caption(f"{peer.ticker}: {peer.inclusion_rationale}")
        if bottom_up.exclusion_rationales:
            st.caption(
                "Explicit exclusions: " + " | ".join(
                    f"{symbol}: {reason}"
                    for symbol, reason in bottom_up.exclusion_rationales
                )
            )

        distribution = bottom_up.raw_unlevered_distribution
        summary_columns = st.columns(5)
        summary_columns[0].metric("Valid Peers", str(bottom_up.valid_peer_count))
        summary_columns[1].metric("Peer Median Unlevered", f"{distribution.median:.3f}" if distribution.median is not None else "N/A")
        summary_columns[2].metric("Peer Mean Unlevered", f"{distribution.mean:.3f}" if distribution.mean is not None else "N/A")
        summary_columns[3].metric("Target Relevered Median", f"{bottom_up.relevered_beta_median:.3f}" if bottom_up.relevered_beta_median is not None else "N/A")
        summary_columns[4].metric("Target Relevered Mean", f"{bottom_up.relevered_beta_mean:.3f}" if bottom_up.relevered_beta_mean is not None else "N/A")
        if distribution.minimum is not None:
            st.caption(
                f"Raw peer unlevered distribution: min {distribution.minimum:.3f} · "
                f"max {distribution.maximum:.3f} · median {distribution.median:.3f} · "
                f"mean {distribution.mean:.3f} · std {distribution.standard_deviation:.3f}. "
                f"Target D/E {bottom_up.target_debt_to_equity:.2%}; relevering changes "
                f"median by {bottom_up.relevered_beta_median - distribution.median:+.3f}."
            )
        loo = bottom_up.raw_leave_one_out
        if loo.median_minimum is not None:
            st.caption(
                f"Leave-one-out target beta: median-based {loo.median_minimum:.3f}–"
                f"{loo.median_maximum:.3f}; mean-based {loo.mean_minimum:.3f}–"
                f"{loo.mean_maximum:.3f}."
            )
        adjusted = bottom_up.adjusted_unlevered_distribution
        if adjusted.median is not None:
            st.caption(
                f"Secondary adjusted-peer framework: unlevered median {adjusted.median:.3f}, "
                f"mean {adjusted.mean:.3f}; target relevered median "
                f"{bottom_up.adjusted_relevered_beta_median:.3f}, mean "
                f"{bottom_up.adjusted_relevered_beta_mean:.3f}."
            )

        evidence_rows = []
        for point in comparison.points:
            beta_label = f"{point.beta:.3f}" if point.beta is not None else "N/A"
            if point.evidence_method == "Provisional DCF Default" and point.beta is not None:
                beta_label = f"implied {point.beta:.3f}"
            evidence_rows.append({
                "Evidence Method": point.evidence_method,
                "Beta": beta_label,
                "Formula-Based WACC": point.formula_based_wacc,
                "DCF Value": point.intrinsic_value_per_share,
            })
        evidence_frame = pd.DataFrame(evidence_rows).set_index("Evidence Method")
        st.markdown("**Evidence reconciliation**")
        st.dataframe(
            evidence_frame.style.format({
                "Formula-Based WACC": "{:.2%}", "DCF Value": "${:.2f}",
            }, na_rep="N/A"),
            width="stretch",
        )
        st.caption(
            f"Historical raw beta tested range: {historical_audit.minimum_raw_beta:.3f}–"
            f"{historical_audit.maximum_raw_beta:.3f}. The first table row is not a beta "
            "method; it shows the beta mathematically implied by the Provisional DCF Default WACC."
        )
        if bottom_up.industry_references:
            industry_rows = [{
                "Industry": reference.industry,
                "Firms": reference.number_of_firms,
                "Levered Beta": reference.levered_beta,
                "Unlevered Beta": reference.unlevered_beta,
                "Debt / Equity": reference.debt_to_equity,
                "Source Date": reference.source_date,
                "Mapping Note": reference.mapping_note,
            } for reference in bottom_up.industry_references]
            st.markdown("**Damodaran industry references (independent, not selected)**")
            st.dataframe(
                pd.DataFrame(industry_rows).set_index("Industry").style.format({
                    "Levered Beta": "{:.3f}", "Unlevered Beta": "{:.3f}",
                    "Debt / Equity": "{:.2%}",
                }, na_rep="N/A"),
                width="stretch",
            )
        if bottom_up.warnings:
            st.caption("Objective warnings: " + "；".join(bottom_up.warnings))
        st.caption(f"Peer-evidence robustness: {bottom_up.classification}")
        st.caption(
            "Formula-Based WACC is the current formula with only beta changed. "
            "No beta method is automatically selected for Research WACC."
        )
    return bottom_up


def render_research_wacc_decision(
    ticker: str,
    wacc_audit: WACCAuditResult,
    historical_audit: BetaRobustnessAudit | None,
    bottom_up: BottomUpBetaResult | None,
    assumptions: MultiStageDCFAssumptions,
    provisional_default_wacc: float,
    sensitivity: WACCTerminalGrowthSensitivity,
) -> ResearchWACCDecision | None:
    """Show the user's single WACC assumption against refreshed evidence."""
    if (
        historical_audit is None
        or not historical_audit.production_estimate.available
    ):
        return None
    keys = research_wacc_session_keys(ticker)
    production = historical_audit.production_estimate
    context = BetaWACCContext(
        risk_free_rate=wacc_audit.risk_free_rate,
        equity_risk_premium=wacc_audit.equity_risk_premium,
        after_tax_cost_of_debt=wacc_audit.after_tax_cost_of_debt,
        equity_weight=wacc_audit.equity_weight,
        debt_weight=wacc_audit.debt_weight,
    )
    try:
        decision = build_research_wacc_decision(
            ticker=ticker,
            wacc_status=st.session_state[keys["status"]],
            research_wacc=assumptions.wacc,
            formula_based_wacc=wacc_audit.calculated_wacc,
            provisional_default_wacc=provisional_default_wacc,
            wacc_context=context,
            cost_of_equity_reference=wacc_audit.cost_of_equity,
            historical_raw_beta=production.raw_beta,
            historical_adjusted_beta=production.adjusted_beta,
            bottom_up_result=bottom_up,
            rationale=st.session_state[keys["rationale"]],
            created_at=st.session_state[keys["created_at"]],
        )
    except (TypeError, ValueError) as exc:
        st.caption(f"Research WACC evidence unavailable: {exc}")
        return None

    with st.expander("Research WACC Evidence & Decision", expanded=True):
        status_label = (
            "Provisional default"
            if decision.wacc_status == "provisional_default"
            else "User-reviewed Research WACC"
        )
        summary = st.columns(5)
        summary[0].metric("Research WACC", f"{decision.research_wacc:.2%}")
        summary[1].metric("Status", status_label)
        summary[2].metric("Formula-Based WACC", f"{decision.formula_based_wacc:.2%}")
        summary[3].metric(
            "Research minus Formula-Based",
            f"{decision.research_minus_formula_wacc * 100:+.2f} pp",
        )
        summary[4].metric(
            "Research WACC implied beta",
            f"{decision.research_wacc_implied_beta:.3f}"
            if decision.research_wacc_implied_beta is not None else "N/A",
        )
        st.caption(
            "The implied beta is a mechanical diagnostic; it is not a separate "
            "user-selected beta assumption."
        )

        evidence_rows = [{
            "Evidence Method": item.method,
            "Beta": item.beta,
            "Formula-Based WACC": item.formula_based_wacc,
        } for item in decision.evidence_methods]
        st.dataframe(
            pd.DataFrame(evidence_rows).set_index("Evidence Method").style.format({
                "Beta": "{:.3f}", "Formula-Based WACC": "{:.2%}",
            }),
            width="stretch",
        )
        st.caption(
            f"Observed WACC evidence range: {decision.observed_wacc_minimum:.2%}–"
            f"{decision.observed_wacc_maximum:.2%}. This is the descriptive span "
            "of displayed mechanical evidence methods."
        )
        bottom_median_label = (
            f"{decision.bottom_up_beta_median:.3f}"
            if decision.bottom_up_beta_median is not None else "N/A"
        )
        bottom_mean_label = (
            f"{decision.bottom_up_beta_mean:.3f}"
            if decision.bottom_up_beta_mean is not None else "N/A"
        )
        st.caption(
            f"Historical raw beta {decision.historical_raw_beta:.3f} · "
            f"Historical adjusted beta {decision.historical_adjusted_beta:.3f} · "
            f"Bottom-up median {bottom_median_label} · "
            f"Bottom-up mean {bottom_mean_label} · "
            f"Risk-free {decision.risk_free_rate:.2%} · ERP "
            f"{decision.equity_risk_premium:.2%}"
        )

        st.markdown("**Research WACC ±50bp valuation context**")
        valuation_columns = st.columns(3)
        for column, label, wacc_value in (
            (valuation_columns[0], "Research WACC -50bp", assumptions.wacc - 0.005),
            (valuation_columns[1], "Research WACC base", assumptions.wacc),
            (valuation_columns[2], "Research WACC +50bp", assumptions.wacc + 0.005),
        ):
            point = sensitivity.point_at(wacc_value, assumptions.terminal_growth)
            value = (
                point.intrinsic_value_per_share
                if point is not None and point.valid else None
            )
            column.metric(label, _sensitivity_value_label(value))

        beta_end = production.end_date.date() if production.end_date is not None else "N/A"
        st.caption(
            f"Evidence dates · Risk-free: {wacc_audit.risk_free_period or 'N/A'} · "
            f"ERP: {wacc_audit.erp_source} / {wacc_audit.erp_period or 'N/A'} · "
            f"Beta market data end: {beta_end} · Tax period: "
            f"{wacc_audit.tax_period or 'N/A'} · Debt period: "
            f"{wacc_audit.debt_period or 'N/A'}"
        )
        if decision.rationale:
            st.markdown("**User-authored rationale**")
            st.write(decision.rationale)
        else:
            st.caption("User-authored rationale: not provided.")
        if decision.created_at:
            st.caption(f"Review recorded: {decision.created_at}")
        if decision.warnings:
            st.info("Informational flags: " + "；".join(decision.warnings))
    return decision


def _diagnostic_display(value: float | None,
                        kind: str = "percent",
                        currency: str | None = "USD") -> str:
    if value is None or not np.isfinite(value):
        return "数据不足"
    if kind == "amount":
        absolute = abs(float(value))
        divisor, suffix = (
            (1_000_000_000_000, "T")
            if absolute >= 1_000_000_000_000
            else (1_000_000_000, "B")
        )
        amount = absolute / divisor
        sign = "-" if value < 0 else ""
        prefix = "$" if currency == "USD" else f"{currency or 'Statement currency'} "
        return f"{sign}{prefix}{amount:.1f}{suffix}"
    if kind == "multiple":
        return f"{value:.2f}x"
    return f"{value * 100:.1f}%"


def _per_security_unavailable_message(reason: str | None) -> str:
    if reason == FOREIGN_LISTING_NORMALIZATION_UNSUPPORTED:
        return (
            "Per-security DCF valuation unavailable: foreign-listing currency / "
            "ADR normalization is not currently supported."
        )
    if reason == "valuation_currency_metadata_unavailable":
        return (
            "Per-security DCF valuation unavailable: statement/security currency "
            "metadata is incomplete."
        )
    return f"Per-security DCF valuation unavailable: {reason or 'unknown_reason'}."


def _profile_evidence_display(
    evidence: ResearchEvidenceItem | None,
    *,
    kind: str = "percent",
    currency: str | None = None,
) -> str:
    if evidence is None or not evidence.available or not isinstance(
        evidence.value, (int, float)
    ):
        return "数据不足"
    return _diagnostic_display(float(evidence.value), kind, currency)


def _profile_assumption_display(
    assumption: ResearchAssumption | None,
    *,
    kind: str = "percent",
) -> str:
    if assumption is None or assumption.value is None:
        return "数据不足"
    if kind == "integer":
        return str(int(assumption.value))
    return _diagnostic_display(float(assumption.value), kind)


NVDA_REVIEW_STATE_KEY = "company_profile_review_NVDA"
NVDA_REVIEW_GROUP_LABELS = {
    "revenue": "Revenue",
    "margin": "Margin",
    "capital": "Capital Efficiency",
    "tax": "Tax",
    "wacc": "WACC",
    "terminal": "Terminal Economics",
}


def profile_review_state_key(ticker: str) -> str:
    issuer_key, _ = issuer_normalization_metadata(ticker)
    return f"company_profile_review_{issuer_key}"


def initialize_profile_review_session_state(
    session_state,
    ticker: str,
    candidate_profile,
) -> CompanyProfileReviewState:
    """Persist a generic issuer-level review state across reruns."""
    key = profile_review_state_key(ticker)
    existing = session_state.get(key)
    if not isinstance(existing, CompanyProfileReviewState):
        state = initialize_profile_review(candidate_profile)
    else:
        state = reconcile_review_state(existing, candidate_profile)
    session_state[key] = state
    return state


def initialize_nvda_review_session_state(
    session_state,
    candidate_profile,
) -> CompanyProfileReviewState:
    """Backward-compatible NVDA wrapper around the generic state helper."""
    return initialize_profile_review_session_state(
        session_state, "NVDA", candidate_profile
    )


def _review_evidence_line(profile, group: str) -> str:
    evidence = {item.evidence_id: item for item in profile.evidence_items}

    def display(evidence_id: str, kind: str = "percent") -> str:
        item = evidence.get(evidence_id)
        if item is None or not item.available:
            return "N/A"
        if not isinstance(item.value, (int, float)):
            return str(item.value)
        if kind == "amount":
            return f"{float(item.value) / 1e9:.3f}B"
        if kind == "multiple":
            return f"{float(item.value):.2f}x"
        if kind == "beta":
            return f"{float(item.value):.3f}"
        return f"{float(item.value):.2%}"

    if group == "revenue":
        anchors = profile.revenue_framework.forward_revenue_anchors
        fy1 = fy2 = "N/A"
        if anchors is not None:
            if anchors.points[0].available:
                fy1 = f"{anchors.points[0].revenue_estimate / 1e9:.3f}B"
            if anchors.points[1].available:
                fy2 = f"{anchors.points[1].revenue_estimate / 1e9:.3f}B"
        return (
            f"TTM Revenue {display('ttm_revenue', 'amount')} · "
            f"FY2027/FY2028 consensus {fy1} / {fy2} · "
            f"Q2 guidance {display('q2_fy27_revenue_guidance', 'amount')} · "
            f"Candidate {_profile_assumption_display(profile.revenue_framework.year1_growth)} / "
            f"{_profile_assumption_display(profile.revenue_framework.year2_growth)} / "
            f"{_profile_assumption_display(profile.revenue_framework.year3_growth)}"
        )
    if group == "margin":
        return (
            f"TTM Operating Margin {display('ttm_operating_margin')} · "
            f"latest annual {display('latest_annual_operating_margin')} · "
            f"Q1 FY2027 {display('q1_fy27_operating_margin')} · "
            f"Candidate mature {_profile_assumption_display(profile.margin_framework.mature_operating_margin)}"
        )
    if group == "capital":
        return (
            f"Latest S/C {display('latest_sales_to_capital', 'multiple')} · "
            f"normalized 3Y {display('sales_to_capital_3y', 'multiple')} · "
            f"Accounting ROIC {display('accounting_roic')} · "
            f"Candidate start/mature "
            f"{_profile_assumption_display(profile.capital_efficiency_framework.starting_sales_to_capital, kind='multiple')} / "
            f"{_profile_assumption_display(profile.capital_efficiency_framework.mature_sales_to_capital, kind='multiple')}"
        )
    if group == "tax":
        return (
            f"Latest annual operating tax {display('latest_operating_tax_rate')} · "
            f"FY2027 guidance midpoint {display('fy27_tax_guidance')} · "
            f"Candidate {_profile_assumption_display(profile.operating_tax_rate)}"
        )
    if group == "wacc":
        return (
            f"Formula WACC {display('formula_based_wacc')} · raw/adjusted beta "
            f"{display('historical_raw_beta', 'beta')} / "
            f"{display('historical_adjusted_beta', 'beta')} · bottom-up median "
            f"{display('bottom_up_beta_median', 'beta')} · Candidate "
            f"{_profile_assumption_display(profile.wacc_framework.research_wacc)}"
        )
    return (
        f"Terminal Growth {_profile_assumption_display(profile.terminal_framework.terminal_growth)} · "
        f"Terminal ROIC {_diagnostic_display(profile.terminal_framework.terminal_roic)} · "
        f"Terminal Reinvestment {_diagnostic_display(profile.terminal_framework.terminal_reinvestment_rate)}"
    )


def _review_group_rationale(profile, group: str) -> str:
    if group == "revenue":
        return profile.revenue_framework.near_term_growth_rationale
    if group == "margin":
        return profile.margin_framework.mature_margin_rationale
    if group == "capital":
        return profile.capital_efficiency_framework.mature_s2c_rationale
    if group == "tax":
        return profile.operating_tax_rate.rationale
    if group == "wacc":
        return profile.wacc_framework.rationale
    return profile.terminal_framework.terminal_growth_rationale


def _one_click_review_apply_callback(
    state,
    ticker: str,
    candidate_profile,
    current_base: MultiStageDCFAssumptions,
    preview_validated: bool,
) -> None:
    reviewed_at = pd.Timestamp.now(tz="UTC").isoformat()
    applied_at = pd.Timestamp.now(tz="UTC").isoformat()
    review_and_apply_profile_to_base_session_state(
        state, ticker, candidate_profile, current_base,
        reviewed_at=reviewed_at, applied_at=applied_at,
        preview_validated=preview_validated,
    )


def render_one_click_profile_workflow(
    ticker: str,
    candidate_profile,
    review_state: CompanyProfileReviewState,
    current_base: MultiStageDCFAssumptions,
    candidate_run: MultiStageDCFRunResult | None,
    *,
    show_comparison: bool = True,
) -> None:
    """Render the generic explicit one-click Review & Apply workflow."""
    st.markdown("### Research Profile Review & Apply")
    translation = build_multistage_assumptions_from_profile(candidate_profile)
    candidate = translation.assumptions
    preview_valid = candidate_run is not None
    complete = translation.available and candidate is not None and preview_valid
    previous = st.session_state.get(base_profile_application_key(ticker))
    if not isinstance(previous, ReviewedProfileApplication):
        previous = None

    snapshot = review_state.reviewed_snapshot
    same_reviewed_candidate = (
        review_state.profile_status == "reviewed"
        and snapshot is not None
        and candidate_assumption_signature(candidate_profile)
        == snapshot.assumption_signature
    )
    candidate_changed = (
        review_state.profile_status == "reviewed"
        and snapshot is not None
        and not same_reviewed_candidate
    )

    if "reviewed_profile_evidence_changed" in review_state.warnings:
        st.info(
            "Research evidence has refreshed since review, while the reviewed "
            "assumptions remain unchanged."
        )

    if candidate is not None and show_comparison:
        changes = []
        for field, label in PROFILE_APPLY_FIELD_LABELS.items():
            candidate_value = {
                "year_1_growth": candidate.near_term_revenue_growth[0],
                "year_2_growth": candidate.near_term_revenue_growth[1],
                "year_3_growth": candidate.near_term_revenue_growth[2],
                "revenue_fade_years": candidate.revenue_fade_years,
                "forecast_years": candidate.forecast_years,
                "starting_operating_margin": candidate.starting_operating_margin,
                "mature_operating_margin": candidate.mature_operating_margin,
                "starting_sales_to_capital": candidate.starting_sales_to_capital,
                "mature_sales_to_capital": candidate.mature_sales_to_capital,
                "operating_tax_rate": candidate.operating_tax_rate,
                "research_wacc": candidate.wacc,
                "terminal_growth": candidate.terminal_growth,
            }[field]
            base_value = {
                "year_1_growth": current_base.near_term_revenue_growth[0],
                "year_2_growth": current_base.near_term_revenue_growth[1],
                "year_3_growth": current_base.near_term_revenue_growth[2],
                "revenue_fade_years": current_base.revenue_fade_years,
                "forecast_years": current_base.forecast_years,
                "starting_operating_margin": current_base.starting_operating_margin,
                "mature_operating_margin": current_base.mature_operating_margin,
                "starting_sales_to_capital": current_base.starting_sales_to_capital,
                "mature_sales_to_capital": current_base.mature_sales_to_capital,
                "operating_tax_rate": current_base.operating_tax_rate,
                "research_wacc": current_base.wacc,
                "terminal_growth": current_base.terminal_growth,
            }[field]
            equal = (
                base_value == candidate_value
                if isinstance(base_value, int) and isinstance(candidate_value, int)
                else np.isclose(float(base_value), float(candidate_value))
            )
            if not equal:
                changes.append({
                    "Assumption": label,
                    "Current Base": _profile_apply_value(field, base_value),
                    "Research Candidate": _profile_apply_value(field, candidate_value),
                })
        st.markdown("**Current Base vs Research Candidate — meaningful differences**")
        if changes:
            st.dataframe(
                pd.DataFrame(changes).set_index("Assumption"), width="stretch"
            )
        else:
            st.caption("Current Base values already match the Research Candidate.")

    if not complete:
        st.warning(
            "Research Candidate is incomplete or its DCF Preview is unavailable; "
            "Review & Apply is disabled."
        )
        st.button(
            "Review & Apply Research Profile",
            key=f"one_click_review_apply_{candidate_profile.issuer_id}",
            disabled=True,
        )
        return

    if candidate_changed:
        st.info(
            "New Research Candidate available. The prior Reviewed Snapshot and "
            "Current Base remain unchanged until explicit Review & Apply."
        )
        st.button(
            "Review & Apply Updated Research Profile",
            key=f"one_click_review_apply_{candidate_profile.issuer_id}",
            type="primary",
            on_click=_one_click_review_apply_callback,
            args=(st.session_state, ticker, candidate_profile, current_base, True),
        )
        return

    if same_reviewed_candidate and snapshot is not None:
        plan = build_profile_apply_plan(
            snapshot, current_base, previous_application=previous
        )
        if plan.already_applied:
            st.success("Reviewed profile already applied")
            st.caption(
                f"Reviewed at: {previous.reviewed_at} · Applied at: "
                f"{previous.applied_at} · Source: {previous.source}"
            )
            return
        if plan.base_diverged:
            st.warning(
                "Current Base has diverged from the applied Reviewed Profile."
            )
            st.button(
                "Reapply Reviewed Profile",
                key=f"one_click_reapply_{candidate_profile.issuer_id}",
                type="primary",
                on_click=_explicit_profile_apply_callback,
                args=(st.session_state, ticker, snapshot, current_base),
            )
            return
        st.info("Reviewed Snapshot is ready and has not yet been applied.")
        st.button(
            "Apply Reviewed Profile",
            key=f"one_click_apply_reviewed_{candidate_profile.issuer_id}",
            type="primary",
            on_click=_explicit_profile_apply_callback,
            args=(st.session_state, ticker, snapshot, current_base),
        )
        return

    st.caption(
        "One deliberate action validates the Candidate, creates an immutable "
        "Reviewed Snapshot, and applies that exact snapshot to Current Base."
    )
    st.button(
        "Review & Apply Research Profile",
        key=f"one_click_review_apply_{candidate_profile.issuer_id}",
        type="primary",
        on_click=_one_click_review_apply_callback,
        args=(st.session_state, ticker, candidate_profile, current_base, True),
    )


def render_nvda_research_review(
    candidate_profile,
    state: CompanyProfileReviewState,
) -> CompanyProfileReviewState:
    """Render explicit human review controls without applying DCF inputs."""
    st.markdown("### NVDA Research Review")
    if state.profile_status == "reviewed":
        snapshot = state.reviewed_snapshot
        st.success("Status: Reviewed Research Profile")
        st.caption(f"Reviewed at: {snapshot.reviewed_at}")
        progress = " · ".join(
            f"{NVDA_REVIEW_GROUP_LABELS[group]} ✓"
            for group in REQUIRED_REVIEW_GROUPS
        )
        st.write("Review progress: " + progress)
        with st.expander("Reviewed notes", expanded=False):
            for item in snapshot.group_reviews:
                st.write(
                    f"**{NVDA_REVIEW_GROUP_LABELS[item.group]}**："
                    f"{item.user_note or 'No user note'}"
                )
            st.write(
                "**Overall review note**："
                + (snapshot.overall_review_note or "No overall note")
            )
        if "reviewed_profile_evidence_changed" in state.warnings:
            st.warning(
                "Current evidence differs from the reviewed snapshot; reviewed "
                "assumptions remain unchanged. A fresh review may be appropriate."
            )
        if "review_refresh_recommended" in state.warnings:
            st.warning(
                "The newly generated Research Candidate differs from the reviewed "
                "assumption snapshot; the reviewed profile remains unchanged."
            )
        st.info(
            "Review and application are separate actions. Review status alone does "
            "not update the Current Base DCF."
        )
        if st.button(
            "Reopen NVDA Research Review",
            key="nvda_review_reopen",
        ):
            state = reopen_profile_review(
                state,
                candidate_profile,
                reopened_at=pd.Timestamp.now(tz="UTC").isoformat(),
            )
            st.session_state[NVDA_REVIEW_STATE_KEY] = state
            for group in REQUIRED_REVIEW_GROUPS:
                st.session_state[f"nvda_review_{group}_checked"] = False
            st.rerun()
        return state

    st.info("Status: Research in Progress")
    for group in REQUIRED_REVIEW_GROUPS:
        group_state = state.group(group)
        check_key = f"nvda_review_{group}_checked"
        note_key = f"nvda_review_{group}_note"
        if check_key not in st.session_state:
            st.session_state[check_key] = group_state.reviewed
        if note_key not in st.session_state:
            st.session_state[note_key] = group_state.user_note
        label = NVDA_REVIEW_GROUP_LABELS[group]
        with st.expander(
            f"{label} {'✓' if group_state.reviewed else 'Not reviewed'}",
            expanded=False,
        ):
            st.caption("Research evidence / candidate rationale")
            st.write(_review_evidence_line(candidate_profile, group))
            st.caption(_review_group_rationale(candidate_profile, group))
            note = st.text_area(
                f"{label} user review note",
                key=note_key,
                placeholder="Optional human review note; candidate rationale remains unchanged.",
            )
            checked = st.checkbox(
                f"{label} reviewed",
                key=check_key,
            )
        if checked != group_state.reviewed or note != group_state.user_note:
            state = set_review_group(
                state,
                candidate_profile,
                group,
                reviewed=checked,
                user_note=note,
                reviewed_at=(
                    group_state.reviewed_at
                    if checked and group_state.reviewed
                    else (
                        pd.Timestamp.now(tz="UTC").isoformat()
                        if checked else None
                    )
                ),
            )

    overall_key = "nvda_review_overall_note"
    if overall_key not in st.session_state:
        st.session_state[overall_key] = state.overall_review_note
    overall_note = st.text_area(
        "Overall profile review note (optional)",
        key=overall_key,
        placeholder="Record the human judgment behind accepting this profile.",
    )
    if overall_note != state.overall_review_note:
        state = set_overall_review_note(state, overall_note)

    st.session_state[NVDA_REVIEW_STATE_KEY] = state
    progress = " · ".join(
        f"{NVDA_REVIEW_GROUP_LABELS[group]} "
        f"{'✓' if state.group(group).reviewed else '○'}"
        for group in REQUIRED_REVIEW_GROUPS
    )
    st.write("Review progress: " + progress)
    if state.incomplete_groups:
        st.caption(
            "Remaining: " + "；".join(
                NVDA_REVIEW_GROUP_LABELS[group]
                for group in state.incomplete_groups
            )
        )
    if st.button(
        "Mark NVDA Research Profile as Reviewed",
        key="nvda_review_finalize",
        disabled=not state.eligible_for_full_review,
    ):
        state = mark_profile_reviewed(
            state,
            candidate_profile,
            reviewed_at=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        st.session_state[NVDA_REVIEW_STATE_KEY] = state
        st.rerun()
    return state


PROFILE_APPLY_FIELD_LABELS = {
    "year_1_growth": "Y1 Growth",
    "year_2_growth": "Y2 Growth",
    "year_3_growth": "Y3 Growth",
    "revenue_fade_years": "Fade Years",
    "forecast_years": "Forecast Horizon",
    "starting_operating_margin": "Starting Margin",
    "mature_operating_margin": "Mature Margin",
    "starting_sales_to_capital": "Starting S/C",
    "mature_sales_to_capital": "Mature S/C",
    "operating_tax_rate": "Operating Tax",
    "research_wacc": "Research WACC",
    "terminal_growth": "Terminal Growth",
}


def _profile_apply_value(field: str, value: float | int) -> str:
    if field in {"revenue_fade_years", "forecast_years"}:
        return str(int(value))
    if field in {"starting_sales_to_capital", "mature_sales_to_capital"}:
        return f"{float(value):.2f}x"
    return f"{float(value):.2%}"


def _explicit_profile_apply_callback(
    state,
    ticker: str,
    snapshot,
    current_base: MultiStageDCFAssumptions,
) -> None:
    apply_reviewed_profile_to_base_session_state(
        state,
        ticker,
        snapshot,
        current_base,
        applied_at=pd.Timestamp.now(tz="UTC").isoformat(),
    )


def render_reviewed_profile_application(
    ticker: str,
    review_state: CompanyProfileReviewState,
    current_base: MultiStageDCFAssumptions,
) -> ProfileApplyPlan | None:
    """Render an explicit all-or-nothing Reviewed Profile application action."""
    application_key = base_profile_application_key(ticker)
    previous = st.session_state.get(application_key)
    if not isinstance(previous, ReviewedProfileApplication):
        previous = None

    if review_state.profile_status != "reviewed":
        if previous is not None:
            if assumptions_match(current_base, previous.assumptions):
                st.info(
                    "Current Base remains based on the previously applied reviewed "
                    "profile while Research Review is reopened."
                )
            else:
                st.warning(
                    "Current Base has been modified since the reviewed profile was "
                    "applied. Reopening review did not alter the Base."
                )
        return None

    snapshot = review_state.reviewed_snapshot
    plan = build_profile_apply_plan(
        snapshot, current_base, previous_application=previous
    )
    st.markdown("### Apply Reviewed Profile to Current Base")
    if not plan.available or snapshot is None:
        st.warning(
            "Reviewed profile cannot be applied: "
            + (plan.reason or "reviewed_profile_unavailable")
        )
        return plan

    if plan.changed_fields:
        changed_frame = pd.DataFrame([
            {
                "Assumption": PROFILE_APPLY_FIELD_LABELS[item.field],
                "Current Base": _profile_apply_value(
                    item.field, item.current_value
                ),
                "Reviewed Profile": _profile_apply_value(
                    item.field, item.reviewed_value
                ),
            }
            for item in plan.changed_fields
        ]).set_index("Assumption")
        st.caption(
            f"{len(plan.changed_fields)} assumption(s) will change; application "
            "is complete and cannot be partial."
        )
        st.dataframe(changed_frame, width="stretch")
    else:
        st.caption(
            "The economically relevant Current Base values already match the "
            "reviewed snapshot. Explicit Apply is still required unless provenance "
            "has already been recorded."
        )

    if plan.newer_review_available:
        st.info(
            "A newer reviewed profile is available. The previously applied Base "
            "has not been replaced automatically."
        )
    if plan.base_diverged:
        st.warning(
            "Current Base has been modified since the reviewed profile was applied. "
            "It will not be restored automatically."
        )

    if plan.already_applied:
        st.success("Reviewed NVDA Research Profile applied to Current Base DCF.")
        st.caption(
            f"Reviewed at: {previous.reviewed_at} · Applied at: "
            f"{previous.applied_at} · Source: Reviewed NVDA Research Profile"
        )
        st.button(
            "Reviewed profile already applied",
            key="nvda_reviewed_profile_apply",
            disabled=True,
        )
        return plan

    label = (
        "Reapply Reviewed NVDA Profile"
        if plan.base_diverged else "Apply Reviewed NVDA Profile to Base DCF"
    )
    st.caption(
        f"Reviewed at: {snapshot.reviewed_at} · Applied at: Not applied"
    )
    st.button(
        label,
        key="nvda_reviewed_profile_apply",
        type="primary",
        on_click=_explicit_profile_apply_callback,
        args=(st.session_state, ticker, snapshot, current_base),
    )
    return plan


def render_nvda_growth_duration_reassessment(
    reassessment: GrowthDurationReassessment,
    comparison: GrowthDurationDCFComparison | None,
) -> None:
    """Render the isolated read-only NVDA growth-duration research shadow."""
    with st.expander(
        "NVDA Growth Duration & Product-Cycle Reassessment",
        expanded=False,
    ):
        st.caption(
            "Read-only research shadow. It does not update the NVDA Research "
            "Candidate, Review state, Reviewed Snapshot, Current Base, or Apply workflow."
        )

        st.markdown("**A. Quarterly Revenue momentum**")
        quarterly = pd.DataFrame([
            {
                "Quarter": point.fiscal_quarter,
                "Period end": point.period_end,
                "Revenue (B)": point.revenue / 1e9,
                "YoY": point.yoy_growth,
                "Sequential": point.sequential_growth,
                "Source": point.source,
            }
            for point in reassessment.quarterly_revenue
        ]).set_index("Quarter")
        st.dataframe(
            quarterly.style.format({
                "Revenue (B)": "{:.3f}", "YoY": "{:.2%}",
                "Sequential": "{:.2%}",
            }, na_rep="N/A"),
            width="stretch",
        )
        st.caption(
            "The eight-quarter path decelerated during the initial Blackwell "
            "transition, then reaccelerated through Q1 FY2027; a single-quarter "
            "annualization is not treated as a forecast."
        )

        st.markdown("**B. Data Center momentum**")
        data_center = pd.DataFrame([
            {
                "Quarter": point.fiscal_quarter,
                "Revenue (B)": point.revenue / 1e9,
                "YoY": point.yoy_growth,
                "Sequential": point.sequential_growth,
                "Share of total Revenue": point.share_of_total_revenue,
            }
            for point in reassessment.data_center_revenue
        ]).set_index("Quarter")
        st.dataframe(
            data_center.style.format({
                "Revenue (B)": "{:.3f}", "YoY": "{:.2%}",
                "Sequential": "{:.2%}", "Share of total Revenue": "{:.2%}",
            }),
            width="stretch",
        )

        st.markdown("**C. Product-cycle timeline**")
        product_cycles = pd.DataFrame([
            {
                "Platform": point.platform,
                "Launch / ramp window": point.ramp_window,
                "Current evidence": point.current_evidence,
                "Revenue relevance": point.revenue_relevance,
                "Confidence": point.confidence,
                "Source": point.source,
            }
            for point in reassessment.product_cycles
        ]).set_index("Platform")
        st.dataframe(product_cycles, width="stretch")

        st.markdown("**D. Consensus, guidance and period alignment**")
        run_rate_columns = st.columns(4)
        for column, label, value in (
            (run_rate_columns[0], "Validated TTM", reassessment.run_rates.validated_ttm_revenue),
            (run_rate_columns[1], "Latest quarter ×4", reassessment.run_rates.latest_quarter_annualized),
            (run_rate_columns[2], "Guidance midpoint ×4", reassessment.run_rates.guidance_midpoint_annualized),
            (run_rate_columns[3], "FY2027 consensus", reassessment.run_rates.fy2027_consensus_revenue),
        ):
            column.metric(label, f"${value / 1e9:.3f}B" if value is not None else "N/A")
        st.caption(
            "Quarter ×4 figures are run-rate diagnostics only. FY consensus ends "
            "before the corresponding TTM-based DCF year and is not copied directly."
        )
        alignment_frame = pd.DataFrame([
            {
                "DCF Year": f"Y{item.dcf_year}",
                "DCF period end": item.dcf_period_end,
                "Fiscal consensus period": item.fiscal_consensus_period_end,
                "Alignment": item.alignment,
                "Note": item.note,
            }
            for item in reassessment.alignments
        ]).set_index("DCF Year")
        st.dataframe(alignment_frame, width="stretch")

        st.markdown("**E. Y1–Y5 research path**")
        current_growth = reassessment.current_implied_first_five_growth
        research_path = pd.DataFrame([
            {
                "Year": f"Y{item.year_index}",
                "Existing candidate / implied fade": current_growth[item.year_index - 1],
                "Slower-normalization shadow": item.growth,
                "Confidence": item.confidence,
                "Evidence": "；".join(item.evidence),
                "Rationale": item.rationale,
            }
            for item in reassessment.research_path
        ]).set_index("Year")
        st.dataframe(
            research_path.style.format({
                "Existing candidate / implied fade": "{:.2%}",
                "Slower-normalization shadow": "{:.2%}",
            }),
            width="stretch",
        )
        st.caption(
            "The shadow is an upper-duration diagnostic, not a revised stored "
            "Research Candidate. Current evidence is insufficient to update the profile."
        )

        if comparison is not None:
            st.markdown("**F–G. Existing versus five-year shadow DCF**")
            year_rows = []
            for existing_year, shadow_year, existing_pv, shadow_pv in zip(
                comparison.existing.operating_forecast.years[:5],
                comparison.shadow.operating_forecast.years[:5],
                comparison.existing.discounted_forecast.years[:5],
                comparison.shadow.discounted_forecast.years[:5],
            ):
                year_rows.append({
                    "Year": f"Y{existing_year.year_index}",
                    "Existing Revenue (B)": existing_year.revenue / 1e9,
                    "Shadow Revenue (B)": shadow_year.revenue / 1e9,
                    "Existing FCFF (B)": existing_year.fcff / 1e9,
                    "Shadow FCFF (B)": shadow_year.fcff / 1e9,
                    "Existing FCFF PV (B)": existing_pv.present_value_fcff / 1e9,
                    "Shadow FCFF PV (B)": shadow_pv.present_value_fcff / 1e9,
                })
            st.dataframe(
                pd.DataFrame(year_rows).set_index("Year").style.format("{:.3f}"),
                width="stretch",
            )

            def summary_row(label, run):
                per_share = run.per_share_value
                return {
                    "Model": label,
                    "Explicit FCFF (B)": run.operating_forecast.total_fcff / 1e9,
                    "Explicit FCFF PV (B)": run.enterprise_value.explicit_forecast_pv / 1e9,
                    "Enterprise Value (B)": run.enterprise_value.enterprise_value / 1e9,
                    "Equity Value (B)": run.equity_value.equity_value / 1e9,
                    "Intrinsic / Share": per_share.intrinsic_value_per_share if per_share else None,
                    "TV / EV": run.enterprise_value.terminal_value_share,
                }

            summary = pd.DataFrame([
                summary_row("Existing Candidate", comparison.existing),
                summary_row("Slower-normalization shadow", comparison.shadow),
            ]).set_index("Model")
            st.dataframe(
                summary.style.format({
                    "Explicit FCFF (B)": "{:.3f}",
                    "Explicit FCFF PV (B)": "{:.3f}",
                    "Enterprise Value (B)": "{:.3f}",
                    "Equity Value (B)": "{:.3f}",
                    "Intrinsic / Share": "${:.2f}",
                    "TV / EV": "{:.2%}",
                }),
                width="stretch",
            )

        st.markdown("**H. Evidence balance and decision**")
        evidence_columns = st.columns(2)
        with evidence_columns[0]:
            st.markdown("Supporting longer duration")
            for item in reassessment.supporting_evidence:
                st.write(f"• {item}")
        with evidence_columns[1]:
            st.markdown("Against longer duration")
            for item in reassessment.opposing_evidence:
                st.write(f"• {item}")
        st.warning(
            f"Growth-duration decision: {reassessment.decision}. The stored "
            "55% / 40% / 25% candidate remains unchanged."
        )


FINAL_MODEL_LIMITATIONS = {
    "AMZN": (
        "Consolidated Sales-to-Capital (S/C) may not fully capture the timing gap between AI/AWS infrastructure spending and later utilization.",
    ),
    "MU": (
        "AI/HBM structural demand reduces, but does not remove, memory pricing, utilization and cycle-normalization risk.",
    ),
    "AVGO": (
        "Software/semiconductor mix, acquisition accounting and leverage make consolidated mature economics less directly observable.",
    ),
    "AAPL": (
        "Economic S/C is a research interpretation because outsourced production, cash management and capital returns distort accounting invested capital.",
    ),
    "AMD": (
        "The consolidated model cannot separately capture GPU deployment timing, customer-warrant dilution, accelerator working capital or the GAAP/non-GAAP margin bridge.",
    ),
}


def _research_details_confidence(research_details) -> dict[str, str]:
    assessments = getattr(research_details, "confidence_assessments", ())
    return {item.category: item.confidence for item in assessments}


def _final_profile_status(
    ticker: str,
    current_assumptions: MultiStageDCFAssumptions,
    review_state: CompanyProfileReviewState | None,
) -> tuple[str, str]:
    application = st.session_state.get(base_profile_application_key(ticker))
    if isinstance(application, ReviewedProfileApplication):
        if assumptions_match(current_assumptions, application.assumptions):
            return "Applied Base", "Applied Reviewed Profile"
        return "Manual Base Divergence", "Current Manual Base"
    if review_state is not None and review_state.profile_status == "reviewed":
        return "Reviewed", "Reviewed Profile — not yet applied"
    return "Research Candidate", "Research Candidate"


def render_final_company_header(
    ticker: str,
    snapshot: CompanySnapshot,
    profile,
    research_base_run: MultiStageDCFRunResult,
    profile_state: str,
    base_source: str,
) -> None:
    """Compact, neutral header for the final research workstation."""
    name = profile.company_name if profile is not None else ticker
    st.header(f"{name} · {ticker}")
    per_share = research_base_run.per_share_value
    dcf_value = per_share.intrinsic_value_per_share if per_share is not None else None
    price = snapshot.price
    ratio = dcf_value / price if dcf_value is not None and price and price > 0 else None
    columns = st.columns(6)
    columns[0].metric("Market Price", f"${price:.2f}" if price is not None else "N/A")
    columns[1].metric("Research Base DCF", f"${dcf_value:.2f}" if dcf_value is not None else "N/A")
    columns[2].metric("DCF / Market Price", f"{ratio:.2f}x" if ratio is not None else "N/A")
    columns[3].metric("Profile State", profile_state)
    columns[4].metric("Base Source", base_source)
    columns[5].metric("Model Risk", profile.model_risk if profile and profile.model_risk else "N/A")
    st.caption(
        "Valuation gap is a neutral research diagnostic, not a recommendation or trading signal."
    )


def render_final_research_profile(
    lookup: CompanyProfileLookupResult,
    *,
    ticker: str,
    current_assumptions: MultiStageDCFAssumptions,
    candidate_run: MultiStageDCFRunResult | None,
    research_details=None,
) -> tuple[str, str]:
    """Concise final Profile summary with one Review & Apply control."""
    st.header("Research Profile")
    if (
        not lookup.available
        or lookup.profile is None
        or lookup.profile.profile_status == "provisional"
    ):
        st.info(
            "No researched Company Profile is available for this ticker. "
            "The Manual Base workspace remains available; profile and market-implied "
            "outputs are not presented as researched assumptions."
        )
        return "Manual Base", "Current Manual Base"

    profile = lookup.profile
    review_state = None
    if profile.profile_status == "research_in_progress":
        review_state = initialize_profile_review_session_state(
            st.session_state, ticker, profile
        )
    profile_state, base_source = _final_profile_status(
        ticker, current_assumptions, review_state
    )
    with st.container(border=True):
        st.markdown(f"**{profile_state}** · {profile.company_name}")
        st.caption(
            f"Research Profile state: {profile.profile_status.replace('_', ' ').title()} · "
            f"Base source: {base_source} · Model risk: {profile.model_risk or 'N/A'}"
        )

    translation = build_multistage_assumptions_from_profile(profile)
    candidate = translation.assumptions
    if candidate is None:
        st.warning("Research Profile assumptions are incomplete; valuation remains unavailable.")
        return profile_state, base_source

    confidence = _research_details_confidence(research_details)
    rows = (
        ("Y1 Growth", f"{candidate.near_term_revenue_growth[0]:.1%}", confidence.get("Y1 Growth", "N/A")),
        ("Y2 Growth", f"{candidate.near_term_revenue_growth[1]:.1%}", confidence.get("Y2 Growth", "N/A")),
        ("Y3 Growth", f"{candidate.near_term_revenue_growth[2]:.1%}", confidence.get("Y3 Growth", "N/A")),
        ("Mature Margin", f"{candidate.mature_operating_margin:.1%}", confidence.get("Mature Margin", "N/A")),
        ("Mature S/C", f"{candidate.mature_sales_to_capital:.2f}x", confidence.get("Mature S/C", "N/A")),
        ("Operating Tax", f"{candidate.operating_tax_rate:.1%}", "N/A"),
        ("Research WACC", f"{candidate.wacc:.2%}", confidence.get("WACC", confidence.get("Research WACC", "N/A"))),
        ("Terminal Growth", f"{candidate.terminal_growth:.1%}", confidence.get("Terminal Economics", "N/A")),
        ("Terminal ROIC", f"{candidate.derived_terminal_roic:.1%}", confidence.get("Terminal Economics", "N/A")),
    )
    st.dataframe(
        pd.DataFrame(rows, columns=("Assumption", "Research Candidate", "Confidence")),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Forecast structure: Y1/Y2/Y3 → {candidate.revenue_fade_years}-year deterministic fade · "
        f"{candidate.forecast_years}-year horizon. Confidence describes evidence strength, not investment quality."
    )

    if review_state is not None:
        render_one_click_profile_workflow(
            ticker,
            profile,
            review_state,
            current_assumptions,
            candidate_run,
            show_comparison=False,
        )
    return profile_state, base_source


def _final_evidence_value(item: ResearchEvidenceItem) -> str:
    if not item.available or item.value is None:
        return "N/A"
    if isinstance(item.value, (int, float)):
        if item.unit == "currency_amount":
            return _diagnostic_display(float(item.value), "amount", "USD")
        if item.unit == "ratio":
            return f"{float(item.value):.1%}"
        if item.unit in {"multiple", "x"}:
            return f"{float(item.value):.2f}x"
        return f"{float(item.value):,.2f}"
    return str(item.value)


def render_final_evidence(profile, research_details=None) -> None:
    st.header("Evidence & Research Interpretation")
    if profile is None:
        st.info("Structured research evidence is unavailable for this ticker.")
        return
    category_labels = {
        "historical_financial": "Revenue / Growth & Reported Financials",
        "forward_consensus": "Revenue / Growth",
        "management_guidance": "Revenue / Growth & Management Guidance",
        "company_specific_research": "Margin & Capital Efficiency",
        "market_risk": "WACC",
        "industry_reference": "Terminal Economics",
    }
    type_labels = {
        "historical_financial": "Reported / Disclosed",
        "management_guidance": "Reported / Disclosed",
        "forward_consensus": "External Evidence",
        "company_specific_research": "Derived Metric",
        "market_risk": "Derived Metric",
        "industry_reference": "Research Context",
    }
    confidence = _research_details_confidence(research_details)
    grouped: dict[str, list[dict]] = {}
    for item in profile.evidence_items:
        group = category_labels.get(item.category, "Other Evidence")
        source_url = item.source if isinstance(item.source, str) and item.source.startswith(("http://", "https://")) else None
        grouped.setdefault(group, []).append({
            "Summary": item.label,
            "Type": type_labels.get(item.category, "Evidence"),
            "Value": _final_evidence_value(item),
            "Period": item.period or "N/A",
            "Source": "Open source" if source_url else item.source,
            "Source URL": source_url,
            "Research Interpretation": item.notes or "Evidence only; not automatically applied.",
        })
    for group in (
        "Revenue / Growth & Reported Financials",
        "Revenue / Growth",
        "Revenue / Growth & Management Guidance",
        "Margin & Capital Efficiency",
        "WACC",
        "Terminal Economics",
        "Other Evidence",
    ):
        rows = grouped.get(group)
        if not rows:
            continue
        with st.expander(group, expanded=False):
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Source URL": st.column_config.LinkColumn(
                        "Source Link", display_text="Open"
                    )
                },
            )
    assumption_rows = []
    translation = build_multistage_assumptions_from_profile(profile)
    if translation.assumptions is not None:
        model = translation.assumptions
        assumption_rows = [
            ("Y1 / Y2 / Y3 Growth", " / ".join(f"{value:.1%}" for value in model.near_term_revenue_growth)),
            ("Mature Margin", f"{model.mature_operating_margin:.1%}"),
            ("Mature S/C", f"{model.mature_sales_to_capital:.2f}x"),
            ("Research WACC", f"{model.wacc:.2%}"),
            ("Terminal Growth", f"{model.terminal_growth:.1%}"),
        ]
    if assumption_rows:
        st.markdown("**Research Assumptions — researcher-selected, not disclosures**")
        st.dataframe(
            pd.DataFrame(assumption_rows, columns=("Assumption", "Selected Value")),
            width="stretch",
            hide_index=True,
        )


def render_final_model_limitations(profile) -> None:
    st.header("Model Limitations")
    if profile is None:
        st.info("No researched Company Profile is available; company-specific limitations are unavailable.")
        return
    ticker = profile.ticker.strip().upper()
    notes = list(FINAL_MODEL_LIMITATIONS.get(ticker, ()))
    for note in profile.uncertainty_notes:
        if "phase" not in note.lower() and "hybrid" not in note.lower() and note not in notes:
            notes.append(note)
        if len(notes) >= 4:
            break
    if not notes:
        notes.append(
            "Long-horizon growth, mature economics and discount rates remain research assumptions rather than observable facts."
        )
    for note in notes:
        st.write(f"• {note}")
    st.caption(
        "The unified production model favors transparent cross-company consistency over company-specific accounting detail."
    )


def render_final_forecast_chart(
    run: MultiStageDCFRunResult,
    statement_currency: str | None,
) -> None:
    years = [row.year_index for row in run.operating_forecast.years]
    revenues = [row.revenue / 1e9 for row in run.operating_forecast.years]
    fcff_values = [row.fcff / 1e9 for row in run.operating_forecast.years]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=years, y=revenues, mode="lines+markers", name="Revenue",
    ))
    figure.add_trace(go.Bar(
        x=years, y=fcff_values, name="FCFF", opacity=0.45,
    ))
    figure.update_layout(
        title="Forecast Revenue and FCFF",
        xaxis_title="Forecast Year",
        yaxis_title=f"{statement_currency or 'Statement currency'} B",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(figure, width="stretch")


def render_final_advanced_diagnostics(
    diagnostics,
    assumptions: MultiStageDCFAssumptions,
    statement_currency: str | None,
) -> None:
    with st.expander("Advanced Diagnostics", expanded=False):
        st.caption(
            "Detailed accounting anchors, terminal mechanics and data-quality flags."
        )
        revenue_diag, margin_diag, capital_diag = st.columns(3)
        with revenue_diag:
            st.markdown("**Revenue**")
            st.write(f"Historical CAGR 3Y：{_diagnostic_display(diagnostics.revenue.historical_cagr_3y)}")
            st.write(
                "Y1 / Y2 / Y3："
                + " / ".join(f"{value:.1%}" for value in assumptions.near_term_revenue_growth)
            )
            st.write(f"Year 5 Revenue：{_diagnostic_display(diagnostics.revenue.year_5_revenue, 'amount', statement_currency)}")
            st.write(f"Final Revenue：{_diagnostic_display(diagnostics.revenue.final_forecast_revenue, 'amount', statement_currency)}")
            st.write(f"Revenue Multiple：{_diagnostic_display(diagnostics.revenue.final_to_starting_revenue_multiple, 'multiple')}")
        with margin_diag:
            st.markdown("**Margin**")
            st.write(f"Annual / TTM：{_diagnostic_display(diagnostics.operating_margin.latest_annual_margin)} / {_diagnostic_display(diagnostics.operating_margin.latest_ttm_margin)}")
            st.write(f"Start / Year 5 / Mature：{_diagnostic_display(diagnostics.operating_margin.starting_forecast_margin)} / {_diagnostic_display(diagnostics.operating_margin.year_5_margin)} / {_diagnostic_display(diagnostics.operating_margin.mature_margin)}")
        with capital_diag:
            st.markdown("**Capital Efficiency**")
            st.write(f"Latest / 3Y：{_diagnostic_display(diagnostics.sales_to_capital.latest_annual, 'multiple')} / {_diagnostic_display(diagnostics.sales_to_capital.historical_normalized_3y, 'multiple')}")
            st.write(f"Start / Year 5 / Mature：{_diagnostic_display(diagnostics.sales_to_capital.starting_forecast, 'multiple')} / {_diagnostic_display(diagnostics.sales_to_capital.year_5, 'multiple')} / {_diagnostic_display(diagnostics.sales_to_capital.mature, 'multiple')}")

        roic_diag, cash_diag, dependency_diag = st.columns(3)
        with roic_diag:
            st.markdown("**ROIC**")
            st.write(f"Accounting：{_diagnostic_display(diagnostics.roic.current_accounting_roic)}")
            st.write(f"Year 1 / Year 5 / Terminal：{_diagnostic_display(diagnostics.roic.year_1_implied_operating_roic)} / {_diagnostic_display(diagnostics.roic.year_5_implied_operating_roic)} / {_diagnostic_display(diagnostics.roic.terminal_derived_roic)}")
        with cash_diag:
            cash = diagnostics.cash_flow_economics
            st.markdown("**Cash Flow**")
            st.write(f"Historical TTM FCF Margin：{_diagnostic_display(cash.historical_fundamental_ttm_fcf_margin)}")
            st.write(f"FCFF Margin Y1 / Y5 / Final：{_diagnostic_display(cash.year_1.fcff_margin)} / {_diagnostic_display(cash.year_5.fcff_margin if cash.year_5 else None)} / {_diagnostic_display(cash.final_year.fcff_margin)}")
            st.write(f"Terminal Reinvestment Rate：{_diagnostic_display(cash.terminal_reinvestment_rate)}")
        with dependency_diag:
            dependency = diagnostics.terminal_dependency
            st.markdown("**Valuation Dependency**")
            st.write(f"Explicit PV：{_diagnostic_display(dependency.explicit_forecast_pv, 'amount', statement_currency)}")
            st.write(f"Terminal PV：{_diagnostic_display(dependency.terminal_value_pv, 'amount', statement_currency)}")
            st.write(f"Terminal / EV：{_diagnostic_display(dependency.terminal_value_share)}")
        if diagnostics.flags:
            st.markdown("**Objective informational flags**")
            for flag in diagnostics.flags:
                st.info(MULTISTAGE_FLAG_LABELS.get(flag, flag))


def render_company_research_profile(
    lookup: CompanyProfileLookupResult,
    *,
    statement_currency: str | None,
    current_assumptions: MultiStageDCFAssumptions | None = None,
    current_run: MultiStageDCFRunResult | None = None,
    candidate_run: MultiStageDCFRunResult | None = None,
    nvda_research: NVDAResearchProfileResult | None = None,
    nvda_growth_reassessment: GrowthDurationReassessment | None = None,
    nvda_growth_comparison: GrowthDurationDCFComparison | None = None,
    alphabet_research: AlphabetResearchProfileResult | None = None,
    hyperscaler_research: HyperscalerResearchProfileResult | None = None,
    amazon_research: AmazonResearchProfileResult | None = None,
    unified_research: UnifiedCompanyResearchResult | None = None,
    current_price: float | None = None,
) -> None:
    """Render research evidence and the explicit reviewed-profile transition."""
    st.subheader("Company Research Profile 公司研究档案")
    if not lookup.available or lookup.profile is None:
        st.info(
            "该公司尚无显式 Company Research Profile；当前 DCF 仍是手动/临时假设工作流。"
        )
        return

    candidate_profile = lookup.profile
    workflow_ticker = (
        current_run.inputs.ticker
        if current_run is not None
        else candidate_profile.ticker
    )
    review_state = None
    if (
        candidate_profile.profile_status == "research_in_progress"
        and candidate_profile.issuer_id in {"NVDA", "ALPHABET_INC", "MSFT", "META", "AMZN", "MU", "AAPL", "AVGO", "AMD"}
    ):
        review_state = initialize_profile_review_session_state(
            st.session_state, workflow_ticker, candidate_profile
        )
    profile = candidate_profile
    status_labels = {
        "provisional": "Provisional 临时",
        "research_in_progress": "Research in progress 研究中",
        "reviewed": "Reviewed 已复核",
    }
    st.caption(
        f"Issuer：{profile.issuer_id} · Status："
        f"{status_labels[profile.profile_status]}"
    )
    if profile.model_risk is not None:
        st.caption(
            f"Production-model abstraction risk：{profile.model_risk} · "
            "research descriptor only, not an investment rating."
        )
    if profile.profile_status == "provisional":
        st.warning(
            "Current operating assumptions are provisional and have not yet "
            "been reviewed as company-specific research assumptions."
        )
    elif profile.profile_status == "research_in_progress":
        research_candidate_label = profile.company_name
        st.info(
            f"{research_candidate_label} Research Candidate is read-only and "
            "unreviewed. It does not "
            "change the editable Base DCF assumptions."
        )

    if (
        current_assumptions is not None
        and profile.profile_status in {"research_in_progress", "reviewed"}
    ):
        translated = build_multistage_assumptions_from_profile(profile)
        candidate = translated.assumptions
        if candidate is not None:
            evidence_labels = {
                item.evidence_id: item.label for item in profile.evidence_items
            }

            def evidence_text(assumption: ResearchAssumption | None) -> str:
                if assumption is None:
                    return "N/A"
                labels = [
                    evidence_labels.get(reference, reference)
                    for reference in assumption.evidence_references[:3]
                ]
                return "；".join(labels) if labels else "N/A"

            assumptions_rows = (
                ("Y1 Growth", current_assumptions.near_term_revenue_growth[0], candidate.near_term_revenue_growth[0], profile.revenue_framework.year1_growth, "percent"),
                ("Y2 Growth", current_assumptions.near_term_revenue_growth[1], candidate.near_term_revenue_growth[1], profile.revenue_framework.year2_growth, "percent"),
                ("Y3 Growth", current_assumptions.near_term_revenue_growth[2], candidate.near_term_revenue_growth[2], profile.revenue_framework.year3_growth, "percent"),
                ("Fade Years", current_assumptions.revenue_fade_years, candidate.revenue_fade_years, profile.revenue_framework.revenue_fade_years, "integer"),
                ("Forecast Horizon", current_assumptions.forecast_years, candidate.forecast_years, profile.forecast_years, "integer"),
                ("Starting Margin", current_assumptions.starting_operating_margin, candidate.starting_operating_margin, profile.margin_framework.starting_operating_margin, "percent"),
                ("Mature Margin", current_assumptions.mature_operating_margin, candidate.mature_operating_margin, profile.margin_framework.mature_operating_margin, "percent"),
                ("Starting S/C", current_assumptions.starting_sales_to_capital, candidate.starting_sales_to_capital, profile.capital_efficiency_framework.starting_sales_to_capital, "multiple"),
                ("Mature S/C", current_assumptions.mature_sales_to_capital, candidate.mature_sales_to_capital, profile.capital_efficiency_framework.mature_sales_to_capital, "multiple"),
                ("Operating Tax", current_assumptions.operating_tax_rate, candidate.operating_tax_rate, profile.operating_tax_rate, "percent"),
                ("Research WACC", current_assumptions.wacc, candidate.wacc, profile.wacc_framework.research_wacc, "percent"),
                ("Terminal Growth", current_assumptions.terminal_growth, candidate.terminal_growth, profile.terminal_framework.terminal_growth, "percent"),
            )
            comparison_label = (
                "Reviewed Research Profile"
                if profile.profile_status == "reviewed"
                else "Research Candidate"
            )
            comparison_frame = pd.DataFrame(
                {
                    "Assumption": [row[0] for row in assumptions_rows],
                    (
                        "Current Provisional"
                        if profile.issuer_id == "ALPHABET_INC"
                        else "Current DCF"
                    ): [
                        str(int(row[1])) if row[4] == "integer" else (
                            f"{row[1]:.2%}" if row[4] == "percent" else f"{row[1]:.2f}x"
                        ) for row in assumptions_rows
                    ],
                    comparison_label: [
                        str(int(row[2])) if row[4] == "integer" else (
                            f"{row[2]:.2%}" if row[4] == "percent" else f"{row[2]:.2f}x"
                        ) for row in assumptions_rows
                    ],
                    "Key Evidence": [evidence_text(row[3]) for row in assumptions_rows],
                }
            ).set_index("Assumption")
            st.markdown(
                f"**Current DCF / Current Base vs {comparison_label}**"
            )
            st.dataframe(comparison_frame, width="stretch")
            st.caption(
                "Research Candidate values remain separate from Base state. Only "
                "an immutable Reviewed Profile can be applied through the explicit "
                "all-or-nothing action below."
            )

            research_details = (
                nvda_research or alphabet_research or hyperscaler_research
                or amazon_research or unified_research
            )
            if research_details is not None:
                issuer_label = (
                    "Alphabet"
                    if profile.issuer_id == "ALPHABET_INC"
                    else profile.company_name
                )
                with st.expander(
                    f"{issuer_label} Revenue Evidence and Period Reconciliation",
                    expanded=False,
                ):
                    revenue_frame = pd.DataFrame([
                        {
                            "Evidence": row.label,
                            "Period": row.period,
                            "Revenue (B)": (
                                row.revenue / 1e9 if row.revenue is not None else None
                            ),
                            "Growth": row.growth,
                            "Source": row.source,
                            "Source / retrieval date": row.source_date or row.retrieved_at,
                            "Analysts": row.analyst_count,
                            "Notes": row.notes,
                        }
                        for row in research_details.revenue_evidence
                    ]).set_index("Evidence")
                    st.dataframe(
                        revenue_frame.style.format(
                            {"Revenue (B)": "{:.3f}", "Growth": "{:.2%}"},
                            na_rep="N/A",
                        ),
                        width="stretch",
                    )

                    if hasattr(research_details, "confidence_assessments"):
                        confidence_frame = pd.DataFrame([
                            {
                                "Category": item.category,
                                "Confidence": item.confidence,
                                "Rationale": item.rationale,
                            }
                            for item in research_details.confidence_assessments
                        ]).set_index("Category")
                        st.markdown("**Evidence confidence — descriptive, not a score**")
                        st.dataframe(confidence_frame, width="stretch")
                    for note in research_details.period_reconciliation:
                        st.caption(note)
                    range_frame = pd.DataFrame([
                        {
                            "DCF Year": item.assumption_id,
                            "Low evidence case": item.low,
                            "Research Candidate": item.central,
                            "High evidence case": item.high,
                            "Rationale": item.rationale,
                        }
                        for item in research_details.growth_ranges
                    ]).set_index("DCF Year")
                    st.markdown("**Research context ranges — not Bear/Base/Bull scenarios**")
                    st.dataframe(
                        range_frame.style.format({
                            "Low evidence case": "{:.2%}",
                            "Research Candidate": "{:.2%}",
                            "High evidence case": "{:.2%}",
                        }),
                        width="stretch",
                    )

                if (
                    unified_research is not None
                    and unified_research.micron_period_alignment is not None
                ):
                    alignment = unified_research.micron_period_alignment
                    with st.expander(
                        "Micron Forecast-Period Alignment Audit",
                        expanded=False,
                    ):
                        alignment_rows = [{
                            "Metric": "Current TTM",
                            "Period": alignment.ttm_period,
                            "Revenue (B)": alignment.ttm_revenue / 1e9,
                            "Overlap / construction": "FY2025 Q4 + FY2026 Q1-Q3",
                        }]
                        alignment_rows.extend({
                            "Metric": point.fiscal_year + " consensus",
                            "Period": point.fiscal_year,
                            "Revenue (B)": point.revenue / 1e9,
                            "Overlap / construction": (
                                f"Fiscal consensus; analysts {point.analyst_count}"
                                if point.analyst_count is not None
                                else "Fiscal consensus; analyst count N/A"
                            ),
                        } for point in alignment.fiscal_consensus)
                        alignment_rows.extend({
                            "Metric": f"DCF Y{point.year_index}",
                            "Period": point.period,
                            "Revenue (B)": point.revenue / 1e9,
                            "Overlap / construction": " + ".join(
                                label for label, _ in point.quarters
                            ),
                        } for point in alignment.rolling_years)
                        st.dataframe(
                            pd.DataFrame(alignment_rows).set_index("Metric").style.format(
                                {"Revenue (B)": "{:.3f}"}
                            ),
                            width="stretch",
                        )
                        st.caption(alignment.interpolation_method)
                        st.warning(
                            "Old 45% Y1 implied Revenue "
                            f"{alignment.old_y1_implied_revenue / 1e9:.3f}B, "
                            "which approximately reproduced FY2026 consensus rather "
                            "than the forward rolling year. Period-alignment error: "
                            f"{'Yes' if alignment.old_y1_alignment_error else 'No'}."
                        )
                        st.caption(
                            "The rolling path is evidence, not a price-calibrated forecast. "
                            "Only Y1/Y2/Y3 enter production; Y4 onward remains deterministic fade."
                        )

                if nvda_growth_reassessment is not None:
                    render_nvda_growth_duration_reassessment(
                        nvda_growth_reassessment,
                        nvda_growth_comparison,
                    )

                if alphabet_research is not None:
                    with st.expander(
                        "Alphabet Growth & Mature Economics Reassessment",
                        expanded=False,
                    ):
                        reassessment = alphabet_research.reassessment
                        st.caption(
                            f"Evidence as of {reassessment.evidence_as_of} · "
                            "Read-only research; no Base or Scenario state is changed."
                        )
                        st.markdown("**A. Growth Momentum**")
                        quarterly_frame = pd.DataFrame([
                            {
                                "Quarter": row.quarter,
                                "Revenue (B)": row.revenue / 1e9,
                                "YoY Growth": row.year_over_year_growth,
                                "Sequential Growth": row.sequential_growth,
                                "Source": row.source,
                            }
                            for row in reassessment.quarterly_revenue
                        ]).set_index("Quarter")
                        st.dataframe(
                            quarterly_frame.style.format({
                                "Revenue (B)": "{:.3f}",
                                "YoY Growth": "{:.2%}",
                                "Sequential Growth": "{:.2%}",
                            }, na_rep="N/A"),
                            width="stretch",
                        )
                        st.caption(
                            "Eight-quarter consolidated YoY growth accelerated "
                            "from 15% to 24%; seasonality makes sequential growth "
                            "contextual rather than a forecast anchor."
                        )
                        segment_momentum_frame = pd.DataFrame([
                            {
                                "Business": row.segment,
                                "Quarter": row.quarter,
                                "Revenue (B)": row.revenue / 1e9,
                                "YoY Growth": row.year_over_year_growth,
                                "Source": row.source,
                            }
                            for row in reassessment.segment_momentum
                        ]).set_index(["Business", "Quarter"])
                        st.dataframe(
                            segment_momentum_frame.style.format({
                                "Revenue (B)": "{:.3f}",
                                "YoY Growth": "{:.2%}",
                            }),
                            width="stretch",
                        )
                        contribution_frame = pd.DataFrame([
                            {
                                "Business": row.segment,
                                "Prior-year Revenue Weight": row.prior_year_revenue_weight,
                                "Q2 2026 YoY Growth": row.year_over_year_growth,
                                "Growth Contribution": row.consolidated_growth_contribution,
                            }
                            for row in reassessment.q2_2026_growth_contributions
                        ]).set_index("Business")
                        st.markdown("**Q2 2026 issuer-level growth contribution diagnostic**")
                        st.dataframe(
                            contribution_frame.style.format({
                                "Prior-year Revenue Weight": "{:.2%}",
                                "Q2 2026 YoY Growth": "{:.2%}",
                                "Growth Contribution": "{:.2%}",
                            }),
                            width="stretch",
                        )
                        st.caption(
                            "Contribution = prior-year segment weight × segment "
                            "growth. This reconciles issuer growth and is not a segment DCF."
                        )

                        st.markdown("**B. Capital Efficiency**")
                        capital_evidence = {
                            item.evidence_id: item for item in profile.evidence_items
                        }
                        latest_sc = profile.capital_efficiency_framework.latest_sales_to_capital
                        normalized_sc = profile.capital_efficiency_framework.normalized_3y_sales_to_capital
                        capital_columns = st.columns(4)
                        capital_columns[0].metric(
                            "Latest accounting S/C",
                            _profile_evidence_display(latest_sc, kind="multiple"),
                        )
                        capital_columns[1].metric(
                            "Normalized 3Y S/C",
                            _profile_evidence_display(normalized_sc, kind="multiple"),
                        )
                        for column, evidence_id, label in (
                            (capital_columns[2], "h1_2026_capex", "H1 2026 CapEx"),
                            (capital_columns[3], "h1_2026_depreciation", "H1 2026 Depreciation"),
                        ):
                            item = capital_evidence.get(evidence_id)
                            column.metric(
                                label,
                                _diagnostic_display(
                                    float(item.value), "amount", statement_currency
                                ) if item is not None and isinstance(item.value, (int, float)) else "N/A",
                            )
                        for index, step in enumerate(reassessment.capex_lead_lag, 1):
                            st.caption(f"{index}. {step}")

                        st.markdown("**C. Mature Economics Matrix**")
                        matrix_frame = pd.DataFrame([
                            {
                                "Mature Margin": point.mature_operating_margin,
                                "Mature S/C": point.mature_sales_to_capital,
                                "Terminal ROIC": point.terminal_roic,
                                "Terminal Reinvestment": point.terminal_reinvestment_rate,
                                "FCFF / NOPAT": point.fcff_to_nopat,
                            }
                            for point in reassessment.terminal_economics_matrix
                        ]).set_index(["Mature Margin", "Mature S/C"])
                        st.dataframe(
                            matrix_frame.style.format({
                                "Terminal ROIC": "{:.2%}",
                                "Terminal Reinvestment": "{:.2%}",
                                "FCFF / NOPAT": "{:.2%}",
                            }),
                            width="stretch",
                        )
                        st.caption(
                            "Terminal ROIC = mature margin × (1 − tax) × S/C; "
                            "reinvestment = terminal growth / ROIC. Diagnostic only."
                        )

                        st.markdown("**D. Existing vs Revised Candidate**")
                        revision_frame = pd.DataFrame([
                            {
                                "Parameter": item.parameter,
                                "Existing Candidate": (
                                    str(int(item.existing_value))
                                    if item.unit == "integer"
                                    else (
                                        f"{item.existing_value:.2%}"
                                        if item.unit == "percent"
                                        else f"{item.existing_value:.2f}x"
                                    )
                                ),
                                "Revised Candidate": (
                                    str(int(item.revised_value))
                                    if item.unit == "integer"
                                    else (
                                        f"{item.revised_value:.2%}"
                                        if item.unit == "percent"
                                        else f"{item.revised_value:.2f}x"
                                    )
                                ),
                                "Evidence": item.evidence,
                                "Why": item.rationale,
                            }
                            for item in reassessment.revisions
                        ]).set_index("Parameter")
                        st.dataframe(revision_frame, width="stretch")
                        st.caption(reassessment.revision_note)

                        if current_run is not None:
                            existing_run = run_multistage_dcf(
                                current_run.inputs,
                                reassessment.existing_candidate,
                            )
                            revised_run = run_multistage_dcf(
                                current_run.inputs,
                                reassessment.revised_candidate,
                            )
                            growth_only = run_multistage_dcf(
                                current_run.inputs,
                                replace(
                                    reassessment.existing_candidate,
                                    near_term_revenue_growth=(0.23, 0.20, 0.17),
                                ),
                            )
                            margin_only = run_multistage_dcf(
                                current_run.inputs,
                                replace(
                                    reassessment.existing_candidate,
                                    mature_operating_margin=0.34,
                                ),
                            )
                            sales_to_capital_only = run_multistage_dcf(
                                current_run.inputs,
                                replace(
                                    reassessment.existing_candidate,
                                    starting_sales_to_capital=0.50,
                                    mature_sales_to_capital=0.70,
                                ),
                            )

                            def reassessment_run_row(label, run):
                                years = run.operating_forecast.years
                                per_share = run.per_share_value
                                return {
                                    "Run": label,
                                    "Intrinsic / Share": (
                                        per_share.intrinsic_value_per_share
                                        if per_share is not None else None
                                    ),
                                    "Enterprise Value (B)": run.enterprise_value.enterprise_value / 1e9,
                                    "Equity Value (B)": run.equity_value.equity_value / 1e9,
                                    "TV / EV": run.enterprise_value.terminal_value_share,
                                    "Y1 Revenue (B)": years[0].revenue / 1e9,
                                    "Y3 Revenue (B)": years[2].revenue / 1e9,
                                    "Y5 Revenue (B)": years[4].revenue / 1e9,
                                    "Final Revenue (B)": years[-1].revenue / 1e9,
                                    "Terminal ROIC": run.assumptions.derived_terminal_roic,
                                    "Terminal Reinvestment": run.assumptions.terminal_reinvestment_rate,
                                }

                            comparison_runs = (
                                ("Existing Candidate", existing_run),
                                ("Revised Candidate", revised_run),
                                ("Growth only", growth_only),
                                ("Mature margin only", margin_only),
                                ("Sales-to-Capital only", sales_to_capital_only),
                            )
                            dcf_frame = pd.DataFrame([
                                reassessment_run_row(label, run)
                                for label, run in comparison_runs
                            ]).set_index("Run")
                            st.markdown("**Full-chain DCF and one-factor diagnostics**")
                            st.dataframe(
                                dcf_frame.style.format({
                                    "Intrinsic / Share": "${:.2f}",
                                    "Enterprise Value (B)": "{:.3f}",
                                    "Equity Value (B)": "{:.3f}",
                                    "TV / EV": "{:.2%}",
                                    "Y1 Revenue (B)": "{:.3f}",
                                    "Y3 Revenue (B)": "{:.3f}",
                                    "Y5 Revenue (B)": "{:.3f}",
                                    "Final Revenue (B)": "{:.3f}",
                                    "Terminal ROIC": "{:.2%}",
                                    "Terminal Reinvestment": "{:.2%}",
                                }),
                                width="stretch",
                            )
                            st.caption(
                                "Each row reruns the complete DCF. One-factor rows "
                                "are read-only diagnostics, not scenarios or causal attribution."
                            )

                    with st.expander(
                        "Alphabet Segment, AI Infrastructure and Capital Context",
                        expanded=False,
                    ):
                        segment_frame = pd.DataFrame([
                            {
                                "Business": row.segment,
                                "Period": row.period,
                                "Revenue (B)": (
                                    row.revenue / 1e9
                                    if row.revenue is not None else None
                                ),
                                "Revenue Growth": row.revenue_growth,
                                "Operating Income (B)": (
                                    row.operating_income / 1e9
                                    if row.operating_income is not None else None
                                ),
                                "Operating Margin": row.operating_margin,
                                "Notes": row.notes,
                            }
                            for row in alphabet_research.segment_evidence
                        ]).set_index("Business")
                        st.dataframe(
                            segment_frame.style.format({
                                "Revenue (B)": "{:.3f}",
                                "Revenue Growth": "{:.2%}",
                                "Operating Income (B)": "{:.3f}",
                                "Operating Margin": "{:.2%}",
                            }, na_rep="N/A"),
                            width="stretch",
                        )
                        evidence = {
                            item.evidence_id: item
                            for item in profile.evidence_items
                        }
                        for evidence_id in (
                            "h1_2026_capex", "ttm_capex",
                            "2026_capex_guidance", "h1_2026_depreciation",
                            "cloud_backlog", "search_ai_monetization",
                            "search_ai_disruption",
                        ):
                            item = evidence.get(evidence_id)
                            if item is not None:
                                value = (
                                    _diagnostic_display(
                                        float(item.value), "amount",
                                        statement_currency,
                                    )
                                    if isinstance(item.value, (int, float))
                                    and item.unit == "currency_amount"
                                    else str(item.value)
                                )
                                st.write(f"**{item.label}**：{value}")
                                if item.notes:
                                    st.caption(item.notes)

            with st.expander("Research Rationale, Sources and Uncertainty", expanded=False):
                for _, _, _, assumption, _ in assumptions_rows:
                    if assumption is not None:
                        st.markdown(f"**{assumption.assumption_id}** — {assumption.rationale}")
                source_rows = [
                    {
                        "Evidence": item.label,
                        "Period": item.period,
                        "Source": item.source,
                        "Source date": item.source_date,
                        "Retrieved": item.retrieved_at,
                    }
                    for item in profile.evidence_items
                    if item.category in {
                        "management_guidance", "company_specific_research",
                        "industry_reference", "market_risk",
                    }
                ]
                if source_rows:
                    st.dataframe(
                        pd.DataFrame(source_rows).set_index("Evidence"),
                        width="stretch",
                    )
                st.markdown("**Uncertainty notes**")
                for note in profile.uncertainty_notes:
                    st.write(f"• {note}")
                st.caption(
                    "Future scenario dimensions: "
                    + "；".join(profile.future_scenario_drivers)
                )

            if review_state is not None:
                render_one_click_profile_workflow(
                    workflow_ticker,
                    candidate_profile,
                    review_state,
                    current_assumptions,
                    candidate_run,
                )

            terminal = profile.terminal_framework
            diagnostic_columns = st.columns(3)
            diagnostic_columns[0].metric(
                "Terminal ROIC",
                _diagnostic_display(terminal.terminal_roic),
            )
            diagnostic_columns[1].metric(
                "Terminal Reinvestment Rate",
                _diagnostic_display(terminal.terminal_reinvestment_rate),
            )
            diagnostic_columns[2].metric(
                "Terminal FCFF / NOPAT",
                _diagnostic_display(terminal.terminal_fcff_conversion),
            )

            preview_label = (
                "Reviewed Research DCF Preview"
                if profile.profile_status == "reviewed"
                else (
                    "Alphabet Research Candidate DCF Preview"
                    if profile.issuer_id == "ALPHABET_INC"
                    else f"{profile.company_name} Research Candidate DCF Preview"
                )
            )
            st.markdown(f"**{preview_label}**")
            preview_run = candidate_run
            if preview_run is None and current_run is not None:
                preview_run = run_multistage_dcf(
                    current_run.inputs, candidate
                )
            if preview_run is None:
                st.warning(f"{preview_label} unavailable.")
            else:
                years = preview_run.operating_forecast.years
                preview_columns = st.columns(4)
                per_share = preview_run.per_share_value
                preview_columns[0].metric(
                    "Intrinsic Value / Share",
                    f"${per_share.intrinsic_value_per_share:.2f}"
                    if per_share is not None else "N/A",
                )
                preview_columns[1].metric(
                    "Enterprise Value",
                    _diagnostic_display(preview_run.enterprise_value.enterprise_value, "amount", statement_currency),
                )
                preview_columns[2].metric(
                    "Equity Value",
                    _diagnostic_display(preview_run.equity_value.equity_value, "amount", statement_currency),
                )
                preview_columns[3].metric(
                    "TV / EV",
                    _diagnostic_display(preview_run.enterprise_value.terminal_value_share),
                )
                revenue_indexes = (1, 3, 5, len(years))
                revenue_columns = st.columns(4)
                for column, index in zip(revenue_columns, revenue_indexes):
                    column.metric(
                        f"Year {index} Revenue" if index != len(years) else "Final Revenue",
                        _diagnostic_display(years[index - 1].revenue, "amount", statement_currency),
                    )
                st.caption(
                    f"Mature Margin {candidate.mature_operating_margin:.2%} · "
                    f"Mature S/C {candidate.mature_sales_to_capital:.2f}x · "
                    f"Terminal ROIC {candidate.derived_terminal_roic:.2%} · "
                    f"Terminal Reinvestment {candidate.terminal_reinvestment_rate:.2%}"
                )
                implied_y4 = years[3].revenue_growth if len(years) >= 4 else None
                implied_y5 = years[4].revenue_growth if len(years) >= 5 else None
                st.caption(
                    "Implied fade growth (not production inputs): "
                    f"Y4 {implied_y4:.2%} · Y5 {implied_y5:.2%} · "
                    f"Final explicit year {years[-1].revenue_growth:.2%} · "
                    f"Terminal {candidate.terminal_growth:.2%}"
                )
                if current_price is not None and current_price > 0 and per_share is not None:
                    st.caption(
                        f"Post-assumption market diagnostic: Candidate DCF / Market Price "
                        f"= {per_share.intrinsic_value_per_share / current_price:.2f}x "
                        f"(${current_price:.2f}); absolute gap "
                        f"${per_share.intrinsic_value_per_share - current_price:+.2f}. "
                        "Market price is not an assumption input."
                    )

                sensitivity_expander = st.expander(
                    "Research Candidate sensitivities", expanded=False
                )
                valuation_inputs = (
                    current_run.inputs if current_run is not None else preview_run.inputs
                )
                sc_step = 0.05
                sc_rows = []
                for sc in (
                    candidate.mature_sales_to_capital - 2 * sc_step,
                    candidate.mature_sales_to_capital - sc_step,
                    candidate.mature_sales_to_capital,
                    candidate.mature_sales_to_capital + sc_step,
                    candidate.mature_sales_to_capital + 2 * sc_step,
                ):
                    sensitivity_run = run_multistage_dcf(
                        valuation_inputs,
                        replace(candidate, mature_sales_to_capital=sc),
                    )
                    sc_rows.append({
                        "Mature S/C": sc,
                        "Terminal ROIC": sensitivity_run.assumptions.derived_terminal_roic,
                        "Intrinsic / Share": sensitivity_run.per_share_value.intrinsic_value_per_share if sensitivity_run.per_share_value else None,
                        "TV / EV": sensitivity_run.enterprise_value.terminal_value_share,
                    })
                sensitivity_expander.markdown("**Mature Sales-to-Capital — full-chain DCF**")
                sensitivity_expander.dataframe(pd.DataFrame(sc_rows).set_index("Mature S/C").style.format({"Terminal ROIC": "{:.2%}", "Intrinsic / Share": "${:.2f}", "TV / EV": "{:.2%}"}), width="stretch")

                y3_rows = []
                for y3 in (
                    candidate.near_term_revenue_growth[2] - 0.02,
                    candidate.near_term_revenue_growth[2],
                    candidate.near_term_revenue_growth[2] + 0.02,
                ):
                    growth = candidate.near_term_revenue_growth[:2] + (y3,)
                    sensitivity_run = run_multistage_dcf(
                        valuation_inputs,
                        replace(candidate, near_term_revenue_growth=growth),
                    )
                    sensitivity_years = sensitivity_run.operating_forecast.years
                    y3_rows.append({
                        "Y3 Growth": y3,
                        "Implied Y4": sensitivity_years[3].revenue_growth,
                        "Implied Y5": sensitivity_years[4].revenue_growth,
                        "Intrinsic / Share": sensitivity_run.per_share_value.intrinsic_value_per_share if sensitivity_run.per_share_value else None,
                        "TV / EV": sensitivity_run.enterprise_value.terminal_value_share,
                    })
                sensitivity_expander.markdown("**Y3 duration — Y1/Y2 fixed, full-chain DCF**")
                sensitivity_expander.dataframe(pd.DataFrame(y3_rows).set_index("Y3 Growth").style.format({"Implied Y4": "{:.2%}", "Implied Y5": "{:.2%}", "Intrinsic / Share": "${:.2f}", "TV / EV": "{:.2%}"}), width="stretch")
                sensitivity_expander.caption("Read-only research support. No Base, review, application, or scenario state is changed.")
                if current_run is not None:
                    current_value = (
                        current_run.per_share_value.intrinsic_value_per_share
                        if current_run.per_share_value is not None else None
                    )
                    candidate_value = (
                        per_share.intrinsic_value_per_share
                        if per_share is not None else None
                    )
                    if current_value is not None and candidate_value is not None:
                        st.caption(
                            f"{comparison_label} minus Current Base DCF: "
                            f"{candidate_value - current_value:+.2f} per share "
                            f"({candidate_value / current_value - 1:+.2%}). "
                            "Primary deltas are shown in the assumption table; no causal attribution is implied."
                        )

    with st.expander("Evidence vs Current Research Assumption", expanded=False):
        revenue, margin, capital, risk = st.columns(4)
        revenue_framework = profile.revenue_framework
        with revenue:
            st.markdown("**Revenue**")
            st.write(
                "TTM evidence："
                + _profile_evidence_display(
                    revenue_framework.ttm_revenue,
                    kind="amount",
                    currency=statement_currency,
                )
            )
            if revenue_framework.forward_revenue_anchors is not None:
                anchor = revenue_framework.forward_revenue_anchors.points[0]
                st.write(
                    "FY consensus evidence："
                    + _diagnostic_display(anchor.implied_revenue_growth)
                    if anchor.available else "FY consensus evidence：数据不足"
                )
                if anchor.fiscal_period is not None:
                    st.caption(
                        f"Fiscal period：{anchor.fiscal_period.date()} · "
                        f"Analysts：{anchor.analyst_count or 'N/A'}"
                    )
            st.write(
                "Research Y1 assumption："
                + _profile_assumption_display(revenue_framework.year1_growth)
            )
            st.caption(
                "Evidence remains read-only and is not automatically applied."
            )
        with margin:
            st.markdown("**Margin**")
            st.write(
                "TTM Operating Margin evidence："
                + _profile_evidence_display(
                    profile.margin_framework.ttm_operating_margin
                )
            )
            st.write(
                "Starting assumption："
                + _profile_assumption_display(
                    profile.margin_framework.starting_operating_margin
                )
            )
            st.write(
                "Mature assumption："
                + _profile_assumption_display(
                    profile.margin_framework.mature_operating_margin
                )
            )
        with capital:
            st.markdown("**Capital Efficiency**")
            st.write(
                "Normalized 3Y S/C evidence："
                + _profile_evidence_display(
                    profile.capital_efficiency_framework.normalized_3y_sales_to_capital,
                    kind="multiple",
                )
            )
            st.write(
                "Starting S/C assumption："
                + _profile_assumption_display(
                    profile.capital_efficiency_framework.starting_sales_to_capital,
                    kind="multiple",
                )
            )
            st.write(
                "Mature S/C assumption："
                + _profile_assumption_display(
                    profile.capital_efficiency_framework.mature_sales_to_capital,
                    kind="multiple",
                )
            )
        with risk:
            st.markdown("**Risk / WACC**")
            audit = profile.wacc_framework.wacc_audit
            st.write(
                "Formula-Based WACC evidence："
                + _diagnostic_display(
                    audit.calculated_wacc
                    if audit is not None and audit.available else None
                )
            )
            st.write(
                "Research WACC assumption："
                + _profile_assumption_display(
                    profile.wacc_framework.research_wacc
                )
            )
            st.write(
                "Operating Tax assumption："
                + _profile_assumption_display(profile.operating_tax_rate)
            )
            st.write(
                "Forecast horizon："
                + _profile_assumption_display(
                    profile.forecast_years, kind="integer"
                )
                + " years"
            )


def build_company_revenue_forecast_anchors(
    ticker: str,
    snapshot: CompanySnapshot,
    history: FundamentalHistory,
):
    """Adapt cached snapshot consensus data to source-independent anchors."""
    ttm_revenue = history.ttm.get(REVENUE)
    if ttm_revenue is not None and ttm_revenue.available and ttm_revenue.value:
        current_base = float(ttm_revenue.value)
        base_period = (
            ttm_revenue.periods_used[-1] if ttm_revenue.periods_used else None
        )
        base_kind = "ttm"
    else:
        current_base = _valid_history_value(history, REVENUE)
        base_period = (
            pd.Timestamp(history.annual.index[-1])
            if current_base is not None and not history.annual.empty else None
        )
        base_kind = "annual"
    if current_base is None:
        return None
    latest_annual = _valid_history_value(history, REVENUE)
    latest_annual_period = (
        pd.Timestamp(history.annual.index[-1])
        if latest_annual is not None and not history.annual.empty else None
    )
    return load_revenue_forecast_anchors(
        ticker=ticker,
        current_revenue_base=current_base,
        base_period=base_period,
        base_kind=base_kind,
        latest_actual_fiscal_revenue=latest_annual,
        latest_actual_fiscal_period=latest_annual_period,
        provider_data=snapshot.revenue_estimates,
        provider_as_of=snapshot.revenue_estimates_as_of,
    )


MULTISTAGE_FLAG_LABELS = {
    "forecast_start_margin_far_from_current": "预测起始利润率与当前 TTM 相差至少 5 个百分点。",
    "forecast_start_sales_to_capital_far_from_historical": "预测起始 Sales-to-Capital 与历史 3Y 相差至少 25%。",
    "terminal_roic_above_current_accounting_roic": "终值隐含 ROIC 高于当前会计 ROIC。",
    "revenue_never_reaches_terminal_growth": "显式预测期内 Revenue Growth 未达到终值增长率。",
    "final_state_not_mature": "显式预测末年尚未达到成熟 Margin 或 Sales-to-Capital。",
    "terminal_value_dominates_enterprise_value": "Terminal Value 占 Enterprise Value 超过 80%。",
}


SCENARIO_EDITABLE_FIELDS = (
    "year_1_growth",
    "year_2_growth",
    "year_3_growth",
    "fade_years",
    "terminal_growth",
    "mature_margin",
    "mature_sales_to_capital",
    "wacc",
)


def scenario_session_keys(ticker: str, scenario: str) -> dict[str, str]:
    """Return issuer-level Bear/Bull editor keys without creating Base state."""
    normalized_scenario = scenario.strip().lower()
    if normalized_scenario not in {"bear", "bull"}:
        raise ValueError("scenario must be bear or bull")
    issuer_key, _ = issuer_normalization_metadata(ticker)
    prefix = f"scenario_{issuer_key}_{normalized_scenario}_"
    keys = {field: prefix + field for field in SCENARIO_EDITABLE_FIELDS}
    keys.update({
        "status": prefix + "status",
        "rationale": prefix + "rationale",
    })
    return keys


def provisional_scenario_values(
    base: MultiStageDCFAssumptions,
    scenario: str,
) -> dict[str, float | int]:
    """Create transparent mechanical editor defaults, not recommendations.

    Bear subtracts 5 percentage points from explicit growth and mature margin,
    shortens the fade by two years, reduces terminal growth by 50bp and mature
    Sales-to-Capital by 0.20x, and adds 100bp to Research WACC. Bull applies
    the opposite economic offsets, keeps the Base fade horizon, and subtracts
    50bp from Research WACC. Values are initialization only and are not
    regenerated after user editing.
    """
    if not isinstance(base, MultiStageDCFAssumptions):
        raise TypeError("base must be MultiStageDCFAssumptions")
    normalized_scenario = scenario.strip().lower()
    if normalized_scenario not in {"bear", "bull"}:
        raise ValueError("scenario must be bear or bull")
    growth = tuple(rate * 100 for rate in base.near_term_revenue_growth)
    if len(growth) != 3:
        raise ValueError("scenario editor currently requires three near-term years")
    if normalized_scenario == "bear":
        growth_offset = -5.0
        fade_years = max(0, base.revenue_fade_years - 2)
        terminal_offset = -0.5
        margin_offset = -5.0
        capital_offset = -0.20
        wacc_offset = 1.0
    else:
        growth_offset = 5.0
        fade_years = base.revenue_fade_years
        terminal_offset = 0.5
        margin_offset = 5.0
        capital_offset = 0.20
        wacc_offset = -0.5
    return {
        "year_1_growth": growth[0] + growth_offset,
        "year_2_growth": growth[1] + growth_offset,
        "year_3_growth": growth[2] + growth_offset,
        "fade_years": fade_years,
        "terminal_growth": base.terminal_growth * 100 + terminal_offset,
        "mature_margin": base.mature_operating_margin * 100 + margin_offset,
        "mature_sales_to_capital": (
            base.mature_sales_to_capital + capital_offset
        ),
        "wacc": base.wacc * 100 + wacc_offset,
    }


def initialize_scenario_session_state(
    state,
    ticker: str,
    base: MultiStageDCFAssumptions,
) -> None:
    """Initialize issuer-level scenarios once and preserve later user edits."""
    for scenario in ("bear", "bull"):
        keys = scenario_session_keys(ticker, scenario)
        defaults = provisional_scenario_values(base, scenario)
        for field, value in defaults.items():
            state.setdefault(keys[field], value)
        state.setdefault(keys["status"], "provisional")
        state.setdefault(keys["rationale"], "")


def mark_scenario_edited(state, ticker: str, scenario: str) -> None:
    """Record only that the user edited a scenario; this is not approval."""
    state[scenario_session_keys(ticker, scenario)["status"]] = "user_edited"


def reset_scenario_session_state(
    state,
    ticker: str,
    base: MultiStageDCFAssumptions,
) -> None:
    """Reset Bear/Bull inputs only; preserve Base, main WACC, and rationales."""
    for scenario in ("bear", "bull"):
        keys = scenario_session_keys(ticker, scenario)
        rationale = state.get(keys["rationale"], "")
        for field, value in provisional_scenario_values(base, scenario).items():
            state[keys[field]] = value
        state[keys["status"]] = "provisional"
        state[keys["rationale"]] = rationale


def scenario_case_from_state(
    state,
    ticker: str,
    scenario: str,
    base: MultiStageDCFAssumptions,
):
    """Resolve UI-unit scenario state into one complete validated case."""
    keys = scenario_session_keys(ticker, scenario)
    return create_scenario_from_base(
        scenario,
        base,
        rationale=state.get(keys["rationale"], ""),
        near_term_revenue_growth=(
            float(state[keys["year_1_growth"]]) / 100,
            float(state[keys["year_2_growth"]]) / 100,
            float(state[keys["year_3_growth"]]) / 100,
        ),
        revenue_fade_years=int(state[keys["fade_years"]]),
        terminal_growth=float(state[keys["terminal_growth"]]) / 100,
        mature_operating_margin=float(state[keys["mature_margin"]]) / 100,
        mature_sales_to_capital=float(
            state[keys["mature_sales_to_capital"]]
        ),
        research_wacc=float(state[keys["wacc"]]) / 100,
    )


def build_scenario_summary_frame(
    result: MultiScenarioDCFResult,
) -> pd.DataFrame:
    """Build numeric Bear/Base/Bull summary data without presentation logic."""
    rows = {}
    for scenario in result.scenarios:
        metrics = scenario.metrics
        delta = scenario.delta_vs_base
        rows[scenario.name.title()] = {
            "Intrinsic Value / Share": (
                metrics.intrinsic_value_per_share if metrics else np.nan
            ),
            "Enterprise Value (B)": (
                metrics.enterprise_value / 1e9 if metrics else np.nan
            ),
            "Equity Value (B)": (
                metrics.equity_value / 1e9 if metrics else np.nan
            ),
            "Terminal Value / EV": (
                metrics.terminal_value_share if metrics else np.nan
            ),
            "Research WACC": metrics.research_wacc if metrics else np.nan,
            "Terminal Growth": metrics.terminal_growth if metrics else np.nan,
            "Value Delta vs Base ($)": (
                delta.intrinsic_value_difference if delta else np.nan
            ),
            "Value Delta vs Base (%)": (
                delta.intrinsic_value_percentage_difference if delta else np.nan
            ),
        }
    return pd.DataFrame(rows)


def build_scenario_economic_frame(
    result: MultiScenarioDCFResult,
) -> pd.DataFrame:
    """Build detailed economic-path comparison data from scenario metrics."""
    rows = {}
    for scenario in result.scenarios:
        metrics = scenario.metrics
        rows[scenario.name.title()] = {
            "Y1 Revenue Growth": metrics.year_1_revenue_growth if metrics else np.nan,
            "Y2 Revenue Growth": metrics.year_2_revenue_growth if metrics else np.nan,
            "Y3 Revenue Growth": metrics.year_3_revenue_growth if metrics else np.nan,
            "Revenue Fade Years": metrics.revenue_fade_years if metrics else np.nan,
            "Year 5 Revenue (B)": (
                metrics.year_5_revenue / 1e9
                if metrics and metrics.year_5_revenue is not None else np.nan
            ),
            "Final Revenue (B)": (
                metrics.final_forecast_revenue / 1e9 if metrics else np.nan
            ),
            "Final / Starting Revenue": (
                metrics.final_revenue_to_starting_revenue if metrics else np.nan
            ),
            "Year 5 Operating Margin": (
                metrics.year_5_operating_margin
                if metrics and metrics.year_5_operating_margin is not None else np.nan
            ),
            "Mature Operating Margin": (
                metrics.mature_operating_margin if metrics else np.nan
            ),
            "Year 5 Sales-to-Capital": (
                metrics.year_5_sales_to_capital
                if metrics and metrics.year_5_sales_to_capital is not None else np.nan
            ),
            "Mature Sales-to-Capital": (
                metrics.mature_sales_to_capital if metrics else np.nan
            ),
            "Terminal ROIC": metrics.terminal_roic if metrics else np.nan,
            "Year 5 FCFF Margin": (
                metrics.year_5_fcff_margin
                if metrics and metrics.year_5_fcff_margin is not None else np.nan
            ),
            "Final FCFF Margin": (
                metrics.final_year_fcff_margin
                if metrics and metrics.final_year_fcff_margin is not None else np.nan
            ),
            "Terminal FCFF / NOPAT": (
                metrics.terminal_fcff_to_nopat
                if metrics and metrics.terminal_fcff_to_nopat is not None else np.nan
            ),
        }
    return pd.DataFrame(rows)


def _render_scenario_editor(
    ticker: str,
    scenario: str,
    base: MultiStageDCFAssumptions,
) -> None:
    label = scenario.title()
    keys = scenario_session_keys(ticker, scenario)
    status = st.session_state[keys["status"]]
    with st.expander(f"{label} Case", expanded=True):
        st.caption(
            "Status: Provisional scenario defaults"
            if status == "provisional"
            else "Status: User-edited scenario"
        )
        revenue_column, economics_column = st.columns(2)
        callback = mark_scenario_edited
        callback_args = (st.session_state, ticker, scenario)
        with revenue_column:
            st.markdown("**Revenue path**")
            st.number_input(
                f"{label} Year 1 Growth (%)", step=0.5,
                key=keys["year_1_growth"], on_change=callback,
                args=callback_args,
            )
            st.number_input(
                f"{label} Year 2 Growth (%)", step=0.5,
                key=keys["year_2_growth"], on_change=callback,
                args=callback_args,
            )
            st.number_input(
                f"{label} Year 3 Growth (%)", step=0.5,
                key=keys["year_3_growth"], on_change=callback,
                args=callback_args,
            )
            st.number_input(
                f"{label} Revenue Fade Years", min_value=0,
                max_value=max(0, base.forecast_years - base.near_term_years),
                step=1, key=keys["fade_years"], on_change=callback,
                args=callback_args,
            )
            st.number_input(
                f"{label} Terminal Growth (%)", step=0.1,
                key=keys["terminal_growth"], on_change=callback,
                args=callback_args,
            )
        with economics_column:
            st.markdown("**Mature economics / risk**")
            st.number_input(
                f"{label} Mature Operating Margin (%)", step=0.5,
                key=keys["mature_margin"], on_change=callback,
                args=callback_args,
            )
            st.number_input(
                f"{label} Mature Sales-to-Capital", step=0.05,
                key=keys["mature_sales_to_capital"], on_change=callback,
                args=callback_args,
            )
            st.number_input(
                f"{label} Research WACC (%)", step=0.1,
                key=keys["wacc"], on_change=callback,
                args=callback_args,
            )
        st.text_area(
            f"{label} rationale (optional, user-authored)",
            key=keys["rationale"], max_chars=500,
            placeholder="Record the economic path represented by this case.",
            on_change=callback, args=callback_args,
        )


def _scenario_unavailable_message(scenario: ScenarioRunResult) -> str:
    reason = scenario.reason or "unknown_reason"
    return f"{scenario.name.title()} Case unavailable: {reason}"


def render_scenario_analysis(
    ticker: str,
    history: FundamentalHistory,
    base_assumptions: MultiStageDCFAssumptions,
    base_run: MultiStageDCFRunResult,
    statement_currency: str | None = "USD",
) -> MultiScenarioDCFResult:
    """Render compact scenario editing and comparison around the pure engine."""
    initialize_scenario_session_state(
        st.session_state, ticker, base_assumptions
    )
    st.header("Bear / Base / Bull Scenario Analysis")
    st.caption(
        "Base is the current Multi-Stage DCF above. Bear and Bull are explicit "
        "alternative economic paths. Provisional defaults are transparent "
        "mechanical starting values, not researched or recommended scenarios."
    )
    st.caption(
        "This scenario analysis is separate from the WACC × Terminal Growth "
        "sensitivity analysis."
    )
    st.button(
        "Reset Bear/Bull to provisional defaults",
        key=scenario_session_keys(ticker, "bear")["status"] + "_reset_both",
        on_click=reset_scenario_session_state,
        args=(st.session_state, ticker, base_assumptions),
        help="Resets Bear/Bull inputs only; Base, main Research WACC, and rationales are preserved.",
    )
    st.info(
        "Base Case uses the current main DCF assumptions directly: "
        f"Y1/Y2/Y3 {base_assumptions.near_term_revenue_growth[0]:.1%} / "
        f"{base_assumptions.near_term_revenue_growth[1]:.1%} / "
        f"{base_assumptions.near_term_revenue_growth[2]:.1%}, "
        f"Mature Margin {base_assumptions.mature_operating_margin:.1%}, "
        f"Mature S/C {base_assumptions.mature_sales_to_capital:.2f}x, "
        f"Research WACC {base_assumptions.wacc:.2%}."
    )
    st.caption(
        "Inherited current-state inputs for Bear and Bull: "
        f"Forecast Years {base_assumptions.forecast_years}; "
        f"Starting Operating Margin {base_assumptions.starting_operating_margin:.2%}; "
        f"Starting Sales-to-Capital {base_assumptions.starting_sales_to_capital:.2f}x; "
        f"Operating Tax Rate {base_assumptions.operating_tax_rate:.2%}."
    )

    bear_column, bull_column = st.columns(2)
    with bear_column:
        _render_scenario_editor(ticker, "bear", base_assumptions)
    with bull_column:
        _render_scenario_editor(ticker, "bull", base_assumptions)

    bear = scenario_case_from_state(
        st.session_state, ticker, "bear", base_assumptions
    )
    bull = scenario_case_from_state(
        st.session_state, ticker, "bull", base_assumptions
    )
    base = create_scenario_from_base(
        "base", base_assumptions, rationale="Current main Multi-Stage DCF"
    )
    result = run_multi_scenario_dcf(
        inputs=base_run.inputs,
        fundamentals=history,
        bear=bear,
        base=base,
        bull=bull,
    )

    unsupported_reasons = {
        scenario.reason
        for scenario in result.scenarios
        if scenario.reason is not None
        and scenario.metrics is not None
        and scenario.metrics.intrinsic_value_per_share is None
    }
    if len(unsupported_reasons) == 1:
        st.warning(_per_security_unavailable_message(unsupported_reasons.pop()))

    for scenario in result.scenarios:
        if not scenario.available or scenario.metrics is None:
            st.warning(_scenario_unavailable_message(scenario))
    if "scenario_value_order_unexpected" in result.warnings:
        st.info(
            "Scenario value ordering is unexpected: Bear is above Base or "
            "Bull is below Base. Review the explicit scenario assumptions."
        )

    st.subheader("Scenario Comparison")
    summary = build_scenario_summary_frame(result)
    summary_display = summary.astype(object)
    for row_name in summary.index:
        for column_name in summary.columns:
            value = summary.loc[row_name, column_name]
            if pd.isna(value):
                display = "N/A"
            elif row_name in {
                "Terminal Value / EV", "Research WACC", "Terminal Growth",
                "Value Delta vs Base (%)",
            }:
                display = f"{value:.2%}"
            elif row_name in {
                "Intrinsic Value / Share", "Value Delta vs Base ($)",
            }:
                display = f"${value:,.2f}"
            else:
                display = f"{value:,.2f}"
            summary_display.loc[row_name, column_name] = display
    st.dataframe(summary_display, width="stretch")
    st.caption(
        "Value Delta vs Base compares intrinsic values only; it is not market "
        "upside/downside or an expected return. Percentage rows are stored as decimals."
    )
    st.caption(
        "Issuer-level Enterprise Value, Equity Value and forecast Revenue are "
        f"shown in {statement_currency or 'statement-currency'} billions."
    )

    with st.expander("Economic Path Comparison", expanded=False):
        economic = build_scenario_economic_frame(result)
        economic_display = economic.astype(object)
        percentage_rows = {
            "Y1 Revenue Growth", "Y2 Revenue Growth", "Y3 Revenue Growth",
            "Year 5 Operating Margin", "Mature Operating Margin",
            "Terminal ROIC", "Year 5 FCFF Margin", "Final FCFF Margin",
            "Terminal FCFF / NOPAT",
        }
        multiple_rows = {
            "Final / Starting Revenue", "Year 5 Sales-to-Capital",
            "Mature Sales-to-Capital",
        }
        for row_name in economic.index:
            for column_name in economic.columns:
                value = economic.loc[row_name, column_name]
                if pd.isna(value):
                    display = "N/A"
                elif row_name in percentage_rows:
                    display = f"{value:.2%}"
                elif row_name in multiple_rows:
                    display = f"{value:.2f}x"
                elif row_name == "Revenue Fade Years":
                    display = f"{int(value)}"
                else:
                    display = f"{value:,.3f}"
                economic_display.loc[row_name, column_name] = display
        st.dataframe(economic_display, width="stretch")

    revenue_figure = go.Figure()
    intrinsic_names = []
    intrinsic_values = []
    for scenario in result.scenarios:
        if scenario.dcf_result is not None:
            years = [
                year.year_index
                for year in scenario.dcf_result.operating_forecast.years
            ]
            revenues = [
                year.revenue / 1e9
                for year in scenario.dcf_result.operating_forecast.years
            ]
            revenue_figure.add_trace(go.Scatter(
                x=years, y=revenues, mode="lines+markers",
                name=scenario.name.title(),
            ))
        if (
            scenario.metrics is not None
            and scenario.metrics.intrinsic_value_per_share is not None
        ):
            intrinsic_names.append(scenario.name.title())
            intrinsic_values.append(scenario.metrics.intrinsic_value_per_share)
    chart_columns = st.columns(2)
    with chart_columns[0]:
        revenue_figure.update_layout(
            title="Scenario Revenue Paths", xaxis_title="Forecast Year",
            yaxis_title="Revenue (B)",
        )
        st.plotly_chart(revenue_figure, width="stretch")
    with chart_columns[1]:
        intrinsic_figure = go.Figure(go.Bar(
            x=intrinsic_names, y=intrinsic_values,
        ))
        intrinsic_figure.update_layout(
            title="Intrinsic Value / Share by Scenario",
            yaxis_title="Intrinsic Value / Share",
        )
        st.plotly_chart(intrinsic_figure, width="stretch")
    return result


def render_multistage_dcf_panel(ticker: str,
                                snapshot: CompanySnapshot | None,
                                history: FundamentalHistory | None,
                                wacc_audit: WACCAuditResult | None = None,
                                *,
                                header_container=None,
                                profile_container=None):
    """Collect assumptions, call pure engines, and render research diagnostics."""
    st.header("Research Base DCF")
    st.caption(
        "Unified production model: Y1/Y2/Y3 Growth → deterministic fade → "
        "Mature Margin → Mature Sales-to-Capital (S/C) → standard reinvestment."
    )
    st.caption(
        "The editable Manual Base is a workspace. When an unapplied Research "
        "Candidate exists, the finished research view uses that Candidate as its Research Base."
    )
    if snapshot is None or history is None:
        st.warning("公司快照或基本面历史不可用，暂时无法运行多阶段 DCF。")
        return

    revenue_anchors = build_company_revenue_forecast_anchors(
        ticker, snapshot, history
    )

    initialize_multistage_session_state(st.session_state, ticker, history)
    prefix = f"multistage_{ticker.strip().upper()}_"
    research_keys = research_wacc_session_keys(ticker)
    provisional_default_wacc = multistage_initial_defaults(ticker, history)["wacc"] / 100
    with st.expander("Manual Base Workspace", expanded=False):
        st.caption(
            "Optional editable workspace. It is not labeled as the Research Candidate "
            "unless a reviewed profile is explicitly applied."
        )
        revenue_column, margin_column, capital_column, terminal_column = st.columns(4)
        with revenue_column:
            st.markdown("**Revenue Growth**")
            y1 = st.number_input("Year 1 Growth (%)", step=0.5, key=prefix + "year_1_growth")
            if revenue_anchors is not None:
                point = revenue_anchors.points[0]
                st.caption(
                    f"FY consensus anchor：{_diagnostic_display(point.implied_revenue_growth)}"
                    if point.available else "FY consensus anchor：数据不足"
                )
            y2 = st.number_input("Year 2 Growth (%)", step=0.5, key=prefix + "year_2_growth")
            if revenue_anchors is not None:
                point = revenue_anchors.points[1]
                st.caption(
                    f"FY consensus anchor：{_diagnostic_display(point.implied_revenue_growth)}"
                    if point.available else "FY consensus anchor：数据不足"
                )
            y3 = st.number_input("Year 3 Growth (%)", step=0.5, key=prefix + "year_3_growth")
            if revenue_anchors is not None:
                point = revenue_anchors.points[2]
                st.caption(
                    f"FY consensus anchor：{_diagnostic_display(point.implied_revenue_growth)}"
                    if point.available else "FY consensus anchor：数据不足"
                )
            fade_years = st.number_input("Revenue Fade Years", min_value=0, max_value=17, step=1, key=prefix + "fade_years")
            forecast_years = st.number_input("Forecast Years", min_value=3, max_value=20, step=1, key=prefix + "forecast_years")
        with margin_column:
            st.markdown("**Operating Margin**")
            starting_margin = st.number_input("Starting Margin (%)", step=0.5, key=prefix + "starting_margin")
            mature_margin = st.number_input("Mature Margin (%)", step=0.5, key=prefix + "mature_margin")
            st.caption("起始值仅首次从当前 TTM/年度 Margin 初始化。")
        with capital_column:
            st.markdown("**Sales-to-Capital**")
            starting_stc = st.number_input("Starting Sales-to-Capital", step=0.05, key=prefix + "starting_sales_to_capital")
            mature_stc = st.number_input("Mature Sales-to-Capital", step=0.05, key=prefix + "mature_sales_to_capital")
        with terminal_column:
            st.markdown("**Tax / WACC / Terminal**")
            tax_rate = st.number_input("Operating Tax Rate (%)", min_value=0.0, max_value=100.0, step=0.5, key=prefix + "tax_rate")
            wacc = st.number_input(
                "Research WACC (%)",
                step=0.1,
                key=research_keys["value"],
                on_change=mark_research_wacc_reviewed,
                args=(st.session_state, ticker),
            )
            status = st.session_state[research_keys["status"]]
            st.caption(
                "Status: Provisional default"
                if status == "provisional_default"
                else "Status: User-reviewed Research WACC"
            )
            st.button(
                "Confirm current Research WACC as reviewed",
                key=research_keys["status"] + "_confirm",
                on_click=mark_research_wacc_reviewed,
                args=(st.session_state, ticker),
            )
            if wacc_audit is not None and wacc_audit.available:
                st.caption(
                    f"Formula-Based WACC {wacc_audit.calculated_wacc:.2%} · "
                    f"Risk-free {wacc_audit.risk_free_rate:.2%} · "
                    f"ERP {wacc_audit.equity_risk_premium:.2%}"
                )
            terminal_growth = st.number_input("Terminal Growth (%)", step=0.1, key=prefix + "terminal_growth")
            st.caption("Near-term explicit growth years：固定为 3 年。")

        st.text_area(
            "Research WACC rationale (optional, user-authored)",
            key=research_keys["rationale"],
            max_chars=500,
            placeholder="Record the business or long-horizon risk judgment behind this WACC.",
        )

    try:
        ui_values = {
            "year_1_growth": y1, "year_2_growth": y2,
            "year_3_growth": y3, "fade_years": fade_years,
            "forecast_years": forecast_years,
            "starting_margin": starting_margin,
            "mature_margin": mature_margin,
            "starting_sales_to_capital": starting_stc,
            "mature_sales_to_capital": mature_stc,
            "tax_rate": tax_rate, "wacc": wacc,
            "terminal_growth": terminal_growth,
        }
        assumptions = build_multistage_assumptions_from_ui(ui_values)
        run = run_real_company_multistage_dcf(snapshot, history, assumptions)
        diagnostics = build_assumption_diagnostics(
            history, run.inputs, assumptions, run.forecast_path,
            run.operating_forecast, run.terminal_value, run.enterprise_value,
        )
        sensitivity = build_wacc_terminal_growth_sensitivity(
            run.inputs, assumptions
        )
    except (TypeError, ValueError) as exc:
        st.error(f"假设无法运行：{exc}")
        return

    applied_profile = st.session_state.get(base_profile_application_key(ticker))

    nvda_research = None
    nvda_growth_reassessment = None
    nvda_growth_comparison = None
    alphabet_research = None
    hyperscaler_research = None
    amazon_research = None
    unified_research = None
    candidate_run = None
    if ticker.strip().upper() == "NVDA":
        beta_audit = None
        bottom_up = None
        if wacc_audit is not None and wacc_audit.available:
            try:
                beta_audit = load_beta_robustness_audit(
                    ticker,
                    wacc_audit.risk_free_rate,
                    wacc_audit.equity_risk_premium,
                    wacc_audit.after_tax_cost_of_debt,
                    wacc_audit.equity_weight,
                    wacc_audit.debt_weight,
                    assumptions.wacc,
                )
                bottom_up = load_bottom_up_beta_audit(ticker)
            except Exception:
                # Research candidate remains usable with the available Phase 2
                # evidence; missing diagnostics stay explicitly absent.
                beta_audit = None
                bottom_up = None
        nvda_research = build_nvda_research_profile(
            assumptions,
            history,
            revenue_anchors=revenue_anchors,
            wacc_audit=wacc_audit,
            beta_audit=beta_audit,
            bottom_up_beta=bottom_up,
            retrieved_at=pd.Timestamp.now(tz="UTC").date().isoformat(),
        )
        profile_lookup = nvda_research.lookup
        translation = build_multistage_assumptions_from_profile(
            profile_lookup.profile
        )
        if translation.available and translation.assumptions is not None:
            candidate_run = run_real_company_multistage_dcf(
                snapshot, history, translation.assumptions
            )
            try:
                consensus_points = tuple(
                    ConsensusRevenuePoint(
                        fiscal_year=f"FY{pd.Timestamp(point.fiscal_period).year}",
                        period_end=pd.Timestamp(point.fiscal_period).date().isoformat(),
                        revenue=float(point.revenue_estimate),
                        implied_growth=(
                            float(point.implied_revenue_growth)
                            if point.implied_revenue_growth is not None else None
                        ),
                        analyst_count=point.analyst_count,
                        source=point.source,
                        retrieved_at=(
                            pd.Timestamp(point.source_as_of).date().isoformat()
                            if point.source_as_of is not None else "N/A"
                        ),
                    )
                    for point in (revenue_anchors.points if revenue_anchors else ())
                    if point.available and point.revenue_estimate is not None
                )
                ttm_period_end = run.inputs.starting_revenue_periods[-1]
                nvda_growth_reassessment = build_nvda_growth_reassessment(
                    translation.assumptions,
                    ttm_revenue=run.inputs.starting_revenue,
                    ttm_period_end=pd.Timestamp(ttm_period_end).date().isoformat(),
                    consensus=consensus_points,
                )
                nvda_growth_comparison = compare_growth_duration_dcf(
                    run.inputs, nvda_growth_reassessment
                )
            except (TypeError, ValueError, IndexError):
                # The existing NVDA candidate remains fully available if the
                # optional read-only reassessment cannot be constructed.
                nvda_growth_reassessment = None
                nvda_growth_comparison = None
    elif ticker.strip().upper() in {"GOOG", "GOOGL"}:
        beta_audit = None
        bottom_up = None
        if wacc_audit is not None and wacc_audit.available:
            try:
                beta_audit = load_beta_robustness_audit(
                    ticker,
                    wacc_audit.risk_free_rate,
                    wacc_audit.equity_risk_premium,
                    wacc_audit.after_tax_cost_of_debt,
                    wacc_audit.equity_weight,
                    wacc_audit.debt_weight,
                    assumptions.wacc,
                )
                bottom_up = load_bottom_up_beta_audit(ticker)
            except Exception:
                beta_audit = None
                bottom_up = None
        alphabet_research = build_alphabet_research_profile(
            assumptions,
            history,
            revenue_anchors=revenue_anchors,
            wacc_audit=wacc_audit,
            beta_audit=beta_audit,
            bottom_up_beta=bottom_up,
            retrieved_at=pd.Timestamp.now(tz="UTC").date().isoformat(),
        )
        profile_lookup = alphabet_research.lookup
        translation = build_multistage_assumptions_from_profile(
            profile_lookup.profile
        )
        if translation.available and translation.assumptions is not None:
            candidate_run = run_real_company_multistage_dcf(
                snapshot, history, translation.assumptions
            )
    elif ticker.strip().upper() in {"MSFT", "META"}:
        builder = (
            build_microsoft_research_profile
            if ticker.strip().upper() == "MSFT"
            else build_meta_research_profile
        )
        hyperscaler_research = builder(
            assumptions,
            history,
            revenue_anchors=revenue_anchors,
            wacc_audit=wacc_audit,
            retrieved_at=pd.Timestamp.now(tz="UTC").date().isoformat(),
        )
        profile_lookup = hyperscaler_research.lookup
        translation = build_multistage_assumptions_from_profile(
            profile_lookup.profile
        )
        if translation.available and translation.assumptions is not None:
            candidate_run = run_real_company_multistage_dcf(
                snapshot, history, translation.assumptions
            )
    elif ticker.strip().upper() == "AMZN":
        try:
            amazon_research = build_amazon_research_profile(
                assumptions,
                history,
                wacc_audit=wacc_audit,
                retrieved_at=pd.Timestamp.now(tz="UTC").date().isoformat(),
            )
            profile_lookup = amazon_research.lookup
            if profile_lookup.profile is not None:
                amazon_preview = run_amazon_candidate_preview(
                    run.inputs, profile_lookup.profile
                )
                amazon_research = replace(
                    amazon_research, candidate_preview=amazon_preview
                )
                candidate_run = amazon_preview
        except (TypeError, ValueError) as exc:
            profile_lookup = CompanyProfileLookupResult(
                None, False, f"amazon_candidate_unavailable:{exc}"
            )
    elif ticker.strip().upper() in {"MU", "AAPL", "AVGO", "AMD"}:
        builder = {
            "MU": build_micron_research_profile,
            "AAPL": build_apple_research_profile,
            "AVGO": build_broadcom_research_profile,
            "AMD": build_amd_research_profile,
        }[ticker.strip().upper()]
        unified_research = builder(
            assumptions,
            history,
            revenue_anchors=revenue_anchors,
            wacc_audit=wacc_audit,
            retrieved_at=pd.Timestamp.now(tz="UTC").date().isoformat(),
        )
        profile_lookup = unified_research.lookup
        translation = build_multistage_assumptions_from_profile(
            profile_lookup.profile
        )
        if translation.available and translation.assumptions is not None:
            candidate_run = run_real_company_multistage_dcf(
                snapshot, history, translation.assumptions
            )
    else:
        profile_lookup = build_provisional_company_profile(
            ticker,
            assumptions,
            history=history,
            revenue_anchors=revenue_anchors,
            wacc_audit=wacc_audit,
        )
    profile = profile_lookup.profile if profile_lookup is not None else None
    research_details = (
        nvda_research or alphabet_research or hyperscaler_research
        or amazon_research or unified_research
    )
    if isinstance(applied_profile, ReviewedProfileApplication):
        if assumptions_match(assumptions, applied_profile.assumptions):
            research_base_run = run
            research_base_source = "Applied Reviewed Profile"
        else:
            research_base_run = run
            research_base_source = "Current Manual Base"
    elif candidate_run is not None:
        research_base_run = candidate_run
        research_base_source = "Research Candidate"
    else:
        research_base_run = run
        research_base_source = "Current Manual Base"

    target_profile_container = profile_container or st.container()
    with target_profile_container:
        profile_state, displayed_source = render_final_research_profile(
            profile_lookup,
            ticker=ticker,
            current_assumptions=assumptions,
            candidate_run=candidate_run,
            research_details=research_details,
        )
    # The explicit selection above is authoritative; the status renderer uses
    # the same session semantics and should normally return the same source.
    research_base_source = displayed_source if displayed_source else research_base_source
    target_header_container = header_container or st.container()
    with target_header_container:
        render_final_company_header(
            ticker,
            snapshot,
            profile,
            research_base_run,
            profile_state,
            research_base_source,
        )

    # From this point onward every visible Base valuation, forecast,
    # sensitivity, scenario and Reverse DCF consumes the same selected Research Base.
    run = research_base_run
    assumptions = run.assumptions
    diagnostics = build_assumption_diagnostics(
        history, run.inputs, assumptions, run.forecast_path,
        run.operating_forecast, run.terminal_value, run.enterprise_value,
    )
    sensitivity = build_wacc_terminal_growth_sensitivity(run.inputs, assumptions)

    st.subheader("Base Valuation")
    st.caption(f"Research Base source: {research_base_source}")
    statement_currency = run.inputs.statement_currency
    security_currency = run.inputs.security_currency
    output_columns = st.columns(6)
    per_share = run.per_share_value
    output_columns[0].metric(
        "Intrinsic Value / Share",
        (
            f"${per_share.intrinsic_value_per_share:.2f}"
            if per_share and security_currency == "USD"
            else (
                f"{security_currency} {per_share.intrinsic_value_per_share:.2f}"
                if per_share and security_currency else "N/A"
            )
        ),
    )
    output_columns[1].metric("Enterprise Value", _diagnostic_display(run.enterprise_value.enterprise_value, "amount", statement_currency))
    output_columns[2].metric("Equity Value", _diagnostic_display(run.equity_value.equity_value, "amount", statement_currency))
    output_columns[3].metric("Explicit Forecast PV", _diagnostic_display(run.enterprise_value.explicit_forecast_pv, "amount", statement_currency))
    output_columns[4].metric("PV Terminal Value", _diagnostic_display(run.enterprise_value.terminal_value_pv, "amount", statement_currency))
    output_columns[5].metric("Terminal Value / EV", _diagnostic_display(run.enterprise_value.terminal_value_share))
    shares = run.inputs.normalized_share_count
    share_period = shares.source_period.date() if shares.source_period is not None else "current metadata"
    if not run.per_security_valuation_supported:
        st.warning(
            _per_security_unavailable_message(run.per_share_unavailable_reason)
        )
        if shares.available:
            st.caption(
                f"Observed issuer-share denominator："
                f"{shares.shares_outstanding / 1_000_000_000:.6f}B · "
                f"{shares.source} · {share_period}; it is not converted to the "
                "displayed security unit."
            )
    elif shares.available:
        st.caption(
            f"Per-share denominator：{shares.shares_outstanding / 1_000_000_000:.6f}B shares · "
            f"{shares.scope} · {shares.source} · {share_period}"
        )
        if "multi_class_issuer" in shares.warnings:
            st.info("使用合并普通股股数；未对不同投票权类别设置溢价或折价。")
    else:
        st.warning("合并普通股数不可用；Enterprise Value 与 Equity Value 可用，但不显示每股价值。")
    st.caption(f"WACC − Terminal Growth：{(assumptions.wacc - assumptions.terminal_growth) * 100:.2f} percentage points")
    st.markdown("**DCF Value Bridge**")
    bridge_rows = [
        ("Explicit FCFF PV", run.enterprise_value.explicit_forecast_pv),
        ("+ Terminal PV", run.enterprise_value.terminal_value_pv),
        ("= Enterprise Value", run.enterprise_value.enterprise_value),
        ("− Net Debt", run.inputs.net_debt),
        ("= Equity Value", run.equity_value.equity_value),
    ]
    bridge_frame = pd.DataFrame(
        {
            "Bridge Step": [label for label, _ in bridge_rows],
            "Value": [
                _diagnostic_display(value, "amount", statement_currency)
                for _, value in bridge_rows
            ],
        }
    )
    if shares.available and per_share is not None:
        bridge_frame.loc[len(bridge_frame)] = (
            "÷ Shares",
            f"{shares.shares_outstanding / 1_000_000_000:.2f}B",
        )
        bridge_frame.loc[len(bridge_frame)] = (
            "= Research Base DCF / Share",
            f"${per_share.intrinsic_value_per_share:.2f}",
        )
    st.dataframe(bridge_frame, width="stretch", hide_index=True)

    if wacc_audit is not None and wacc_audit.available:
        with st.expander("WACC Calculation Details", expanded=False):
            st.markdown(
                f"CAPM：{wacc_audit.risk_free_rate:.2%} + "
                f"{wacc_audit.beta:.3f} × {wacc_audit.equity_risk_premium:.2%} "
                f"= **{wacc_audit.cost_of_equity:.2%}**  \n"
                f"Debt：{wacc_audit.pre_tax_cost_of_debt:.2%} × "
                f"(1 − {wacc_audit.tax_rate:.2%}) = "
                f"**{wacc_audit.after_tax_cost_of_debt:.2%}**  \n"
                f"WACC：{wacc_audit.equity_weight:.2%} × "
                f"{wacc_audit.cost_of_equity:.2%} + "
                f"{wacc_audit.debt_weight:.2%} × "
                f"{wacc_audit.after_tax_cost_of_debt:.2%} = "
                f"**{wacc_audit.calculated_wacc:.2%}**"
            )
            audit_columns = st.columns(3)
            audit_columns[0].metric(
                "Formula-Based WACC", f"{wacc_audit.calculated_wacc:.2%}"
            )
            applied_wacc_label = (
                "Reviewed Profile Research WACC"
                if isinstance(applied_profile, ReviewedProfileApplication)
                and assumptions_match(assumptions, applied_profile.assumptions)
                else "Current Research WACC"
            )
            audit_columns[1].metric(applied_wacc_label, f"{assumptions.wacc:.2%}")
            audit_columns[2].metric(
                "Research minus Formula-Based",
                f"{(assumptions.wacc - wacc_audit.calculated_wacc) * 100:+.2f} pp",
            )
            st.caption(
                f"Risk-free：{wacc_audit.risk_free_source} · "
                f"{wacc_audit.risk_free_period or 'N/A'} | "
                f"Beta：{wacc_audit.beta_source} · "
                f"{wacc_audit.beta_observations} observations | "
                f"ERP：{wacc_audit.erp_source} · {wacc_audit.erp_period or 'N/A'}"
            )
            st.caption(
                f"Equity contribution：{wacc_audit.equity_contribution:.2%} · "
                f"Debt contribution：{wacc_audit.debt_contribution:.2%} · "
                f"Market cap：{_diagnostic_display(wacc_audit.market_cap, 'amount', security_currency)} · "
                f"Gross debt：{_diagnostic_display(wacc_audit.debt_value, 'amount', statement_currency)}"
            )
            if wacc_audit.fallbacks_used:
                st.caption("Fallbacks：" + "；".join(wacc_audit.fallbacks_used))
            else:
                st.caption("Fallbacks：none")

    path_rows = []
    for operating, discounted in zip(
        run.operating_forecast.years, run.discounted_forecast.years
    ):
        path_rows.append({
            "Year": operating.year_index, "Stage": operating.stage,
            "Revenue Growth": operating.revenue_growth,
            "Revenue (B)": operating.revenue / 1e9,
            "Operating Margin": operating.operating_margin,
            "NOPAT (B)": operating.nopat / 1e9,
            "S/C": operating.sales_to_capital,
            "Reinvestment (B)": operating.reinvestment / 1e9,
            "FCFF (B)": operating.fcff / 1e9,
        })
    path_frame = pd.DataFrame(path_rows).set_index("Year")
    st.subheader("Forecast & Assumption Diagnostics")
    with st.expander("Annual Forecast Path", expanded=True):
        st.dataframe(
            path_frame.style.format({
                "Revenue Growth": "{:.1%}", "Revenue (B)": "{:.1f}",
                "Operating Margin": "{:.1%}", "NOPAT (B)": "{:.1f}",
                "S/C": "{:.2f}x", "Reinvestment (B)": "{:.1f}",
                "FCFF (B)": "{:.1f}",
            }),
            width="stretch",
        )

    if revenue_anchors is not None:
        st.subheader("Near-Term Revenue Forecast Anchors")
        st.caption(
            "External analyst consensus is evidence only. Current DCF uses a TTM starting base, "
            "while these estimates are fiscal-year levels; mismatched periods are not forced into a delta."
        )
        normalized_estimates = revenue_anchors_to_forward_estimate_set(
            revenue_anchors, retrieved_at=snapshot.revenue_estimates_as_of
        )
        dcf_periods = build_dcf_revenue_forecast_periods(
            run.inputs.starting_revenue_periods[-1],
            run.operating_forecast.years,
        )
        anchor_rows = []
        prior_consensus_period = normalized_estimates.latest_actual_fiscal_period
        for dcf_period, estimate in zip(dcf_periods, normalized_estimates.estimates):
            alignment = align_dcf_and_consensus_period(
                dcf_period, estimate, prior_consensus_period,
            )
            point = compare_aligned_forward_estimate(
                dcf_period, estimate, alignment
            )
            if estimate.fiscal_period_end is not None:
                prior_consensus_period = estimate.fiscal_period_end
            anchor_rows.append({
                "DCF Year": point.forecast_year_index,
                "DCF Period": (
                    f"{alignment.dcf_period_start.date()} → "
                    f"{alignment.dcf_period_end.date()}"
                ),
                "Fiscal Period": (
                    point.fiscal_period.date() if point.fiscal_period else "N/A"
                ),
                "Consensus Revenue (B)": (
                    point.consensus_revenue / 1e9
                    if point.consensus_revenue is not None else np.nan
                ),
                "Consensus FY Growth": point.consensus_fiscal_growth,
                "DCF Revenue (B)": point.dcf_revenue / 1e9,
                "DCF Growth": point.dcf_growth,
                "Growth Difference (pp)": (
                    point.assumption_minus_consensus_growth * 100
                    if point.assumption_minus_consensus_growth is not None
                    else np.nan
                ),
                "Revenue Difference (B)": (
                    point.dcf_minus_consensus_revenue / 1e9
                    if point.dcf_minus_consensus_revenue is not None else np.nan
                ),
                "Analysts": estimate.analyst_count,
                "Source": estimate.source,
                "Provider As-of": (
                    estimate.source_as_of.date()
                    if estimate.source_as_of is not None else "N/A"
                ),
                "Retrieved At": (
                    estimate.retrieved_at.date()
                    if estimate.retrieved_at is not None else "N/A"
                ),
                "Overlap": alignment.overlap_fraction,
                "Alignment": alignment.alignment_status,
            })
        anchor_frame = pd.DataFrame(anchor_rows).set_index("DCF Year")
        st.dataframe(
            anchor_frame.style.format({
                "Consensus Revenue (B)": "{:.3f}",
                "Consensus FY Growth": "{:.2%}",
                "DCF Revenue (B)": "{:.3f}",
                "DCF Growth": "{:.2%}",
                "Growth Difference (pp)": "{:+.2f}",
                "Revenue Difference (B)": "{:+.3f}",
                "Overlap": "{:.1%}",
            }, na_rep="N/A"),
            width="stretch",
        )
        with st.expander("Forecast anchor source details", expanded=False):
            st.write(f"Issuer anchor ticker：{revenue_anchors.issuer_ticker}")
            st.write(f"Source：{revenue_anchors.source}")
            st.write("Statistic：mean analyst consensus Revenue")
            for point in revenue_anchors.points:
                period = point.fiscal_period.date() if point.fiscal_period else "N/A"
                st.write(
                    f"FY{point.forecast_year_index} · {period} · analysts: "
                    f"{point.analyst_count if point.analyst_count is not None else 'N/A'} · "
                    f"status: {'available' if point.available else point.reason}"
                )
            as_of = next(
                (point.source_as_of for point in revenue_anchors.points if point.source_as_of),
                None,
            )
            st.write(f"Retrieved as of：{as_of if as_of is not None else 'N/A'}")
            if revenue_anchors.warnings:
                st.write("Warnings：" + ", ".join(revenue_anchors.warnings))

    render_final_forecast_chart(run, statement_currency)
    render_final_advanced_diagnostics(
        diagnostics, assumptions, statement_currency
    )

    st.subheader("Sensitivity & Scenario Diagnostics")
    render_multistage_sensitivity(run, assumptions, sensitivity)
    render_scenario_analysis(
        ticker, history, assumptions, run, statement_currency
    )

    reverse_analysis = None
    if profile is None or profile.profile_status == "provisional":
        st.header("Reverse DCF — Market-Implied Expectations")
        st.info(
            "Reverse DCF is unavailable because this ticker does not have a "
            "researched Company Profile. The Manual Base remains available above."
        )
    else:
        reverse_ranges = research_ranges_from_profile(profile)
        range_items = tuple(
            (variable, research_range.lower, research_range.upper)
            for variable, research_range in sorted(reverse_ranges.items())
        )
        reverse_analysis = calculate_reverse_dcf_cached(
            run.inputs,
            assumptions,
            snapshot.price,
            ticker,
            research_base_source,
            range_items,
        )
        render_reverse_dcf(
            reverse_analysis,
            model_risk=profile.model_risk if profile is not None else None,
            limitations=FINAL_MODEL_LIMITATIONS.get(ticker.strip().upper(), ()),
        )
    return {
        "profile": (
            profile
            if profile is not None and profile.profile_status != "provisional"
            else None
        ),
        "research_details": research_details,
        "research_base_run": run,
        "base_source": research_base_source,
        "reverse_analysis": reverse_analysis,
    }

# ================= 4. Streamlit UI =================
def main():
    st.set_page_config(page_title="Stock Valuation Research Workstation", layout="wide")
    st.title("Stock Valuation Research Workstation")
    st.caption(
        "Fundamentals → Research Base DCF → Sensitivity → Reverse DCF → Evidence"
    )

    # 侧边栏：参数输入
    with st.sidebar:
        st.header("⚙️ 参数设置")
        ticker = st.text_input("股票代码 Ticker (如 AAPL, MSFT)", "AAPL").strip().upper()

        try:
            snapshot = load_company_snapshot(ticker)
            wacc_reference = fetch_wacc_reference(ticker, snapshot)
            wacc_audit = build_wacc_audit_result(ticker, wacc_reference)
            annual_financials, quarterly_financials, health_checks = fetch_financial_overview(ticker, snapshot)
            fundamental_history = build_company_fundamentals(snapshot)
        except Exception as exc:
            snapshot = None
            wacc_reference = {"wacc": None, "error": str(exc)}
            wacc_audit = build_wacc_audit_result(ticker, wacc_reference)
            annual_financials, quarterly_financials, health_checks = pd.DataFrame(), pd.DataFrame(), []
            fundamental_history = None
            st.warning("Company data could not be loaded. Valuation is unavailable until the data source recovers.")
            with st.expander("Technical details", expanded=False):
                st.code(str(exc), language=None)
        st.caption("Research assumptions and diagnostics are managed on the main page.")

    header_slot = st.container()
    profile_slot = st.container()
    statement_currency = snapshot.financial_currency if snapshot else None
    st.divider()
    render_fundamental_quality(
        ticker, fundamental_history, statement_currency
    )
    st.divider()
    render_financial_trends(
        ticker,
        annual_financials,
        quarterly_financials,
        statement_currency,
    )
    st.divider()
    context = render_multistage_dcf_panel(
        ticker,
        snapshot,
        fundamental_history,
        wacc_audit,
        header_container=header_slot,
        profile_container=profile_slot,
    )
    st.divider()
    profile = context.get("profile") if context else None
    research_details = context.get("research_details") if context else None
    render_final_evidence(profile, research_details)
    st.divider()
    render_final_model_limitations(profile)
    st.divider()
    render_health_checks(ticker, health_checks)

if __name__ == "__main__":
    main()
