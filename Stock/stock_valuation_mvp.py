import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
from difflib import SequenceMatcher
from io import StringIO

import requests

warnings.filterwarnings("ignore")

# ================= 1. 数据获取层 =================
def _find_statement_row(statement: pd.DataFrame, candidates: tuple[str, ...]):
    """按 yfinance 财务报表的行名查找科目，兼容名称的小幅变化。"""
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    normalized = {normalize(label): label for label in statement.index}
    for candidate in candidates:
        normalized_candidate = normalize(candidate)
        if normalized_candidate in normalized:
            return statement.loc[normalized[normalized_candidate]]
    for normalized_label, original_label in normalized.items():
        if any(normalize(candidate) in normalized_label for candidate in candidates):
            return statement.loc[original_label]
    return None


def _statement_series(statement: pd.DataFrame,
                      candidates: tuple[str, ...]) -> pd.Series:
    """读取财务报表科目并按日期升序返回数值序列。"""
    if statement is None or statement.empty:
        return pd.Series(dtype=float)
    row = _find_statement_row(statement, candidates)
    if row is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(row, errors="coerce").dropna().sort_index()


def _calculate_fcff_series(income: pd.DataFrame,
                           cashflow: pd.DataFrame) -> tuple[pd.Series, str]:
    """用 CFO 口径从年度或季度报表计算 FCFF 原始美元序列。"""
    operating_cf = _statement_series(
        cashflow,
        ("Operating Cash Flow", "Total Cash From Operating Activities"),
    )
    interest_expense = _statement_series(
        income,
        ("Interest Expense", "Interest Expense Non Operating"),
    )
    pretax = _statement_series(income, ("Pretax Income", "Income Before Tax"))
    tax = _statement_series(income, ("Tax Provision", "Income Tax Expense"))
    capex = _statement_series(
        cashflow, ("Capital Expenditure", "Capital Expenditures")
    )

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
        effective_tax = effective_tax.where(frame["pretax"] > 0)
        effective_tax.loc[loss_period] = 0.0
        effective_tax = effective_tax.clip(lower=0.0, upper=0.35).fillna(0.21)
        valid_period = frame["operating_cf"].notna() & frame["capex"].notna()
        fcff = (
            frame["operating_cf"]
            + frame["capex"]
            + frame["interest_expense"].abs().fillna(0) * (1 - effective_tax)
        ).loc[valid_period].dropna()
        if not fcff.empty:
            return fcff, "FCFF = CFO + CapEx + 税后利息"

    fallback = _statement_series(cashflow, ("Free Cash Flow",))
    if fallback.empty:
        if not operating_cf.empty and not capex.empty:
            fallback = operating_cf.add(capex, fill_value=0)
    return fallback, "yfinance FCF 回退口径" if not fallback.empty else "无数据"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fcff_data(ticker: str) -> tuple[pd.Series, str]:
    """返回年度 FCFF，并用最近四个季度追加最新 TTM FCFF。"""
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.Series(dtype=float), "无数据"

    try:
        ticker_obj = yf.Ticker(ticker)
        annual, annual_source = _calculate_fcff_series(
            ticker_obj.get_income_stmt(freq="yearly"),
            ticker_obj.get_cash_flow(freq="yearly"),
        )
        quarterly, quarterly_source = _calculate_fcff_series(
            ticker_obj.get_income_stmt(freq="quarterly"),
            ticker_obj.get_cash_flow(freq="quarterly"),
        )

        result = annual.iloc[-5:] / 1_000_000_000
        source = annual_source
        if len(quarterly) >= 4:
            latest_four = quarterly.iloc[-4:]
            ttm_end = pd.Timestamp(latest_four.index[-1])
            ttm_fcff = float(latest_four.sum()) / 1_000_000_000
            if result.empty or ttm_end > pd.Timestamp(result.index[-1]):
                result.loc[ttm_end] = ttm_fcff
                source = (
                    f"{annual_source}；最新值为截至 {ttm_end.date()} 的 TTM "
                    f"（季度口径：{quarterly_source}）"
                )
        return result.sort_index(), source
    except Exception:
        return pd.Series(dtype=float), "无数据"


def fetch_fcf_data(ticker: str, debug: bool = False) -> pd.Series:
    """兼容旧调用；返回 FCFF/FCF 序列，单位为十亿美元。"""
    data, source = fetch_fcff_data(ticker)
    if debug and data.empty:
        st.warning(f"⚠️ 无法获取现金流数据（{source}）。")
    return data


def _latest_statement_value(statement: pd.DataFrame,
                            candidates: tuple[str, ...]) -> float:
    """读取财务报表科目的最近一期有效数值。"""
    row = _find_statement_row(statement, candidates)
    if row is None:
        return 0.0
    values = pd.to_numeric(row, errors="coerce").dropna().sort_index()
    return float(values.iloc[-1]) if not values.empty else 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker: str) -> tuple[float, float, float]:
    """获取股价、净债务（十亿美元）和总股本（十亿股）。"""
    ticker = ticker.strip().upper()
    if not ticker:
        return 0.0, 0.0, 0.0

    ticker_obj = yf.Ticker(ticker)
    info = {}
    balance_sheet = pd.DataFrame()

    try:
        fast_info = ticker_obj.fast_info
        price = float(
            fast_info.get("last_price") or fast_info.get("lastPrice") or 0
        )
    except Exception:
        price = 0.0

    try:
        info = ticker_obj.info or {}
        if price <= 0:
            price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    except Exception:
        info = {}

    try:
        balance_sheet = ticker_obj.get_balance_sheet(freq="yearly")
        if balance_sheet is None:
            balance_sheet = pd.DataFrame()
    except Exception:
        balance_sheet = pd.DataFrame()

    # 当前流通股本优先使用 info；报表股本和市值推算作为回退。
    shares_raw = float(
        info.get("sharesOutstanding")
        or info.get("impliedSharesOutstanding")
        or 0
    )
    if shares_raw <= 0 and not balance_sheet.empty:
        shares_raw = _latest_statement_value(
            balance_sheet, ("Ordinary Shares Number", "Share Issued")
        )
    if shares_raw <= 0 and price > 0:
        shares_raw = float(info.get("marketCap") or 0) / price

    # 优先采用 yfinance 资产负债表中的 NetDebt。若缺失，则逐级计算。
    net_debt_raw = 0.0
    if not balance_sheet.empty:
        net_debt_raw = _latest_statement_value(balance_sheet, ("Net Debt",))
        if net_debt_raw == 0:
            total_debt = _latest_statement_value(balance_sheet, ("Total Debt",))
            cash = _latest_statement_value(
                balance_sheet, ("Cash And Cash Equivalents",)
            )
            if total_debt or cash:
                net_debt_raw = total_debt - cash
    if net_debt_raw == 0:
        total_debt = float(info.get("totalDebt") or 0)
        total_cash = float(info.get("totalCash") or 0)
        if total_debt or total_cash:
            net_debt_raw = total_debt - total_cash

    if price <= 0:
        try:
            closes = ticker_obj.history(period="5d")["Close"].dropna()
            price = float(closes.iloc[-1]) if not closes.empty else 0.0
        except Exception:
            price = 0.0
    return price, net_debt_raw / 1_000_000_000, shares_raw / 1_000_000_000


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
    except Exception:
        try:
            treasury_yield = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
            if not treasury_yield.empty:
                risk_free = float(treasury_yield.iloc[-1]) / 100
                treasury_date = "^TNX 回退"
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
    except Exception:
        pass

    return {
        "risk_free": risk_free,
        "erp": erp,
        "treasury_date": treasury_date,
        "erp_date": erp_date,
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wacc_reference(ticker: str) -> dict:
    """计算公司级 WACC，并返回可解释的各项输入。"""
    empty = {"wacc": None, "error": "数据不足"}
    ticker = ticker.strip().upper()
    if not ticker:
        return empty
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}
        income = ticker_obj.get_income_stmt(freq="yearly")
        balance = ticker_obj.get_balance_sheet(freq="yearly")
        price, _, shares_b = fetch_market_data(ticker)

        beta, beta_months = _regression_beta(ticker)
        if beta is None:
            beta = float(info.get("beta") or 1.0)
        macro = fetch_macro_assumptions()

        total_debt = _latest_statement_value(balance, ("Total Debt",))
        if total_debt <= 0:
            total_debt = float(info.get("totalDebt") or 0)
        market_cap = float(info.get("marketCap") or 0)
        if market_cap <= 0 and price > 0 and shares_b > 0:
            market_cap = price * shares_b * 1_000_000_000
        if market_cap <= 0:
            return empty

        ebit = _latest_statement_value(income, ("EBIT", "Operating Income"))
        interest = abs(_latest_statement_value(
            income, ("Interest Expense", "Interest Expense Non Operating")
        ))
        pretax = _latest_statement_value(income, ("Pretax Income", "Income Before Tax"))
        tax = _latest_statement_value(income, ("Tax Provision", "Income Tax Expense"))
        tax_rate = tax / pretax if pretax > 0 and tax >= 0 else 0.21
        tax_rate = float(np.clip(tax_rate, 0.0, 0.35))
        coverage = ebit / interest if interest > 0 else np.inf
        financial = info.get("sector") == "Financial Services"
        spread, rating = _default_spread(coverage, financial)

        cost_equity = macro["risk_free"] + beta * macro["erp"]
        pretax_cost_debt = macro["risk_free"] + spread
        after_tax_cost_debt = pretax_cost_debt * (1 - tax_rate)
        equity_weight = market_cap / (market_cap + max(total_debt, 0))
        debt_weight = 1 - equity_weight
        wacc = cost_equity * equity_weight + after_tax_cost_debt * debt_weight
        industry_ref = fetch_industry_wacc(str(info.get("industry") or ""))
        return {
            "wacc": float(wacc),
            "cost_equity": float(cost_equity),
            "pretax_cost_debt": float(pretax_cost_debt),
            "after_tax_cost_debt": float(after_tax_cost_debt),
            "risk_free": macro["risk_free"],
            "erp": macro["erp"],
            "beta": float(beta),
            "beta_months": beta_months,
            "tax_rate": tax_rate,
            "coverage": float(coverage),
            "rating": rating,
            "equity_weight": float(equity_weight),
            "debt_weight": float(debt_weight),
            "industry_wacc": industry_ref["wacc"],
            "matched_industry": industry_ref["matched_industry"],
            "treasury_date": macro["treasury_date"],
            "erp_date": macro["erp_date"],
            "error": None,
        }
    except Exception as exc:
        return {"wacc": None, "error": str(exc)}


# ================= 2. 估值计算引擎 =================
def calculate_dcf(historical_fcf: pd.Series,
                  growth_rate: float,
                  wacc: float,
                  terminal_growth: float,
                  forecast_years: int,
                  net_debt: float,
                  shares_outstanding: float) -> dict:
    """
    计算DCF内在价值。现金流、净债务使用十亿美元，股本使用十亿股。
    返回：字典包含当前价、内在价值、安全边际、投影数据等
    """
    if len(historical_fcf) == 0 or shares_outstanding <= 0:
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

# ================= 4. Streamlit UI =================
def main():
    st.set_page_config(page_title="美股估值分析器", layout="wide")
    st.title("📊 美股 DCF 估值分析器 (MVP)")
    st.markdown("> 结合数值模拟思维：参数投影 + 折现积分 + 敏感性网格")

    # 侧边栏：参数输入
    with st.sidebar:
        st.header("⚙️ 参数设置")
        ticker = st.text_input("股票代码 (如 AAPL, MSFT)", "AAPL").strip().upper()

        try:
            current_price, fetched_net_debt, fetched_shares = fetch_market_data(ticker)
            fcff_data, fcff_source = fetch_fcff_data(ticker)
            wacc_reference = fetch_wacc_reference(ticker)
        except Exception as exc:
            current_price, fetched_net_debt, fetched_shares = 0.0, 0.0, 0.0
            fcff_data, fcff_source = pd.Series(dtype=float), "无数据"
            wacc_reference = {"wacc": None, "error": str(exc)}
            st.warning(f"yfinance 公司数据读取失败: {exc}")

        st.subheader("📈 增长假设")
        growth_rate = st.slider("未来N年增长率 (%)", 0.0, 100.0, 8.0, 0.1) / 100
        terminal_growth = st.slider("终值增长率 (%)", 0.0, 5.0, 2.5, 0.1) / 100
        forecast_years = st.slider("预测年限", 5, 15, 5)

        st.subheader("💰 资本成本")
        wacc = st.slider("WACC (%)", 5.0, 15.0, 9.0, 0.1) / 100
        if wacc_reference.get("wacc") is not None:
            wacc_notes = [
                f"模型 WACC {wacc_reference['wacc']*100:.2f}%",
                f"股权成本 {wacc_reference['cost_equity']*100:.2f}%",
                f"税后债务成本 {wacc_reference['after_tax_cost_debt']*100:.2f}%",
                f"Beta {wacc_reference['beta']:.2f}",
            ]
            if wacc_reference.get("industry_wacc") is not None:
                wacc_notes.append(
                    f"{wacc_reference['matched_industry']} 行业 {wacc_reference['industry_wacc']*100:.2f}%"
                )
            st.caption("参考：" + " · ".join(wacc_notes))
            with st.expander("WACC 计算明细", expanded=False):
                st.markdown(
                    f"无风险利率 **{wacc_reference['risk_free']*100:.2f}%** "
                    f"（{wacc_reference['treasury_date']}）  \n"
                    f"成熟市场 ERP **{wacc_reference['erp']*100:.2f}%** "
                    f"（{wacc_reference['erp_date']}）  \n"
                    f"股权/债务权重 **{wacc_reference['equity_weight']*100:.1f}% / "
                    f"{wacc_reference['debt_weight']*100:.1f}%**  \n"
                    f"有效税率 **{wacc_reference['tax_rate']*100:.1f}%** · "
                    f"合成评级 **{wacc_reference['rating']}**"
                )
        else:
            st.caption(f"WACC 参考暂不可用：{wacc_reference.get('error') or '数据不足'}")

        st.subheader("🏦 资产负债表")
        net_debt = st.number_input(
            "净债务 (十亿美元)",
            value=float(fetched_net_debt),
            step=0.1,
            format="%.3f",
            key=f"net_debt_{ticker}",
        )
        shares = st.number_input(
            "总股本 (十亿股)",
            value=float(fetched_shares),
            step=0.01,
            format="%.3f",
            key=f"shares_{ticker}",
        )
        if fetched_net_debt or fetched_shares:
            st.caption("以上默认值已从 yfinance 自动获取，可手动覆盖。")

    # 主界面
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("🚀 运行估值", type="primary"):
            if not ticker:
                st.error("请输入股票代码。")
                return

            if len(fcff_data) == 0:
                st.warning("无法获取该股票现金流数据，请检查代码或更换股票。")
                return

            if shares <= 0:
                st.error("无法自动读取总股本，请在左侧手动填写“总股本 (十亿股)”。")
                return

            # 计算
            res = calculate_dcf(
                fcff_data, growth_rate, wacc, terminal_growth,
                forecast_years, net_debt, shares
            )

            if "error" in res:
                st.error(res["error"])
                return

            intrinsic = res["intrinsic_value"]
            margin_safety = (intrinsic - current_price) / current_price * 100 if current_price > 0 else 0

            st.metric("📉 当前股价", f"${current_price:.2f}")
            st.metric("🎯 内在价值", f"${intrinsic:.2f}")
            st.metric("🛡️ 安全边际", f"{margin_safety:+.1f}%")

            st.markdown(
                f"**核心假设**：FCFF = ${res['last_fcf']:.2f}B | "
                f"净债务 = ${net_debt:.2f}B | 股本 = {shares:.3f}B | "
                f"增长率={growth_rate*100:.1f}% | WACC={wacc*100:.1f}% | "
                f"终值g={terminal_growth*100:.1f}%"
            )
            st.caption(f"现金流口径：{fcff_source}")

            # 敏感性分析
            st.subheader("📊 参数敏感性热力图")
            wacc_grid = np.linspace(max(0.05, wacc-0.03), min(0.15, wacc+0.03), 10)
            growth_grid = np.linspace(max(0.01, growth_rate-0.03), min(0.15, growth_rate+0.03), 10)

            grid_vals = sensitivity_grid(
                fcff_data, wacc_grid, growth_grid, terminal_growth,
                forecast_years, net_debt, shares
            )

            fig_heat = go.Figure(data=go.Heatmap(
                z=grid_vals,
                x=[f"{g*100:.1f}%" for g in growth_grid],
                y=[f"{w*100:.1f}%" for w in wacc_grid],
                colorscale="RdYlGn",
                colorbar=dict(title="内在价值 ($)")
            ))
            fig_heat.update_layout(xaxis_title="增长率", yaxis_title="WACC", height=400)
            st.plotly_chart(fig_heat, width="stretch")

        st.info("💡 提示：DCF对假设极度敏感。建议用滑块观察价值区间变化，类似物理系统的相变分析。")

    with col2:
        st.subheader("📈 FCFF 投影与折现")
        if "res" in locals() and "error" not in res:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                row_heights=[0.6, 0.4],
                                subplot_titles=["FCFF 投影", "折现因子与PV"])

            # 历史 FCFF
            fig.add_trace(go.Scatter(x=fcff_data.index, y=fcff_data.values, name="历史FCFF", mode="lines+markers"), row=1, col=1)
            # 投影 FCFF
            # 从最新年度/TTM 报告期向后逐年投影，保留相同月日。
            latest_fcff_date = pd.Timestamp(fcff_data.index[-1])
            proj_idx = pd.DatetimeIndex(
                [latest_fcff_date + pd.DateOffset(years=i)
                 for i in range(1, len(res["projected_fcf"]) + 1)]
            )
            fig.add_trace(go.Scatter(x=proj_idx, y=res["projected_fcf"], name="投影FCFF", line=dict(dash="dash")), row=1, col=1)

            # 折现因子
            years = np.arange(1, forecast_years+1)
            disc_factors = 1 / (1 + wacc) ** years
            fig.add_trace(go.Bar(x=proj_idx, y=res["pv_fcf"], name="FCFF现值", opacity=0.7), row=2, col=1)
            fig.add_trace(go.Scatter(x=proj_idx, y=disc_factors, name="折现因子", line=dict(color="red")), row=2, col=1)

            fig.update_layout(height=500, showlegend=True)
            fig.update_yaxes(title_text="十亿美元", row=1, col=1)
            fig.update_yaxes(title_text="十亿美元", row=2, col=1)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("点击左侧「运行估值」查看图表")

if __name__ == "__main__":
    main()
