import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re

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


def fetch_fcf_data(ticker: str, debug: bool = False) -> pd.Series:
    """获取最近5个年度报告期的自由现金流，返回单位为百万美元。"""
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return pd.Series(dtype=float)
        ticker_obj = yf.Ticker(ticker)
        cf = ticker_obj.get_cash_flow(freq="yearly")

        # 降级 fallback
        if cf is None or cf.empty:
            cf = ticker_obj.cashflow
            if cf is None or cf.empty:
                return pd.Series(dtype=float)

        # yfinance 的科目位于行索引，日期才位于列。优先使用其已计算的 FCF。
        fcf = _find_statement_row(cf, ("Free Cash Flow",))
        if fcf is None:
            ocf = _find_statement_row(
                cf, ("Operating Cash Flow", "Total Cash From Operating Activities")
            )
            capex = _find_statement_row(
                cf, ("Capital Expenditure", "Capital Expenditures")
            )
            if ocf is not None and capex is not None:
                # yfinance 中 CapEx 通常为负值。
                fcf = ocf.add(capex, fill_value=0)

        if fcf is None:
            if debug:
                st.warning(f"⚠️ 无法识别现金流科目。可用科目: {cf.index.tolist()}")
            return pd.Series(dtype=float)

        fcf = pd.to_numeric(fcf, errors="coerce").dropna().sort_index()
        fcf = fcf / 1_000_000  # 与界面中的“百万美元”保持一致

        # 返回最近5个报告期（通常为年度，若为季度可后续加过滤逻辑）
        return fcf.iloc[-5:] if len(fcf) >= 5 else fcf

    except Exception as e:
        st.error(f"❌ 数据获取异常: {e}")
        return pd.Series(dtype=float)


def fetch_market_data(ticker: str) -> tuple[float, float]:
    """获取当前价格和总股本（百万股），字段缺失时返回 0。"""
    ticker = ticker.strip().upper()
    ticker_obj = yf.Ticker(ticker)
    try:
        fast_info = ticker_obj.fast_info
        price = float(
            fast_info.get("last_price") or fast_info.get("lastPrice") or 0
        )
    except Exception:
        price = 0.0

    try:
        info = ticker_obj.info
        if price <= 0:
            price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        shares = float(info.get("sharesOutstanding") or 0) / 1_000_000
    except Exception:
        shares = 0.0

    if price <= 0:
        try:
            closes = ticker_obj.history(period="5d")["Close"].dropna()
            price = float(closes.iloc[-1]) if not closes.empty else 0.0
        except Exception:
            price = 0.0
    return price, shares

# ================= 2. 估值计算引擎 =================
def calculate_dcf(historical_fcf: pd.Series,
                  growth_rate: float,
                  wacc: float,
                  terminal_growth: float,
                  forecast_years: int,
                  net_debt: float,
                  shares_outstanding: float) -> dict:
    """
    计算DCF内在价值。参数单位需一致（如均为百万美元）。
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

        st.subheader("📈 增长假设")
        growth_rate = st.slider("未来N年增长率 (%)", 0.0, 20.0, 8.0, 0.1) / 100
        terminal_growth = st.slider("终值增长率 (%)", 0.0, 5.0, 2.5, 0.1) / 100
        forecast_years = st.slider("预测年限", 5, 15, 10)

        st.subheader("💰 资本成本")
        wacc = st.slider("WACC (%)", 5.0, 15.0, 9.0, 0.1) / 100

        st.subheader("🏦 资产负债表")
        net_debt = st.number_input("净债务 (百万美元)", 0.0, 100000.0, 0.0, 1.0)
        shares = st.number_input("总股本 (百万股)", 0.0, 10000.0, 0.0, 0.1)

    # 主界面
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("🚀 运行估值", type="primary"):
            if not ticker:
                st.error("请输入股票代码。")
                return

            fcf_data = fetch_fcf_data(ticker, debug=True)
            if len(fcf_data) == 0:
                st.warning("无法获取该股票现金流数据，请检查代码或更换股票。")
                return

            # 获取当前价格；用户未填写股本时自动读取。
            try:
                current_price, fetched_shares = fetch_market_data(ticker)
                shares_for_calc = shares if shares > 0 else fetched_shares
            except Exception as exc:
                st.warning(f"行情参数读取失败: {exc}")
                current_price, shares_for_calc = 0.0, shares

            if shares_for_calc <= 0:
                st.error("无法自动读取总股本，请在左侧手动填写“总股本 (百万股)”。")
                return

            # 计算
            res = calculate_dcf(
                fcf_data, growth_rate, wacc, terminal_growth,
                forecast_years, net_debt, shares_for_calc
            )

            if "error" in res:
                st.error(res["error"])
                return

            intrinsic = res["intrinsic_value"]
            margin_safety = (intrinsic - current_price) / current_price * 100 if current_price > 0 else 0

            st.metric("📉 当前股价", f"${current_price:.2f}")
            st.metric("🎯 内在价值", f"${intrinsic:.2f}")
            st.metric("🛡️ 安全边际", f"{margin_safety:+.1f}%")

            st.markdown(f"**核心假设**：FCF = {res['last_fcf']:.1f}M | 增长率={growth_rate*100:.1f}% | WACC={wacc*100:.1f}% | 终值g={terminal_growth*100:.1f}%")

            # 敏感性分析
            st.subheader("📊 参数敏感性热力图")
            wacc_grid = np.linspace(max(0.05, wacc-0.03), min(0.15, wacc+0.03), 10)
            growth_grid = np.linspace(max(0.01, growth_rate-0.03), min(0.15, growth_rate+0.03), 10)

            grid_vals = sensitivity_grid(
                fcf_data, wacc_grid, growth_grid, terminal_growth,
                forecast_years, net_debt, shares_for_calc
            )

            fig_heat = go.Figure(data=go.Heatmap(
                z=grid_vals,
                x=[f"{g*100:.1f}%" for g in growth_grid],
                y=[f"{w*100:.1f}%" for w in wacc_grid],
                colorscale="RdYlGn",
                colorbar=dict(title="内在价值 ($)")
            ))
            fig_heat.update_layout(xaxis_title="增长率", yaxis_title="WACC", height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.info("💡 提示：DCF对假设极度敏感。建议用滑块观察价值区间变化，类似物理系统的相变分析。")

    with col2:
        st.subheader("📈 现金流投影与折现")
        if "res" in locals() and "error" not in res:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                row_heights=[0.6, 0.4],
                                subplot_titles=["自由现金流投影", "折现因子与PV"])

            # 历史FCF
            fig.add_trace(go.Scatter(x=fcf_data.index, y=fcf_data.values, name="历史FCF", mode="lines+markers"), row=1, col=1)
            # 投影FCF
            # pandas 3.0 移除了旧的 "Y" 别名，"YE" 表示年末频率。
            proj_idx = pd.date_range(
                start=fcf_data.index[-1],
                periods=len(res["projected_fcf"]) + 1,
                freq="YE",
            )[1:]
            fig.add_trace(go.Scatter(x=proj_idx, y=res["projected_fcf"], name="投影FCF", line=dict(dash="dash")), row=1, col=1)

            # 折现因子
            years = np.arange(1, forecast_years+1)
            disc_factors = 1 / (1 + wacc) ** years
            fig.add_trace(go.Bar(x=proj_idx, y=res["pv_fcf"], name="FCF现值", opacity=0.7), row=2, col=1)
            fig.add_trace(go.Scatter(x=proj_idx, y=disc_factors, name="折现因子", line=dict(color="red")), row=2, col=1)

            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("点击左侧「运行估值」查看图表")

if __name__ == "__main__":
    main()
