import numpy as np
import pandas as pd
import pytest

from Stock import fundamentals as f
from Stock import stock_valuation_mvp as app


def series(values, dates):
    return pd.Series(values, index=pd.to_datetime(dates), dtype=float)


def period_metrics(
    dates,
    *,
    revenue=None,
    gross_profit=None,
    operating_income=None,
    cfo=None,
    capex=None,
):
    def supplied(values):
        return series(values, dates) if values is not None else pd.Series(dtype=float)

    return f.build_period_fundamentals(
        revenue=supplied(revenue),
        gross_profit=supplied(gross_profit),
        operating_income=supplied(operating_income),
        cfo=supplied(cfo),
        capex=supplied(capex),
        periods=dates,
    )


def test_revenue_growth_for_consecutive_years():
    frame = period_metrics(
        ["2024-12-31", "2025-12-31"], revenue=[100.0, 120.0]
    )

    assert frame.loc["2025-12-31", f.REVENUE_GROWTH] == pytest.approx(0.20)


def test_negative_revenue_growth():
    frame = period_metrics(
        ["2024-12-31", "2025-12-31"], revenue=[100.0, 80.0]
    )

    assert frame.loc["2025-12-31", f.REVENUE_GROWTH] == pytest.approx(-0.20)


def test_genuine_zero_current_revenue_produces_negative_one_growth():
    frame = period_metrics(
        ["2024-12-31", "2025-12-31"], revenue=[100.0, 0.0]
    )

    assert frame.loc["2025-12-31", f.REVENUE] == 0.0
    assert frame.loc["2025-12-31", f.REVENUE_GROWTH] == -1.0


def test_zero_previous_revenue_makes_growth_unavailable():
    frame = period_metrics(
        ["2024-12-31", "2025-12-31"], revenue=[0.0, 100.0]
    )

    assert pd.isna(frame.loc["2025-12-31", f.REVENUE_GROWTH])


def test_revenue_growth_does_not_cross_missing_middle_year():
    dates = ["2022-12-31", "2023-12-31", "2024-12-31"]
    frame = period_metrics(dates, revenue=[100.0, np.nan, 140.0])

    assert pd.isna(frame.loc["2023-12-31", f.REVENUE_GROWTH])
    assert pd.isna(frame.loc["2024-12-31", f.REVENUE_GROWTH])


def test_latest_missing_revenue_remains_in_history():
    dates = ["2024-12-31", "2025-12-31"]
    frame = period_metrics(dates, revenue=[100.0, np.nan])

    assert pd.Timestamp("2025-12-31") in frame.index
    assert pd.isna(frame.loc["2025-12-31", f.REVENUE])
    assert pd.isna(frame.loc["2025-12-31", f.REVENUE_GROWTH])


def test_unordered_annual_periods_are_sorted_before_growth():
    dates = ["2025-12-31", "2023-12-31", "2024-12-31"]
    frame = period_metrics(dates, revenue=[120.0, 80.0, 100.0])

    assert list(frame.index) == list(
        pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])
    )
    assert frame.loc["2024-12-31", f.REVENUE_GROWTH] == pytest.approx(0.25)
    assert frame.loc["2025-12-31", f.REVENUE_GROWTH] == pytest.approx(0.20)


def test_revenue_growth_requires_consecutive_fiscal_years():
    dates = ["2022-12-31", "2024-12-31"]
    frame = period_metrics(dates, revenue=[100.0, 140.0])

    assert pd.isna(frame.loc["2024-12-31", f.REVENUE_GROWTH])


def test_normal_gross_and_operating_margins():
    frame = period_metrics(
        ["2025-12-31"],
        revenue=[200.0],
        gross_profit=[80.0],
        operating_income=[30.0],
    )

    assert frame.iloc[0][f.GROSS_MARGIN] == pytest.approx(0.40)
    assert frame.iloc[0][f.OPERATING_MARGIN] == pytest.approx(0.15)


def test_zero_margin_numerators_are_preserved():
    frame = period_metrics(
        ["2025-12-31"],
        revenue=[200.0],
        gross_profit=[0.0],
        operating_income=[0.0],
    )

    assert frame.iloc[0][f.GROSS_MARGIN] == 0.0
    assert frame.iloc[0][f.OPERATING_MARGIN] == 0.0


def test_zero_revenue_makes_all_margins_unavailable():
    frame = period_metrics(
        ["2025-12-31"],
        revenue=[0.0],
        gross_profit=[10.0],
        operating_income=[5.0],
        cfo=[4.0],
        capex=[-1.0],
    )

    assert pd.isna(frame.iloc[0][f.GROSS_MARGIN])
    assert pd.isna(frame.iloc[0][f.OPERATING_MARGIN])
    assert pd.isna(frame.iloc[0][f.FCF_MARGIN])


def test_missing_margin_numerator_is_unavailable():
    frame = period_metrics(
        ["2025-12-31"], revenue=[100.0], gross_profit=[np.nan]
    )

    assert pd.isna(frame.iloc[0][f.GROSS_MARGIN])


def test_missing_revenue_makes_margin_unavailable():
    frame = period_metrics(
        ["2025-12-31"], revenue=[np.nan], operating_income=[10.0]
    )

    assert pd.isna(frame.iloc[0][f.OPERATING_MARGIN])


def test_mismatched_periods_do_not_create_margin():
    frame = f.build_period_fundamentals(
        revenue=series([100.0], ["2024-12-31"]),
        gross_profit=series([40.0], ["2025-12-31"]),
        operating_income=pd.Series(dtype=float),
        cfo=pd.Series(dtype=float),
        capex=pd.Series(dtype=float),
        periods=["2024-12-31", "2025-12-31"],
    )

    assert frame[f.GROSS_MARGIN].isna().all()


@pytest.mark.parametrize(
    ("cfo", "capex", "expected"),
    [
        (100.0, -30.0, 70.0),
        (10.0, -30.0, -20.0),
        (30.0, -30.0, 0.0),
    ],
)
def test_fcf_is_cfo_plus_negative_capex(cfo, capex, expected):
    frame = period_metrics(["2025-12-31"], cfo=[cfo], capex=[capex])

    assert frame.iloc[0][f.FCF] == expected


@pytest.mark.parametrize(
    ("cfo", "capex"),
    [(np.nan, -30.0), (100.0, np.nan)],
)
def test_fcf_missing_required_side_is_unavailable(cfo, capex):
    frame = period_metrics(["2025-12-31"], cfo=[cfo], capex=[capex])

    assert pd.isna(frame.iloc[0][f.FCF])


def test_fcf_requires_cfo_and_capex_in_same_period():
    frame = f.build_period_fundamentals(
        revenue=pd.Series(dtype=float),
        gross_profit=pd.Series(dtype=float),
        operating_income=pd.Series(dtype=float),
        cfo=series([100.0], ["2024-12-31"]),
        capex=series([-30.0], ["2025-12-31"]),
        periods=["2024-12-31", "2025-12-31"],
    )

    assert frame[f.FCF].isna().all()


def test_normal_fcf_margin():
    frame = period_metrics(
        ["2025-12-31"], revenue=[200.0], cfo=[100.0], capex=[-30.0]
    )

    assert frame.iloc[0][f.FCF_MARGIN] == pytest.approx(0.35)


@pytest.mark.parametrize(
    ("revenue", "cfo", "capex"),
    [
        (0.0, 100.0, -30.0),
        (200.0, np.nan, -30.0),
        (np.nan, 100.0, -30.0),
    ],
)
def test_fcf_margin_requires_nonzero_revenue_and_available_fcf(
    revenue,
    cfo,
    capex,
):
    frame = period_metrics(
        ["2025-12-31"], revenue=[revenue], cfo=[cfo], capex=[capex]
    )

    assert pd.isna(frame.iloc[0][f.FCF_MARGIN])


def quarterly_history(
    dates,
    *,
    revenue,
    gross_profit,
    operating_income,
    cfo,
    capex,
):
    empty = pd.Series(dtype=float)
    return f.build_fundamental_history(
        annual_revenue=empty,
        annual_gross_profit=empty,
        annual_operating_income=empty,
        annual_cfo=empty,
        annual_capex=empty,
        quarterly_revenue=series(revenue, dates),
        quarterly_gross_profit=series(gross_profit, dates),
        quarterly_operating_income=series(operating_income, dates),
        quarterly_cfo=series(cfo, dates),
        quarterly_capex=series(capex, dates),
        quarterly_income_periods=dates,
        quarterly_cashflow_periods=dates,
    )


def test_valid_ttm_fundamentals_and_aggregate_margins():
    dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    history = quarterly_history(
        dates,
        revenue=[100.0, 200.0, 300.0, 400.0],
        gross_profit=[40.0, 80.0, 120.0, 160.0],
        operating_income=[10.0, 20.0, 90.0, 80.0],
        cfo=[30.0, 40.0, 50.0, 60.0],
        capex=[-10.0, -10.0, -10.0, -10.0],
    )

    assert history.ttm[f.REVENUE].value == 1000.0
    assert history.ttm[f.GROSS_MARGIN].value == pytest.approx(0.40)
    assert history.ttm[f.OPERATING_MARGIN].value == pytest.approx(0.20)
    assert history.ttm[f.FCF].value == 140.0
    assert history.ttm[f.FCF_MARGIN].value == pytest.approx(0.14)
    assert history.ttm[f.OPERATING_MARGIN].value != pytest.approx(0.175)
    assert history.ttm[f.REVENUE].periods_used == tuple(pd.to_datetime(dates))


def test_ttm_missing_recent_quarter_is_unavailable():
    expected = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    empty = pd.Series(dtype=float)
    history = f.build_fundamental_history(
        annual_revenue=empty,
        annual_gross_profit=empty,
        annual_operating_income=empty,
        annual_cfo=empty,
        annual_capex=empty,
        quarterly_revenue=series([100.0, 100.0, 100.0], expected[:3]),
        quarterly_gross_profit=empty,
        quarterly_operating_income=empty,
        quarterly_cfo=empty,
        quarterly_capex=empty,
        quarterly_income_periods=expected,
        quarterly_cashflow_periods=expected,
    )

    assert history.ttm[f.REVENUE].available is False
    assert history.ttm[f.REVENUE].reason == "missing_quarter_value"


def test_ttm_metric_missing_one_required_quarter_is_unavailable():
    dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    history = quarterly_history(
        dates,
        revenue=[100.0] * 4,
        gross_profit=[40.0] * 4,
        operating_income=[10.0, np.nan, 10.0, 10.0],
        cfo=[20.0] * 4,
        capex=[-5.0] * 4,
    )

    assert history.ttm[f.REVENUE].available is True
    assert history.ttm[f.OPERATING_INCOME].available is False
    assert history.ttm[f.OPERATING_MARGIN].available is False


def test_ttm_fcf_rejects_missing_cfo_or_capex_quarter():
    dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    history = quarterly_history(
        dates,
        revenue=[100.0] * 4,
        gross_profit=[40.0] * 4,
        operating_income=[10.0] * 4,
        cfo=[20.0, np.nan, 20.0, 20.0],
        capex=[-5.0] * 4,
    )

    assert history.ttm[f.FCF].available is False
    assert history.ttm[f.FCF_MARGIN].available is False


def test_ttm_non_consecutive_quarters_are_rejected():
    dates = ["2024-12-31", "2025-03-31", "2025-09-30", "2025-12-31"]
    history = quarterly_history(
        dates,
        revenue=[100.0] * 4,
        gross_profit=[40.0] * 4,
        operating_income=[10.0] * 4,
        cfo=[20.0] * 4,
        capex=[-5.0] * 4,
    )

    assert history.ttm[f.REVENUE].reason == "non_consecutive_quarters"
    assert history.ttm[f.FCF].reason == "non_consecutive_quarters"


def test_app_snapshot_adapter_uses_cfo_plus_capex_not_reported_fcf(
    statement_factory,
    snapshot_factory,
):
    dates = ["2024-12-31", "2025-12-31"]
    snapshot = snapshot_factory(
        annual_income=statement_factory(
            {
                "Total Revenue": [100.0, 120.0],
                "Gross Profit": [40.0, 48.0],
                "Operating Income": [10.0, 12.0],
            },
            dates,
        ),
        annual_cashflow=statement_factory(
            {
                "Operating Cash Flow": [30.0, 40.0],
                "Capital Expenditure": [-10.0, -15.0],
                "Free Cash Flow": [999.0, 999.0],
            },
            dates,
        ),
    )

    history = app.build_company_fundamentals(snapshot)

    assert history.annual.loc["2025-12-31", f.FCF] == 25.0
    assert history.annual.loc["2025-12-31", f.REVENUE_GROWTH] == pytest.approx(0.20)


def test_nopat_uses_effective_operating_tax_rate():
    result = f.calculate_nopat(100.0, 80.0, 16.0)

    assert result.available is True
    assert result.tax_rate == pytest.approx(0.20)
    assert result.nopat == pytest.approx(80.0)
    assert result.assumption_used is False


@pytest.mark.parametrize(
    ("operating_income", "expected"),
    [(0.0, 0.0), (-50.0, -40.0)],
)
def test_nopat_preserves_zero_and_negative_operating_income(
    operating_income,
    expected,
):
    result = f.calculate_nopat(operating_income, 100.0, 20.0)

    assert result.available is True
    assert result.nopat == pytest.approx(expected)


@pytest.mark.parametrize(
    ("pretax", "tax"),
    [(None, 10.0), (100.0, None), (np.nan, 10.0), (100.0, np.nan)],
)
def test_nopat_missing_tax_inputs_is_unavailable(pretax, tax):
    result = f.calculate_nopat(100.0, pretax, tax)

    assert result.available is False
    assert result.reason == "missing_tax_inputs"


@pytest.mark.parametrize(
    ("pretax", "tax", "reason"),
    [
        (0.0, 0.0, "non_positive_pretax_income"),
        (-10.0, 2.0, "non_positive_pretax_income"),
        (100.0, 60.0, "unreasonable_tax_rate"),
        (100.0, -5.0, "unreasonable_tax_rate"),
    ],
)
def test_nopat_rejects_unreliable_tax_rate(pretax, tax, reason):
    result = f.calculate_nopat(100.0, pretax, tax)

    assert result.available is False
    assert result.reason == reason


def test_invested_capital_is_equity_plus_debt_minus_cash():
    result = f.calculate_invested_capital(100.0, 40.0, 20.0)

    assert result.available is True
    assert result.value == 120.0


def test_invested_capital_preserves_true_zero_debt():
    result = f.calculate_invested_capital(100.0, 0.0, 20.0)

    assert result.value == 80.0


@pytest.mark.parametrize(
    ("equity", "debt", "cash", "reason"),
    [
        (None, 40.0, 20.0, "missing_total_equity"),
        (100.0, None, 20.0, "missing_total_debt"),
        (100.0, 40.0, None, "missing_cash"),
    ],
)
def test_invested_capital_missing_component_is_unavailable(
    equity,
    debt,
    cash,
    reason,
):
    result = f.calculate_invested_capital(equity, debt, cash)

    assert result.available is False
    assert result.reason == reason


def test_negative_invested_capital_is_preserved_as_edge_case():
    result = f.calculate_invested_capital(10.0, 5.0, 30.0)

    assert result.available is True
    assert result.value == -15.0


def test_average_invested_capital_uses_two_consecutive_years():
    result = f.calculate_average_invested_capital(
        120.0, 100.0, "2025-12-31", "2024-12-31"
    )

    assert result.available is True
    assert result.value == 110.0


def test_average_invested_capital_requires_prior_year():
    result = f.calculate_average_invested_capital(
        120.0, None, "2025-12-31", "2024-12-31"
    )

    assert result.available is False
    assert result.reason == "missing_prior_invested_capital"


def test_average_invested_capital_rejects_non_consecutive_years():
    result = f.calculate_average_invested_capital(
        120.0, 80.0, "2025-12-31", "2023-12-31"
    )

    assert result.available is False
    assert result.reason == "non_consecutive_fiscal_years"


@pytest.mark.parametrize(
    ("nopat", "average_capital", "expected"),
    [(20.0, 100.0, 0.20), (0.0, 100.0, 0.0), (-10.0, 100.0, -0.10)],
)
def test_roic_preserves_positive_zero_and_negative_nopat(
    nopat,
    average_capital,
    expected,
):
    result = f.calculate_roic(nopat, average_capital)

    assert result.available is True
    assert result.value == pytest.approx(expected)


@pytest.mark.parametrize("denominator", [0.0, -10.0])
def test_roic_rejects_non_positive_denominator(denominator):
    result = f.calculate_roic(20.0, denominator)

    assert result.available is False
    assert result.reason == "non_positive_average_invested_capital"


def test_roic_missing_denominator_is_unavailable():
    result = f.calculate_roic(20.0, None)

    assert result.available is False
    assert result.reason == "missing_average_invested_capital"


@pytest.mark.parametrize(
    ("capex", "depreciation", "expected"),
    [(-50.0, 20.0, 30.0), (-10.0, 20.0, -10.0)],
)
def test_simplified_net_investment_formula(capex, depreciation, expected):
    result = f.calculate_simplified_net_investment(capex, depreciation)

    assert result.available is True
    assert result.value == expected


@pytest.mark.parametrize(
    ("capex", "depreciation", "reason"),
    [
        (None, 20.0, "missing_capex"),
        (-50.0, None, "missing_depreciation_amortization"),
    ],
)
def test_simplified_net_investment_missing_input_is_unavailable(
    capex,
    depreciation,
    reason,
):
    result = f.calculate_simplified_net_investment(capex, depreciation)

    assert result.available is False
    assert result.reason == reason


def test_normal_reinvestment_rate():
    result = f.calculate_reinvestment_rate(30.0, 60.0)

    assert result.value == pytest.approx(0.50)


@pytest.mark.parametrize("nopat", [0.0, -10.0])
def test_reinvestment_rate_requires_positive_nopat(nopat):
    result = f.calculate_reinvestment_rate(30.0, nopat)

    assert result.available is False
    assert result.reason == "non_positive_nopat"


def test_reinvestment_rate_allows_negative_reinvestment():
    result = f.calculate_reinvestment_rate(-10.0, 50.0)

    assert result.available is True
    assert result.value == pytest.approx(-0.20)


def test_fundamental_growth_capacity_is_roic_times_reinvestment_rate():
    result = f.calculate_fundamental_growth_capacity(0.25, 0.40)

    assert result.available is True
    assert result.value == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("roic", "reinvestment_rate", "reason"),
    [
        (None, 0.40, "missing_roic"),
        (0.25, None, "missing_reinvestment_rate"),
    ],
)
def test_growth_capacity_propagates_unavailable_component(
    roic,
    reinvestment_rate,
    reason,
):
    result = f.calculate_fundamental_growth_capacity(roic, reinvestment_rate)

    assert result.available is False
    assert result.reason == reason


def test_history_calculates_average_capital_roic_and_reinvestment():
    dates = ["2024-12-31", "2025-12-31"]
    empty = pd.Series(dtype=float)
    history = f.build_fundamental_history(
        annual_revenue=series([200.0, 240.0], dates),
        annual_gross_profit=empty,
        annual_operating_income=series([40.0, 50.0], dates),
        annual_cfo=empty,
        annual_capex=series([-30.0, -40.0], dates),
        annual_pretax_income=series([32.0, 40.0], dates),
        annual_tax_provision=series([6.4, 8.0], dates),
        annual_total_equity=series([80.0, 100.0], dates),
        annual_total_debt=series([30.0, 40.0], dates),
        annual_cash=series([10.0, 20.0], dates),
        annual_depreciation_amortization=series([10.0, 15.0], dates),
        quarterly_revenue=empty,
        quarterly_gross_profit=empty,
        quarterly_operating_income=empty,
        quarterly_cfo=empty,
        quarterly_capex=empty,
        annual_periods=dates,
    )
    latest = history.annual.loc["2025-12-31"]

    assert latest[f.NOPAT] == pytest.approx(40.0)
    assert latest[f.INVESTED_CAPITAL] == 120.0
    assert latest[f.AVERAGE_INVESTED_CAPITAL] == 110.0
    assert latest[f.ROIC] == pytest.approx(40.0 / 110.0)
    assert latest[f.NET_INVESTMENT] == 25.0
    assert latest[f.REINVESTMENT_RATE] == pytest.approx(0.625)
    assert latest[f.FUNDAMENTAL_GROWTH_CAPACITY] == pytest.approx(
        (40.0 / 110.0) * 0.625
    )
    assert history.annual_reasons.loc["2025-12-31", f.ROIC] is None


def test_history_rejects_non_consecutive_prior_capital():
    dates = ["2023-12-31", "2025-12-31"]
    empty = pd.Series(dtype=float)
    history = f.build_fundamental_history(
        annual_revenue=empty,
        annual_gross_profit=empty,
        annual_operating_income=series([40.0, 50.0], dates),
        annual_cfo=empty,
        annual_capex=empty,
        annual_pretax_income=series([32.0, 40.0], dates),
        annual_tax_provision=series([6.4, 8.0], dates),
        annual_total_equity=series([80.0, 100.0], dates),
        annual_total_debt=series([30.0, 40.0], dates),
        annual_cash=series([10.0, 20.0], dates),
        annual_depreciation_amortization=empty,
        quarterly_revenue=empty,
        quarterly_gross_profit=empty,
        quarterly_operating_income=empty,
        quarterly_cfo=empty,
        quarterly_capex=empty,
        annual_periods=dates,
    )

    assert pd.isna(history.annual.loc["2025-12-31", f.ROIC])
    assert history.annual_reasons.loc[
        "2025-12-31", f.AVERAGE_INVESTED_CAPITAL
    ] == "non_consecutive_fiscal_years"


def test_roic_field_aliases_are_conservative(statement_factory):
    balance = statement_factory(
        {
            "Stockholders Equity": [100.0],
            "Common Stock Equity": [90.0],
            "Total Equity Gross Minority Interest": [110.0],
            "Cash And Cash Equivalents": [20.0],
            "Cash Cash Equivalents And Short Term Investments": [30.0],
        },
        ["2025-12-31"],
    )
    cashflow = statement_factory(
        {
            "Depreciation And Amortization": [15.0],
            "Depreciation Amortization Depletion": [16.0],
            "Depreciation": [12.0],
        },
        ["2025-12-31"],
    )

    assert app.resolve_financial_field(
        balance, "total_equity"
    ).row_name == "Stockholders Equity"
    assert app.resolve_financial_field(
        balance, "roic_cash"
    ).row_name == "Cash And Cash Equivalents"
    assert app.resolve_financial_field(
        cashflow, "depreciation_amortization"
    ).row_name == "Depreciation And Amortization"


def test_roic_cash_does_not_accept_combined_cash_alias(statement_factory):
    balance = statement_factory(
        {"Cash Cash Equivalents And Short Term Investments": [30.0]},
        ["2025-12-31"],
    )

    assert app.resolve_financial_field(balance, "roic_cash").row is None


@pytest.mark.parametrize(
    ("years", "dates", "values", "expected"),
    [
        (3, ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"],
         [100.0, 110.0, 121.0, 133.1], 0.10),
        (5, ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31",
             "2024-12-31", "2025-12-31"],
         [100.0, 110.0, 121.0, 133.1, 146.41, 161.051], 0.10),
        (3, ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"],
         [100.0, 95.0, 90.0, 80.0], (0.8 ** (1 / 3)) - 1),
        (3, ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"],
         [100.0, 80.0, 40.0, 0.0], -1.0),
    ],
)
def test_revenue_cagr_valid_year_count_and_growth(
    years, dates, values, expected
):
    result = f.calculate_revenue_cagr(series(values, dates), years)

    assert result.available is True
    assert result.value == pytest.approx(expected)
    assert result.years == years
    assert result.start_period == pd.Timestamp(dates[0])
    assert result.end_period == pd.Timestamp(dates[-1])


def test_revenue_cagr_sorts_unordered_periods():
    result = f.calculate_revenue_cagr(
        series([133.1, 100.0, 121.0, 110.0],
               ["2025-12-31", "2022-12-31", "2024-12-31", "2023-12-31"]),
        3,
    )

    assert result.value == pytest.approx(0.10)
    assert result.start_period == pd.Timestamp("2022-12-31")


@pytest.mark.parametrize(
    ("values", "dates", "reason"),
    [
        ([0.0, 10.0, 20.0, 30.0],
         ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"],
         "non_positive_start_revenue"),
        ([100.0, np.nan, 120.0, 130.0],
         ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"],
         "missing_intermediate_revenue"),
        ([100.0, 110.0, 120.0, np.nan],
         ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"],
         "missing_end_revenue"),
        ([100.0, 120.0, 130.0],
         ["2022-12-31", "2024-12-31", "2025-12-31"],
         "insufficient_history"),
        ([100.0, 110.0, 120.0, 130.0],
         ["2021-12-31", "2022-12-31", "2024-12-31", "2025-12-31"],
         "non_consecutive_fiscal_years"),
    ],
)
def test_revenue_cagr_unavailable_cases(values, dates, reason):
    result = f.calculate_revenue_cagr(series(values, dates), 3)

    assert result.available is False
    assert result.reason == reason


@pytest.mark.parametrize(
    ("start_revenue", "end_revenue", "start_capital", "end_capital", "expected"),
    [
        (100.0, 130.0, 50.0, 60.0, 3.0),
        (100.0, 80.0, 50.0, 60.0, -2.0),
        (100.0, 130.0, 60.0, 50.0, -3.0),
        (0.0, 0.0, 0.0, 10.0, 0.0),
    ],
)
def test_annual_sales_to_capital_preserves_economic_signs_and_zero(
    start_revenue, end_revenue, start_capital, end_capital, expected
):
    result = f.calculate_sales_to_capital(
        start_revenue, end_revenue, start_capital, end_capital,
        "2024-12-31", "2025-12-31",
    )

    assert result.available is True
    assert result.value == pytest.approx(expected)
    assert result.delta_revenue == pytest.approx(end_revenue - start_revenue)
    assert result.delta_invested_capital == pytest.approx(end_capital - start_capital)


@pytest.mark.parametrize("capital_delta", [0.0, 1e-10])
def test_sales_to_capital_rejects_zero_or_near_zero_capital_delta(capital_delta):
    result = f.calculate_sales_to_capital(
        100.0, 120.0, 50.0, 50.0 + capital_delta,
        "2024-12-31", "2025-12-31",
    )

    assert result.available is False
    assert result.reason == "zero_or_near_zero_delta_invested_capital"
    assert result.delta_revenue == 20.0


@pytest.mark.parametrize(
    ("args", "reason"),
    [
        ((None, 120.0, 50.0, 60.0), "missing_start_revenue"),
        ((100.0, None, 50.0, 60.0), "missing_end_revenue"),
        ((100.0, 120.0, None, 60.0), "missing_start_invested_capital"),
        ((100.0, 120.0, 50.0, None), "missing_end_invested_capital"),
    ],
)
def test_sales_to_capital_missing_components_are_unavailable(args, reason):
    result = f.calculate_sales_to_capital(
        *args, "2024-12-31", "2025-12-31"
    )

    assert result.available is False
    assert result.reason == reason


def test_annual_sales_to_capital_rejects_non_consecutive_years():
    result = f.calculate_sales_to_capital(
        100.0, 130.0, 50.0, 60.0, "2023-12-31", "2025-12-31"
    )

    assert result.available is False
    assert result.reason == "non_consecutive_fiscal_years"


def test_normalized_sales_to_capital_uses_cumulative_deltas_not_ratio_average():
    dates = ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    revenue = series([100.0, 120.0, 125.0, 160.0], dates)
    capital = series([50.0, 51.0, 60.0, 80.0], dates)

    result = f.calculate_normalized_sales_to_capital(revenue, capital, 3)
    annual_ratios = [20.0 / 1.0, 5.0 / 9.0, 35.0 / 20.0]

    assert result.available is True
    assert result.value == pytest.approx(60.0 / 30.0)
    assert result.value != pytest.approx(np.mean(annual_ratios))
    assert result.delta_revenue == 60.0
    assert result.delta_invested_capital == 30.0


def test_valid_five_year_normalized_sales_to_capital():
    dates = pd.date_range("2020-12-31", periods=6, freq="YE")
    result = f.calculate_normalized_sales_to_capital(
        series([100, 110, 120, 130, 140, 150], dates),
        series([50, 55, 60, 65, 70, 75], dates),
        5,
    )

    assert result.available is True
    assert result.value == pytest.approx(2.0)
    assert result.years == 5


@pytest.mark.parametrize(
    ("revenues", "capitals", "expected"),
    [
        ([100.0, 110.0, 120.0, 130.0], [80.0, 70.0, 60.0, 50.0], -1.0),
        ([130.0, 120.0, 110.0, 100.0], [50.0, 60.0, 70.0, 80.0], -1.0),
    ],
)
def test_normalized_sales_to_capital_preserves_negative_deltas(
    revenues, capitals, expected
):
    dates = ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    result = f.calculate_normalized_sales_to_capital(
        series(revenues, dates), series(capitals, dates), 3
    )

    assert result.available is True
    assert result.value == expected


def test_normalized_sales_to_capital_rejects_missing_year_and_bad_endpoints():
    gap_dates = ["2021-12-31", "2022-12-31", "2024-12-31", "2025-12-31"]
    gap = f.calculate_normalized_sales_to_capital(
        series([100, 110, 120, 130], gap_dates),
        series([50, 55, 60, 65], gap_dates), 3,
    )
    missing_endpoint = f.calculate_normalized_sales_to_capital(
        series([100, 110, 120, np.nan],
               ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]),
        series([50, 55, 60, 65],
               ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]), 3,
    )

    assert gap.reason == "non_consecutive_fiscal_years"
    assert missing_endpoint.reason == "missing_end_revenue"


def test_normalized_sales_to_capital_rejects_zero_denominator():
    dates = ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    result = f.calculate_normalized_sales_to_capital(
        series([100, 110, 120, 130], dates),
        series([50, 60, 55, 50], dates), 3,
    )

    assert result.available is False
    assert result.reason == "zero_or_near_zero_delta_invested_capital"


def test_approximate_roic_is_after_tax_margin_times_sales_to_capital():
    result = f.calculate_approximate_roic(0.20, 0.25, 2.0)

    assert result.available is True
    assert result.value == pytest.approx(0.30)


def test_history_exposes_annual_and_normalized_dcf_anchors():
    dates = ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    empty = pd.Series(dtype=float)
    history = f.build_fundamental_history(
        annual_revenue=series([100, 120, 140, 160], dates),
        annual_gross_profit=empty,
        annual_operating_income=series([20, 24, 28, 32], dates),
        annual_cfo=empty, annual_capex=empty,
        annual_pretax_income=series([16, 19.2, 22.4, 25.6], dates),
        annual_tax_provision=series([4, 4.8, 5.6, 6.4], dates),
        annual_total_equity=series([50, 55, 60, 65], dates),
        annual_total_debt=series([10, 10, 10, 10], dates),
        annual_cash=series([10, 10, 10, 10], dates),
        quarterly_revenue=empty, quarterly_gross_profit=empty,
        quarterly_operating_income=empty, quarterly_cfo=empty,
        quarterly_capex=empty, annual_periods=dates,
    )

    assert history.dcf_anchors.revenue_cagr[3].available is True
    assert history.dcf_anchors.normalized_sales_to_capital[3].value == 4.0
    assert set(history.dcf_anchors.revenue_cagr) == {3}
    assert set(history.dcf_anchors.normalized_sales_to_capital) == {3}
    assert history.annual.loc["2025-12-31", f.SALES_TO_CAPITAL] == 4.0
    assert history.annual.loc["2025-12-31", f.APPROXIMATE_ROIC] == pytest.approx(0.60)
