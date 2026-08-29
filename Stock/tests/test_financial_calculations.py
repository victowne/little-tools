import numpy as np
import pandas as pd
import pytest

from Stock import stock_valuation_mvp as app


def test_financial_trend_frame_keeps_amounts_in_billions_and_margins_as_ratios():
    period = pd.Timestamp("2026-06-30")
    income = pd.DataFrame(
        {
            period: [
                100_000_000_000.0,
                40_000_000_000.0,
                20_000_000_000.0,
                15_000_000_000.0,
            ]
        },
        index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
    )
    cashflow = pd.DataFrame(
        {period: [30_000_000_000.0, -5_000_000_000.0]},
        index=["Operating Cash Flow", "Capital Expenditure"],
    )

    result = app._financial_trend_frame(income, cashflow, pd.DataFrame())

    assert result.loc[period, "Revenue"] == pytest.approx(100.0)
    assert result.loc[period, "Gross Profit"] == pytest.approx(40.0)
    assert result.loc[period, "Gross Margin"] == pytest.approx(0.40)
    assert result.loc[period, "Operating Margin"] == pytest.approx(0.20)
    assert result.loc[period, "Free Cash Flow"] == pytest.approx(25.0)


def test_exact_field_name_is_resolved(statement_factory):
    statement = statement_factory(
        {
            "Operating Cash Flow": [100.0],
            "Total Cash From Operating Activities": [999.0],
        },
        ["2025-12-31"],
    )

    result = app._statement_series(
        statement,
        ("Operating Cash Flow", "Total Cash From Operating Activities"),
    )

    assert result.iloc[0] == 100.0


def test_alternative_field_name_is_resolved(statement_factory):
    statement = statement_factory(
        {"Total Cash From Operating Activities": [125.0]},
        ["2025-12-31"],
    )

    result = app._statement_series(
        statement,
        ("Operating Cash Flow", "Total Cash From Operating Activities"),
    )

    assert result.iloc[0] == 125.0


def test_field_normalization_ignores_case_spaces_and_punctuation(statement_factory):
    statement = statement_factory(
        {"OPERATING-CASH_FLOW": [42.0]},
        ["2025-12-31"],
    )

    result = app._statement_series(statement, ("Operating Cash Flow",))

    assert result.iloc[0] == 42.0


def test_no_candidate_returns_empty_series(statement_factory):
    statement = statement_factory({"Revenue": [50.0]}, ["2025-12-31"])

    result = app._statement_series(statement, ("Operating Cash Flow",))

    assert result.empty


def test_real_zero_is_preserved(statement_factory):
    statement = statement_factory({"Total Debt": [0.0]}, ["2025-12-31"])

    series = app._statement_series(statement, ("Total Debt",))

    assert len(series) == 1
    assert series.iloc[0] == 0.0
    assert app._latest_statement_optional(statement, ("Total Debt",)) == 0.0


def test_nan_is_dropped_but_other_periods_remain(statement_factory):
    statement = statement_factory(
        {"Net Income": [np.nan, 12.0]},
        ["2024-12-31", "2025-12-31"],
    )

    result = app._statement_series(statement, ("Net Income",))

    assert result.to_dict() == {pd.Timestamp("2025-12-31"): 12.0}


def test_all_nan_is_missing_for_scalar_helpers(statement_factory):
    statement = statement_factory({"Total Debt": [np.nan]}, ["2025-12-31"])

    assert app._latest_statement_optional(statement, ("Total Debt",)) is None


def test_missing_field_and_real_zero_are_distinct(statement_factory):
    zero_statement = statement_factory({"Total Debt": [0.0]}, ["2025-12-31"])
    missing_statement = statement_factory({"Revenue": [10.0]}, ["2025-12-31"])

    assert app._latest_statement_optional(zero_statement, ("Total Debt",)) == 0.0
    assert app._latest_statement_optional(missing_statement, ("Total Debt",)) is None


def test_latest_statement_nan_does_not_fall_back_to_older_period(statement_factory):
    statement = statement_factory(
        {"Total Debt": [12.0, np.nan]},
        ["2024-12-31", "2025-12-31"],
    )

    assert app._latest_statement_optional(statement, ("Total Debt",)) is None


def test_reported_zero_net_debt_is_preserved():
    assert app._derive_net_debt(0.0, 10.0, 3.0, 20.0, 4.0) == 0.0


def test_net_debt_is_missing_when_no_complete_debt_cash_pair_exists():
    assert app._derive_net_debt(None, 10.0, None, None, 4.0) is None


def test_net_income_continuous_operations_is_not_substring_matched(statement_factory):
    statement = statement_factory(
        {"Net Income Continuous Operations": [17.0]},
        ["2025-12-31"],
    )

    result = app._statement_series(statement, "net_income")

    assert result.empty


def test_net_income_canonical_wins_among_similar_rows(statement_factory):
    statement = statement_factory(
        {
            "Net Income": [10.0],
            "Net Income Common Stockholders": [20.0],
            "Net Income Continuous Operations": [30.0],
            "Net Income Including Noncontrolling Interests": [40.0],
        },
        ["2025-12-31"],
    )

    match = app.resolve_financial_field(statement, "net_income")

    assert match.row_name == "Net Income"
    assert match.tier == 1
    assert match.row.iloc[0] == 10.0


def test_net_income_common_stockholders_is_approved_alias(statement_factory):
    statement = statement_factory(
        {"Net Income Common Stockholders": [20.0]}, ["2025-12-31"]
    )

    match = app.resolve_financial_field(statement, "net_income")

    assert match.row_name == "Net Income Common Stockholders"
    assert match.tier == 2


@pytest.mark.parametrize(
    "row_name",
    [
        "Net Income Continuous Operations",
        "Net Income Including Noncontrolling Interests",
    ],
)
def test_financially_different_net_income_rows_are_rejected(
    statement_factory,
    row_name,
):
    statement = statement_factory({row_name: [30.0]}, ["2025-12-31"])

    match = app.resolve_financial_field(statement, "net_income")

    assert match.row is None
    assert match.reason == "not_found"


def test_operating_income_and_ebit_resolve_separate_rows(statement_factory):
    statement = statement_factory(
        {
            "Operating Income": [10.0],
            "EBIT": [20.0],
            "Total Operating Income As Reported": [30.0],
            "Pretax Income": [40.0],
        },
        ["2025-12-31"],
    )

    operating = app.resolve_financial_field(statement, "operating_income")
    ebit = app.resolve_financial_field(statement, "ebit")

    assert operating.row_name == "Operating Income"
    assert operating.row.iloc[0] == 10.0
    assert ebit.row_name == "EBIT"
    assert ebit.row.iloc[0] == 20.0


def test_total_operating_income_is_explicit_operating_alias(statement_factory):
    statement = statement_factory(
        {
            "Total Operating Income As Reported": [30.0],
            "Pretax Income": [40.0],
        },
        ["2025-12-31"],
    )

    match = app.resolve_financial_field(statement, "operating_income")

    assert match.row_name == "Total Operating Income As Reported"
    assert match.tier == 2


def test_pretax_income_is_not_operating_income_or_ebit(statement_factory):
    statement = statement_factory({"Pretax Income": [40.0]}, ["2025-12-31"])

    assert app.resolve_financial_field(statement, "operating_income").row is None
    assert app.resolve_financial_field(statement, "ebit").row is None


def test_interest_expense_ignores_net_interest_income(statement_factory):
    statement = statement_factory(
        {
            "Interest Expense": [10.0],
            "Interest Expense Non Operating": [20.0],
            "Net Interest Income": [30.0],
        },
        ["2025-12-31"],
    )

    match = app.resolve_financial_field(statement, "interest_expense")

    assert match.row_name == "Interest Expense"
    assert match.row.iloc[0] == 10.0


def test_interest_non_operating_is_alias_but_net_interest_is_not(statement_factory):
    alias_statement = statement_factory(
        {"Interest Expense Non Operating": [20.0]}, ["2025-12-31"]
    )
    wrong_statement = statement_factory(
        {"Net Interest Income": [30.0]}, ["2025-12-31"]
    )

    assert app.resolve_financial_field(
        alias_statement, "interest_expense"
    ).row_name == "Interest Expense Non Operating"
    assert app.resolve_financial_field(
        wrong_statement, "interest_expense"
    ).row is None


def test_debt_concepts_resolve_only_their_approved_rows(statement_factory):
    statement = statement_factory(
        {
            "Total Debt": [10.0],
            "Long Term Debt": [20.0],
            "Current Debt": [30.0],
            "Net Debt": [40.0],
        },
        ["2025-12-31"],
    )

    assert app.resolve_financial_field(statement, "total_debt").row_name == "Total Debt"
    assert app.resolve_financial_field(statement, "long_term_debt").row_name == "Long Term Debt"
    assert app.resolve_financial_field(statement, "net_debt").row_name == "Net Debt"


def test_current_debt_is_not_total_or_long_term_debt(statement_factory):
    statement = statement_factory({"Current Debt": [30.0]}, ["2025-12-31"])

    assert app.resolve_financial_field(statement, "total_debt").row is None
    assert app.resolve_financial_field(statement, "long_term_debt").row is None


def test_cash_concept_prefers_cash_equivalents_among_similar_rows(statement_factory):
    statement = statement_factory(
        {
            "Cash Cash Equivalents And Short Term Investments": [10.0],
            "Cash And Cash Equivalents": [20.0],
            "Cash Financial": [30.0],
        },
        ["2025-12-31"],
    )

    match = app.resolve_financial_field(statement, "cash")

    assert match.row_name == "Cash And Cash Equivalents"
    assert match.tier == 1


def test_cash_combined_field_is_alias_but_cash_financial_is_not(statement_factory):
    combined = statement_factory(
        {"Cash Cash Equivalents And Short Term Investments": [10.0]},
        ["2025-12-31"],
    )
    financial = statement_factory({"Cash Financial": [30.0]}, ["2025-12-31"])

    assert app.resolve_financial_field(combined, "cash").tier == 2
    assert app.resolve_financial_field(financial, "cash").row is None


def test_cash_flow_concepts_do_not_cross_match(statement_factory):
    statement = statement_factory(
        {
            "Operating Cash Flow": [10.0],
            "Total Cash From Operating Activities": [20.0],
            "Free Cash Flow": [30.0],
            "Capital Expenditure": [-40.0],
        },
        ["2025-12-31"],
    )

    assert app.resolve_financial_field(
        statement, "operating_cash_flow"
    ).row_name == "Operating Cash Flow"
    assert app.resolve_financial_field(
        statement, "free_cash_flow"
    ).row_name == "Free Cash Flow"
    assert app.resolve_financial_field(
        statement, "capital_expenditure"
    ).row_name == "Capital Expenditure"


def test_operating_cash_flow_alias_is_explicit(statement_factory):
    statement = statement_factory(
        {"Total Cash From Operating Activities": [20.0]}, ["2025-12-31"]
    )

    match = app.resolve_financial_field(statement, "operating_cash_flow")

    assert match.row_name == "Total Cash From Operating Activities"
    assert match.tier == 2


def test_duplicate_normalized_canonical_rows_are_ambiguous():
    statement = pd.DataFrame(
        [[10.0], [20.0]],
        index=["Net Income", "NET-INCOME"],
        columns=pd.to_datetime(["2025-12-31"]),
    )

    match = app.resolve_financial_field(statement, "net_income")

    assert match.row is None
    assert match.reason == "ambiguous_normalized_match"


def test_validated_ttm_uses_four_consecutive_quarters():
    dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    quarterly = pd.Series([10.0, 20.0, 30.0, 40.0], index=dates)

    result = app.build_validated_ttm(quarterly)

    assert result.available is True
    assert result.value == 100.0
    assert result.periods_used == tuple(pd.to_datetime(dates))
    assert result.reason is None


def test_validated_ttm_selects_latest_four_from_five():
    dates = ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    quarterly = pd.Series([50.0, 10.0, 20.0, 30.0, 40.0], index=dates)

    result = app.build_validated_ttm(quarterly)

    assert result.available is True
    assert result.value == 100.0
    assert result.periods_used == tuple(pd.to_datetime(dates[-4:]))


def test_validated_ttm_sorts_out_of_order_dates():
    dates = ["2025-12-31", "2025-03-31", "2025-09-30", "2025-06-30"]
    quarterly = pd.Series([40.0, 10.0, 30.0, 20.0], index=dates)

    result = app.build_validated_ttm(quarterly)

    assert result.available is True
    assert result.value == 100.0
    assert result.periods_used == tuple(pd.to_datetime([
        "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"
    ]))


def test_validated_ttm_duplicate_period_keeps_last_occurrence():
    dates = ["2025-03-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    quarterly = pd.Series([1.0, 10.0, 20.0, 30.0, 40.0], index=dates)

    result = app.build_validated_ttm(quarterly)

    assert result.available is True
    assert result.value == 100.0
    assert result.periods_used == tuple(pd.to_datetime([
        "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"
    ]))


def test_validated_ttm_fewer_than_four_quarters_is_unavailable():
    dates = ["2025-03-31", "2025-06-30", "2025-09-30"]
    quarterly = pd.Series([10.0, 20.0, 30.0], index=dates)

    result = app.build_validated_ttm(quarterly)

    assert result.available is False
    assert result.value is None
    assert result.periods_used == tuple(pd.to_datetime(dates))
    assert result.reason == "fewer_than_four_quarters"


def test_validated_ttm_gap_in_middle_is_unavailable():
    dates = ["2024-12-31", "2025-03-31", "2025-06-30", "2025-12-31"]
    quarterly = pd.Series([10.0, 20.0, 30.0, 40.0], index=dates)

    result = app.build_validated_ttm(quarterly)

    assert result.available is False
    assert result.reason == "non_consecutive_quarters"
    assert result.periods_used == tuple(pd.to_datetime(dates))


def test_validated_ttm_latest_quarter_missing_is_unavailable():
    dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    quarterly = pd.Series([10.0, 20.0, 30.0], index=dates[:3])

    result = app.build_validated_ttm(quarterly, expected_periods=dates)

    assert result.available is False
    assert result.reason == "missing_quarter_value"
    assert result.periods_used == tuple(pd.to_datetime(dates))


def test_validated_ttm_nan_in_required_four_is_unavailable():
    dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    quarterly = pd.Series([10.0, 20.0, np.nan, 40.0], index=dates)

    result = app.build_validated_ttm(quarterly, expected_periods=dates)

    assert result.available is False
    assert result.reason == "missing_quarter_value"


def test_validated_ttm_does_not_replace_recent_missing_with_older_quarter():
    expected = ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    quarterly = pd.Series(
        [50.0, 10.0, 20.0, 30.0],
        index=expected[:4],
    )

    result = app.build_validated_ttm(quarterly, expected_periods=expected)

    assert result.available is False
    assert result.reason == "missing_quarter_value"
    assert result.periods_used == tuple(pd.to_datetime(expected[-4:]))


def test_validated_ttm_all_values_missing():
    dates = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    quarterly = pd.Series([np.nan, np.nan, np.nan, np.nan], index=dates)

    result = app.build_validated_ttm(quarterly, expected_periods=dates)

    assert result.available is False
    assert result.value is None
    assert result.periods_used == ()
    assert result.reason == "all_values_missing"


def test_validated_ttm_invalid_reporting_date():
    quarterly = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
        index=["2025-03-31", "not-a-date", "2025-09-30", "2025-12-31"],
    )

    result = app.build_validated_ttm(quarterly)

    assert result.available is False
    assert result.value is None
    assert result.periods_used == ()
    assert result.reason == "invalid_dates"


def test_latest_flow_value_fewer_than_four_uses_latest_annual():
    quarterly = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]),
    )
    annual = pd.Series([75.0, 90.0], index=pd.to_datetime(["2024-12-31", "2025-12-31"]))

    value, basis = app._latest_flow_value(
        quarterly,
        annual,
        expected_periods=quarterly.index,
    )

    assert value == 90.0
    assert basis == "财年截至 2025-12-31"


def test_latest_flow_value_fewer_than_four_without_annual_returns_missing():
    quarterly = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]),
    )

    value, basis = app._latest_flow_value(
        quarterly,
        pd.Series(dtype=float),
        expected_periods=quarterly.index,
    )

    assert value is None
    assert basis == "无可用数据"


def test_latest_flow_value_non_consecutive_quarters_uses_annual_fallback():
    quarterly = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
        index=pd.to_datetime(["2024-12-31", "2025-03-31", "2025-09-30", "2025-12-31"]),
    )
    annual = pd.Series([88.0], index=pd.to_datetime(["2025-12-31"]))

    value, basis = app._latest_flow_value(
        quarterly,
        annual,
        expected_periods=quarterly.index,
    )

    assert value == 88.0
    assert basis == "财年截至 2025-12-31"


def build_health_fixture(statement_factory, long_term_debt_marker):
    balance_rows = {
        "Total Assets": [100_000_000_000.0],
        "Total Liabilities": [50_000_000_000.0],
    }
    if long_term_debt_marker is not None:
        balance_rows["Long Term Debt"] = [long_term_debt_marker]
    return app._build_health_checks(
        annual_income=statement_factory(
            {"Net Income": [10_000_000_000.0]}, ["2025-12-31"]
        ),
        annual_cashflow=statement_factory(
            {
                "Operating Cash Flow": [30_000_000_000.0],
                "Investing Cash Flow": [-10_000_000_000.0],
                "Financing Cash Flow": [-5_000_000_000.0],
            },
            ["2025-12-31"],
        ),
        annual_balance=statement_factory(balance_rows, ["2025-12-31"]),
        quarterly_income=pd.DataFrame(),
        quarterly_cashflow=pd.DataFrame(),
        quarterly_balance=pd.DataFrame(),
    )


def test_health_check_missing_debt_is_unavailable(statement_factory):
    checks = build_health_fixture(statement_factory, None)

    assert checks[1]["status"] is None
    assert "数据缺失" in checks[1]["detail"]


def test_health_check_missing_assets_is_unavailable(statement_factory):
    checks = app._build_health_checks(
        annual_income=pd.DataFrame(),
        annual_cashflow=pd.DataFrame(),
        annual_balance=statement_factory(
            {"Total Liabilities": [50_000_000_000.0]}, ["2025-12-31"]
        ),
        quarterly_income=pd.DataFrame(),
        quarterly_cashflow=pd.DataFrame(),
        quarterly_balance=pd.DataFrame(),
    )

    assert checks[0]["status"] is None


def test_health_check_real_zero_debt_passes(statement_factory):
    checks = build_health_fixture(statement_factory, 0.0)

    assert checks[1]["status"] is True
    assert "0.00B" in checks[1]["detail"]


def test_health_check_latest_missing_net_income_is_unavailable(statement_factory):
    checks = app._build_health_checks(
        annual_income=statement_factory(
            {"Net Income": [10_000_000_000.0, np.nan]},
            ["2024-12-31", "2025-12-31"],
        ),
        annual_cashflow=pd.DataFrame(),
        annual_balance=statement_factory(
            {
                "Total Assets": [100_000_000_000.0],
                "Total Liabilities": [50_000_000_000.0],
                "Long Term Debt": [5_000_000_000.0],
            },
            ["2025-12-31"],
        ),
        quarterly_income=pd.DataFrame(),
        quarterly_cashflow=pd.DataFrame(),
        quarterly_balance=pd.DataFrame(),
    )

    assert checks[1]["status"] is None
    assert "数据缺失" in checks[1]["detail"]


def test_market_adapter_preserves_missing_values(snapshot_factory):
    snapshot = snapshot_factory(
        price=None,
        shares_outstanding=None,
        net_debt=None,
    )

    assert app.fetch_market_data("TEST", snapshot) == (None, None, None)


def test_market_adapter_preserves_real_zeros(snapshot_factory):
    snapshot = snapshot_factory(
        price=0.0,
        shares_outstanding=0.0,
        net_debt=0.0,
    )

    assert app.fetch_market_data("TEST", snapshot) == (0.0, 0.0, 0.0)


def patch_wacc_external_inputs(monkeypatch):
    monkeypatch.setattr(app, "_regression_beta", lambda ticker: (1.0, 60))
    monkeypatch.setattr(
        app,
        "fetch_macro_assumptions",
        lambda: {
            "risk_free": 0.04,
            "erp": 0.05,
            "treasury_date": "fixture",
            "erp_date": "fixture",
        },
    )
    monkeypatch.setattr(
        app,
        "fetch_industry_wacc",
        lambda industry: {"wacc": None, "matched_industry": None},
    )


def test_wacc_distinguishes_missing_interest_from_zero(
    statement_factory,
    snapshot_factory,
    monkeypatch,
):
    patch_wacc_external_inputs(monkeypatch)
    base_income = {
        "EBIT": [100.0],
        "Pretax Income": [80.0],
        "Tax Provision": [16.0],
    }
    missing = snapshot_factory(
        total_debt=0.0,
        annual_income=statement_factory(base_income, ["2025-12-31"]),
    )
    zero = snapshot_factory(
        total_debt=0.0,
        annual_income=statement_factory(
            {**base_income, "Interest Expense": [0.0]}, ["2025-12-31"]
        ),
    )

    missing_result = app.fetch_wacc_reference("TEST", missing)
    zero_result = app.fetch_wacc_reference("TEST", zero)

    assert missing_result["interest_expense"] is None
    assert missing_result["interest_assumption_used"] is True
    assert zero_result["interest_expense"] == 0.0
    assert zero_result["interest_assumption_used"] is False


def test_wacc_distinguishes_missing_tax_from_zero_tax(
    statement_factory,
    snapshot_factory,
    monkeypatch,
):
    patch_wacc_external_inputs(monkeypatch)
    base_income = {
        "EBIT": [100.0],
        "Interest Expense": [5.0],
        "Pretax Income": [80.0],
    }
    missing = snapshot_factory(
        total_debt=10.0,
        annual_income=statement_factory(base_income, ["2025-12-31"]),
    )
    zero = snapshot_factory(
        total_debt=10.0,
        annual_income=statement_factory(
            {**base_income, "Tax Provision": [0.0]}, ["2025-12-31"]
        ),
    )

    missing_result = app.fetch_wacc_reference("TEST", missing)
    zero_result = app.fetch_wacc_reference("TEST", zero)

    assert missing_result["tax_provision"] is None
    assert missing_result["tax_rate"] == pytest.approx(0.21)
    assert missing_result["tax_assumption_used"] is True
    assert zero_result["tax_provision"] == 0.0
    assert zero_result["tax_rate"] == 0.0
    assert zero_result["tax_assumption_used"] is False
