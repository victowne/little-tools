import pandas as pd
import pytest

from Stock import stock_valuation_mvp as app
from Stock.fundamentals import (
    OPERATING_MARGIN,
    OPERATING_TAX_RATE,
    FundamentalHistory,
    HistoricalDCFAnchors,
    TTMResult,
)
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.share_normalization import NormalizedShareCount
from Stock.valuation_sensitivity import build_wacc_terminal_growth_sensitivity
from Stock.valuation_scenarios import (
    create_scenario_from_base,
    run_multi_scenario_dcf,
)


def history(ttm_margin=0.64, tax_rate=0.16):
    annual = pd.DataFrame(
        {OPERATING_MARGIN: [0.60], OPERATING_TAX_RATE: [tax_rate]},
        index=[pd.Timestamp("2025-12-31")],
    )
    return FundamentalHistory(
        annual=annual,
        ttm={
            OPERATING_MARGIN: TTMResult(
                ttm_margin,
                True,
                tuple(pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"])),
                None,
            )
        },
        annual_reasons=pd.DataFrame(index=annual.index),
        dcf_anchors=HistoricalDCFAnchors(),
    )


def company_inputs(shares=10.0):
    normalized = NormalizedShareCount(
        ticker="TEST",
        shares_outstanding=shares,
        source="fixture",
        source_period=pd.Timestamp("2025-12-31"),
        scope="consolidated_common",
        method="fixture",
        components=(), warnings=(), available=True, reason=None,
    )
    return RealCompanyDCFInputs(
        ticker="TEST", starting_revenue=100.0,
        starting_revenue_source="ttm", starting_revenue_periods=(),
        net_debt=5.0, net_debt_source="fixture", net_debt_period=None,
        shares_outstanding=shares, normalized_share_count=normalized,
        historical_sales_to_capital_3y=1.4,
        current_accounting_roic=0.30,
    )


def test_nvda_and_alphabet_provisional_defaults_are_explicit():
    nvda = app.multistage_initial_defaults("NVDA", history())
    googl = app.multistage_initial_defaults("GOOGL", history(ttm_margin=0.3311))
    goog = app.multistage_initial_defaults("GOOG", history(ttm_margin=0.3311))

    assert (nvda["year_1_growth"], nvda["year_2_growth"], nvda["year_3_growth"]) == (30.0, 25.0, 20.0)
    assert nvda["starting_margin"] == pytest.approx(64.0)
    assert googl == goog
    assert googl["starting_margin"] == pytest.approx(33.11)
    assert googl["starting_sales_to_capital"] == 0.8


def test_generic_defaults_use_current_margin_and_tax_only_as_initial_values():
    defaults = app.multistage_initial_defaults("OTHER", history(0.28, 0.19))
    assert defaults["starting_margin"] == pytest.approx(28.0)
    assert defaults["tax_rate"] == pytest.approx(19.0)
    assert defaults["year_1_growth"] == app.MULTISTAGE_GENERIC_DEFAULTS["year_1_growth"]


def test_ticker_keyed_state_preserves_edits_and_initializes_new_ticker():
    state = {}
    app.initialize_multistage_session_state(state, "NVDA", history())
    state["multistage_NVDA_year_1_growth"] = 17.5

    rerun = app.initialize_multistage_session_state(state, "NVDA", history(0.20))
    alphabet = app.initialize_multistage_session_state(state, "GOOGL", history(0.3311))

    assert rerun["year_1_growth"] == 17.5
    assert rerun["starting_margin"] == 64.0
    assert alphabet["year_1_growth"] == 15.0
    assert "multistage_GOOGL_year_1_growth" in state


def test_research_wacc_initializes_as_provisional_default():
    state = {}
    values = app.initialize_multistage_session_state(state, "NVDA", history())
    keys = app.research_wacc_session_keys("NVDA")

    assert values["wacc"] == 9.0
    assert state[keys["value"]] == 9.0
    assert state[keys["status"]] == "provisional_default"
    assert state[keys["rationale"]] == ""


def test_explicit_research_wacc_edit_switches_status_and_survives_rerun():
    state = {}
    app.initialize_multistage_session_state(state, "NVDA", history())
    keys = app.research_wacc_session_keys("NVDA")
    state[keys["value"]] = 11.5
    app.mark_research_wacc_reviewed(
        state, "NVDA", "2026-08-13T12:00:00+00:00"
    )

    rerun = app.initialize_multistage_session_state(state, "NVDA", history())
    assert rerun["wacc"] == 11.5
    assert state[keys["status"]] == "user_reviewed"
    assert state[keys["created_at"]] == "2026-08-13T12:00:00+00:00"


def test_ticker_switching_preserves_separate_issuer_research_state():
    state = {}
    app.initialize_multistage_session_state(state, "NVDA", history())
    nvda_keys = app.research_wacc_session_keys("NVDA")
    state[nvda_keys["value"]] = 11.0
    app.mark_research_wacc_reviewed(state, "NVDA", "time-a")

    app.initialize_multistage_session_state(state, "MSFT", history())
    msft_keys = app.research_wacc_session_keys("MSFT")
    state[msft_keys["value"]] = 8.8
    app.mark_research_wacc_reviewed(state, "MSFT", "time-b")

    assert state[nvda_keys["value"]] == 11.0
    assert state[msft_keys["value"]] == 8.8
    assert nvda_keys != msft_keys


def test_goog_and_googl_share_research_wacc_status_and_rationale():
    state = {}
    app.initialize_multistage_session_state(state, "GOOG", history(0.3311))
    goog_keys = app.research_wacc_session_keys("GOOG")
    googl_keys = app.research_wacc_session_keys("GOOGL")
    assert goog_keys == googl_keys

    state[goog_keys["value"]] = 9.4
    state[goog_keys["rationale"]] = "Issuer-level long-horizon judgment"
    app.mark_research_wacc_reviewed(state, "GOOG", "review-time")
    values = app.initialize_multistage_session_state(
        state, "GOOGL", history(0.3311)
    )

    assert values["wacc"] == 9.4
    assert state[googl_keys["status"]] == "user_reviewed"
    assert state[googl_keys["rationale"]] == "Issuer-level long-horizon judgment"


def test_legacy_non_default_wacc_migrates_as_reviewed_without_reset():
    state = {"multistage_NVDA_wacc": 10.7}
    values = app.initialize_multistage_session_state(state, "NVDA", history())
    keys = app.research_wacc_session_keys("NVDA")
    assert values["wacc"] == 10.7
    assert state[keys["status"]] == "user_reviewed"


def ui_values(**overrides):
    values = app.multistage_initial_defaults("NVDA", history())
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("year_1_growth", 40.0),
        ("mature_margin", 50.0),
        ("mature_sales_to_capital", 2.0),
    ],
)
def test_editing_core_ui_assumptions_changes_valuation(override, value):
    base_model = app.build_multistage_assumptions_from_ui(ui_values())
    changed_model = app.build_multistage_assumptions_from_ui(
        ui_values(**{override: value})
    )
    base = run_multistage_dcf(company_inputs(), base_model)
    changed = run_multistage_dcf(company_inputs(), changed_model)

    assert changed.per_share_value.intrinsic_value_per_share != pytest.approx(
        base.per_share_value.intrinsic_value_per_share
    )


def test_invalid_wacc_terminal_growth_is_reported_by_existing_validation():
    with pytest.raises(ValueError, match="wacc must be greater"):
        app.build_multistage_assumptions_from_ui(
            ui_values(wacc=3.0, terminal_growth=3.5)
        )


def test_forecast_horizon_validation_is_not_silently_adjusted():
    with pytest.raises(ValueError, match="forecast_years must cover"):
        app.build_multistage_assumptions_from_ui(
            ui_values(forecast_years=9, fade_years=7)
        )


def test_sensitivity_display_frame_has_wacc_rows_and_growth_columns():
    model = app.build_multistage_assumptions_from_ui(ui_values())
    sensitivity = build_wacc_terminal_growth_sensitivity(company_inputs(), model)

    frame = app.build_sensitivity_display_frame(sensitivity)

    assert frame.shape == (5, 5)
    assert frame.loc[model.wacc, model.terminal_growth] == pytest.approx(
        sensitivity.base_case_point.intrinsic_value_per_share
    )


def test_changing_wacc_updates_sensitivity_base_and_grid():
    base_model = app.build_multistage_assumptions_from_ui(ui_values())
    changed_model = app.build_multistage_assumptions_from_ui(ui_values(wacc=9.5))
    base = build_wacc_terminal_growth_sensitivity(company_inputs(), base_model)
    changed = build_wacc_terminal_growth_sensitivity(company_inputs(), changed_model)

    assert changed.wacc_values != base.wacc_values
    assert changed.base_case_point.intrinsic_value_per_share != pytest.approx(
        base.base_case_point.intrinsic_value_per_share
    )


def test_changing_terminal_growth_updates_sensitivity_base_and_grid():
    base_model = app.build_multistage_assumptions_from_ui(ui_values())
    changed_model = app.build_multistage_assumptions_from_ui(
        ui_values(terminal_growth=4.0)
    )
    base = build_wacc_terminal_growth_sensitivity(company_inputs(), base_model)
    changed = build_wacc_terminal_growth_sensitivity(company_inputs(), changed_model)

    assert changed.terminal_growth_values != base.terminal_growth_values
    assert changed.base_case_point.intrinsic_value_per_share != pytest.approx(
        base.base_case_point.intrinsic_value_per_share
    )


def test_invalid_sensitivity_cells_are_nan_not_zero_in_display_frame():
    model = app.build_multistage_assumptions_from_ui(
        ui_values(wacc=4.0, terminal_growth=3.5)
    )
    sensitivity = build_wacc_terminal_growth_sensitivity(company_inputs(), model)
    frame = app.build_sensitivity_display_frame(sensitivity)
    invalid_wacc = sensitivity.wacc_values[0]
    invalid_growth = sensitivity.terminal_growth_values[-1]
    invalid = sensitivity.point_at(invalid_wacc, invalid_growth)

    assert invalid is not None and not invalid.valid
    assert pd.isna(frame.loc[invalid_wacc, invalid_growth])
    assert frame.loc[invalid_wacc, invalid_growth] != 0


def test_sensitivity_ui_data_contains_no_market_price_comparison():
    model = app.build_multistage_assumptions_from_ui(ui_values())
    sensitivity = build_wacc_terminal_growth_sensitivity(company_inputs(), model)
    frame = app.build_sensitivity_display_frame(sensitivity)

    assert not hasattr(sensitivity, "market_price")
    assert "market" not in " ".join(map(str, frame.columns)).lower()


def scenario_result(state=None, ticker="NVDA", base=None):
    state = {} if state is None else state
    base = base or app.build_multistage_assumptions_from_ui(ui_values())
    app.initialize_scenario_session_state(state, ticker, base)
    return run_multi_scenario_dcf(
        inputs=company_inputs(), fundamentals=history(),
        bear=app.scenario_case_from_state(state, ticker, "bear", base),
        base=create_scenario_from_base("base", base),
        bull=app.scenario_case_from_state(state, ticker, "bull", base),
    )


def test_scenario_provisional_initialization_is_transparent_and_complete():
    base = app.build_multistage_assumptions_from_ui(ui_values())
    state = {}
    app.initialize_scenario_session_state(state, "NVDA", base)
    bear_keys = app.scenario_session_keys("NVDA", "bear")
    bull_keys = app.scenario_session_keys("NVDA", "bull")

    assert state[bear_keys["year_1_growth"]] == pytest.approx(25.0)
    assert state[bear_keys["fade_years"]] == 5
    assert state[bear_keys["mature_margin"]] == pytest.approx(35.0)
    assert state[bear_keys["mature_sales_to_capital"]] == pytest.approx(1.0)
    assert state[bear_keys["wacc"]] == pytest.approx(10.0)
    assert state[bull_keys["year_1_growth"]] == pytest.approx(35.0)
    assert state[bull_keys["mature_margin"]] == pytest.approx(45.0)
    assert state[bull_keys["mature_sales_to_capital"]] == pytest.approx(1.4)
    assert state[bull_keys["wacc"]] == pytest.approx(8.5)

    bear = app.scenario_case_from_state(state, "NVDA", "bear", base)
    assert bear.assumptions.forecast_years == base.forecast_years
    assert bear.assumptions.starting_operating_margin == base.starting_operating_margin
    assert bear.assumptions.starting_sales_to_capital == base.starting_sales_to_capital
    assert bear.assumptions.operating_tax_rate == base.operating_tax_rate


def test_scenario_edits_survive_reruns_without_rebasing():
    state = {}
    first_base = app.build_multistage_assumptions_from_ui(ui_values())
    app.initialize_scenario_session_state(state, "NVDA", first_base)
    bear_keys = app.scenario_session_keys("NVDA", "bear")
    state[bear_keys["mature_margin"]] = 22.0
    app.mark_scenario_edited(state, "NVDA", "bear")

    changed_base = app.build_multistage_assumptions_from_ui(
        ui_values(year_1_growth=45.0, mature_margin=50.0)
    )
    app.initialize_scenario_session_state(state, "NVDA", changed_base)

    assert state[bear_keys["mature_margin"]] == 22.0
    assert state[bear_keys["year_1_growth"]] == 25.0
    assert state[bear_keys["status"]] == "user_edited"


def test_bear_edit_changes_only_bear_scenario_result():
    state = {}
    before = scenario_result(state)
    bear_keys = app.scenario_session_keys("NVDA", "bear")
    state[bear_keys["mature_margin"]] = 20.0
    app.mark_scenario_edited(state, "NVDA", "bear")
    after = scenario_result(state)

    assert after.bear.metrics.intrinsic_value_per_share != pytest.approx(
        before.bear.metrics.intrinsic_value_per_share
    )
    assert after.base.metrics == before.base.metrics
    assert after.bull.metrics == before.bull.metrics


def test_bull_edit_changes_only_bull_scenario_result():
    state = {}
    before = scenario_result(state)
    bull_keys = app.scenario_session_keys("NVDA", "bull")
    state[bull_keys["year_1_growth"]] = 50.0
    app.mark_scenario_edited(state, "NVDA", "bull")
    after = scenario_result(state)

    assert after.bull.metrics.intrinsic_value_per_share != pytest.approx(
        before.bull.metrics.intrinsic_value_per_share
    )
    assert after.base.metrics == before.base.metrics
    assert after.bear.metrics == before.bear.metrics


@pytest.mark.parametrize("scenario", ["bear", "bull"])
def test_invalid_alternative_keeps_other_scenarios_available(scenario):
    state = {}
    base = app.build_multistage_assumptions_from_ui(ui_values())
    app.initialize_scenario_session_state(state, "NVDA", base)
    keys = app.scenario_session_keys("NVDA", scenario)
    state[keys["wacc"]] = 2.0
    state[keys["terminal_growth"]] = 3.0
    result = scenario_result(state, base=base)

    invalid = getattr(result, scenario)
    other = result.bull if scenario == "bear" else result.bear
    assert not invalid.available
    assert "wacc must be greater" in invalid.reason
    assert result.base.available
    assert other.available


def test_scenario_rationale_and_issuer_state_persist_between_goog_classes():
    state = {}
    base = app.build_multistage_assumptions_from_ui(
        app.multistage_initial_defaults("GOOGL", history(0.3311))
    )
    app.initialize_scenario_session_state(state, "GOOGL", base)
    googl_keys = app.scenario_session_keys("GOOGL", "bull")
    goog_keys = app.scenario_session_keys("GOOG", "bull")
    assert googl_keys == goog_keys

    state[googl_keys["year_1_growth"]] = 27.0
    state[googl_keys["rationale"]] = "User-authored issuer thesis"
    app.mark_scenario_edited(state, "GOOGL", "bull")
    app.initialize_scenario_session_state(state, "GOOG", base)

    assert state[goog_keys["year_1_growth"]] == 27.0
    assert state[goog_keys["rationale"]] == "User-authored issuer thesis"
    assert state[goog_keys["status"]] == "user_edited"


def test_scenario_state_survives_switching_between_distinct_issuers():
    state = {}
    nvda_base = app.build_multistage_assumptions_from_ui(ui_values())
    app.initialize_scenario_session_state(state, "NVDA", nvda_base)
    nvda_keys = app.scenario_session_keys("NVDA", "bear")
    state[nvda_keys["year_1_growth"]] = 12.0
    app.mark_scenario_edited(state, "NVDA", "bear")

    msft_values = app.multistage_initial_defaults("MSFT", history(0.40))
    msft_base = app.build_multistage_assumptions_from_ui(msft_values)
    app.initialize_scenario_session_state(state, "MSFT", msft_base)
    msft_keys = app.scenario_session_keys("MSFT", "bear")
    state[msft_keys["year_1_growth"]] = 8.0
    app.mark_scenario_edited(state, "MSFT", "bear")
    app.initialize_scenario_session_state(state, "NVDA", nvda_base)

    assert state[nvda_keys["year_1_growth"]] == 12.0
    assert state[msft_keys["year_1_growth"]] == 8.0
    assert nvda_keys != msft_keys


def test_scenario_reset_changes_only_alternatives_and_preserves_rationales():
    state = {}
    base = app.build_multistage_assumptions_from_ui(ui_values())
    app.initialize_scenario_session_state(state, "NVDA", base)
    bear_keys = app.scenario_session_keys("NVDA", "bear")
    bull_keys = app.scenario_session_keys("NVDA", "bull")
    state[bear_keys["year_1_growth"]] = 1.0
    state[bull_keys["wacc"]] = 12.0
    state[bear_keys["rationale"]] = "Keep bear rationale"
    state[bull_keys["rationale"]] = "Keep bull rationale"
    main_wacc_keys = app.research_wacc_session_keys("NVDA")
    state[main_wacc_keys["value"]] = 9.7

    app.reset_scenario_session_state(state, "NVDA", base)

    assert state[bear_keys["year_1_growth"]] == pytest.approx(25.0)
    assert state[bull_keys["wacc"]] == pytest.approx(8.5)
    assert state[bear_keys["rationale"]] == "Keep bear rationale"
    assert state[bull_keys["rationale"]] == "Keep bull rationale"
    assert state[main_wacc_keys["value"]] == 9.7
    assert base.wacc == pytest.approx(0.09)


def test_scenario_comparison_reconciles_base_and_contains_no_forbidden_outputs():
    base = app.build_multistage_assumptions_from_ui(ui_values())
    standalone = run_multistage_dcf(company_inputs(), base)
    result = scenario_result(base=base)
    summary = app.build_scenario_summary_frame(result)
    economic = app.build_scenario_economic_frame(result)

    assert summary.loc["Intrinsic Value / Share", "Base"] == pytest.approx(
        standalone.per_share_value.intrinsic_value_per_share
    )
    assert summary.loc["Enterprise Value (B)", "Base"] == pytest.approx(
        standalone.enterprise_value.enterprise_value / 1e9
    )
    assert "Year 5 Revenue (B)" in economic.index
    assert "Terminal ROIC" in economic.index
    forbidden = " ".join([*summary.index, *economic.index]).lower()
    assert "probability" not in forbidden
    assert "expected value" not in forbidden
    assert "market price" not in forbidden
    assert not hasattr(result, "market_price")
