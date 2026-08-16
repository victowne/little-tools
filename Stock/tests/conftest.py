import pandas as pd
import pytest

from Stock import stock_valuation_mvp as app


@pytest.fixture
def statement_factory():
    """Build a Yahoo-style statement: line items in rows, periods in columns."""
    def build(rows: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame(rows, index=pd.to_datetime(dates)).T

    return build


@pytest.fixture
def snapshot_factory():
    """Build a deterministic snapshot without calling yfinance."""
    def build(**overrides) -> app.CompanySnapshot:
        empty = pd.DataFrame()
        values = {
            "ticker": "TEST",
            "price": 100.0,
            "market_cap": 1_000_000_000_000.0,
            "shares_outstanding": 10_000_000_000.0,
            "cash": 10_000_000_000.0,
            "total_debt": 20_000_000_000.0,
            "net_debt": 10_000_000_000.0,
            "sector": "Technology",
            "industry": "Software - Application",
            "beta": 1.0,
            "annual_income": empty,
            "quarterly_income": empty,
            "annual_balance": empty,
            "quarterly_balance": empty,
            "annual_cashflow": empty,
            "quarterly_cashflow": empty,
            "financial_currency": "USD",
            "price_currency": "USD",
        }
        values.update(overrides)
        return app.CompanySnapshot(**values)

    return build
