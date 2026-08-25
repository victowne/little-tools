# little-tools
custom tools

## Stock valuation research workstation

The `Stock` application combines historical fundamentals with one unified
multi-stage DCF architecture: researched Y1/Y2/Y3 revenue growth, deterministic
fade, mature operating margin, mature Sales-to-Capital, standard reinvestment,
and a terminal-value framework.

Nine researched Company Profiles (NVDA, GOOGL, META, MSFT, AMZN, MU, AAPL,
AVGO, and AMD) currently support an explicit Review & Apply
workflow. The workstation also provides Bear/Base/Bull diagnostics,
WACC/terminal-growth sensitivity, and single-variable Reverse DCF market-implied
expectations. Market price is diagnostic only: the application does not produce
BUY/HOLD/SELL labels or automatically calibrate research assumptions to price.

Run from the repository root:

```powershell
streamlit run Stock/stock_valuation_mvp.py
```
