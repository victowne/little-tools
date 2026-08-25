# Stock Valuation Research Workstation

A transparent, fundamentals-first equity research application for studying historical financial performance, building a unified multi-stage DCF, testing valuation sensitivity, and comparing research assumptions with market-implied expectations through Reverse DCF.

The workstation is designed to help answer four questions:

1. How are a company's revenue, margins, cash generation, and capital efficiency changing?
2. What intrinsic value is supported by an explicit set of operating assumptions?
3. How dependent is that value on WACC, terminal growth, mature margins, and capital efficiency?
4. What growth, margin, capital-efficiency, or discount-rate assumption is implied by the current market price?

The project favors transparency, conservative missing-data handling, and one comparable production valuation architecture across companies. It does not issue `BUY`, `HOLD`, or `SELL` ratings, and it does not tune research assumptions to match the market price.

> **Important disclaimer**
>
> This project is a research tool, not investment advice. Financial data may be incomplete, delayed, revised, or incorrectly mapped by a third-party provider. Company Profile assumptions involve judgment, and DCF outputs can be highly sensitive to long-duration assumptions. Always verify material inputs against original company filings and investor-relations materials before relying on a result.

## Contents

1. [What the application can do](#1-what-the-application-can-do)
2. [Quick start](#2-quick-start)
3. [Suggested research workflow](#3-suggested-research-workflow)
4. [User interface guide](#4-user-interface-guide)
5. [The multi-stage DCF model](#5-the-multi-stage-dcf-model)
6. [The Reverse DCF model](#6-the-reverse-dcf-model)
7. [How Company Profile assumptions are developed](#7-how-company-profile-assumptions-are-developed)
8. [Evidence used to challenge assumptions](#8-evidence-used-to-challenge-assumptions)
9. [Data sources and refresh behavior](#9-data-sources-and-refresh-behavior)
10. [Data-integrity rules](#10-data-integrity-rules)
11. [Model limitations](#11-model-limitations)
12. [Frequently asked questions](#12-frequently-asked-questions)
13. [Testing and project structure](#13-testing-and-project-structure)
14. [License](#14-license)

## 1. What the application can do

### 1.1 Explore historical fundamentals

The application displays available annual and quarterly trends for:

- Revenue
- Gross Profit and Gross Margin
- Operating Income and Operating Margin
- Net Income
- Operating Cash Flow
- Free Cash Flow and FCF Margin
- Retained Earnings
- Shares Outstanding

The Key Fundamentals panel also exposes:

- Latest annual and validated TTM values
- Revenue Growth and 3-year Revenue CAGR
- 3-year normalized Sales-to-Capital and latest annual Sales-to-Capital
- NOPAT
- Accounting ROIC
- Simplified Net Investment
- Simplified Reinvestment Rate
- Fundamental Growth Capacity

Historical metrics are evidence. They do not automatically populate or overwrite forward DCF assumptions.

### 1.2 Run a unified multi-stage DCF

Every production Company Profile uses the same architecture:

```text
Researched Y1 / Y2 / Y3 Revenue Growth
                    ↓
          Deterministic linear fade
                    ↓
          Mature Operating Margin
                    ↓
          Mature Sales-to-Capital
                    ↓
            Standard reinvestment
                    ↓
FCFF → Enterprise Value → Equity Value → Value per Share
```

Differences between companies are expressed through assumptions, evidence, confidence, and model-risk disclosures—not through ticker-specific valuation engines.

### 1.3 Inspect sensitivity and scenario risk

The application includes:

- WACC × Terminal Growth sensitivity
- Bear / Base / Bull operating-path comparison
- Explicit FCFF and present-value diagnostics
- Enterprise Value and Equity Value bridges
- Terminal Value / Enterprise Value
- Terminal ROIC and Terminal Reinvestment Rate
- Annual Revenue, Margin, Sales-to-Capital, NOPAT, Reinvestment, and FCFF paths

The default Bear and Bull cases are transparent mechanical starting points. They are not probability-weighted forecasts, researched ratings, or recommendations.

### 1.4 Run a Reverse DCF

Reverse DCF changes the question from:

> What is the company worth under our assumptions?

to:

> Holding everything else constant, what value must one assumption reach for the DCF to equal the current market price?

The current implementation independently solves for:

- A common uplift applied to Y1/Y2/Y3 Revenue Growth
- Mature Operating Margin
- Mature Sales-to-Capital
- WACC

Each candidate point reruns the full production DCF. Reverse DCF results are market-implied expectation diagnostics, not forecasts.

### 1.5 Use researched Company Profiles

The current release includes nine researched profiles:

| Ticker | Company |
|---|---|
| NVDA | NVIDIA |
| GOOGL / GOOG | Alphabet |
| META | Meta Platforms |
| MSFT | Microsoft |
| AMZN | Amazon |
| MU | Micron Technology |
| AAPL | Apple |
| AVGO | Broadcom |
| AMD | Advanced Micro Devices |

Other securities covered by Yahoo Finance can still expose historical data and a manual DCF workspace, but they do not receive a project-researched Company Profile automatically.

## 2. Quick start

### 2.1 Requirements

- Python 3.10 or later is recommended.
- Internet access is required for live financial, market, and macro data.
- Run all commands from the repository root—the directory that contains `Stock`.

### 2.2 Install

On PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2.3 Start the application

From the repository root:

```powershell
python -m streamlit run Stock/stock_valuation_mvp.py
```

Then open:

```text
http://127.0.0.1:8501/
```

The shorter command also works when the correct environment is active:

```powershell
streamlit run Stock/stock_valuation_mvp.py
```

If you see `ModuleNotFoundError: No module named 'Stock'`, the command was probably launched from inside the `Stock` directory or another working directory. Return to the repository root and run the command again.

## 3. Suggested research workflow

1. Enter a ticker such as `NVDA`, `GOOGL`, or `AMD` in the sidebar.
2. Start with Key Fundamentals. Verify the TTM periods, Revenue base, margins, FCF, and historical capital efficiency.
3. Read the Research Profile evidence, assumption rationale, confidence, and model-risk disclosures.
4. Inspect the complete Research Base DCF path—not only the per-share output.
5. Review explicit FCFF, Terminal Value / EV, Terminal ROIC, and the equity bridge.
6. Use WACC × Terminal Growth sensitivity and Bear/Base/Bull cases to identify fragile assumptions.
7. Use Reverse DCF to compare the Research Base with expectations embedded in the market price.
8. Finish with Evidence, Model Limitations, and the operating-health checks.

A valuation number without its operating path, evidence, and sensitivity context is not the intended output of this project.

## 4. User interface guide

### 4.1 Company summary

The page header presents the selected issuer, market price, Research Base DCF, DCF/Market Price ratio, profile state, Base source, and model-risk descriptor.

The valuation gap is a neutral research diagnostic. It is not expected return, upside/downside guidance, or a recommendation.

### 4.2 Research Profile

The profile summarizes the company's business model, research assumptions, evidence references, confidence levels, and known abstraction risks.

Major assumptions include:

- Starting Revenue
- Y1/Y2/Y3 Revenue Growth
- Revenue fade and forecast horizon
- Starting and mature Operating Margin
- Starting and mature Sales-to-Capital
- Operating Tax Rate
- Research WACC
- Terminal Growth

### 4.3 Review & Apply workflow

Company Profiles follow an explicit state transition:

```text
Research Candidate
        ↓ explicit user action: Review & Apply
Reviewed Snapshot
        ↓
Current Base
```

- **Research Candidate** is read-only research output. It does not silently change the Base.
- **Reviewed Snapshot** is the immutable profile captured at the moment of acceptance.
- **Current Base** is the assumption set used by the primary valuation, sensitivity, and Reverse DCF views.
- If the Current Base is edited later, the UI identifies that it has diverged from the applied snapshot.

Review and Apply state currently lives in the Streamlit session. It is not persisted to a research database. Restarting the server or clearing the session may require the profile to be reviewed again.

### 4.4 Key Fundamentals

This section presents latest annual data, validated TTM metrics, annual quality trends, and historical DCF anchors. Monetary values are generally normalized to billions in the financial-statement currency.

### 4.5 Financial Statement Trends

This section is collapsed by default. It can display annual or quarterly Revenue, profits, margins, cash flow, Retained Earnings, and Shares Outstanding.

### 4.6 Research Base DCF

This section contains the assumption path, annual operating forecast, explicit FCFF present value, terminal value, Enterprise Value, net-debt bridge, Equity Value, and per-share intrinsic value.

The Manual Base Workspace remains editable, but a separate Research Candidate is never presented as if the user had already accepted it.

### 4.7 Sensitivity and scenario diagnostics

- **WACC × Terminal Growth** reruns the full valuation chain at every grid point.
- **Bear / Base / Bull** compares coherent operating paths rather than changing only the terminal formula.
- Forecast diagnostics expose where growth, margins, capital efficiency, and cash conversion may become economically implausible.

### 4.8 Reverse DCF

The Reverse DCF panel solves one assumption at a time against the current market price. It does not write implied values back to a Company Profile or Current Base.

### 4.9 Evidence, limitations, and operating health

The final sections separate disclosure from research judgment, describe company-specific limitations of the unified model, and show quick operating checks covering:

- Total Assets versus Total Liabilities
- Long-term Debt / Net Income
- Operating Cash Flow coverage of investing and financing cash flows

These checks are screening evidence, not conclusions about business quality.

## 5. The multi-stage DCF model

### 5.1 Core inputs

The production model consumes:

- Starting Revenue
- Y1/Y2/Y3 Revenue Growth
- Revenue Fade Years
- Forecast Years
- Starting Operating Margin
- Mature Operating Margin
- Starting Sales-to-Capital
- Mature Sales-to-Capital
- Operating Tax Rate
- Research WACC
- Terminal Growth
- Net Debt
- Current Common Shares Outstanding

Starting Revenue prefers a validated four-quarter TTM. If a valid TTM is unavailable, only an explicitly identified latest-annual fallback is accepted. Missing net debt or share count is not replaced with zero to manufacture a per-share value.

### 5.2 Revenue growth path

Y1/Y2/Y3 use explicit researched growth rates. After the final explicit year, growth fades linearly toward Terminal Growth:

```text
Growth(fade, k)
    = Last Explicit Growth
      + (Terminal Growth - Last Explicit Growth) × k / Fade Years
```

where `k = 1 ... Fade Years`. The final fade year therefore equals Terminal Growth without repeating the last explicit rate.

Current Company Profiles generally use:

```text
3 researched years + 8 deterministic fade years = 11 explicit forecast years
```

Y4 and Y5 are generated by the deterministic fade. They are not hidden analyst estimates or additional researched forecast fields.

### 5.3 Operating Margin and Sales-to-Capital paths

Operating Margin and Sales-to-Capital move linearly from their starting values in Year 1 to their mature values at the end of the economic transition horizon:

```text
X(t)
    = X(start)
      + [X(mature) - X(start)] × (t - 1) / (Transition Years - 1)
```

`X` represents either Operating Margin or Sales-to-Capital. Values are not rounded inside the engine. A Sales-to-Capital path that touches or crosses zero is rejected because the reinvestment equation would become undefined or misleading.

### 5.4 Operating forecast and FCFF

For each forecast year `t`:

```text
Revenue(t)
    = Revenue(t-1) × [1 + Growth(t)]

Operating Income(t)
    = Revenue(t) × Operating Margin(t)

NOPAT(t)
    = Operating Income(t) × [1 - Operating Tax Rate]

Reinvestment(t)
    = Change in Revenue(t) / Sales-to-Capital(t)

FCFF(t)
    = NOPAT(t) - Reinvestment(t)
```

Sales-to-Capital measures how much incremental Revenue is supported by one unit of incremental invested capital:

- A higher S/C requires less reinvestment for a given growth rate.
- A lower S/C requires more reinvestment for the same growth rate.
- High Revenue growth does not automatically create value. Growth can destroy value when incremental returns do not cover the cost of capital.

The production DCF does not separately forecast CapEx, D&A, working capital, or acquisitions. It represents growth investment through `Change in Revenue / Sales-to-Capital`. This improves comparability, but it may abstract poorly when infrastructure spending and later Revenue realization are separated by a long lead time.

### 5.5 Explicit-period discounting

The model uses year-end discounting:

```text
Discount Factor(t) = 1 / (1 + WACC)^t

PV of FCFF(t) = FCFF(t) × Discount Factor(t)
```

The current engine does not use a mid-year convention.

### 5.6 Terminal economics

Terminal FCFF is not assumed to equal Terminal NOPAT. The model explicitly recognizes the reinvestment required to sustain terminal growth:

```text
Terminal ROIC
    = Mature Operating Margin
      × (1 - Operating Tax Rate)
      × Mature Sales-to-Capital

Terminal Reinvestment Rate
    = Terminal Growth / Terminal ROIC

Terminal FCFF
    = Terminal NOPAT × (1 - Terminal Reinvestment Rate)

Terminal Value at Year N
    = Terminal FCFF in Year N+1 / (WACC - Terminal Growth)
```

The model requires `WACC > Terminal Growth`.

Terminal diagnostics should be read together:

- A very high Terminal ROIC may indicate an aggressive mature margin or S/C assumption.
- A Terminal Reinvestment Rate above 100% signals internally strained mature economics.
- A high Terminal Value / EV indicates that most estimated value depends on distant assumptions.

### 5.7 Enterprise Value, Equity Value, and per-share value

```text
Enterprise Value
    = Sum of PV Explicit FCFF + PV Terminal Value

Net Debt = Total Debt - Cash

Equity Value = Enterprise Value - Net Debt

Intrinsic Value per Share
    = Equity Value / Current Common Shares Outstanding
```

Negative Net Debt represents net cash and increases Equity Value. The denominator is the current consolidated common share count, not the historical weighted-average EPS denominator. The current model does not forecast future dilution, buybacks, or a forward share-count path.

## 6. The Reverse DCF model

### 6.1 Forward versus reverse valuation

A normal DCF runs in this direction:

```text
Operating assumptions → Intrinsic value
```

Reverse DCF runs in the opposite direction:

```text
Current market price → Implied value of one assumption
```

It helps determine whether the market price appears to require stronger growth, higher mature margins, better capital efficiency, or a lower discount rate than the Research Base.

### 6.2 Variables solved by the current engine

Each calculation changes exactly one lever:

1. **Near-Term Growth Uplift** adds the same percentage-point adjustment to Y1, Y2, and Y3.
2. **Mature Operating Margin** changes only the mature margin.
3. **Mature Sales-to-Capital** changes only mature capital efficiency.
4. **WACC** changes only the discount rate.

For every trial value, the solver reruns:

```text
Forecast Path
→ Operating Forecast
→ Explicit FCFF Discounting
→ Terminal Value
→ Enterprise Value
→ Equity Value
→ Per-Share Value
```

Reverse DCF is not implemented as a terminal-value-only adjustment.

### 6.3 Example interpretation

Suppose the Research Base growth path is:

```text
Y1 20% / Y2 15% / Y3 10%
```

If Reverse DCF solves a `+8 percentage point` growth uplift, the implied path is:

```text
Y1 28% / Y2 23% / Y3 18%
```

This does not prove that the market expects those exact growth rates. It means that, if every other Base assumption stays unchanged, near-term growth alone would need that uplift to reconcile the DCF with the market price.

The four Reverse DCF outputs are independent one-at-a-time diagnostics. They are not joint conditions that must all hold simultaneously. In reality, price may reflect a combination of growth, margins, S/C, WACC, and business optionality that the unified model does not capture.

### 6.4 Solver safeguards

The solver scans a transparent bounded range, checks monotonicity, and only uses root finding when it identifies one valid solution interval. Possible unavailable states include:

- `OUTSIDE_REASONABLE_RANGE`
- `NO_BRACKET`
- `NON_MONOTONIC`
- `AMBIGUOUS`
- `VALUATION_FAILED`

These states are preferable to presenting an apparently precise but unreliable implied assumption.

## 7. How Company Profile assumptions are developed

### 7.1 Research principle

A Company Profile stores a traceable set of research judgments. Its purpose is not to manufacture a correct target price. Market price is introduced only after assumptions have been selected and is excluded from Profile construction.

The evidence hierarchy is generally:

1. **Tier 1:** SEC filings, company investor-relations materials, earnings releases, and formal management guidance.
2. **Tier 2:** High-quality industry, regulatory, and comparable-company evidence.
3. **Tier 3:** Analyst consensus or third-party research when direct evidence is unavailable.

The application attempts to distinguish:

- **Disclosure:** directly reported by a company or regulator.
- **Derived Metric:** calculated from reported information.
- **Research Assumption:** a forward-looking parameter selected by the researcher.

### 7.2 Revenue base

- A validated TTM Revenue constructed from four distinct consecutive fiscal quarters is preferred.
- The exact reporting periods are retained and displayed.
- A missing or NaN quarter is not removed and replaced with an older quarter.
- Some dedicated company research may use an explicitly validated SEC TTM bridge with preserved source periods.

### 7.3 Y1/Y2/Y3 Revenue Growth

Near-term growth research may consider:

- Recent annual and quarterly growth
- Management guidance
- Backlog, contracted demand, capacity, and product cycles
- Segment and product mix
- Available fiscal-year Revenue consensus
- Company scale, competition, cyclicality, and execution risk

Consensus is supporting evidence, not an automatic assumption. The DCF typically begins from a rolling TTM base, while consensus estimates normally represent company fiscal-year Revenue endpoints. When periods differ, the application does not pretend that they are directly comparable one-year growth rates.

### 7.4 Mature Operating Margin

Mature Margin is not a mechanical copy of the latest quarter or a management non-GAAP target. Research may consider:

- Long-run business mix
- Scale economics and pricing power
- Cycle normalization
- Competition and customer concentration
- GAAP versus non-GAAP differences
- The economics of asset-light and capital-intensive activities

### 7.5 Mature Sales-to-Capital

Mature S/C is one of the most consequential and easily overlooked assumptions in the model. Research may use:

- Historical 3-year normalized S/C
- Latest annual S/C
- CapEx, D&A, PP&E, and working-capital context
- Outsourced manufacturing or asset-light operating structures
- Hardware and software mix
- Acquisition, goodwill, buyback, and accounting-capital distortions

The selected mature S/C is an economic research assumption and may differ materially from historical accounting S/C. Companies such as Amazon, Micron, Broadcom, and Apple exhibit different capital-measurement distortions. The production DCF still applies one unified S/C formula and exposes the uncertainty through evidence and model-risk disclosures.

### 7.6 Operating Tax Rate

The research rate considers historical effective taxes and business structure while treating losses, tax benefits, and unusual provisions carefully. The historical fundamentals engine does not force a default tax rate merely to display ROIC. The Company Profile's future Operating Tax Rate is a separate explicit assumption.

### 7.7 Research WACC

The application deliberately separates two concepts:

- **Formula-Based WACC** is a mechanical, auditable result based on current market and financial inputs.
- **Research WACC** is the long-horizon discount-rate assumption selected for the Company Profile.

Research WACC may differ from Formula-Based WACC, but the difference should be explained through beta robustness, industry risk, capital structure, and long-duration business risk. It should not be changed simply to make DCF equal market price.

### 7.8 Terminal Growth and mature economics

Terminal Growth represents mature nominal growth and should generally be far below the company's high-growth phase. It must form a coherent set of mature economics with margin, S/C, tax, Terminal ROIC, and Terminal Reinvestment Rate.

### 7.9 Confidence and model risk

Profiles assign evidence confidence—typically High, Medium, or Low—to major assumptions. Confidence describes evidence strength, not investment quality.

A separate model-risk descriptor communicates how much economic complexity is compressed by the unified DCF. It is not a security rating.

## 8. Evidence used to challenge assumptions

### 8.1 Revenue Growth

Useful evidence includes:

- Latest quarterly and annual Revenue Growth
- 3-year Revenue CAGR
- Segment growth and business mix
- Management guidance, backlog, capacity, and product cycles
- Fiscal-year Revenue consensus
- Explicit alignment between DCF years and consensus periods

Historical growth is an anchor, not a forecast. High growth should also be tested against reinvestment capacity, capital efficiency, competition, and addressable-market constraints.

### 8.2 Operating Margin

Review:

- Gross Margin trends
- Annual and quarterly Operating Margin
- GAAP/non-GAAP reconciliation
- Segment profit pools
- R&D, stock-based compensation, operating leverage, and cycle position

If valuation depends heavily on Mature Margin, use margin sensitivity and avoid relying on one point estimate without a supporting range.

### 8.3 Sales-to-Capital and reinvestment

Historical accounting anchors use:

```text
Historical Sales-to-Capital
    = Change in Revenue / Change in Invested Capital

Accounting Invested Capital
    = Equity + Debt - Cash
```

The UI can provide:

- Latest annual S/C
- 3-year normalized S/C
- Simplified Net Investment = CapEx cash outlay - D&A
- Simplified Reinvestment Rate = Simplified Net Investment / NOPAT
- Revenue and invested-capital changes over matching periods

These metrics may be distorted by acquisitions, goodwill, full deduction of accounting cash, uncapitalized R&D, buybacks, negative equity, and working-capital timing. They should not be copied mechanically into mature assumptions.

### 8.4 ROIC

Historical Accounting ROIC is calculated as:

```text
NOPAT
    = Operating Income × (1 - Effective Operating Tax Rate)

Invested Capital
    = Equity + Debt - Cash

ROIC
    = NOPAT / Average Invested Capital
```

The valuation model's Terminal ROIC is:

```text
Terminal ROIC
    = Mature Operating Margin
      × (1 - Operating Tax Rate)
      × Mature Sales-to-Capital
```

They serve different purposes:

- Accounting ROIC is a historical accounting diagnostic.
- Terminal ROIC is the structural return implied by forward mature assumptions.

If Terminal ROIC is materially above historical ROIC, the profile should explain why future margin, capital efficiency, or business mix will improve. Conversely, Accounting ROIC may appear unusually high for asset-light technology companies because cash is fully deducted, R&D is not capitalized, and accounting invested capital can be very small.

### 8.5 WACC

The Formula-Based WACC framework is:

```text
Cost of Equity
    = Risk-free Rate + Beta × Equity Risk Premium

Pre-tax Cost of Debt
    = Risk-free Rate + Synthetic Default Spread

After-tax Cost of Debt
    = Pre-tax Cost of Debt × (1 - Tax Rate)

WACC
    = Equity Weight × Cost of Equity
      + Debt Weight × After-tax Cost of Debt
```

The audit can include:

- U.S. 10-year Treasury yield
- Damodaran mature-market Equity Risk Premium
- Five-year monthly regression beta versus the S&P 500
- Yahoo metadata beta fallback
- Industry and bottom-up beta references
- EBIT / Interest Expense coverage
- Damodaran synthetic spread and rating
- Market Capitalization, Total Debt, and capital weights
- Mechanical beta, ERP, and risk-free-rate sensitivities

Users should compare Formula WACC, Research WACC, implied beta, and the valuation effect of at least ±50 basis points.

### 8.6 Terminal-value dependency

At minimum, inspect:

- The `WACC - Terminal Growth` spread
- Terminal ROIC
- Terminal Reinvestment Rate
- Terminal FCFF / NOPAT
- Terminal Value / Enterprise Value
- WACC × Terminal Growth sensitivity

A high terminal-value share does not automatically invalidate a valuation, but it increases the importance of mature-economics evidence.

## 9. Data sources and refresh behavior

| Data | Primary source | Use |
|---|---|---|
| Annual and quarterly statements | yfinance / Yahoo Finance | Historical fundamentals, TTM, balance sheet, and cash flow |
| Price, Market Cap, Beta, and Shares | yfinance | Market diagnostics, WACC, and per-share value |
| Revenue consensus reference | Yahoo analyst estimates | Near-term anchors; never auto-applied |
| U.S. 10-year Treasury | U.S. Treasury; `^TNX` fallback | Risk-free Rate |
| ERP, industry WACC, and synthetic spreads | Damodaran datasets | WACC evidence and cross-checks |
| Company research evidence | SEC filings, company IR, earnings releases, and guidance | Company Profile research |
| Alpha Vantage adapter | Optional provider audit | Not required by the production UI |

During one Streamlit page run, annual statements, quarterly statements, and market metadata are reused through a centralized company snapshot. Streamlit caching reduces repeated requests. Live values may change after a cache refresh, provider revision, or application restart.

The application does not guarantee the completeness, timing, or accuracy of third-party data.

## 10. Data-integrity rules

The project follows a conservative principle: **missing is preferable to confidently wrong**.

- A genuinely reported `0.0` remains zero.
- Missing fields, NaN values, and failed lookups remain unavailable instead of becoming zero.
- Financial concepts use canonical names and concept-specific aliases, not generic fuzzy matching.
- TTM requires four distinct consecutive fiscal quarters with usable values.
- A missing recent quarter cannot be replaced by an older quarter to manufacture TTM.
- Missing Debt does not mean zero Debt.
- Missing Shares does not mean zero Shares.
- Missing Price does not produce a fake market comparison.
- A health check with insufficient inputs is unavailable, not PASS.

As a result, `N/A` can be an intentional conservative outcome. For example, a negative Tax Provision or an economically unreasonable effective tax rate may make NOPAT and Accounting ROIC unavailable even when Revenue, Margin, and FCF are present.

## 11. Model limitations

The production model intentionally remains unified and explainable. It does not capture every accounting or business-model detail:

- No segment DCF or sum-of-the-parts valuation
- No ticker-specific production engine for Amazon, Micron, or other companies
- No separate CapEx, D&A, working-capital, or acquisition forecast
- No future dilution, buyback, or share-count path
- No separate stock-based compensation adjustment
- No probability-weighted price target or expected-return model
- No automatic conversion for every ADR, foreign statement currency, or trading currency
- No guarantee of complete Yahoo statements or analyst-estimate coverage
- No persistent database for Profile reviews or user research notes
- No investment recommendation

One consistent architecture improves comparability and auditability. The tradeoff is reduced precision for heterogeneous businesses, capital-spending lead/lag effects, extreme cycles, and acquisition accounting.

Company-specific research artifacts may document these limitations without changing the production engine.

## 12. Frequently asked questions

### Why can DCF be far below the market price?

The Research Base may imply lower growth, lower mature margins, lower S/C, or a higher WACC than the market. The unified model may also omit business optionality or represent investment timing imperfectly. Review Reverse DCF, WACC/terminal sensitivity, explicit FCFF, and Terminal Value / EV before changing assumptions.

Do not tune the Base merely to reproduce market price.

### Why can faster growth reduce FCFF?

Growth requires reinvestment:

```text
Reinvestment = Change in Revenue / Sales-to-Capital
```

When S/C is low, high growth consumes substantial capital. Growth creates value only when the return on incremental investment adequately exceeds the cost of capital.

### Why is Accounting ROIC extremely high or unavailable for some companies?

Asset-light companies can report very high ROIC when accounting invested capital is small. Negative tax provisions, non-positive pretax income, non-positive invested capital, and missing fields can make ROIC unavailable. An unavailable value is not zero, and an unusually high historical ROIC should not be copied directly into Terminal ROIC.

### Should I use Formula-Based WACC or Research WACC?

Formula-Based WACC is an auditable current-data reference. Research WACC is the long-duration valuation assumption. They can differ, but the difference should have an explicit risk rationale. Sensitivity analysis should cover the plausible range between them.

### Is consensus Revenue a forward rolling twelve-month estimate?

Usually not. Yahoo generally provides company fiscal-year Revenue estimates. DCF Year 1 begins after the current validated TTM period. The UI exposes period alignment and does not force a fiscal endpoint into a mismatched rolling-year comparison.

### Must all Reverse DCF implied values hold at the same time?

No. They are independent one-variable diagnostics. Each result holds all other Base assumptions constant and is not a joint market forecast.

### Why does a Company Profile not update the Base automatically?

Research evidence and accepted assumptions are intentionally separated. The user must explicitly Review & Apply a Candidate before it becomes Current Base. This prevents a live data refresh or revised research module from silently changing the primary valuation.

## 13. Testing and project structure

### 13.1 Deterministic tests

Run the full suite from the repository root:

```powershell
python -m pytest
```

The main test suite uses fixed fixtures and does not require live Yahoo Finance data.

### 13.2 Syntax check

```powershell
python -m compileall Stock
```

Live-data checks are separate smoke validations and may be affected by network access, provider throttling, or data refreshes.

### 13.3 Project structure

```text
Stock/
├── stock_valuation_mvp.py       Streamlit UI and data-acquisition integration
├── fundamentals.py              Pure historical fundamental calculations
├── valuation.py                 Pure multi-stage DCF layers
├── multistage_integration.py    Real-company input adapter and orchestration
├── valuation_sensitivity.py     WACC × Terminal Growth sensitivity
├── valuation_scenarios.py       Bear / Base / Bull scenarios
├── reverse_dcf.py               Single-variable Reverse DCF
├── company_profiles.py          Company Profile structures and workflow
├── *_research.py                Company-specific evidence and Candidate builders
├── wacc_audit.py                Formula WACC audit and diagnostics
└── tests/                       Deterministic regression and Streamlit AppTests
```

The core fundamentals and valuation layers do not depend on Streamlit or network access. The UI presents their results and should not duplicate financial or valuation formulas.

## 14. License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the complete terms.

GPL-3.0 allows users to run, study, share, and modify the software, subject to its copyleft and source-distribution requirements. This README summary is not a substitute for the license text.

---

**Release:** `v1.0-stock-valuation` — the first formal release of the Stock Valuation Research Workstation.
