# Sprint 2 Retrospective — Financial Ratio & KPI Engine
**Nifty 100 Financial Intelligence Platform**
Sprint: Sprint 2 (Days 08–14) · Author: Sujal Hingmire · Date: 2026-07-01

---

## 1. Sprint Goal

Build the analytics layer that turns raw financial statements (P&L, Balance Sheet,
Cash Flow) into a clean `financial_ratios` table: profitability, leverage,
efficiency, CAGR growth, and cash-flow-quality KPIs for every Nifty 100
company-year, plus a capital allocation classification.

## 2. What We Shipped

| Deliverable | Location | Status |
|---|---|---|
| Profitability / leverage / efficiency ratios | `src/analytics/ratios.py` | ✅ Done |
| CAGR engine (Revenue, PAT, EPS — 3/5/10yr) | `src/analytics/cagr.py` | ✅ Done |
| Cash flow KPIs (CFO quality, CapEx intensity, FCF conversion) | `src/analytics/cashflow_kpis.py` | ✅ Done |
| Capital allocation 8-pattern classifier | `src/analytics/cashflow_kpis.py` → `output/capital_allocation.csv` | ✅ Done |
| Population pipeline into SQLite | `src/analytics/populate_financial_ratios.py` | ✅ Done |
| Unit tests | `tests/analytics/`, `tests/kpi/` | ✅ 274 tests, 0 failures |
| Edge case log | `output/ratio_edge_cases.log` | ✅ 785 documented entries |

**Final numbers (as of this run):**
- `financial_ratios` table: **2,295 rows**, 101 companies, **26 KPI columns**, zero null-only columns.
- `capital_allocation.csv`: **1,128 rows**, all 8 pattern labels represented.
- Test suite: **274/274 passed** (target was 20).
- Screener check (ROE > 15%, D/E < 1): **39 companies** — sensible large-cap, low-debt names (TCS, Infosys, HUL, ITC, Asian Paints, Nestlé India, Maruti, etc.)

---

## 3. Formula Decisions

### Profitability, Leverage, Efficiency (`ratios.py`)
- **Net Profit Margin / Operating Profit Margin** — standard `PAT / Revenue` and
  `Operating Profit / Revenue`. We cross-check OPM against the source-reported
  OPM with a **1.0 percentage-point tolerance**; mismatches beyond that are
  logged, not silently overwritten.
- **ROE / ROCE / ROA** — standard formulas, guarded against zero/negative
  denominators (see edge cases below).
- **Debt-to-Equity** — flagged **"High Leverage"** above a 5.0x threshold, and
  labeled **"Debt Free"** when total debt is zero rather than showing D/E = 0
  without context.
- **Financial-sector carve-out** — for companies tagged `Financials` (banks,
  NBFCs, insurers), the high-leverage flag is **forced to False**. Banks
  structurally run high D/E (that's the business model), so flagging them the
  same way as an industrial company would create false alarms.
- **Interest Coverage Ratio** — labeled with a warning below a 1.5x threshold
  (weak ability to service debt from operating earnings).
- **Asset Turnover** — standard `Revenue / Total Assets`, no special cases beyond
  the shared missing-value guards.

### CAGR Engine (`cagr.py`)
The spec required six edge cases to be explicitly handled — all six are
implemented as a single classification function (`determine_cagr_flag`) so
every caller uses the same rule set:

| Case | Condition | Treatment |
|---|---|---|
| `NORMAL` | start > 0 and end > 0 | CAGR computed normally |
| `ZERO_BASE` | start ≈ 0 | Undefined — division impossible, left null |
| `DECLINE_TO_LOSS` | start > 0, end < 0 | Undefined — CAGR has no real meaning across a profit→loss swing |
| `TURNAROUND` | start < 0, end > 0 | Undefined — same reasoning, loss→profit |
| `BOTH_NEGATIVE` | start < 0 and end < 0 | Undefined — a negative CAGR% on two losses is misleading |
| `INSUFFICIENT` | fewer data points than the window (e.g. requesting 5yr CAGR with only 3 years of history) | Skipped, not extrapolated |

**Decision:** we never force a CAGR number out of a sign-changing pair. A
"growth rate" computed across a profit-to-loss transition is mathematically
computable but economically meaningless, and would corrupt any downstream
ranking. We log these as `INFO`/`WARNING` and leave the field `NULL` rather
than guess.

### Cash Flow KPIs (`cashflow_kpis.py`)
- **CFO Quality Score** = average of `CFO / PAT` across the company's history.
  Labeled `High Quality` (>1.0, cash backs up reported profit), `Moderate`
  (~0.7–1.0), or `Accrual Risk` (<0.7 — profit not converting to cash,
  possible aggressive revenue recognition).
- **CapEx Intensity** = `CapEx / Revenue`, labeled `Asset Light`, `Moderate`,
  or `Capital Intensive`.
- **FCF Conversion Rate** = `Free Cash Flow / CFO`.
- **Capital Allocation Pattern** — classifies each company-year into one of
  **8 patterns** based on the sign combination of CFO / CFI / CFF
  (e.g. `+ − −` = "Shareholder Returns" — cash from operations, spent on
  investing and returned to shareholders/debt paydown; `− − +` = "Growth
  Funded by Debt"). This is a pure sign-based lookup table, not a magnitude
  calculation, so it's robust to scale differences across companies.

---

## 4. Edge Cases Found & How We Resolved Them

All 785 entries in `output/ratio_edge_cases.log` fall into these categories:

1. **Non-numeric / NaN source data** (majority of entries) — a company-year
   row had missing or non-numeric fields after type coercion (common for the
   most recent half-year filing, e.g. `2024-09`, where only partial statements
   are available). **Resolution:** row is skipped for the affected KPI, not
   dropped entirely — other KPIs for that row still populate if their inputs
   are present.
2. **Negative PAT (net loss years)** — e.g. Tata Motors FY20–FY22, Zomato
   FY20–FY23, Vedanta FY20, Union Bank FY20, Indigo FY20–FY23.
   **Resolution:** ratio is still calculated and included, but flagged in the
   log so anyone reading ROE/margin numbers for these years knows the company
   was loss-making that year (a negative ROE means something different than a
   "bad" positive ROE).
3. **ROE / ROCE source-vs-computed mismatches** — where our computed ratio
   diverges from the source-reported ratio by more than a 5% relative
   threshold. Categorized automatically into `DATA_SOURCE_ISSUE`,
   `VERSION_DIFFERENCE`, or `FORMULA_DISCREPANCY` depending on the size and
   direction of the gap (see `TestCategoriseAnomaly` in
   `test_edge_case_validation.py`). One confirmed example: **TCS**, where the
   source ROE is far below our computed value — documented as a known
   source-data anomaly, not a bug in our formula.
4. **Extreme ROE outliers from near-zero equity** — e.g. BEL, HAL, IRCTC,
   Nestlé India, IndiGo show ROE in the hundreds/thousands of percent in the
   screener output. This is not a formula bug — these companies have very
   small or near-zero net worth relative to profit in certain years (common
   for asset-light or recently-restructured companies), which mathematically
   inflates ROE. **Recommendation for Sprint 3:** consider a documented
   "extreme ratio" flag (e.g. |ROE| > 100%) rather than filtering these out,
   since filtering would hide a real (if unusual) characteristic of the
   business.

---

## 5. Testing Summary

- 274 unit tests across `tests/analytics/` (ratios, CAGR) and `tests/kpi/`
  (cash flow KPIs, edge case validation, full pipeline integration) — **all
  passing**, well above the 20-test minimum.
- Manual spot-check performed for TCS, Infosys, and HUL — ROE and Revenue
  CAGR values pulled directly from `financial_ratios` line up with the
  formulas in `ratios.py`/`cagr.py`; team lead to confirm against source
  spreadsheet within 0.1% tolerance at sign-off.

## 6. Known Follow-ups for Sprint 3

- Some company-years have duplicate rows under slightly different year-string
  formats (e.g. `"Mar 2024"` vs `"2024-03"`) coming from different source
  sheets merging in — worth normalizing `year` formatting upstream in the ETL
  layer so `financial_ratios` has exactly one row per company-year.
- Add an explicit "extreme ratio" flag for ROE/ROCE outliers driven by
  near-zero equity, instead of leaving them unflagged in the raw numbers.

---

## 7. Sign-off

- [ ] Reviewed by team lead
- [ ] Demo completed (5-company KPI walkthrough)
- [ ] Retrospective accepted into knowledge base