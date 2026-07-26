# Data availability check — Strategy #2+ families 7–26

Provider in this repo: **Massive.com** (ex-Polygon), not FMP. Probed with the
configured Massive key on 2026-07-26. Existing local artifacts: price panels,
`pit_history.pkl` (347 tickers, filing-date index), Fixed CS v2 / no-chase engine.

| # | Family | Status | Reason |
|---|---|---|---|
| 7 | Insider buying (Form 4) | **NOT VIABLE** | Form-4 / insider endpoints return HTTP 404 on this plan; no PIT insider feed in local infra. |
| 8 | Analyst estimate revision momentum | **NOT VIABLE** | Benzinga estimates 403; no analyst-estimate endpoints available. |
| 9 | Buyback *announcement* drift | **NOT VIABLE** | No buyback-announcement endpoint. CF statements lack a clean repurchase line item; Δshares is family 15, not announcement drift. |
| 10 | Ex-dividend seasonality | **VIABLE** | `/v3/reference/dividends` returns `ex_dividend_date` (+ declaration/record/pay). |
| 11 | Turn-of-month seasonality | **VIABLE** | Calendar overlay on existing daily price panel only. |
| 12 | Analyst price-target dispersion | **NOT VIABLE** | No price-target / estimate-dispersion endpoint on this plan. |
| 13 | Accruals anomaly | **DEGRADED** | PIT has `op_cf`, `total_assets`, `operating_income`; stored `net_income` is 0% populated (Massive field-name gap). Use `(operating_income − op_cf) / assets` as accruals proxy — not Sloan NI−CFO. |
| 14 | Asset growth | **VIABLE** | PIT `total_assets` by filing_date; YoY growth PIT-safe. |
| 15 | Net stock issuance | **VIABLE** | PIT `diluted_shares_outstanding` YoY change. |
| 16 | Short interest / days-to-cover | **DEGRADED** | `/stocks/v1/short-interest` works (`settlement_date`, `days_to_cover`). No publish timestamp — apply **+14 calendar-day lag** after settlement (FINRA bi-monthly release proxy) for PIT safety. |
| 17 | 52-week-high proximity | **VIABLE** | Trailing 252d high from price panel. |
| 18 | Piotroski F-Score | **DEGRADED** | Partial score from available PIT fields (ROA proxy via op_income/assets, accruals proxy, leverage, shares, margin, asset turnover). Not full 9-signal Piotroski; elevated corr risk vs S1 quality noted. |
| 19 | Distress (Altman Z proxy) | **DEGRADED** | Z-lite from PIT + price: WC≈0 (current assets/liab not in pit_history), use `(op_income, retained proxy via equity, sales, leverage, equity/assets)`. Flag as Z-proxy, not full Altman. |
| 20 | Neglected-firm (low coverage) | **NOT VIABLE** | No analyst-coverage count endpoint. |
| 21 | EPS-revision acceleration | **NOT VIABLE** | Same as #8 — no estimate time series. |
| 22 | Capex / investment anomaly | **VIABLE** | PIT `capex` / `total_assets`. |
| 23 | Net operating assets | **DEGRADED** | Proxy `NOA ≈ (total_assets − cash_eq) − (total_assets − total_debt − total_equity)` = `total_debt + total_equity − cash_eq` rearranged; use `(total_assets − cash_eq − (total_equity))` wait — standard proxy: `(total_assets − cash) − (total_liabilities − debt)`. With pit: `total_liab ≈ total_assets − total_equity`, so `NOA ≈ (A − cash) − (A − E − D) = E + D − cash`. PIT-safe proxy. |
| 24 | Institutional ownership (13F) | **NOT VIABLE** | 13F / institutional endpoints HTTP 404 on this plan. |
| 25 | Gross-margin trend | **DEGRADED** | No gross-profit in pit_history; use **op_margin** and **gross_profitability** trailing changes as margin-trend proxies (not literal gross-margin). |
| 26 | MAX / lottery-demand | **VIABLE** | Max daily return over trailing 21 sessions from price panel; long low-MAX. |

**Summary:** VIABLE=7, DEGRADED=6, NOT VIABLE=7. Only VIABLE+DEGRADED are implemented below.

**Skipped (not implemented):** 7, 8, 9, 12, 20, 21, 24.
