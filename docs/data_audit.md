# Data audit (phase 1)

Answers to audit questions A1–A4 in `AGENTS.md` section 4, from real calls. This file holds
evidence only (row counts, date ranges, column names, item codes, aggregate ratios). No raw
market data is reproduced here.

| | |
| :-- | :-- |
| Run | 2026-11-13, 20:00–21:00 Vietnam time |
| Library | `vnstock` 6.0.0, `vnstock-oapi` 0.1.1, Python 3.12.3 |
| Sources | **`fhsc` only.** No `ssi` key on this machine, so every `ssi` call is untested |
| Symbols | FPT, HPG, HSG (corporate template), VCB (bank), SSI (securities), BVH (insurance) |
| Script | `python scripts/data_audit.py --sources fhsc` (sequential, 1 s apart; 62 calls, one `NetworkError` retried once) |
| Evidence | `data/audit/audit_summary.json` (git-ignored, aggregates only) |
| End date | `2026-10-30`, the last session in `configs/base.yaml` |

## Decisions (approved by the owner on 2026-11-13, frozen in `configs/base.yaml`)

| Decision | Value | Based on |
| :-- | :-- | :-- |
| Sources | `fhsc` only for version 1 | no `ssi` key |
| Price data start | `2017-01-03`, the first bar `fhsc` returns | A1 |
| First rebalance date | `2018-01-31`, the first month-end with 252 sessions of history (it has 270; `2017-12-29` has 250) | A1 |
| Last rebalance date | `2026-08-31`, the last one whose forward return ends by `2026-10-30` | A1 |
| Research period | 2018-01 to 2024-01 (73 rebalance months) | 70% of 104 months |
| Holdout period | 2024-02 to 2026-08 (31 rebalance months) | 30% of 104 months |
| Market cap | Option 2: adjusted close × current `outstanding_shares` | A2, A3 |
| Traded value (ADV, universe) | Anchored proxy, see A2 option (e) | A2 |
| Equity for `bp`, `roe` | Total `BS_EQUITY`; `roe` uses total net profit, see A3 option (a) | A3 |

Open: the survivorship illustration needs a replacement for "current VN100", which `fhsc`
does not have (see follow-up checks). Owner decision needed before phase 7.

With 73 research months, ML walk-forward (36 training months, retrain every 12, one-month label
gap) has 36 out-of-sample research months (2021-02 to 2024-01). The holdout has 31 months, so
holdout statistics will have low power.

## A1. How deep do daily bars go?

| Symbol | Rows | First bar | Last bar | Years | Zero-volume days | Rows before first bar |
| :-- | --: | :-- | :-- | --: | --: | --: |
| FPT | 2451 | 2017-01-03 | 2026-10-30 | 9.82 | 0 | 0 |
| HPG | 2451 | 2017-01-03 | 2026-10-30 | 9.82 | 0 | 0 |
| VCB | 2451 | 2017-01-03 | 2026-10-30 | 9.82 | 0 | 0 |
| SSI | 2451 | 2017-01-03 | 2026-10-30 | 9.82 | 0 | 0 |
| BVH | 2451 | 2017-01-03 | 2026-10-30 | 9.82 | 0 | 0 |
| HSG | 2451 | 2017-01-03 | 2026-10-30 | 9.82 | 0 | 0 |

- One call (`start=2000-01-01`) returns the full range: `attrs["request_count"] == 1`, no
  `truncated` flag, no duplicate days, `last_bar_partial` is False.
- All six symbols start on the same day although FPT and VCB were listed long before 2017, and a
  second call for the range before 2017-01-03 returns 0 rows. **`fhsc` keeps daily history from
  2017-01-03 only.** `ssi` depth is unknown (no key).
- Columns with `compat="v6"`: `time, open, high, low, close, volume`. `time` is timezone-aware
  (Asia/Ho_Chi_Minh); `columns="all"` adds no extra field.
- 9.8 years is above the 5-year stop threshold, so the project continues.

## A2. Unadjusted close and historical traded value

- **Unadjusted close: not available.** `fhsc` `ohlcv` prices are adjusted for corporate actions.
- **`ohlcv` volume is unadjusted:** it equals `volume_history.matched_volume` (ratio 1.0 at
  the 5th, 50th and 95th percentiles for all six symbols; lowest single-day ratio 0.88, HSG).
- **Traded value:** `Market("fhsc").equity(sym).volume_history()` returns `time,
  matched_volume, matched_value, block_volume, block_value, total_volume, total_value`, with
  values in VND. The source rejects ranges over 5 years (HTTP 400 "range must not exceed 5
  years"), so the script walks back in 5-year chunks (1245, 63, then 0 rows). **History starts
  on 2021-08-02** for all six symbols: 1308 sessions to 2026-10-30, nothing earlier.
- `matched_value / matched_volume` is an unadjusted VWAP. Ratio of adjusted close to raw VWAP,
  2021-08-02 to 2026-10-30:

| Symbol | Median | Minimum | Median, last 60 sessions |
| :-- | --: | --: | --: |
| FPT | 0.66 | 0.44 | 0.96 |
| HPG | 0.69 | 0.49 | 1.00 |
| VCB | 0.66 | 0.42 | 1.00 |
| SSI | 0.56 | 0.33 | 1.00 |
| BVH | 0.96 | 0.82 | 1.00 |
| HSG | 0.74 | 0.55 | 1.00 |

- The adjusted series is back-adjusted, by a large amount for frequent stock-dividend payers
  (SSI, VCB, FPT) and little for BVH. Which event types are adjusted was not checked (deferred,
  `AGENTS.md` section 11).
- `ssi` candidates (`trade_history`, `trading_stats`) are untested.

Consequences:

1. **Market cap option 1 is not possible** before 2021-08 (no unadjusted price) and not at all
   with the reduced item set (no paid-in capital, see A3). **Option 2 is available:** adjusted
   close × current `outstanding_shares` from `Reference("fhsc").company(sym).listing()`.
   For stock dividends and splits this roughly equals the historical market cap, because the
   adjustment factor equals the share-count ratio. It is biased by cash-dividend adjustments
   (past market cap understated for high-yield stocks) and by shares issued at market price
   (past market cap overstated). Both use information from after `t`; this goes in
   Limitations.
2. **ADV before 2021-08 can only be approximated** as adjusted close × 1000 × raw volume. This
   understates past traded value by the adjustment factor (down to 0.33 for SSI), so stocks
   that later pay many stock dividends are under-ranked in early universes. Options for the
   owner:
   - (a) Use the proxy for the whole period (one consistent definition) and measure its error
     against real `matched_value` on 2022–2026, where both exist.
   - (b) Use real `matched_value` from 2021-08 and the proxy before (more accurate later,
     definition changes mid-sample).
   - (c) Rebuild adjustment factors from `Reference("fhsc").company(sym).events()`. This is
     deferred work in `AGENTS.md` section 11.
   - (d) Get an `ssi` key and audit `trade_history` depth.
   - **(e) Chosen: anchored proxy.** From 2021-08-02 use `matched_value`. Before that use
     adjusted close × 1000 × volume / k, where k per symbol is the median of
     (adjusted close × 1000 × volume / `matched_value`) over its first 20 sessions with
     `matched_value`. This removes every price adjustment made after the anchor (for scale,
     see the "Minimum" column above) and keeps the series continuous at the splice. Adjustments for events between 2017 and 2021-08 remain; this
     goes in Limitations. The series is built at ingest as the `value` column, so universe
     and factor code never see the proxy logic. Option (a)'s error check on 2022–2026 was
     dropped because the proxy error grows going back in time, so it would understate the
     error in 2018–2021.

## A3. Which items does the reduced set provide?

`df.attrs["taxonomy"]` is `{"variant": "reduced", "version": "vas_v6.1", "origin": "server"}`
for every symbol, and every item has `item_level == 1`. Templates and company types are
detected correctly:

| Symbol | `company_type` | `template` | Balance-sheet items | Income items |
| :-- | :-- | :-- | --: | --: |
| FPT, HPG, HSG | `regular` | `DN` | 21 | 19 |
| VCB | `bank` | `TCTD` | 28 | 16 |
| SSI | `securities` | `CTCK` | 17 | 21 |
| BVH | `insurance` | `DNPNT` | 23 | 28 |

Long-format columns (`compat="v6"`): `report_period, item_id, item_name, line_code, item_order,
item_level, off_balance, item_unit, item_value`. Values are in VND.

| Needed (internal `item_id`) | Source code | DN | TCTD | CTCK | DNPNT |
| :-- | :-- | :-: | :-: | :-: | :-: |
| `npat_parent` | `IS_PROFIT_AFTER_TAX_FOR_SHAREHOLDERS_OF_PARENT_COMPANY` | ✅ | ✅ | ✅ | ✅ |
| `equity` (incl. minority) | `BS_EQUITY` | ✅ | ✅ | ✅ | ✅ |
| `minority_interest` (balance sheet) | `BS_MINORITY_INTEREST` | ❌ | ✅ | ❌ | ❌ |
| `paid_in_capital` | none at level 1 | ❌ | ❌ | ❌ | ❌ |

Also present: `IS_NET_PROFIT_AFTER_TAX` (total, all templates) and `IS_MINORITY_INTEREST`
(the income-statement minority share, all templates). `BS_CHARTER_CAPITAL_AND_FUNDS` on the bank
template is charter capital plus funds, not paid-in capital. EPS items exist but are always
empty.

Consequences:

- `ep` = TTM `npat_parent` / market cap: **computable**.
- `bp` and `roe` need parent equity = `BS_EQUITY − BS_MINORITY_INTEREST`, which only the bank
  template allows. Options for the owner:
  - **(a) Chosen.** Use total `BS_EQUITY` for every template: `bp` = total equity / market cap, and
    `roe` = TTM `IS_NET_PROFIT_AFTER_TAX` / average total equity, a consistent total-company
    pair. Overstates book value for firms with large minority interests; goes in Limitations.
  - (b) Keep `roe` = TTM `npat_parent` / average total equity (numerator and denominator
    mismatch), which understates ROE for those same firms.
- Option 1 for market cap is impossible (no paid-in capital).

## A4. How deep do quarterly statements go, and does TTM work?

| Symbol | First quarter | Last quarter | Quarters | Missing quarters | Annual periods |
| :-- | :-- | :-- | --: | :-- | :-- |
| FPT | 2013-Q1 | 2026-Q3 | 55 | none | 2004–2025 |
| HPG | 2013-Q1 | 2026-Q3 | 55 | none | 2008–2025 |
| VCB | 2009-Q1 | 2026-Q3 | 71 | none | 2004–2025 |
| SSI | 2013-Q1 | 2026-Q3 | 55 | none | 2007–2025 |
| BVH | 2009-Q1 | 2026-Q3 | 71 | none | 2008–2025 |
| HSG | 2009-Q4 | 2026-Q3 | 68 | none | 2006–2026 |

- Quarter labels are `"YYYY-Qn"`; annual labels are `"YYYY"`. There is no publication-date
  column, so `available_date = period_end + lag_days` stays as designed.
- Quarterly statements start 4+ years before the first price bar, so fundamental factors are
  available from the first rebalance (TTM to 2017-Q3, average equity from 2016-Q3).
- **Sum of four quarters vs the annual figure**, relative gap for `npat_parent`:

| Symbol | Years compared | Median gap | Max gap |
| :-- | --: | --: | --: |
| FPT | 13 | 0.0% | 0.2% |
| HPG | 13 | 0.0% | 0.3% |
| VCB | 17 | 0.0% | 0.0% |
| SSI | 13 | 0.0% | 4.7% |
| BVH | 17 | 1.6% | 19.9% |
| HSG | 16 | 15.3% | 2611% |

- Quarterly figures are standalone, so TTM = sum of the last four available quarters, as
  planned. Insurance (BVH) quarters do not always add up to the audited annual figure. HSG does
  not match because its fiscal year ends in September (see below).

## Follow-up checks (approved by the owner, run 2026-11-13 20:30–20:32, 9 calls over two runs)

- **HSG quarter labels are calendar quarters.** HSG's fiscal year ends in September. The annual
  figure for year Y equals Q4 of Y−1 plus Q1–Q3 of Y: net revenue matches within 1% in 17 of
  17 years, parent net profit in 15 of 17 (median gap 0.02%). Calendar-year sums match in 1
  of 16 years. So `period_end` = calendar quarter end is correct for an off-cycle fiscal
  year, and the 15% gap in the A4 table is a mismatch of 12-month windows, not a data error.
- **Session counts** (FPT bars): 250 sessions through 2017-12-29, 270 through 2018-01-31.
- **`fhsc` has no VN100.** `index("VN100").constituents()` returns `INDEX_NOT_FOUND`. Its
  index list has 19 entries; the Vietnamese equity ones are `VNINDEX`, `VN30`, `HNX30`,
  `HNXINDEX`, `UPCOMINDEX`. `VN30` constituents work (30 rows).

## Incidental observations (recorded, not investigated)

- `fhsc` raises `DataWarning VNSTOCK_VAS_IDENTITY` for a few periods where reported totals do not
  equal the sum of their parts (e.g. FPT balance sheet 2015-Q1 to Q3, HPG short-term assets
  2026-Q1 to Q3). The numbers are passed through as the source reports them.
- `Reference("fhsc").index("VNINDEX").constituents()` returns 409 symbols, all `HOSE`: a
  current listing usable as the candidate list, with no delisted symbols.
- `Reference("fhsc").company(sym).listing()` returns `listed_shares, outstanding_shares,
  treasury_shares, free_float_shares, free_float_percent` (current values only).

## Ingest findings (phase 2, run 2026-11-14 16:21–20:15 Vietnam time)

`python -m vnfactor ingest` finished with exit code 0 after 1,843 calls; a second run made
0 calls. Aggregate figures only:

| Item | Result |
| :-- | :-- |
| Symbols with bars | 408 of 409 candidates. `DVV` returned no bars |
| Date range | 2017-01-03 .. 2026-10-30, 2,453 sessions (249–252 per full year) |
| `prices_daily` | 854,183 rows; 43.1% proxy traded value, 0.04% without traded value, no zero-volume rows |
| Anchor factor `k` | median 0.679 (p5 0.317, p95 1.001), defined for every symbol |
| Statements | 269 symbols ever in the top 150 by ADV, 39,900 rows: 224 regular, 22 bank, 20 securities, 3 insurance; median first quarter 2013-Q1; 32 missing values; 27 equity rows ≤ 0 |
| Missing bars | 6.2% of sessions inside each symbol's own first-to-last span; per symbol median 0.4%, 46 symbols above 20% |

**Daily returns beyond the HOSE limit.** 3,765 close-to-close returns exceed ±7.5% in 94
symbols. 80% fall in 2017–2021, the median size is 10.5% and the 90th percentile 14.9%,
which matches the ±10% (HNX) and ±15% (UPCoM) limits. The most affected symbols are most
likely companies that traded on HNX or UPCoM before moving to HOSE: the candidate list is
today's HOSE listing, and `fhsc` returns each symbol's full history without an exchange
column or transfer date. This was not verified symbol by symbol. Four returns exceed 40%
and may be adjustment errors; not investigated.

**Decision (owner, 2026-11-14): keep the pre-transfer history and record it as a
limitation.** Consequences:

- Before a symbol's transfer, the universe can include a stock that was then listed on HNX
  or UPCoM, against the rule "common stocks on HOSE" (AGENTS.md section 5.2).
- Selection bias on top of survivorship: companies move to HOSE after they have grown, so
  their pre-transfer history is conditioned on later success.
- The liquidity filter (trading on at least 80% of the last 126 sessions, top 100 by median
  traded value) removes most thinly traded pre-transfer periods but not all of them.

**Slow responses.** One request after `DCM` took about 25 minutes before it returned. The
client sets no request timeout; whether vnstock exposes one has not been checked yet.

## Not tested

- Every `ssi` call (no key): HOSE listing, VN100 members, `trade_history`, `trading_stats`, and
  `ssi` bar depth.
- Delisted symbols, adjustment method per event type, `ratio()` (deferred in `AGENTS.md`
  section 11).
- Fiscal-year labels were checked for HSG only; other off-cycle companies are assumed to follow
  the same calendar-quarter labelling.
