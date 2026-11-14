# Methodology

Everything below was fixed on 2026-11-14, at the end of phase 4 and before any forward
return was compared with a signal. Parameters live in `configs/base.yaml`; this file states
them in words. Changes after this point are logged at the bottom with a reason.

## Data

- Source: `fhsc` through vnstock 6.0.0 (bring-your-own-key Open API). Version 1 uses no
  other source. Details and evidence in `docs/data_audit.md`.
- Daily bars from 2017-01-03 to 2026-10-30: prices in thousand VND, adjusted for corporate
  actions by the source; volume unadjusted.
- Daily traded value (VND): the source's matched value from 2021-08-02; before that an
  anchored proxy `close_adj * 1000 * volume / k`, where `k` per symbol is the median ratio
  of `close_adj * 1000 * volume` to matched value over its first 20 sessions with a matched
  value.
- Quarterly statements (standalone quarters, VND), reduced item set: total equity
  (`BS_EQUITY`, minority interest included), net profit after tax attributable to the
  parent, total net profit after tax. Company type: regular, bank, securities, insurance.
- Current outstanding shares (no history).

## Timing

- Rebalance date `t`: last trading session of each month, 2018-01-31 to 2026-08-28
  (104 dates; the config bound 2026-08-31 is inclusive and 2026-08-28 is the last session
  of August 2026 in the data).
- Signals use data up to the close of `t`; statements count from
  `available_date = period_end + 45 days`.
- Execution at the close of the session after `t`. Forward return: from that close to the
  close of the session after the next rebalance date. The same return is used for IC,
  machine-learning labels and the backtest.
- Split: research = the first 73 rebalance dates (to 2024-01-31); holdout = the last 31
  (from 2024-02-29). The holdout is evaluated once, with the command `holdout`, after this
  file is final.

## Universe (at each `t`)

1. Candidates: today's VNINDEX constituents on HOSE (409; 408 have bars).
2. At least 252 bars up to `t`, and a bar with positive volume on at least 80% of the last
   126 market sessions.
3. The 100 symbols with the highest median daily traded value over those 126 sessions
   (ties by symbol).

Sector group: financial (bank, securities, insurance) or non-financial, from the latest
company type.

## Factors and hypotheses

`P` is the adjusted close on the market calendar; a session without a bar keeps the last
close. A window missing more than 20% of its sessions gives NaN.

| Factor | Definition at `t` | Hypothesis (sign) |
| :-- | :-- | :-- |
| `mom_12_1` | `P[t-21] / P[t-252] - 1` | + (winners keep winning) |
| `rev_1m` | `P[t] / P[t-21] - 1` | − (short-term reversal) |
| `vol_126` | standard deviation (ddof 1) of daily log returns over 126 sessions; a return after a missing bar spans the gap | − (low-volatility anomaly) |
| `adv_63` | log of the median daily traded value over 63 sessions | − (illiquidity premium) |
| `ep` | TTM parent net profit / market cap | + (value) |
| `bp` | total equity / market cap | + (value) |
| `roe` | TTM total net profit / average total equity | + (quality) |

- Market cap (VND) = `P[t] * 1000 * current outstanding shares`.
- TTM = sum of the latest four consecutive available quarters; NaN if one is missing.
- Average equity = mean of equity at the latest available quarter end and four quarters
  earlier.
- Equity ≤ 0 (latest or average) makes `bp` and `roe` NaN.
- E/P and B/P rather than P/E and P/B: P/E has no meaning for losses and cannot be ranked
  around zero.

## Preprocessing and composite

At each `t`, over the universe:

1. Winsorize at the 1st and 99th percentiles, then z-score (population standard
   deviation). Fewer than three values or zero dispersion gives NaN.
2. `ep`, `bp`, `roe` are winsorized and z-scored within each sector group.
3. Multiply by the hypothesised sign, so a higher signed z always means a higher expected
   return. Missing values are not filled.

Composite score = equal-weight mean of four group scores: momentum (`mom_12_1`),
volatility (`vol_126`), value (mean of `ep` and `bp`), quality (`roe`). A group score is
the mean of its available members. A stock needs at least two groups to be ranked.
`rev_1m` and `adv_63` are reported as single factors and used as model features only.
The weights are fixed here and are not tuned on results.

## Evaluation

For each single factor (signed z), the composite and each model:

- Monthly rank IC (Spearman) between score and forward return across universe members:
  mean, standard deviation, ICIR = mean / standard deviation (monthly),
  t-stat = ICIR × √months, share of months with IC > 0.
- Quintiles by score each month (Q5 = highest); mean forward return per quintile over time
  and the Q5 − Q1 spread with its t-stat.
- Seven factors are tested, so a few t-stats above 2 can arise by chance; no
  multiple-testing correction is applied in version 1.

## Backtest

- Long-only, equal weight in the top quintile (20 stocks), rebalanced monthly.
- Benchmark: equal weight in the whole universe, rebalanced monthly, same costs.
- Costs: 0.15% of traded value on each buy and sell, plus 0.1% tax on sell value, charged
  on the change between drifted and target weights; the first month buys from cash.
  Results are shown before and after costs. Net return = gross return − cost.
- Metrics: CAGR, annualised volatility, Sharpe ratio with a zero risk-free rate, maximum
  drawdown on monthly values, excess CAGR over the benchmark, mean monthly two-sided
  turnover (sum of absolute weight changes).
- Execution is assumed at the close of `t+1`, with no price-limit or liquidity
  constraint.

## Machine learning

- Features: signed z of all seven factors. Label: percentile rank of the forward return
  within the month.
- Ridge: `alpha = 10`, missing features set to 0 (the cross-sectional mean of a z-score;
  the only exception to the no-filling rule). LightGBM: regression objective,
  300 trees, learning rate 0.03, 15 leaves, depth 4, at least 200 samples per leaf,
  row subsample 0.8 every iteration, column subsample 0.8, L2 1.0, one thread,
  deterministic, seed 42; missing features passed as NaN.
- Walk-forward with an expanding window. The model predicting month `k` (0-based) is trained
  on months `0 … k−2`: the label of month `k−1` is only known one session after `t_k`.
  The first prediction needs 36 training months, so it is month 37 (2021-02-26). Models
  are refitted every 12 months (months 37, 49, 61, 73, 85, 97) and reused in between.
- Research comparison: months 37–72 (36 months), against the composite on exactly the same
  months: IC, quintile spread and top-quintile return after costs. Holdout predictions
  continue the same schedule. No hyperparameter is searched.

## Robustness (research period, one change at a time)

1. Reporting lag 90 days instead of 45.
2. Costs doubled.
3. Survivorship illustration. `fhsc` has no VN100, so the stand-in is a "current liquid
   basket": the 100 symbols with the highest 126-session median traded value at the last
   session (2026-10-30), among symbols passing the history and trading filters on that
   date. This basket is applied to every past date (members that pass the history and
   trading filters at `t`, no liquidity ranking), and compared with the rule-based
   universe. Chosen by the agent on 2026-11-14 under the owner's instruction to build the
   remaining phases without waiting; it mirrors VN100, which also selects by size and
   liquidity today.

## Known limitations

These go into the Limitations section of the README.

- **Survivorship bias, partly remaining:** candidates are today's VNINDEX constituents, so
  delisted symbols are missing.
- **Pre-transfer history:** a symbol that moved from HNX or UPCoM to HOSE brings its earlier
  history. The universe can hold a stock that was not on HOSE at the time, and that history
  is conditioned on the company later qualifying for HOSE (selection bias). About 3,765
  daily returns beyond ±7.5%, 80% of them in 2017–2021, point to this.
- **Restated figures:** statement values may have been restated retroactively by the source.
- **Market cap:** today's share count applied to past adjusted prices puts future
  information into `ep` and `bp` around cash dividends and new issuance.
- **Traded value before 2021-08-02 is estimated:** corporate actions between 2017 and the
  anchor are not undone.
- **Total equity:** `bp` and `roe` include minority interest.
- **Short sample:** 73 research months, 31 holdout months, about 100 stocks per month.
- **No short selling;** Q5 − Q1 is a signal measure only.
- **Execution:** trades at the close of `t+1`, ignoring price limits (±7% on HOSE),
  lot sizes and market impact.

## Decision log

| Date | Decision | Why | Evidence |
| :-- | :-- | :-- | :-- |
| 2026-11-13 | Use `fhsc` as the only source; first rebalance 2018-01-31, last 2026-08-31; research to 2024-01-31 (73 months), holdout from 2024-02-29 (31 months) | Daily bars start 2017-01-03; no `ssi` key | `docs/data_audit.md`, A1 |
| 2026-11-13 | Market cap = adjusted close × current outstanding shares | No unadjusted prices before 2021 and no paid-in capital in the reduced item set | `docs/data_audit.md`, A2–A3 |
| 2026-11-13 | Traded value before 2021-08-02 = anchored proxy | Matched value exists only from 2021-08-02 | `docs/data_audit.md`, A2 |
| 2026-11-13 | `bp` and `roe` use total equity (incl. minority interest) | Parent-only equity is not in the reduced item set for non-banks | `docs/data_audit.md`, A3 |
| 2026-11-14 | Keep the history of symbols from before they moved to HOSE; record it as a limitation | The source has no exchange history or transfer date; a reconstruction from price-limit breaches would be a guess | `docs/data_audit.md`, Ingest findings |
| 2026-11-14 | Survivorship stand-in = current liquid basket (top 100 by traded value on 2026-10-30) | `fhsc` has no VN100; owner asked to proceed without waiting | this file, Robustness |
| 2026-11-14 | Methodology frozen | End of phase 4, before evaluation | this file |
| 2026-11-14 | Holdout evaluated once at 21:32 Vietnam time; nothing changed afterwards | Phase 8 | `reports/tables/holdout_run.txt` |
| 2026-11-14 | Report wording only (figure titles, summary sentences computed from the tables) after the holdout; no method, parameter or data change | Phase 9 | `src/vnfactor/report.py` |
