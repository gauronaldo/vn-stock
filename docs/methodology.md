# Methodology

Definitions, pre-registered hypotheses and frozen parameters are completed in phase 4, before
any evaluation. Until then this file holds the decision log.

## Decision log

| Date | Decision | Why | Evidence |
| :-- | :-- | :-- | :-- |
| 2026-11-13 | Use `fhsc` as the only source; first rebalance 2018-01-31, last 2026-08-31; research to 2024-01-31 (73 months), holdout from 2024-02-29 (31 months) | Daily bars start 2017-01-03; no `ssi` key | `docs/data_audit.md`, A1 |
| 2026-11-13 | Market cap = adjusted close × current outstanding shares | No unadjusted prices before 2021 and no paid-in capital in the reduced item set | `docs/data_audit.md`, A2–A3 |
| 2026-11-13 | Traded value before 2021-08-02 = anchored proxy | Matched value exists only from 2021-08-02 | `docs/data_audit.md`, A2 |
| 2026-11-13 | `bp` and `roe` use total equity (incl. minority interest) | Parent-only equity is not in the reduced item set for non-banks | `docs/data_audit.md`, A3 |
| 2026-11-14 | Keep the history of symbols from before they moved to HOSE; record it as a limitation | The source has no exchange history or transfer date; a reconstruction from price-limit breaches would be a guess | `docs/data_audit.md`, Ingest findings |

## Known limitations so far

These go into the Limitations section of the README.

- **Survivorship bias, partly remaining:** candidates are today's VNINDEX constituents, so
  delisted symbols are missing.
- **Pre-transfer history:** a symbol that moved from HNX or UPCoM to HOSE brings its earlier
  history. The universe can therefore hold a stock that was not on HOSE at the time, and that
  history is conditioned on the company later qualifying for HOSE (selection bias). About
  3,765 daily returns beyond ±7.5%, 80% of them in 2017–2021, point to this.
- **Restated figures:** statement values may be restated retroactively by the source.
- **Market cap:** current share count applied to past adjusted prices puts future
  information into `ep` and `bp` around cash dividends and new issuance.
- **Traded value before 2021-08-02 is estimated:** corporate actions between 2017 and the
  anchor are not undone.
- **Total equity:** `bp` and `roe` include minority interest.
