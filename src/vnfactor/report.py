"""Figures, ``reports/report.md`` and the results block of ``README.md``.

Every number comes from the aggregate tables written by ``evaluate``, ``ml``,
``robustness`` and ``holdout`` in ``reports/tables``; nothing is typed by hand.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reference palette (dataviz skill, light mode): fixed categorical order, text and grid tokens.
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3: blue, orange, aqua
REFERENCE = "#8a8984"  # neutral ink for the benchmark line

FACTOR_ORDER = ["mom_12_1", "rev_1m", "vol_126", "adv_63", "ep", "bp", "roe"]
MODEL_ORDER = ["composite", "ridge", "lightgbm"]
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


# --- loading ------------------------------------------------------------------------------


def load_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every CSV in ``tables_dir``; raise if one of the required tables is missing."""
    required = [
        "research_signals",
        "research_backtest",
        "research_returns",
        "research_ml_signals",
        "research_ml_backtest",
        "research_robustness",
        "holdout_signals",
        "holdout_backtest",
        "holdout_returns",
        "holdout_ml_signals",
        "holdout_ml_backtest",
        "research_ml_training_log",
    ]
    missing = [n for n in required if not (tables_dir / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"missing tables in {tables_dir}: {missing}; run the earlier steps")
    out = {p.stem: pd.read_csv(p) for p in sorted(tables_dir.glob("*.csv"))}
    for name, df in out.items():
        for col in ("rebalance_date", "refit_date", "predict_date", "train_first", "train_last"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        out[name] = df
    return out


# --- formatting ---------------------------------------------------------------------------


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


def md_table(rows: list[list[str]], header: list[str], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else ":--" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _net(bt: pd.DataFrame, strategy: str, basis: str = "net") -> pd.Series:
    return bt[(bt["strategy"] == strategy) & (bt["basis"] == basis)].iloc[0]


# --- figures ------------------------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure(width: float = 8.0, height: float = 4.2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def equity_figure(t: dict[str, pd.DataFrame], path: Path, holdout_start: pd.Timestamp) -> str:
    """Growth of 1 VND after costs: top quintile of the composite vs the benchmark."""
    series = {}
    for strategy in ("composite", "benchmark"):
        parts = [
            t[f"{p}_returns"][t[f"{p}_returns"]["strategy"] == strategy]
            for p in ("research", "holdout")
        ]
        r = pd.concat(parts).sort_values("rebalance_date").set_index("rebalance_date")["net"]
        series[strategy] = (1 + r).cumprod()
    r_ex = _net(t["research_backtest"], "composite")["excess_cagr"]
    h_ex = _net(t["holdout_backtest"], "composite")["excess_cagr"]
    title = (
        f"Top quintile {'trailed' if r_ex < 0 else 'beat'} the equal-weight universe in research "
        f"({pct(r_ex)} a year) and {'beat' if h_ex > 0 else 'trailed'} it in the holdout "
        f"({pct(h_ex)} a year), after costs"
    )
    fig, ax = _figure()
    ax.axvspan(holdout_start, series["composite"].index.max(), color=GRID, alpha=0.45, lw=0)
    ax.plot(series["composite"], color=SERIES[0], lw=2, label="Top quintile of composite")
    ax.plot(series["benchmark"], color=REFERENCE, lw=2, label="Equal-weight universe (benchmark)")
    for s in series.values():
        ax.annotate(
            f"{s.iloc[-1]:.2f}",
            (s.index[-1], s.iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    ax.text(holdout_start, ax.get_ylim()[1], "  holdout", va="top", fontsize=9, color=TEXT_2)
    ax.set_ylabel("Growth of 1 VND, after costs", color=TEXT_2, fontsize=9)
    ax.set_xlabel("Rebalance date (month end)", color=TEXT_2, fontsize=9)
    ax.set_title(textwrap.fill(title, 90), color=TEXT, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return title


def ic_figure(t: dict[str, pd.DataFrame], path: Path) -> str:
    """Mean monthly rank IC per signal, research vs holdout."""
    rs = t["research_signals"].set_index("model")
    hs = t["holdout_signals"].set_index("model")
    rm = t["research_ml_signals"].set_index("model")
    hm = t["holdout_ml_signals"].set_index("model")
    names = [*FACTOR_ORDER, "composite", "ridge", "lightgbm"]
    research = [rs.loc[n, "mean_ic"] if n in rs.index else rm.loc[n, "mean_ic"] for n in names]
    holdout = [hs.loc[n, "mean_ic"] if n in hs.index else hm.loc[n, "mean_ic"] for n in names]
    strongest = max(abs(v) for v in research + holdout)
    title = f"No signal is strong: every mean monthly rank IC is within ±{strongest:.2f}"
    fig, ax = _figure(8.0, 4.6)
    y = np.arange(len(names))
    h = 0.38
    ax.barh(y - h / 2, research, height=h - 0.04, color=SERIES[0], label="Research")
    ax.barh(y + h / 2, holdout, height=h - 0.04, color=SERIES[1], label="Holdout")
    ax.axvline(0, color=TEXT_2, lw=0.8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("Mean monthly rank IC (Spearman)", color=TEXT_2, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return title


def quintile_figure(t: dict[str, pd.DataFrame], path: Path) -> str:
    """Mean monthly forward return by composite quintile, research vs holdout."""
    cols = ["q1", "q2", "q3", "q4", "q5"]
    r = t["research_signals"].set_index("model").loc["composite", cols].astype(float)
    h = t["holdout_signals"].set_index("model").loc["composite", cols].astype(float)
    title = (
        f"Top-minus-bottom quintile spread of the composite: {pct(r['q5'] - r['q1'], 2)} a month "
        f"in research, {pct(h['q5'] - h['q1'], 2)} in the holdout"
    )
    fig, ax = _figure(7.0, 4.0)
    x = np.arange(5)
    w = 0.38
    ax.bar(x - w / 2, r * 100, width=w - 0.04, color=SERIES[0], label="Research")
    ax.bar(x + w / 2, h * 100, width=w - 0.04, color=SERIES[1], label="Holdout")
    ax.axhline(0, color=TEXT_2, lw=0.8)
    ax.set_xticks(x, ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    ax.set_ylabel("Mean monthly forward return (%)", color=TEXT_2, fontsize=9)
    ax.set_title(textwrap.fill(title, 80), color=TEXT, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return title


# --- text ---------------------------------------------------------------------------------


def _ic_cell(df: pd.DataFrame, model: str) -> str:
    row = df.loc[model]
    return f"{num(row['mean_ic'])} ({num(row['t_stat'], 2)})"


def ic_rows(t: dict[str, pd.DataFrame]) -> list[list[str]]:
    rs, hs = t["research_signals"].set_index("model"), t["holdout_signals"].set_index("model")
    rm, hm = t["research_ml_signals"].set_index("model"), t["holdout_ml_signals"].set_index("model")
    rows = [[f"`{f}`", _ic_cell(rs, f), _ic_cell(hs, f)] for f in FACTOR_ORDER]
    rows.append(
        ["Composite (all research months)", _ic_cell(rs, "composite"), _ic_cell(hs, "composite")]
    )
    for m in MODEL_ORDER:
        label = {"composite": "Composite", "ridge": "Ridge", "lightgbm": "LightGBM"}[m]
        rows.append([f"{label} (out-of-sample months)", _ic_cell(rm, m), _ic_cell(hm, m)])
    return rows


def backtest_rows(bt: pd.DataFrame, strategies: list[str]) -> list[list[str]]:
    rows = []
    for s in strategies:
        g, n = _net(bt, s, "gross"), _net(bt, s, "net")
        rows.append(
            [
                s,
                pct(g["cagr"]),
                pct(n["cagr"]),
                pct(n["vol"]),
                num(n["sharpe"], 2),
                pct(n["max_drawdown"]),
                pct(n["excess_cagr"]),
                pct(n["mean_turnover"], 0),
            ]
        )
    return rows


BT_HEADER = [
    "Strategy",
    "CAGR gross",
    "CAGR net",
    "Volatility",
    "Sharpe (net)",
    "Max drawdown",
    "Excess CAGR (net)",
    "Turnover / month",
]


def facts(
    t: dict[str, pd.DataFrame], universe: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Headline numbers used in both the report and the README."""
    r_bt, h_bt = t["research_backtest"], t["holdout_backtest"]
    hm = t["holdout_ml_signals"].set_index("model")
    hmb = t["holdout_ml_backtest"]
    rmb = t["research_ml_backtest"]
    dates = sorted(universe["rebalance_date"].unique())
    cut = pd.Timestamp(cfg["split"]["holdout_start"])
    return {
        "months": len(dates),
        "research_months": sum(d < cut for d in dates),
        "holdout_months": sum(d >= cut for d in dates),
        "first": pd.Timestamp(dates[0]),
        "last": pd.Timestamp(dates[-1]),
        "symbols": universe["symbol"].nunique(),
        "h_ic": hm.loc["composite", "mean_ic"],
        "h_t": hm.loc["composite", "t_stat"],
        "h_top": _net(h_bt, "composite")["cagr"],
        "h_bench": _net(h_bt, "benchmark")["cagr"],
        "r_top": _net(r_bt, "composite")["cagr"],
        "r_bench": _net(r_bt, "benchmark")["cagr"],
        "r_ic": t["research_signals"].set_index("model").loc["composite", "mean_ic"],
        "r_t": t["research_signals"].set_index("model").loc["composite", "t_stat"],
        "h_ridge": _net(hmb, "ridge")["cagr"],
        "h_lgbm": _net(hmb, "lightgbm")["cagr"],
        "r_ridge": _net(rmb, "ridge")["cagr"],
        "r_lgbm": _net(rmb, "lightgbm")["cagr"],
        "r_comp_oos": _net(rmb, "composite")["cagr"],
        "max_abs_t": max(
            t[n]["t_stat"].abs().max()
            for n in (
                "research_signals",
                "holdout_signals",
                "research_ml_signals",
                "holdout_ml_signals",
            )
        ),
    }


def _vs(model: float, composite: float) -> str:
    return "above" if model > composite else "below"


def summary_paragraph(f: dict[str, Any]) -> str:
    beat_h = f["h_top"] > f["h_bench"]
    beat_r = f["r_top"] > f["r_bench"]
    t_note = (
        "No t-statistic in this study reaches 2, so the evidence for any signal is weak."
        if f["max_abs_t"] < 2
        else f"The largest |t| in this study is {num(f['max_abs_t'], 2)}; with this many tests,"
        " that is weak evidence."
    )
    ml_wins = [
        f[f"h_{m}"] > f["h_top"] and f[f"r_{m}"] > f["r_comp_oos"] for m in ("ridge", "lgbm")
    ]
    ml_verdict = "beat" if any(ml_wins) else "did not reliably beat"
    return (
        f"Each month from {f['first']:%b %Y} to {f['last']:%b %Y} ({f['months']} rebalances), the "
        f"100 most liquid HOSE stocks are ranked on seven factors, and the ranking is tested "
        f"against next-month returns without look-ahead. In the {f['holdout_months']}-month "
        f"holdout, evaluated once, the equal-weight composite had a mean rank IC of "
        f"{num(f['h_ic'])} (t = {num(f['h_t'], 2)}) and its top quintile returned "
        f"{pct(f['h_top'])} a year after costs, against {pct(f['h_bench'])} for the "
        f"equal-weight universe. In the {f['research_months']}-month research period the same "
        f"portfolio {'beat' if beat_r else 'trailed'} the benchmark ({pct(f['r_top'])} vs "
        f"{pct(f['r_bench'])}) and the composite IC was {num(f['r_ic'])} "
        f"(t = {num(f['r_t'], 2)}){'' if beat_h == beat_r else ', so the sign of the edge flips between periods'}. "
        f"Machine learning {ml_verdict} simple score averaging: in the holdout Ridge's top "
        f"quintile returned {pct(f['h_ridge'])} ({_vs(f['h_ridge'], f['h_top'])} the composite) and "
        f"LightGBM's {pct(f['h_lgbm'])} ({_vs(f['h_lgbm'], f['h_top'])}); on the research "
        f"out-of-sample months they returned {pct(f['r_ridge'])} and {pct(f['r_lgbm'])} against "
        f"{pct(f['r_comp_oos'])} for the composite. {t_note}"
    )


LIMITATIONS = """- **Survivorship bias, partly remaining.** Candidates are today's VNINDEX constituents;
  delisted stocks are missing. The robustness check shows how much a basket picked with
  hindsight flatters returns.
- **Pre-transfer history.** Stocks that moved to HOSE from HNX or UPCoM bring their earlier
  history, so the universe can hold a stock that was not on HOSE at the time, and that history
  is conditioned on later success (selection bias).
- **Restated figures.** Statement values may have been restated by the source.
- **Market cap formula.** Today's share count times past adjusted prices puts future
  information into E/P and B/P around cash dividends and share issuance.
- **Estimated traded value before August 2021.** An anchored proxy from adjusted prices and
  raw volume; corporate actions between 2017 and the anchor are not undone.
- **Total equity.** B/P and ROE include minority interest.
- **Short sample.** About 100 stocks, 73 research months and 31 holdout months.
- **No short selling.** Q5 − Q1 is a signal measure, not a tradable return.
- **Execution.** Trades at the close of the next session, ignoring the ±7% price limit, lot
  sizes and market impact."""


def build_report(
    t: dict[str, pd.DataFrame],
    f: dict[str, Any],
    figure_titles: dict[str, str],
    cfg: dict[str, Any],
    holdout_note: str,
) -> str:
    """Full ``reports/report.md`` text."""
    rs = t["research_signals"].set_index("model")
    hs = t["holdout_signals"].set_index("model")
    signif = [m for m in FACTOR_ORDER if abs(rs.loc[m, "t_stat"]) >= 2]
    q_rows = []
    for name, df in (("Research", rs), ("Holdout", hs)):
        row = df.loc["composite"]
        q_rows.append(
            [
                name,
                *[pct(row[c], 2) for c in ("q1", "q2", "q3", "q4", "q5")],
                pct(row["spread"], 2),
                num(row["spread_t"], 2),
            ]
        )
    single_rows = []
    for m in FACTOR_ORDER:
        r, h = rs.loc[m], hs.loc[m]
        single_rows.append(
            [
                f"`{m}`",
                num(r["mean_ic"]),
                num(r["icir"], 2),
                num(r["t_stat"], 2),
                pct(r["pct_positive"], 0),
                pct(r["spread"], 2),
                num(h["mean_ic"]),
                num(h["t_stat"], 2),
            ]
        )
    rob = t["research_robustness"]
    rob_rows = [
        [
            f"`{r['variant']}`",
            num(r["mean_ic"]),
            num(r["ic_t_stat"], 2),
            pct(r["top_cagr_net"]),
            pct(r["benchmark_cagr_net"]),
            pct(r["excess_cagr_net"]),
            pct(r["overlap_with_base"], 0),
        ]
        for _, r in rob.iterrows()
    ]
    by_variant = rob.set_index("variant")
    base = by_variant.loc["baseline"]
    basket = by_variant.loc["current_liquid_basket"]
    lag = by_variant.loc[[v for v in by_variant.index if v.startswith("lag_")][0]]
    costs = by_variant.loc[[v for v in by_variant.index if v.startswith("costs_")][0]]
    steadiest = max(FACTOR_ORDER, key=lambda m: min(rs.loc[m, "t_stat"], hs.loc[m, "t_stat"]))
    net_cagr = {
        m: (_net(t["research_ml_backtest"], m)["cagr"], _net(t["holdout_ml_backtest"], m)["cagr"])
        for m in MODEL_ORDER
    }
    worst = [min(MODEL_ORDER, key=lambda m: net_cagr[m][i]) for i in (0, 1)]
    ml_beats = [
        [m for m in ("ridge", "lightgbm") if net_cagr[m][i] > net_cagr["composite"][i]]
        for i in (0, 1)
    ]
    log = t["research_ml_training_log"]
    ridge_coef = log[log["model"] == "ridge"].drop_duplicates("refit_date")[FACTOR_ORDER].mean()
    lgbm_gain = log[log["model"] == "lightgbm"].drop_duplicates("refit_date")[FACTOR_ORDER].mean()
    ml_rows = [[f"`{c}`", f"{ridge_coef[c]:+.4f}", pct(lgbm_gain[c], 0)] for c in FACTOR_ORDER]
    rm = t["research_ml_signals"].set_index("model")
    hm = t["holdout_ml_signals"].set_index("model")
    cmp_rows = []
    for m in MODEL_ORDER:
        cmp_rows.append(
            [
                m,
                num(rm.loc[m, "mean_ic"]),
                num(rm.loc[m, "t_stat"], 2),
                pct(_net(t["research_ml_backtest"], m)["cagr"]),
                num(hm.loc[m, "mean_ic"]),
                num(hm.loc[m, "t_stat"], 2),
                pct(_net(t["holdout_ml_backtest"], m)["cagr"]),
            ]
        )
    cmp_rows.append(
        [
            "benchmark",
            "",
            "",
            pct(_net(t["research_ml_backtest"], "benchmark")["cagr"]),
            "",
            "",
            pct(_net(t["holdout_ml_backtest"], "benchmark")["cagr"]),
        ]
    )
    oos = log["predict_date"]
    fee, tax = cfg["backtest"]["fee_rate"], cfg["backtest"]["sell_tax_rate"]

    return f"""# Factor ranking on liquid HOSE stocks: full report

Generated by `python -m vnfactor report` from the tables in `reports/tables`. Method and
parameters are in `docs/methodology.md` (frozen before evaluation); data evidence is in
`docs/data_audit.md`. For education only; not investment advice.

## Summary

{summary_paragraph(f)}

## Setup

- {f["months"]} monthly rebalances, {f["first"]:%Y-%m-%d} to {f["last"]:%Y-%m-%d}:
  {f["research_months"]} research months, then {f["holdout_months"]} holdout months.
- Universe: the 100 HOSE stocks with the highest 126-session median traded value, among
  stocks with 252 sessions of history that traded on 80% of recent sessions;
  {f["symbols"]} distinct stocks enter it at some point.
- Signals at the month-end close; trades at the next close; returns to the close after the
  next month end. Statements count 45 days after quarter end.
- Costs: {pct(fee, 2)} per side plus {pct(tax, 2)} sell tax, on traded weight.

## 1. Single factors (research period)

Mean monthly rank IC, ICIR (mean / standard deviation), t-stat (ICIR × √months), share of
months with IC > 0, and the top-minus-bottom quintile spread per month. Signs are the
hypotheses fixed in advance, so a positive IC means the hypothesis held.

{md_table(single_rows, ["Factor", "Mean IC", "ICIR", "t-stat", "IC > 0", "Q5 − Q1", "Holdout IC", "Holdout t"])}

Seven factors were tested, so one or two t-statistics above 2 could appear by chance alone.
{"No factor reaches |t| ≥ 2 in the research period." if not signif else "Factors with |t| ≥ 2 in research: " + ", ".join(f"`{s}`" for s in signif) + "."}

![IC by signal](figures/ic_by_signal.png)

*{figure_titles["ic"]}*

## 2. Composite score

{md_table(q_rows, ["Period", "Q1", "Q2", "Q3", "Q4", "Q5", "Q5 − Q1", "t-stat"])}

![Quintile returns](figures/quintile_returns.png)

*{figure_titles["quintile"]}*

## 3. Backtest: top quintile vs the equal-weight universe

Research period ({f["research_months"]} months):

{md_table(backtest_rows(t["research_backtest"], ["composite", "benchmark"]), BT_HEADER)}

Holdout ({f["holdout_months"]} months):

{md_table(backtest_rows(t["holdout_backtest"], ["composite", "benchmark"]), BT_HEADER)}

Turnover is the two-sided sum of absolute weight changes per month (the first month counts
the initial purchase). Sharpe uses a zero risk-free rate.

![Growth of 1 VND](figures/equity_curve.png)

*{figure_titles["equity"]}*

## 4. Machine learning vs the simple composite

Walk-forward, expanding window, refitted every 12 months, trained only on labels known at
the prediction date. Research out-of-sample months: {oos.min():%Y-%m-%d} to
{oos.max():%Y-%m-%d} ({oos.nunique()} months); the composite is evaluated on exactly the
same months. CAGR is the top quintile after costs.

{md_table(cmp_rows, ["Model", "Research IC", "t", "Research CAGR", "Holdout IC", "t", "Holdout CAGR"])}

Average Ridge coefficients and LightGBM gain shares across research refits:

{md_table(ml_rows, ["Feature", "Ridge coefficient", "LightGBM gain share"])}

With about 100 stocks a month and at most a few thousand training rows, the label is mostly
noise. Ridge coefficients stay close to zero, and LightGBM spreads its splits almost evenly
across features, which is what fitting noise looks like. Its higher turnover also makes
costs weigh more. Ridge uses NaN → 0 for missing features (the cross-sectional mean of a
z-score), the only place where missing values are filled.

## 5. Robustness (research period, one change at a time)

{md_table(rob_rows, ["Variant", "Mean IC", "t", "Top CAGR net", "Benchmark CAGR net", "Excess", "Overlap with base universe"])}

- Reporting lag of 90 days instead of 45: mean IC {num(lag["mean_ic"])} vs
  {num(base["mean_ic"])}, top-quintile CAGR {pct(lag["top_cagr_net"])} vs
  {pct(base["top_cagr_net"])}.
- Doubling costs lowers the top-quintile CAGR from {pct(base["top_cagr_net"])} to
  {pct(costs["top_cagr_net"])} a year.
- Survivorship: picking the universe with hindsight (the 100 most liquid stocks on the
  last day, applied to the past) raises the benchmark's research CAGR from
  {pct(base["benchmark_cagr_net"])} to {pct(basket["benchmark_cagr_net"])}. That gap is the
  size of the bias a hindsight universe would have added.

## 6. Holdout

{holdout_note}

The holdout used the frozen method and parameters, and nothing was changed after it ran.
With {f["holdout_months"]} months, a difference in CAGR of a few points a year is within
the range that chance alone produces; read the holdout as a consistency check, not as proof.

## 7. Conclusions

1. Single factors: the evidence for cross-sectional predictability is weak. The factor
   with the best worse-of-two-periods t-stat is `{steadiest}` (research
   {num(rs.loc[steadiest, "t_stat"], 2)}, holdout {num(hs.loc[steadiest, "t_stat"], 2)}); no
   factor clears |t| ≥ 2 in both periods, and seven tests were run.
2. Composite vs machine learning, top-quintile CAGR after costs on the same months: research
   {", ".join(f"{m} {pct(net_cagr[m][0])}" for m in MODEL_ORDER)}; holdout
   {", ".join(f"{m} {pct(net_cagr[m][1])}" for m in MODEL_ORDER)}. Models that beat the
   composite: research {", ".join(ml_beats[0]) or "none"}; holdout
   {", ".join(ml_beats[1]) or "none"}. Worst: {worst[0]} in research, {worst[1]} in the
   holdout.
3. After costs, the top quintile {"beat" if f["h_top"] > f["h_bench"] else "trailed"} the
   equal-weight universe in the holdout and {"beat" if f["r_top"] > f["r_bench"] else "trailed"}
   it in research{"" if (f["h_top"] > f["h_bench"]) == (f["r_top"] > f["r_bench"]) else "; the sign changes between periods, so no stable edge is shown"}.

## Limitations

{LIMITATIONS}
"""


def results_block(t: dict[str, pd.DataFrame], f: dict[str, Any]) -> str:
    """Markdown placed between the results markers of ``README.md``."""
    rows = ic_rows(t)
    return f"""{README_START}
{summary_paragraph(f)}

![Growth of 1 VND after costs](reports/figures/equity_curve.png)

Mean monthly rank IC (t-stat in brackets). Ridge, LightGBM and the second composite row use
the {t["research_ml_signals"]["months"].iloc[0]:.0f} research months where the models have
out-of-sample predictions, and all {f["holdout_months"]} holdout months.

{md_table(rows, ["Signal", "Research", "Holdout"])}

Top quintile, CAGR after costs: research {pct(f["r_top"])} vs benchmark {pct(f["r_bench"])};
holdout {pct(f["h_top"])} vs benchmark {pct(f["h_bench"])}.
{README_END}"""


def update_readme(readme: str, block: str) -> str:
    """Replace the text between the results markers; raise if the markers are missing."""
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.S)
    if not pattern.search(readme):
        raise ValueError("README.md has no results markers")
    return pattern.sub(lambda _: block, readme)
