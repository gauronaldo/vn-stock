"""Signal evaluation: monthly rank IC and quantile returns (AGENTS.md section 5.6).

``scores`` tables hold ``rebalance_date, symbol, model, score, quintile``; ``fwd`` holds
``rebalance_date, symbol, fwd_ret_1m`` (simple monthly return, see ``calendar``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_NAMES = 10  # fewer scored names with a return in a month gives no IC for that month


def _with_returns(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    m = scores.merge(fwd, on=["rebalance_date", "symbol"], how="inner", validate="many_to_one")
    return m.dropna(subset=["score", "fwd_ret_1m"])


def rank_ic(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between score and forward return, per model and month.

    Returns ``model, rebalance_date, ic, n``.
    """
    m = _with_returns(scores, fwd)
    m["score_rank"] = m.groupby(["model", "rebalance_date"])["score"].rank()
    m["ret_rank"] = m.groupby(["model", "rebalance_date"])["fwd_ret_1m"].rank()
    rows = []
    for (model, date), g in m.groupby(["model", "rebalance_date"], sort=True):
        ic = g["score_rank"].corr(g["ret_rank"]) if len(g) >= MIN_NAMES else np.nan
        rows.append({"model": model, "rebalance_date": date, "ic": ic, "n": len(g)})
    return pd.DataFrame(rows, columns=["model", "rebalance_date", "ic", "n"])


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """Per model: months, mean IC, standard deviation, ICIR, t-stat, share of IC > 0."""

    def one(s: pd.Series) -> pd.Series:
        s = s.dropna()
        n = len(s)
        std = s.std(ddof=1) if n > 1 else np.nan
        icir = s.mean() / std if n > 1 and std > 0 else np.nan
        return pd.Series(
            {
                "months": n,
                "mean_ic": s.mean(),
                "std_ic": std,
                "icir": icir,
                "t_stat": icir * np.sqrt(n),
                "pct_positive": (s > 0).mean() if n else np.nan,
            }
        )

    return ic.groupby("model")["ic"].apply(one).unstack()


def quantile_returns(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """Mean forward return per model, month and quintile: ``model, rebalance_date, q1..qn``."""
    m = _with_returns(scores, fwd).dropna(subset=["quintile"])
    m["quintile"] = m["quintile"].astype(int)
    table = m.pivot_table(
        index=["model", "rebalance_date"], columns="quintile", values="fwd_ret_1m", aggfunc="mean"
    )
    table.columns = [f"q{int(c)}" for c in table.columns]
    return table.reset_index()


def quantile_summary(qret: pd.DataFrame) -> pd.DataFrame:
    """Per model: mean monthly return per quintile, top-minus-bottom spread and its t-stat."""
    cols = sorted([c for c in qret.columns if c.startswith("q")], key=lambda c: int(c[1:]))
    top, bottom = cols[-1], cols[0]

    def one(g: pd.DataFrame) -> pd.Series:
        spread = (g[top] - g[bottom]).dropna()
        n = len(spread)
        se = spread.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        out = g[cols].mean()
        out["spread"] = spread.mean()
        out["spread_t"] = spread.mean() / se if n > 1 and se > 0 else np.nan
        out["months"] = n
        return out

    return pd.DataFrame({model: one(g) for model, g in qret.groupby("model")}).T


def signal_table(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """IC summary and quintile summary side by side, one row per model."""
    ic = ic_summary(rank_ic(scores, fwd))
    q = quantile_summary(quantile_returns(scores, fwd)).drop(columns="months")
    out = ic.join(q, how="left")
    out.index.name = "model"
    return out
