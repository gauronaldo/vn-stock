"""Monthly long-only backtest as weight matrix x return matrix (AGENTS.md section 5.7).

At each rebalance date the portfolio moves from its drifted weights to the target
weights. Cost = ``fee_rate * (buys + sells) + sell_tax_rate * sells``, as a fraction of
portfolio value; net return = gross return − cost. The first period buys from cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BACKTEST_COLUMNS = ["rebalance_date", "gross", "cost", "net", "turnover", "n_holdings"]


def top_quantile_weights(scores: pd.DataFrame, quantile: int) -> pd.DataFrame:
    """Equal weights over names in ``quantile`` each month: ``rebalance_date, symbol, weight``."""
    top = scores[scores["quintile"] == quantile][["rebalance_date", "symbol"]].copy()
    top["weight"] = 1.0 / top.groupby("rebalance_date")["symbol"].transform("size")
    return top.reset_index(drop=True)


def equal_weights(universe: pd.DataFrame) -> pd.DataFrame:
    """Equal weights over every universe member each month (the benchmark)."""
    w = universe[["rebalance_date", "symbol"]].copy()
    w["weight"] = 1.0 / w.groupby("rebalance_date")["symbol"].transform("size")
    return w.reset_index(drop=True)


def run_backtest(
    weights: pd.DataFrame, fwd: pd.DataFrame, fee_rate: float, sell_tax_rate: float
) -> pd.DataFrame:
    """Per rebalance date: gross and net return, cost, two-sided turnover, holdings.

    Raises when a held name has no forward return: a missing return must not count as 0.
    """
    w = weights.pivot(index="rebalance_date", columns="symbol", values="weight").sort_index()
    r = fwd.pivot(index="rebalance_date", columns="symbol", values="fwd_ret_1m")
    r = r.reindex(index=w.index, columns=w.columns)
    held = w.notna() & (w != 0)
    if (held & r.isna()).any().any():
        missing = int((held & r.isna()).sum().sum())
        raise ValueError(f"{missing} held positions have no forward return")
    w = w.fillna(0.0)  # not held = zero weight
    r = r.where(held, 0.0)
    if not np.allclose(w.sum(axis=1), 1.0):
        raise ValueError("weights must sum to 1 at every rebalance date")

    rows = []
    drifted = pd.Series(0.0, index=w.columns)  # start in cash
    for date in w.index:
        target, ret = w.loc[date], r.loc[date]
        trade = target - drifted
        buys, sells = trade.clip(lower=0).sum(), (-trade).clip(lower=0).sum()
        cost = fee_rate * (buys + sells) + sell_tax_rate * sells
        gross = float((target * ret).sum())
        rows.append(
            {
                "rebalance_date": date,
                "gross": gross,
                "cost": cost,
                "net": gross - cost,
                "turnover": buys + sells,
                "n_holdings": int((target > 0).sum()),
            }
        )
        drifted = target * (1.0 + ret) / (1.0 + gross)
    return pd.DataFrame(rows, columns=BACKTEST_COLUMNS)


def performance(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    """CAGR, annualised volatility, Sharpe (zero risk-free rate), maximum drawdown."""
    r = returns.dropna()
    n = len(r)
    wealth = (1.0 + r).cumprod()
    cagr = wealth.iloc[-1] ** (periods_per_year / n) - 1.0 if n else np.nan
    vol = r.std(ddof=1) * np.sqrt(periods_per_year) if n > 1 else np.nan
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year) if n > 1 else np.nan
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    max_dd = float((wealth / peak - 1.0).min()) if n else np.nan
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": max_dd, "months": n}


def summarize(
    results: dict[str, pd.DataFrame], benchmark: str, periods_per_year: int
) -> pd.DataFrame:
    """One row per strategy and cost basis, with excess CAGR over the benchmark."""
    rows = []
    for basis in ("gross", "net"):
        bench = performance(results[benchmark].set_index("rebalance_date")[basis], periods_per_year)
        for name, bt in results.items():
            perf = performance(bt.set_index("rebalance_date")[basis], periods_per_year)
            rows.append(
                {
                    "strategy": name,
                    "basis": basis,
                    **perf,
                    "excess_cagr": perf["cagr"] - bench["cagr"],
                    "mean_turnover": bt["turnover"].mean(),
                    "mean_holdings": bt["n_holdings"].mean(),
                }
            )
    return pd.DataFrame(rows)
