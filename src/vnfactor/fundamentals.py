"""Point-in-time financial statement data (AGENTS.md section 5.3).

``available_date = period_end + lag_days``. At rebalance date ``t`` only quarters with
``available_date <= t`` are used, through an as-of join on ``available_date``.
TTM = sum of the latest ``ttm_quarters`` consecutive quarters; NaN when one is missing.
Average equity = mean of equity at the latest quarter end and ``ttm_quarters`` earlier.
Values are in VND.
"""

from __future__ import annotations

import pandas as pd

ITEMS = ("equity", "npat_parent", "npat_total")
PIT_COLUMNS = ["npat_parent_ttm", "npat_total_ttm", "equity", "equity_avg"]


def add_available_date(statements: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """``fundamentals_q`` table: statements plus ``available_date``."""
    out = statements.copy()
    out["available_date"] = out["period_end"] + pd.Timedelta(days=int(lag_days))
    cols = ["symbol", "report_period", "item_id", "period_end", "available_date", "value"]
    return out[[*cols, "company_type"]]


def company_types(statements: pd.DataFrame) -> pd.Series:
    """Latest ``company_type`` per symbol."""
    latest = statements.sort_values("period_end").groupby("symbol")["company_type"].last()
    return latest.rename("company_type")


def ttm_table(fundamentals: pd.DataFrame, ttm_quarters: int = 4) -> pd.DataFrame:
    """One row per reported quarter: TTM earnings, equity and average equity.

    Columns: ``symbol, period_end, available_date`` and ``PIT_COLUMNS``. A gap in the quarter
    sequence or a missing value makes the affected TTM and average NaN.
    """
    keys = ["symbol", "period_end", "item_id"]
    if fundamentals.duplicated(keys).any():
        raise ValueError("fundamentals hold duplicate (symbol, period_end, item_id) rows")
    wide = fundamentals.set_index(keys)["value"].unstack("item_id")  # keeps NaN values
    wide = wide.reindex(columns=list(ITEMS))
    available = fundamentals.groupby(["symbol", "period_end"])["available_date"].first()
    frames = []
    for symbol, g in wide.groupby(level="symbol"):
        g = g.droplevel("symbol")
        quarters = g.index.to_period("Q")
        full = pd.period_range(quarters.min(), quarters.max(), freq="Q")
        q = g.set_axis(quarters).reindex(full)  # missing quarters become NaN rows
        n = int(ttm_quarters)
        out = pd.DataFrame(
            {
                "npat_parent_ttm": q["npat_parent"].rolling(n, min_periods=n).sum(),
                "npat_total_ttm": q["npat_total"].rolling(n, min_periods=n).sum(),
                "equity": q["equity"],
                "equity_avg": (q["equity"] + q["equity"].shift(n)) / 2.0,
            }
        )
        out = out.loc[quarters]  # keep reported quarters only
        out.index = g.index
        out.insert(0, "symbol", symbol)
        frames.append(out.reset_index())
    table = pd.concat(frames, ignore_index=True)
    table["available_date"] = table.set_index(["symbol", "period_end"]).index.map(available)
    return table[["symbol", "period_end", "available_date", *PIT_COLUMNS]]


def asof_join(table: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """For each ``(rebalance_date, symbol)`` in ``keys``, the latest row of ``table`` with
    ``available_date <= rebalance_date``. Symbols with no such row get NaN.
    """
    left = keys[["rebalance_date", "symbol"]].sort_values("rebalance_date")
    right = table.sort_values("available_date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="rebalance_date",
        right_on="available_date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values(["rebalance_date", "symbol"], ignore_index=True)
