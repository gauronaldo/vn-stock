"""Trading calendar, rebalance dates and forward returns (AGENTS.md section 5.1).

Conventions: the signal at rebalance date ``t`` (last session of a month) uses data up to
the close of ``t``. The trade executes at the close of the session after ``t``. The forward
return runs from that close to the close of the session after the next rebalance date, so
consecutive holding periods chain without gaps or overlaps.
"""

from __future__ import annotations

import pandas as pd


def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted market sessions: every date on which at least one symbol has a bar."""
    return pd.DatetimeIndex(sorted(prices["date"].unique()), name="date")


def month_ends(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last session of each calendar month in ``calendar``."""
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.groupby(calendar.to_period("M")).max().to_numpy(), name="date")


def rebalance_dates(calendar: pd.DatetimeIndex, first: str, last: str) -> pd.DatetimeIndex:
    """Month-end sessions between ``first`` and ``last`` (inclusive)."""
    ends = month_ends(calendar)
    return ends[(ends >= pd.Timestamp(first)) & (ends <= pd.Timestamp(last))]


def holding_periods(
    calendar: pd.DatetimeIndex, rebalances: pd.DatetimeIndex, lag: int = 1
) -> pd.DataFrame:
    """Entry and exit session of the holding period that starts at each rebalance date.

    Entry is ``lag`` sessions after ``t``; exit is ``lag`` sessions after the next month end.
    Raises when the calendar does not reach the exit of the last period.
    """
    ends = month_ends(calendar)
    pos = calendar.get_indexer(rebalances)
    end_idx = ends.get_indexer(rebalances)
    if (pos < 0).any() or (end_idx < 0).any():
        raise ValueError("rebalance dates must be month-end sessions of the calendar")
    if len(end_idx) and end_idx.max() + 1 >= len(ends):
        raise ValueError("the calendar has no month end after the last rebalance date")
    exit_pos = calendar.get_indexer(ends[end_idx + 1]) + lag
    if len(exit_pos) and exit_pos.max() >= len(calendar):
        raise ValueError("the calendar ends before the exit of the last holding period")
    return pd.DataFrame(
        {
            "rebalance_date": rebalances,
            "entry_date": calendar[pos + lag],
            "exit_date": calendar[exit_pos],
        }
    )


def close_panel(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide adjusted close (date x symbol) carried forward inside each symbol's own span.

    A session without a bar keeps the last close (no trade, price unchanged). Before the
    first bar and after the last bar the price stays NaN.
    """
    wide = prices.pivot(index="date", columns="symbol", values="close_adj").reindex(calendar)
    filled = wide.ffill()
    last_bar = wide.notna()[::-1].cummax()[::-1]  # True up to each symbol's last bar
    return filled.where(last_bar)


def forward_returns(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    rebalances: pd.DatetimeIndex,
    lag: int = 1,
) -> pd.DataFrame:
    """Long table ``rebalance_date, symbol, fwd_ret_1m`` (simple return, adjusted prices).

    Symbols without a price at entry or exit get NaN; they are dropped from the output.
    """
    periods = holding_periods(calendar, rebalances, lag)
    close = close_panel(prices, calendar)
    entry = close.loc[periods["entry_date"]].to_numpy()
    exit_ = close.loc[periods["exit_date"]].to_numpy()
    ret = pd.DataFrame(exit_ / entry - 1.0, index=periods["rebalance_date"], columns=close.columns)
    out = ret.stack().rename("fwd_ret_1m").reset_index()
    out.columns = ["rebalance_date", "symbol", "fwd_ret_1m"]
    return out.sort_values(["rebalance_date", "symbol"], ignore_index=True)


def split_label(rebalances: pd.Series, holdout_start: str) -> pd.Series:
    """``research`` before ``holdout_start``, ``holdout`` on or after it."""
    return pd.Series(
        ["holdout" if d >= pd.Timestamp(holdout_start) else "research" for d in rebalances],
        index=rebalances.index,
    )
