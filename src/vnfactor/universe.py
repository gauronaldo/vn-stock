"""Rule-based liquid universe at each rebalance date (AGENTS.md section 5.2).

At ``t``, using only sessions up to ``t``:

1. candidates are the symbols in ``prices`` (today's HOSE listing, see the limitations);
2. keep symbols with at least ``min_history_sessions`` bars and with trades (a bar with
   positive volume) on at least ``min_trading_fraction`` of the last ``liquidity_window``
   market sessions;
3. keep the ``top_n`` symbols by median daily traded value (VND) over that window.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FINANCIAL = "financial"
NON_FINANCIAL = "non_financial"
UNKNOWN = "unknown"


def sector_groups(company_types: pd.Series, financial_types: list[str]) -> pd.Series:
    """Map ``company_type`` per symbol to ``financial`` / ``non_financial``."""
    return company_types.map(lambda c: FINANCIAL if c in financial_types else NON_FINANCIAL)


def liquidity_panels(
    prices: pd.DataFrame, calendar: pd.DatetimeIndex, window: int
) -> dict[str, pd.DataFrame]:
    """Wide panels on the market calendar used by the universe filters.

    ``history``: number of bars up to each session. ``traded_share``: share of the last
    ``window`` sessions with positive volume. ``adv``: median traded value over the last
    ``window`` sessions, ignoring sessions without a bar or without a value.
    """
    volume = prices.pivot(index="date", columns="symbol", values="volume").reindex(calendar)
    value = prices.pivot(index="date", columns="symbol", values="value").reindex(calendar)
    has_bar = volume.notna()
    traded = (volume > 0).astype(float)
    return {
        "history": has_bar.cumsum(),
        "traded_share": traded.rolling(window, min_periods=window).sum() / window,
        "adv": value.where(volume > 0).rolling(window, min_periods=1).median(),
    }


def build_universe(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    rebalances: pd.DatetimeIndex,
    settings: dict[str, Any],
    groups: pd.Series,
    members: list[str] | None = None,
) -> pd.DataFrame:
    """Long table ``rebalance_date, symbol, adv_126, sector_group``.

    ``groups`` maps symbol to sector group; symbols without statements get ``unknown``.
    ``members`` optionally restricts the candidates (used for the survivorship check).
    """
    window = int(settings["liquidity_window"])
    panels = liquidity_panels(prices, calendar, window)
    eligible = (panels["history"] >= int(settings["min_history_sessions"])) & (
        panels["traded_share"] >= float(settings["min_trading_fraction"])
    )
    adv = panels["adv"].where(eligible).loc[rebalances]
    if members is not None:
        adv = adv[adv.columns.intersection(members)]
    long = adv.stack().rename("adv_126").reset_index()
    long.columns = ["rebalance_date", "symbol", "adv_126"]
    long = long[np.isfinite(long["adv_126"])]
    long = long.sort_values(["rebalance_date", "adv_126", "symbol"], ascending=[True, False, True])
    out = long.groupby("rebalance_date").head(int(settings["top_n"])).copy()
    out["sector_group"] = out["symbol"].map(groups).fillna(UNKNOWN)
    return out.sort_values(["rebalance_date", "symbol"], ignore_index=True)
