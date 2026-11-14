"""Factor definitions, cross-sectional preprocessing and the composite score.

AGENTS.md sections 5.4 and 5.5. ``P`` is the adjusted close (thousand VND), indexed by
market sessions; a session without a bar keeps the last close. A window that misses more
than ``max_missing_fraction`` of its sessions gives NaN. Every value at rebalance date
``t`` uses prices up to ``t`` and statements with ``available_date <= t`` only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from vnfactor.calendar import close_panel

PRICE_FACTORS = ("mom_12_1", "rev_1m", "vol_126", "adv_63")
FUNDAMENTAL_FACTORS = ("ep", "bp", "roe")
FACTOR_COLUMNS = ["rebalance_date", "symbol", "factor", "raw", "z"]
SCORE_COLUMNS = ["rebalance_date", "symbol", "model", "score", "quintile"]


def _min_count(window: int, max_missing: float) -> int:
    return math.ceil(window * (1.0 - max_missing) - 1e-9)


def price_factors(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    rebalances: pd.DatetimeIndex,
    definitions: dict[str, dict[str, Any]],
    max_missing: float,
) -> pd.DataFrame:
    """Wide-by-factor table indexed by ``(rebalance_date, symbol)`` with raw price factors.

    * ``mom_12_1 = P[t-skip] / P[t-lookback] - 1``
    * ``rev_1m = P[t] / P[t-lookback] - 1``
    * ``vol_126``: standard deviation of daily log returns over ``window`` sessions
    * ``adv_63``: log of the median daily traded value (VND) over ``window`` sessions
    """
    close = close_panel(prices, calendar)
    has_bar = (
        prices.pivot(index="date", columns="symbol", values="close_adj").reindex(calendar).notna()
    )
    value = prices.pivot(index="date", columns="symbol", values="value").reindex(calendar)
    pos = calendar.get_indexer(rebalances)
    if (pos < 0).any():
        raise ValueError("rebalance dates must be sessions of the calendar")
    bars = np.vstack([np.zeros((1, has_bar.shape[1])), has_bar.to_numpy().cumsum(axis=0)])
    p = close.to_numpy()

    def bars_between(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Number of bars in sessions [start, end] per rebalance row."""
        return bars[end + 1] - bars[start]

    def lagged(k: int) -> np.ndarray:
        idx = pos - k
        out = np.full((len(pos), p.shape[1]), np.nan)
        ok = idx >= 0
        out[ok] = p[idx[ok]]
        return out, np.where(ok, idx, 0), ok

    out: dict[str, np.ndarray] = {}
    if "mom_12_1" in definitions:
        spec = definitions["mom_12_1"]
        lookback, skip = int(spec["lookback"]), int(spec["skip"])
        start, start_idx, ok = lagged(lookback)
        end, _, _ = lagged(skip)
        count = bars_between(start_idx, pos - skip)
        need = _min_count(lookback - skip + 1, max_missing)
        out["mom_12_1"] = np.where(ok[:, None] & (count >= need), end / start - 1.0, np.nan)
    if "rev_1m" in definitions:
        lookback = int(definitions["rev_1m"]["lookback"])
        start, start_idx, ok = lagged(lookback)
        count = bars_between(start_idx, pos)
        need = _min_count(lookback + 1, max_missing)
        out["rev_1m"] = np.where(ok[:, None] & (count >= need), p[pos] / start - 1.0, np.nan)
    if "vol_126" in definitions:
        window = int(definitions["vol_126"]["window"])
        log_ret = np.log(close).diff().where(has_bar)
        vol = log_ret.rolling(window, min_periods=_min_count(window, max_missing)).std()
        out["vol_126"] = vol.to_numpy()[pos]
    if "adv_63" in definitions:
        window = int(definitions["adv_63"]["window"])
        med = value.where(has_bar).rolling(window, min_periods=_min_count(window, max_missing))
        adv = med.median().to_numpy()[pos]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["adv_63"] = np.where(adv > 0, np.log(adv), np.nan)

    index = pd.MultiIndex.from_product(
        [rebalances, close.columns], names=["rebalance_date", "symbol"]
    )
    return pd.DataFrame({k: v.reshape(-1) for k, v in out.items()}, index=index)


def fundamental_factors(pit: pd.DataFrame, close_t: pd.Series, shares: pd.Series) -> pd.DataFrame:
    """``ep``, ``bp``, ``roe`` from point-in-time statement rows.

    ``pit`` holds ``rebalance_date, symbol`` and the TTM columns of ``fundamentals``.
    ``close_t`` is the adjusted close at ``t`` (thousand VND) indexed like ``pit``;
    ``shares`` is the current outstanding share count per symbol. Market cap (VND) =
    ``close_t * 1000 * shares``. Equity <= 0 makes ``bp`` and ``roe`` NaN.
    """
    mcap = close_t.to_numpy() * 1000.0 * pit["symbol"].map(shares).to_numpy(dtype=float)
    equity = pit["equity"].to_numpy(dtype=float)
    equity_avg = pit["equity_avg"].to_numpy(dtype=float)
    positive = (equity > 0) & (equity_avg > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ep = pit["npat_parent_ttm"].to_numpy(dtype=float) / mcap
        bp = np.where(equity > 0, equity / mcap, np.nan)
        roe = np.where(positive, pit["npat_total_ttm"].to_numpy(dtype=float) / equity_avg, np.nan)
    index = pd.MultiIndex.from_frame(pit[["rebalance_date", "symbol"]])
    return pd.DataFrame({"ep": ep, "bp": bp, "roe": roe}, index=index)


def _winsorized_z(x: pd.Series, lower: float, upper: float) -> pd.Series:
    valid = x.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=x.index)
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    clipped = x.clip(lo, hi)
    std = clipped.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=x.index)
    return (clipped - clipped.mean()) / std


def preprocess(
    raw: pd.DataFrame,
    universe: pd.DataFrame,
    definitions: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Long ``factors`` table (``FACTOR_COLUMNS``) for the universe members.

    At each date: winsorize at ``winsor_lower``/``winsor_upper`` quantiles, z-score, then
    multiply by the hypothesised sign. Factors in ``group_zscore_factors`` are processed
    within each ``sector_group``. NaN stays NaN.
    """
    lower, upper = float(settings["winsor_lower"]), float(settings["winsor_upper"])
    grouped = set(settings.get("group_zscore_factors", []))
    keys = universe[["rebalance_date", "symbol", "sector_group"]]
    wide = keys.merge(raw.reset_index(), on=["rebalance_date", "symbol"], how="left")
    frames = []
    for factor, spec in definitions.items():
        part = wide[["rebalance_date", "symbol", "sector_group"]].copy()
        part["factor"] = factor
        part["raw"] = wide[factor].astype(float)
        by = ["rebalance_date", "sector_group"] if factor in grouped else ["rebalance_date"]
        z = part.groupby(by, group_keys=False)["raw"].apply(
            lambda s: _winsorized_z(s, lower, upper)
        )
        part["z"] = z.reindex(part.index) * int(spec["sign"])
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)[FACTOR_COLUMNS]
    return out.sort_values(["rebalance_date", "symbol", "factor"], ignore_index=True)


def assign_quantiles(scores: pd.DataFrame, n: int) -> pd.Series:
    """Quantile 1..n of ``score`` within each date (n = highest); NaN scores stay NaN."""

    def one(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        out = pd.Series(np.nan, index=s.index)
        if len(valid) < n:
            return out
        ranks = valid.rank(method="first")
        out.loc[valid.index] = pd.qcut(ranks, n, labels=False) + 1
        return out

    return (
        scores.groupby("rebalance_date", group_keys=False)["score"].apply(one).reindex(scores.index)
    )


def composite_scores(
    factors: pd.DataFrame, groups: dict[str, list[str]], min_groups: int, n_quantiles: int
) -> pd.DataFrame:
    """Equal-weight composite: mean of group scores, each the mean of its members' z.

    Symbols with fewer than ``min_groups`` available groups get no score.
    """
    z = factors.set_index(["rebalance_date", "symbol", "factor"])["z"].unstack("factor")
    group_scores = pd.DataFrame(
        {name: z[members].mean(axis=1, skipna=True) for name, members in groups.items()}
    )
    available = group_scores.notna().sum(axis=1)
    score = group_scores.mean(axis=1, skipna=True).where(available >= int(min_groups))
    out = score.rename("score").reset_index()
    out["model"] = "composite"
    out = out.dropna(subset=["score"]).reset_index(drop=True)
    out["quintile"] = assign_quantiles(out, n_quantiles)
    return out[SCORE_COLUMNS].sort_values(["rebalance_date", "symbol"], ignore_index=True)


def factor_scores(factors: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """Each single factor's signed z as a score table, ``model`` = factor name."""
    out = factors.dropna(subset=["z"]).rename(columns={"z": "score", "factor": "model"})
    frames = []
    for _, part in out.groupby("model", sort=True):
        part = part[["rebalance_date", "symbol", "model", "score"]].reset_index(drop=True)
        part["quintile"] = assign_quantiles(part, n_quantiles)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)[SCORE_COLUMNS]
