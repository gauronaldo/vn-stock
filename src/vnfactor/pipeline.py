"""Pure orchestration of the build step: raw tables in, research tables out.

No file access here; ``cli.py`` reads and writes the parquet files.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from vnfactor import backtest as bt
from vnfactor import calendar as cal
from vnfactor import evaluation as ev
from vnfactor import factors as fac
from vnfactor import fundamentals as fund
from vnfactor import universe as uni


def build_signals(
    prices: pd.DataFrame,
    statements: pd.DataFrame,
    shares: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    lag_days: int | None = None,
    members: list[str] | None = None,
    last_rebalance: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Universe, point-in-time fundamentals, factors and composite scores.

    ``lag_days`` overrides the reporting lag and ``members`` restricts the candidates; both
    exist for the robustness checks. Only data up to each rebalance date is used.
    """
    days = cal.trading_calendar(prices)
    rebalances = cal.rebalance_dates(
        days,
        cfg["calendar"]["first_rebalance"],
        last_rebalance or cfg["calendar"]["last_rebalance"],
    )
    groups = uni.sector_groups(fund.company_types(statements), cfg["preprocess"]["financial_types"])
    universe = uni.build_universe(prices, days, rebalances, cfg["universe"], groups, members)

    lag = int(cfg["fundamentals"]["lag_days"] if lag_days is None else lag_days)
    fundamentals_q = fund.add_available_date(statements, lag)
    ttm = fund.ttm_table(fundamentals_q, int(cfg["fundamentals"]["ttm_quarters"]))
    pit = fund.asof_join(ttm, universe)

    close = cal.close_panel(prices, days).stack()
    close_t = close.reindex(pd.MultiIndex.from_frame(pit[["rebalance_date", "symbol"]]))
    shares_by_symbol = shares.set_index("symbol")["outstanding_shares"]
    definitions = cfg["factors"]["definitions"]
    raw = fac.price_factors(
        prices, days, rebalances, definitions, float(cfg["factors"]["max_missing_fraction"])
    ).join(fac.fundamental_factors(pit, close_t, shares_by_symbol), how="left")

    factors = fac.preprocess(raw, universe, definitions, cfg["preprocess"])
    scores = fac.composite_scores(
        factors,
        cfg["composite"]["groups"],
        int(cfg["composite"]["min_groups"]),
        int(cfg["evaluation"]["n_quantiles"]),
    )
    return {
        "universe": universe,
        "fundamentals_q": fundamentals_q,
        "factors": factors,
        "scores": scores,
    }


def build_returns(prices: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """``fwd_returns`` for every symbol at every rebalance date."""
    days = cal.trading_calendar(prices)
    rebalances = cal.rebalance_dates(
        days, cfg["calendar"]["first_rebalance"], cfg["calendar"]["last_rebalance"]
    )
    lag = int(cfg["calendar"]["execution_lag_sessions"])
    return cal.forward_returns(prices, days, rebalances, lag)


def period_dates(dates: pd.Series | pd.DatetimeIndex, cfg: dict[str, Any], period: str) -> list:
    """Rebalance dates of ``research`` (before ``split.holdout_start``) or ``holdout``."""
    dates = pd.DatetimeIndex(sorted(pd.unique(pd.Series(dates))))
    cut = pd.Timestamp(cfg["split"]["holdout_start"])
    if period == "research":
        return list(dates[dates < cut])
    if period == "holdout":
        return list(dates[dates >= cut])
    raise ValueError(f"unknown period {period!r}")


def evaluate_period(
    scores: pd.DataFrame,
    factors: pd.DataFrame,
    universe: pd.DataFrame,
    fwd: pd.DataFrame,
    dates: list,
    cfg: dict[str, Any],
    *,
    cost_multiplier: float = 1.0,
    include_factors: bool = True,
) -> dict[str, pd.DataFrame]:
    """Signal statistics and backtests over ``dates``.

    ``scores`` may hold several models (composite, ridge, lightgbm); each gets a top-quintile
    backtest. Single factors get signal statistics only. The benchmark is the equal-weight
    universe. Returns ``signals``, ``ic_monthly``, ``backtest`` and ``returns`` tables.
    """
    n_q = int(cfg["evaluation"]["n_quantiles"])
    fee = float(cfg["backtest"]["fee_rate"]) * cost_multiplier
    tax = float(cfg["backtest"]["sell_tax_rate"]) * cost_multiplier
    ppy = int(cfg["backtest"]["periods_per_year"])

    def sel(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["rebalance_date"].isin(dates)]

    model_scores = sel(scores)
    parts = [model_scores]
    if include_factors:
        parts.insert(0, fac.factor_scores(sel(factors), n_q))
    all_scores = pd.concat(parts, ignore_index=True)
    ic_monthly = ev.rank_ic(all_scores, fwd)
    signals = ev.signal_table(all_scores, fwd)

    runs = {
        model: bt.run_backtest(bt.top_quantile_weights(g, n_q), fwd, fee, tax)
        for model, g in model_scores.groupby("model", sort=True)
    }
    runs["benchmark"] = bt.run_backtest(bt.equal_weights(sel(universe)), fwd, fee, tax)
    returns = pd.concat([r.assign(strategy=name) for name, r in runs.items()], ignore_index=True)
    return {
        "signals": signals.reset_index(),
        "ic_monthly": ic_monthly,
        "backtest": bt.summarize(runs, "benchmark", ppy),
        "returns": returns,
    }


def current_liquid_basket(
    prices: pd.DataFrame, statements: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.Timestamp, list[str]]:
    """Survivorship stand-in for "today's VN100": the universe rule applied on the last
    session only. Returns the as-of date and the member symbols.
    """
    days = cal.trading_calendar(prices)
    last = days[-1]
    groups = uni.sector_groups(fund.company_types(statements), cfg["preprocess"]["financial_types"])
    basket = uni.build_universe(prices, days, pd.DatetimeIndex([last]), cfg["universe"], groups)
    return last, sorted(basket["symbol"])


def robustness_row(
    name: str, result: dict[str, pd.DataFrame], universe: pd.DataFrame, base_universe: pd.DataFrame
) -> dict[str, Any]:
    """One summary row: composite signal quality and net backtest against its benchmark."""
    sig = result["signals"].set_index("model").loc["composite"]
    net = result["backtest"][result["backtest"]["basis"] == "net"].set_index("strategy")
    pairs = universe.merge(base_universe, on=["rebalance_date", "symbol"])
    overlap = (
        pairs.groupby("rebalance_date").size() / base_universe.groupby("rebalance_date").size()
    )
    return {
        "variant": name,
        "mean_ic": sig["mean_ic"],
        "ic_t_stat": sig["t_stat"],
        "spread": sig["spread"],
        "spread_t": sig["spread_t"],
        "top_cagr_net": net.loc["composite", "cagr"],
        "benchmark_cagr_net": net.loc["benchmark", "cagr"],
        "excess_cagr_net": net.loc["composite", "excess_cagr"],
        "top_sharpe_net": net.loc["composite", "sharpe"],
        "mean_turnover": net.loc["composite", "mean_turnover"],
        "universe_size": universe.groupby("rebalance_date").size().mean(),
        "overlap_with_base": overlap.reindex(base_universe["rebalance_date"].unique())
        .fillna(0)
        .mean(),
    }
