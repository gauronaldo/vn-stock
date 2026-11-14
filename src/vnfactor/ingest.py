"""Download, cache and assemble the raw tables (AGENTS.md section 7.3).

Order: index constituents -> daily bars -> matched traded value in chunks of at most
``max_range_years`` -> ``value`` column -> ADV ranking -> statements and current shares for
symbols that were ever in the top ``fundamentals_top_n`` by ADV.

Requests are sequential and spaced, and only sent inside the configured Vietnam-time
window. Every response is cached in one parquet file per symbol, so an interrupted run
resumes where it stopped and a completed run makes no network call.

A failure for one symbol is recorded and the run moves on, but the processed tables are
written only when no symbol failed. Nothing is replaced by an empty table.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vnfactor import client

VN_TIME = timezone(timedelta(hours=7))  # Vietnam has no daylight saving time
PRICE_COLUMNS = [*client.BAR_COLUMNS, "value", "value_source"]
STATEMENTS = ("balance_sheet", "income_statement")


class IngestStop(RuntimeError):
    """The run cannot continue now; cached progress is kept and the run can be resumed."""


def vietnam_now() -> datetime:
    """Current wall-clock time in Vietnam (UTC+7)."""
    return datetime.now(VN_TIME)


# --- pure helpers -------------------------------------------------------------------------


def date_chunks(start: str, end: str, max_years: int) -> list[tuple[str, str]]:
    """Split the inclusive range [start, end] into pieces shorter than ``max_years`` years."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    while first <= last:
        stop = min(first + pd.DateOffset(years=max_years) - pd.Timedelta(days=1), last)
        chunks.append((first.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        first = stop + pd.Timedelta(days=1)
    return chunks


def _merge_traded(bars: pd.DataFrame, traded_value: pd.DataFrame) -> pd.DataFrame:
    if len(traded_value) == 0:
        df = bars.assign(matched_value=np.nan)
        return df.sort_values(["symbol", "date"], ignore_index=True)
    df = bars.merge(
        traded_value[["date", "symbol", "matched_value"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    return df.sort_values(["symbol", "date"], ignore_index=True)


def _anchor_ratio(df: pd.DataFrame, anchor_sessions: int) -> pd.Series:
    """Per row: the anchor factor k of the row's symbol (see ``anchor_factors``)."""
    notional = df["close_adj"] * 1000.0 * df["volume"]
    usable = (df["matched_value"] > 0) & (notional > 0)
    in_anchor = usable & (usable.groupby(df["symbol"]).cumsum() <= anchor_sessions)
    ratio = (notional / df["matched_value"]).where(in_anchor)
    return ratio.groupby(df["symbol"]).transform("median")


def anchor_factors(
    bars: pd.DataFrame, traded_value: pd.DataFrame, anchor_sessions: int
) -> pd.Series:
    """Anchor factor k per symbol, indexed by symbol.

    k is the median of ``close_adj * 1000 * volume / matched_value`` over the first
    ``anchor_sessions`` sessions of the symbol where both are positive. It measures how far
    the adjusted close sits below the traded price at the anchor. NaN without such sessions.
    """
    df = _merge_traded(bars, traded_value)
    k = _anchor_ratio(df, anchor_sessions)
    return k.groupby(df["symbol"]).first()


def build_prices(
    bars: pd.DataFrame, traded_value: pd.DataFrame, anchor_sessions: int
) -> pd.DataFrame:
    """Join bars with matched value and add ``value`` (VND) and ``value_source``.

    From the first session of a symbol that has a matched value: ``value`` is the matched
    value of the source (``matched``); NaN when the source has no record for that session.
    Before it: ``close_adj * 1000 * volume / k`` (``proxy``), k from ``anchor_factors``;
    NaN when k cannot be estimated. Dividing by k removes price adjustments made after the
    anchor; adjustments for events before the anchor remain.
    """
    df = _merge_traded(bars, traded_value)
    notional = df["close_adj"] * 1000.0 * df["volume"]
    k = _anchor_ratio(df, anchor_sessions)
    first_matched = (
        df["date"].where(df["matched_value"].notna()).groupby(df["symbol"]).transform("min")
    )
    is_proxy = first_matched.isna() | (df["date"] < first_matched)
    df["value"] = np.where(is_proxy, notional / k, df["matched_value"]).astype(float)
    df["value_source"] = np.where(is_proxy, "proxy", "matched")
    return df.sort_values(["date", "symbol"], ignore_index=True)[PRICE_COLUMNS]


def symbols_ever_in_top_n(
    prices: pd.DataFrame,
    top_n: int,
    window: int,
    min_periods: int,
    first_date: str,
    last_date: str,
) -> list[str]:
    """Symbols ranked in the top ``top_n`` by rolling median ``value`` at any month end.

    The median covers the last ``window`` market sessions and needs ``min_periods`` values.
    Month ends are the last sessions of each month between ``first_date`` and ``last_date``.
    This only decides which symbols get statements downloaded; the universe is built later.
    """
    wide = prices.pivot(index="date", columns="symbol", values="value").sort_index()
    adv = wide.rolling(window, min_periods=min_periods).median()
    month_end = adv.groupby(adv.index.to_period("M")).tail(1).loc[first_date:last_date]
    ranks = month_end.rank(axis=1, ascending=False, method="first")
    ever = (ranks <= top_n).any(axis=0)
    return sorted(ever.index[ever])


def summarize_prices(prices: pd.DataFrame, daily_limit: float, tolerance: float) -> dict:
    """Aggregate quality figures for the assembled ``prices_daily`` table.

    ``missing_bar_ratio``: share of market sessions between the first and last bar of each
    symbol that have no bar. ``limit_breaches``: per symbol, the number of close-to-close
    returns beyond ``daily_limit + tolerance`` in absolute value.
    """
    calendar = np.sort(prices["date"].unique())
    span = prices.groupby("symbol")["date"].agg(["min", "max", "size"])
    expected = np.searchsorted(calendar, span["max"].to_numpy(), side="right") - np.searchsorted(
        calendar, span["min"].to_numpy(), side="left"
    )
    ordered = prices.sort_values(["symbol", "date"])
    ret = ordered.groupby("symbol")["close_adj"].pct_change()
    breach = ret.abs() > daily_limit + tolerance
    breaches = breach.groupby(ordered["symbol"]).sum()
    return {
        "rows": int(len(prices)),
        "symbols": int(len(span)),
        "first_date": pd.Timestamp(calendar[0]).strftime("%Y-%m-%d"),
        "last_date": pd.Timestamp(calendar[-1]).strftime("%Y-%m-%d"),
        "sessions": int(len(calendar)),
        "missing_bar_ratio": float(1.0 - span["size"].sum() / expected.sum()),
        "missing_value_ratio": float(prices["value"].isna().mean()),
        "proxy_row_ratio": float((prices["value_source"] == "proxy").mean()),
        "zero_volume_rows": int((prices["volume"] == 0).sum()),
        "limit_breach_rows": int(breach.sum()),
        "limit_breaches": {s: int(n) for s, n in breaches[breaches > 0].items()},
    }


# --- network pacing -----------------------------------------------------------------------


class Fetcher:
    """Sequential, spaced calls with backoff on retryable source errors.

    Before every attempt it checks the Vietnam-time window and waits ``interval`` seconds.
    Only errors that ``describe_error`` marks retryable (rate limit, network) are retried,
    with a wait of ``backoff_base * 2**attempt`` seconds, at most ``max_retries`` times.
    Any other exception propagates unchanged.
    """

    def __init__(
        self,
        settings: dict[str, Any],
        describe_error: Callable[[BaseException], dict[str, Any] | None],
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = vietnam_now,
    ) -> None:
        self.interval = float(settings["request_interval_seconds"])
        self.max_retries = int(settings["max_retries"])
        self.backoff_base = float(settings["backoff_base_seconds"])
        self.earliest_hour = int(settings["earliest_local_hour"])
        self.latest_hour = int(settings["latest_local_hour"])
        self.describe_error = describe_error
        self.sleep = sleep
        self.now = now
        self.calls = 0

    def _check_window(self) -> None:
        now = self.now()
        if not self.earliest_hour <= now.hour < self.latest_hour:
            raise IngestStop(
                f"outside the download window {self.earliest_hour}:00-{self.latest_hour}:00 "
                f"Vietnam time (now {now:%H:%M})"
            )

    def call(self, fn: Callable[..., pd.DataFrame], *args: Any) -> pd.DataFrame:
        """Run ``fn(*args)`` under the pacing and retry policy."""
        attempt = 0
        while True:
            self._check_window()
            self.sleep(self.interval)
            self.calls += 1
            try:
                return fn(*args)
            except Exception as exc:
                info = self.describe_error(exc)
                if info is None or not info["retryable"]:
                    raise
                if attempt >= self.max_retries:
                    raise IngestStop(
                        f"{info['error_class']} persists after {self.max_retries} retries"
                    ) from exc
                self.sleep(self.backoff_base * 2**attempt)
                attempt += 1


class _Failures:
    """Per-symbol failures; too many in a row means a systemic problem and stops the run."""

    def __init__(self, max_consecutive: int) -> None:
        self.max_consecutive = max_consecutive
        self.items: list[tuple[str, str, str]] = []
        self.streak = 0

    def ok(self) -> None:
        self.streak = 0

    def add(self, symbol: str, stage: str, reason: str) -> None:
        self.items.append((symbol, stage, reason))
        self.streak += 1
        if self.streak >= self.max_consecutive:
            raise IngestStop(
                f"{self.streak} symbols failed in a row at stage {stage!r}; "
                f"last: {symbol}: {reason}"
            )


# --- file cache ---------------------------------------------------------------------------


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _read_all(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    frames = [df for df in (pd.read_parquet(p) for p in paths) if len(df) > 0]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


# --- the run ------------------------------------------------------------------------------


def run(
    cfg: dict[str, Any],
    refresh: bool = False,
    *,
    api: Any = client,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = vietnam_now,
    log: Callable[[str], None] = print,
) -> int:
    """Run the ingest step; return the process exit code (0 only when complete).

    ``api`` must offer the ``fetch_*`` functions and ``describe_error`` of ``client``.
    """
    fetcher = Fetcher(cfg["ingest"], api.describe_error, sleep=sleep, now=now)
    try:
        code = _run(cfg, refresh, api, fetcher, log)
    except IngestStop as exc:
        log(f"ingest stopped: {exc}")
        log("progress is cached; run the same command again to resume")
        code = 1
    log(f"network calls: {fetcher.calls}")
    return code


def _run(
    cfg: dict[str, Any],
    refresh: bool,
    api: Any,
    fetcher: Fetcher,
    log: Callable[[str], None],
) -> int:
    ing, src, uni = cfg["ingest"], cfg["sources"], cfg["universe"]
    raw, processed = Path(cfg["paths"]["raw_dir"]), Path(cfg["paths"]["processed_dir"])
    start, end = ing["start_date"], ing["end_date"]
    true_value_start = pd.Timestamp(cfg["traded_value"]["true_value_start"])
    failures = _Failures(int(ing["max_consecutive_failures"]))

    def attempt(symbol: str, stage: str, work: Callable[[str], None]) -> None:
        """Run ``work(symbol)``; record source and schema failures, stop on credential ones."""
        try:
            work(symbol)
        except IngestStop:
            raise
        except Exception as exc:
            info = api.describe_error(exc)
            if info is not None and info["credential"]:
                raise IngestStop(
                    f"credential problem ({info['error_class']}); run `vnstock keys check`"
                ) from exc
            if info is None and not isinstance(exc, client.DataError):
                raise
            name = info["error_class"] if info is not None else type(exc).__name__
            failures.add(symbol, stage, f"{name}: {exc}")
        else:
            failures.ok()

    # 1. candidates: current index constituents on the exchange
    listing_path = raw / "listing.parquet"
    if refresh or not listing_path.exists():
        listing = fetcher.call(
            api.fetch_candidates, uni["candidates_index"], uni["exchange"], src["listing"]
        )
        if len(listing) == 0:
            raise client.DataError("the candidate listing is empty")
        _write(listing, listing_path)
    listing = pd.read_parquet(listing_path)
    skip = set(ing.get("skip_symbols") or [])
    symbols = [s for s in listing["symbol"] if s not in skip]
    log(f"candidates: {len(listing)} listed, {len(listing) - len(symbols)} skipped by config")

    # 2. daily bars, then matched traded value for the part of each history that has it
    def bars_path(symbol: str) -> Path:
        return raw / "bars" / f"{symbol}.parquet"

    def traded_path(symbol: str) -> Path:
        return raw / "traded_value" / f"{symbol}.parquet"

    def prices_for(symbol: str) -> None:
        if refresh or not bars_path(symbol).exists():
            bars = fetcher.call(api.fetch_prices, symbol, start, end, src["prices"])
            _write(bars, bars_path(symbol))
        if refresh or not traded_path(symbol).exists():
            bars = pd.read_parquet(bars_path(symbol))
            parts = []
            if len(bars) > 0 and bars["date"].max() >= true_value_start:
                first = max(bars["date"].min(), true_value_start).strftime("%Y-%m-%d")
                for a, b in date_chunks(first, end, int(ing["max_range_years"])):
                    parts.append(
                        fetcher.call(api.fetch_traded_value, symbol, a, b, src["traded_value"])
                    )
            parts = [p for p in parts if len(p) > 0]
            if parts:
                traded = pd.concat(parts, ignore_index=True)
            else:
                traded = pd.DataFrame(columns=client.TRADED_VALUE_COLUMNS)
            if traded["date"].duplicated().any():
                raise client.DataError(f"{symbol} traded value: duplicate dates across chunks")
            _write(traded, traded_path(symbol))

    for i, symbol in enumerate(symbols, start=1):
        attempt(symbol, "prices", prices_for)
        if i % 25 == 0 or i == len(symbols):
            log(f"prices: {i}/{len(symbols)} symbols, {fetcher.calls} calls")
    if failures.items:
        return _report_failures(failures, log)

    bars = _read_all([bars_path(s) for s in symbols], client.BAR_COLUMNS)
    traded = _read_all([traded_path(s) for s in symbols], client.TRADED_VALUE_COLUMNS)
    if len(bars) == 0:
        raise client.DataError("no daily bars were returned for any symbol")
    anchor_sessions = int(cfg["traded_value"]["anchor_sessions"])
    prices = build_prices(bars, traded, anchor_sessions)
    _write(prices, processed / "prices_daily.parquet")
    log(f"prices_daily: {len(bars)} bars + {len(traded)} traded-value rows -> {len(prices)} rows")

    # 3. statements and current shares for symbols ever in the top N by ADV
    window = int(uni["liquidity_window"])
    selected = symbols_ever_in_top_n(
        prices,
        top_n=int(ing["fundamentals_top_n"]),
        window=window,
        min_periods=math.ceil(window * float(uni["min_trading_fraction"])),
        first_date=cfg["calendar"]["first_rebalance"],
        last_date=cfg["calendar"]["last_rebalance"],
    )
    log(f"fundamentals: {len(selected)} symbols were ever in the top {ing['fundamentals_top_n']}")

    def statements_path(symbol: str) -> Path:
        return raw / "fundamentals" / f"{symbol}.parquet"

    def shares_path(symbol: str) -> Path:
        return raw / "shares" / f"{symbol}.parquet"

    def fundamentals_for(symbol: str) -> None:
        if refresh or not statements_path(symbol).exists():
            parts = [
                fetcher.call(api.fetch_statement, symbol, statement, src["fundamentals"])
                for statement in STATEMENTS
            ]
            _write(pd.concat(parts, ignore_index=True), statements_path(symbol))
        if refresh or not shares_path(symbol).exists():
            _write(fetcher.call(api.fetch_shares, symbol, src["shares"]), shares_path(symbol))

    for i, symbol in enumerate(selected, start=1):
        attempt(symbol, "fundamentals", fundamentals_for)
        if i % 25 == 0 or i == len(selected):
            log(f"fundamentals: {i}/{len(selected)} symbols, {fetcher.calls} calls")
    if failures.items:
        return _report_failures(failures, log)

    statements = _read_all([statements_path(s) for s in selected], client.STATEMENT_COLUMNS)
    shares = _read_all([shares_path(s) for s in selected], client.SHARES_COLUMNS)
    _write(statements, processed / "statements_q.parquet")
    _write(shares, processed / "shares_current.parquet")
    log(f"statements_q: {len(statements)} rows, {statements['symbol'].nunique()} symbols")
    log(f"shares_current: {len(shares)} rows")

    _log_summary(prices, anchor_factors(bars, traded, anchor_sessions), cfg, log)
    return 0


def _report_failures(failures: _Failures, log: Callable[[str], None]) -> int:
    log(f"{len(failures.items)} symbol(s) failed; processed tables were not written:")
    for symbol, stage, reason in failures.items:
        log(f"  {symbol} [{stage}] {reason}")
    log("run again to retry them, or list them under ingest.skip_symbols with a reason")
    return 1


def _log_summary(
    prices: pd.DataFrame, k: pd.Series, cfg: dict[str, Any], log: Callable[[str], None]
) -> None:
    ing = cfg["ingest"]
    limit, tol = float(ing["daily_limit"]), float(ing["daily_limit_tolerance"])
    s = summarize_prices(prices, limit, tol)
    log("summary")
    log(f"  symbols: {s['symbols']}, rows: {s['rows']}, sessions: {s['sessions']}")
    log(f"  date range: {s['first_date']} .. {s['last_date']}")
    log(f"  missing bars inside the listed span of each symbol: {s['missing_bar_ratio']:.2%}")
    log(f"  rows without traded value: {s['missing_value_ratio']:.2%}")
    log(f"  rows with proxy traded value: {s['proxy_row_ratio']:.2%}")
    log(f"  zero-volume rows: {s['zero_volume_rows']}")
    if k.notna().any():
        q = k.quantile([0.05, 0.5, 0.95])
        log(
            f"  anchor factor k: p5 {q[0.05]:.3f}, median {q[0.5]:.3f}, p95 {q[0.95]:.3f}, "
            f"undefined for {int(k.isna().sum())} symbols"
        )
    breaches = s["limit_breaches"]
    log(
        f"  daily returns beyond +/-{limit:.0%} (tolerance {tol:.1%}): "
        f"{s['limit_breach_rows']} rows in {len(breaches)} symbols"
    )
    if breaches:
        ranked = sorted(breaches.items(), key=lambda item: (-item[1], item[0]))
        log("  " + ", ".join(f"{symbol}({n})" for symbol, n in ranked))
