"""One-off data audit (phase 1, AGENTS.md section 4). Run by hand, 15:00-21:00 Vietnam time.

Answers A1-A4 with real calls on six symbols and writes a JSON summary of
evidence (row counts, date ranges, column names, item codes, aggregate
statistics) to ``data/audit/audit_summary.json``. ``data/`` is git-ignored;
no raw market data leaves it.

Usage (PowerShell, repo root, venv active):
    python scripts/data_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from vnfactor import client
from vnfactor.config import DEFAULT_CONFIG_PATH, load_config

SYMBOLS = ["FPT", "HPG", "VCB", "SSI", "BVH", "HSG"]
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEEP_START = "2000-01-01"
OUT_PATH = Path("data/audit/audit_summary.json")
PRICE_META = (
    "source",
    "request_count",
    "truncated",
    "last_bar_partial",
    "column_basis",
    "rows",
    "empty_result",
    "extra_columns",
    "source_columns",
)
STATEMENT_META = ("company_type", "template", "taxonomy", "source", "unmapped_codes")


class Caller:
    """Sequential calls with a fixed pause and backoff on retryable vnstock errors."""

    def __init__(self, interval: float, max_retries: int, backoff_base: float) -> None:
        self.interval = interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.n_calls = 0

    def __call__(
        self, label: str, fn: Callable[..., pd.DataFrame], *args: Any, **kwargs: Any
    ) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
        """Return ``(df, None)``, or ``(None, error_info)`` for a non-credential vnstock
        error so the audit can record it as evidence. Credential errors and any
        non-vnstock exception propagate and stop the run."""
        for attempt in range(self.max_retries + 1):
            time.sleep(self.interval)
            self.n_calls += 1
            try:
                df = fn(*args, **kwargs)
            except Exception as exc:
                info = client.describe_error(exc)
                if info is None or info["credential"]:
                    raise
                if info["retryable"] and attempt < self.max_retries:
                    wait = self.backoff_base * 2**attempt
                    print(f"  retry {label}: {info['error_class']}, waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                print(f"  fail  {label}: {info['error_class']} {info['code']}", flush=True)
                return None, info
            print(f"  ok    {label}: {len(df)} rows", flush=True)
            return df, None
        raise AssertionError("unreachable")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _attrs(df: pd.DataFrame, keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: _jsonable(df.attrs[k]) for k in keys if k in df.attrs}


def _naive_day(values: pd.Series) -> pd.Series:
    t = pd.to_datetime(values)
    if t.dt.tz is not None:
        t = t.dt.tz_convert(VN_TZ).dt.tz_localize(None)
    return t.dt.normalize()


def _time_info(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if len(df) == 0:
        return {"rows": 0}
    t = _naive_day(df[col])
    first, last = t.min(), t.max()
    return {
        "rows": int(len(df)),
        "first": str(first.date()),
        "last": str(last.date()),
        "years": round((last - first).days / 365.25, 2),
        "duplicate_days": int(t.duplicated().sum()),
    }


def _frame_summary(df: pd.DataFrame, time_col: str) -> dict[str, Any]:
    return {
        **_time_info(df, time_col),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "attrs": _attrs(df, PRICE_META),
    }


NOT_ENABLED = {"skipped": "source not enabled (no key on this machine)"}


def audit_prices(call: Caller, symbol: str, end: str, sources: set[str]) -> dict[str, Any]:
    """A1 (history depth) and A2 (unadjusted close, traded value) for one symbol."""
    out: dict[str, Any] = {}
    bars: dict[str, pd.DataFrame] = {}
    for source in ("ssi", "fhsc"):
        if source not in sources:
            out[f"ohlcv_{source}"] = NOT_ENABLED
            continue
        df, err = call(
            f"{symbol} ohlcv {source}",
            client.ohlcv,
            symbol,
            DEEP_START,
            end,
            source,
            columns="all",
        )
        out[f"ohlcv_{source}"] = {"error": err} if err else _frame_summary(df, "time")
        if df is not None and len(df):
            bars[source] = df
            if "volume" in df.columns:
                out[f"ohlcv_{source}"]["zero_volume_days"] = int((df["volume"] == 0).sum())
            # Depth probe: is there anything before the first bar returned above?
            before = (_naive_day(df["time"]).min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            older, err = call(
                f"{symbol} ohlcv {source} before {before}",
                client.ohlcv,
                symbol,
                DEEP_START,
                before,
                source,
            )
            out[f"ohlcv_{source}"]["rows_before_first_bar"] = (
                {"error": err} if err else int(len(older))
            )

    if {"ssi", "fhsc"} <= bars.keys():
        out["ohlcv_ssi_vs_fhsc"] = _compare_close(bars["ssi"], bars["fhsc"], "close", "close")

    if "ssi" in sources:
        th, err = call(
            f"{symbol} trade_history ssi",
            client.trade_history,
            symbol,
            DEEP_START,
            end,
            "ssi",
            columns="all",
        )
        if err:
            out["trade_history_ssi"] = {"error": err}
        else:
            out["trade_history_ssi"] = _frame_summary(th, "time")
            out["trade_history_ssi"]["adjustment_check"] = _adjustment_check(th)
            if "ssi" in bars:
                out["trade_history_ssi"]["close_vs_ohlcv_ssi"] = _compare_close(
                    bars["ssi"], th, "close", "close"
                )
        ts, err = call(
            f"{symbol} trading_stats ssi", client.trading_stats, symbol, DEEP_START, end, "ssi"
        )
        if err:
            out["trading_stats_ssi"] = {"error": err}
        else:
            out["trading_stats_ssi"] = {"rows": int(len(ts)), "columns": list(ts.columns)}
            if "time" in ts.columns and len(ts):
                out["trading_stats_ssi"].update(_time_info(ts, "time"))
    else:
        out["trade_history_ssi"] = NOT_ENABLED
        out["trading_stats_ssi"] = NOT_ENABLED

    if "fhsc" in sources:
        out["volume_history_fhsc"] = _volume_history_chunked(call, symbol, end)
        vh = out["volume_history_fhsc"].pop("_frame", None)
        if vh is not None and "fhsc" in bars:
            out["volume_history_fhsc"]["raw_vwap_check"] = _raw_vwap_check(bars["fhsc"], vh)
    else:
        out["volume_history_fhsc"] = NOT_ENABLED
    return out


def _volume_history_chunked(call: Caller, symbol: str, end: str) -> dict[str, Any]:
    """fhsc rejects ranges over 5 years, so walk back in 5-year chunks until one is empty."""
    chunks: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    chunk_end = pd.Timestamp(end)
    while chunk_end > pd.Timestamp(DEEP_START):
        chunk_start = chunk_end - pd.DateOffset(years=5) + pd.Timedelta(days=1)
        s, e = chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        df, err = call(
            f"{symbol} volume_history fhsc {s}..{e}",
            client.volume_history,
            symbol,
            s,
            e,
            "fhsc",
            columns="all",
        )
        if err:
            chunks.append({"start": s, "end": e, "error": err})
            break
        chunks.append({"start": s, "end": e, "rows": int(len(df))})
        if len(df) == 0:
            break
        frames.append(df)
        chunk_end = chunk_start - pd.Timedelta(days=1)
    if not frames:
        return {"chunks": chunks}
    vh = pd.concat(frames, ignore_index=True)
    return {**_frame_summary(vh, "time"), "chunks": chunks, "_frame": vh}


def _raw_vwap_check(bars: pd.DataFrame, vh: pd.DataFrame) -> dict[str, Any]:
    """Is matched_value / matched_volume an unadjusted price? Aggregates only.

    ratio = adjusted close / raw VWAP. Without adjustments it stays near 1; before stock
    dividends or splits it drops below 1. A raw series also shows ex-date jumps that the
    adjusted close does not.
    """
    needed = {"matched_value", "matched_volume", "time"}
    if not needed <= set(vh.columns):
        return {"skipped": f"missing columns {sorted(needed - set(vh.columns))}"}
    left = pd.DataFrame(
        {
            "day": _naive_day(bars["time"]),
            "close": bars["close"].to_numpy(),
            "volume": bars["volume"].to_numpy(),
        }
    )
    vwap = vh["matched_value"] / vh["matched_volume"].where(vh["matched_volume"] > 0) / 1000.0
    right = pd.DataFrame(
        {
            "day": _naive_day(vh["time"]),
            "vwap": vwap.to_numpy(),
            "matched_volume": vh["matched_volume"].to_numpy(),
        }
    )
    m = left.merge(right, on="day").sort_values("day").reset_index(drop=True)
    if m.empty:
        return {"overlap_days": 0}
    vol_ratio = m["volume"] / m["matched_volume"]
    ratio = m["close"] / m["vwap"]
    adj_ret = m["close"].pct_change()
    raw_ret = m["vwap"].pct_change()
    jumps = (raw_ret.abs() > 0.15) & (adj_ret.abs() < 0.07)
    recent = ratio.tail(60)
    return {
        "overlap_days": int(len(m)),
        "ratio_quantiles": {
            str(q): round(float(ratio.quantile(q)), 4) for q in (0.0, 0.05, 0.5, 0.95, 1.0)
        },
        "ratio_last_60_median": round(float(recent.median()), 4),
        "share_days_ratio_below_0_95": round(float((ratio < 0.95).mean()), 4),
        "raw_jump_days_beyond_15pct_while_adjusted_within_7pct": int(jumps.sum()),
        "ohlcv_volume_over_matched_volume_quantiles": {
            str(q): round(float(vol_ratio.quantile(q)), 4) for q in (0.0, 0.05, 0.5, 0.95, 1.0)
        },
    }


def _compare_close(a: pd.DataFrame, b: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    left = pd.DataFrame({"d": _naive_day(a["time"]), "x": a[col_a].to_numpy()})
    right = pd.DataFrame({"d": _naive_day(b["time"]), "y": b[col_b].to_numpy()})
    m = left.merge(right, on="d")
    if m.empty:
        return {"overlap_days": 0}
    rel = (m["x"] / m["y"] - 1).abs()
    return {
        "overlap_days": int(len(m)),
        "share_within_0_5pct": round(float((rel < 0.005).mean()), 4),
        "max_abs_rel_diff": round(float(rel.max()), 4),
    }


def _adjustment_check(th: pd.DataFrame) -> dict[str, Any]:
    """Can an unadjusted close be recovered from trade_history? Aggregates only.

    factor = adjusted average price / raw VWAP, raw VWAP = matched_value / matched_volume.
    A raw close series should show ex-date jumps beyond the +/-7% HOSE limit that the
    adjusted series does not.
    """
    needed = {"matched_value", "matched_volume", "average_price", "close", "time"}
    if not needed <= set(th.columns):
        return {"skipped": f"missing columns {sorted(needed - set(th.columns))}"}
    d = th.assign(day=_naive_day(th["time"])).sort_values("day").reset_index(drop=True)
    traded = d["matched_volume"] > 0
    raw_vwap = d["matched_value"] / d["matched_volume"] / 1000.0  # thousand VND
    factor = (d["average_price"] / raw_vwap).where(traded)
    raw_close = d["close"] / factor
    adj_ret = d["close"].pct_change()
    raw_ret = raw_close.pct_change()
    jumps = (raw_ret.abs() > 0.075) & (adj_ret.abs() < 0.07)
    return {
        "factor_quantiles": {
            str(q): round(float(factor.quantile(q)), 4) for q in (0.0, 0.05, 0.5, 0.95, 1.0)
        },
        "factor_last_60_median": round(float(factor.tail(60).median()), 4),
        "share_days_factor_below_0_99": round(float((factor < 0.99).mean()), 4),
        "raw_jump_days_beyond_7pct_while_adjusted_within": int(jumps.sum()),
        "first_day_factor_within_1pct_of_1": _first_day(d["day"], (factor - 1).abs() < 0.01),
    }


def _first_day(days: pd.Series, mask: pd.Series) -> str | None:
    hit = days[mask.fillna(False)]
    return str(hit.iloc[0].date()) if len(hit) else None


def audit_fundamentals(call: Caller, symbol: str) -> dict[str, Any]:
    """A3 (available items) and A4 (depth, TTM consistency) for one symbol."""
    out: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    statements = (("balance_sheet", client.balance_sheet), ("income", client.income_statement))
    for statement, fn in statements:
        for period in ("quarter", "year"):
            key = f"{statement}_{period}"
            df, err = call(f"{symbol} {key} fhsc", fn, symbol, period, "fhsc")
            if err:
                out[key] = {"error": err}
                continue
            frames[key] = df
            periods = sorted(df["report_period"].astype(str).unique())
            items = df.drop_duplicates("item_id")
            out[key] = {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "n_periods": len(periods),
                "first_period": periods[0] if periods else None,
                "last_period": periods[-1] if periods else None,
                "attrs": _attrs(df, STATEMENT_META),
                "items": [
                    {
                        "item_id": r.item_id,
                        "item_name": getattr(r, "item_name", None),
                        "item_level": _jsonable(getattr(r, "item_level", None)),
                        "non_null_periods": int(
                            df.loc[df["item_id"] == r.item_id, "item_value"].notna().sum()
                        ),
                    }
                    for r in items.itertuples()
                ],
            }
            if period == "quarter":
                out[key]["missing_quarters"] = _missing_quarters(periods)
    if "income_quarter" in frames and "income_year" in frames:
        out["ttm_check_income"] = _ttm_check(frames["income_quarter"], frames["income_year"])
    return out


def _parse_quarter(label: str) -> pd.Period | None:
    try:
        return pd.Period(label.replace("-", ""), freq="Q")
    except ValueError:
        return None


def _missing_quarters(periods: list[str]) -> list[str] | str:
    parsed = [_parse_quarter(x) for x in periods]
    bad = [x for x, p in zip(periods, parsed, strict=True) if p is None]
    if bad:
        return f"unparsed labels: {bad[:5]}"
    if not parsed:
        return []
    have = set(parsed)
    return [str(p) for p in pd.period_range(min(parsed), max(parsed), freq="Q") if p not in have]


def _ttm_check(quarter: pd.DataFrame, year: pd.DataFrame) -> dict[str, Any]:
    """Relative gap between the sum of four quarters and the annual figure, per item."""
    q = quarter.assign(p=quarter["report_period"].astype(str).map(_parse_quarter))
    q = q.dropna(subset=["p", "item_value"])
    q = q.assign(year=q["p"].map(lambda p: p.year))
    counts = q.groupby(["item_id", "year"])["p"].nunique()
    sums = q.groupby(["item_id", "year"])["item_value"].sum()[counts == 4]
    y = year.assign(year=pd.to_numeric(year["report_period"].astype(str).str[:4], errors="coerce"))
    annual = y.dropna(subset=["year", "item_value"]).groupby(["item_id", "year"])["item_value"]
    joined = pd.concat({"q_sum": sums, "annual": annual.sum()}, axis=1, join="inner")
    joined = joined[joined["annual"].abs() > 0]
    rel = (joined["q_sum"] / joined["annual"] - 1).abs()
    per_item = rel.groupby(level="item_id").agg(["count", "median", "max"])
    return {
        "n_item_years": int(len(rel)),
        "share_within_1pct": round(float((rel < 0.01).mean()), 4) if len(rel) else None,
        "per_item": {
            str(item): {
                "n_years": int(r["count"]),
                "median": round(float(r["median"]), 4),
                "max": round(float(r["max"]), 4),
            }
            for item, r in per_item.iterrows()
        },
    }


def check_window(cfg: dict[str, Any], force: bool) -> None:
    now = datetime.now(VN_TZ)
    earliest = cfg["ingest"]["earliest_local_hour"]
    latest = cfg["ingest"]["latest_local_hour"]
    if not earliest <= now.hour < latest and not force:
        sys.exit(f"refusing to run at {now:%H:%M} Vietnam time; allowed {earliest}:00-{latest}:00")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["ssi", "fhsc"],
        default=["ssi", "fhsc"],
        help="sources with a key on this machine; others are recorded as not tested",
    )
    parser.add_argument("--force-time", action="store_true", help="skip the time-window check")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_window(cfg, args.force_time)
    ing = cfg["ingest"]
    call = Caller(ing["request_interval_seconds"], ing["max_retries"], ing["backoff_base_seconds"])
    end = ing["end_date"]

    summary: dict[str, Any] = {
        "run_at": datetime.now(VN_TZ).isoformat(timespec="seconds"),
        "end_date": end,
        "deep_start_requested": DEEP_START,
        "symbols": {},
    }
    sources = set(args.sources)
    summary["sources_enabled"] = sorted(sources)
    if "ssi" in sources:
        listing, err = call("HOSE listing ssi", client.list_equities, "HOSE", "ssi")
        summary["hose_listing_ssi"] = (
            {"error": err} if err else {"rows": int(len(listing)), "columns": list(listing.columns)}
        )
    if "fhsc" in sources:
        cons, err = call("VNINDEX constituents fhsc", client.index_constituents, "VNINDEX", "fhsc")
        summary["vnindex_constituents_fhsc"] = (
            {"error": err}
            if err
            else {
                "rows": int(len(cons)),
                "columns": list(cons.columns),
                "exchange_counts": _jsonable(cons["exchange"].value_counts().to_dict())
                if "exchange" in cons.columns
                else None,
            }
        )

    for symbol in args.symbols:
        print(f"[{symbol}]", flush=True)
        entry: dict[str, Any] = {"prices": audit_prices(call, symbol, end, sources)}
        if "fhsc" in sources:
            entry["fundamentals"] = audit_fundamentals(call, symbol)
            shares, err = call(
                f"{symbol} company listing fhsc", client.company_listing, symbol, "fhsc"
            )
            entry["company_listing_fhsc"] = (
                {"error": err} if err else {"columns": list(shares.columns)}
            )
        summary["symbols"][symbol] = entry
        _write(summary, call.n_calls)  # keep partial results if the run is cut short

    print(f"wrote {OUT_PATH} after {call.n_calls} calls")
    return 0


def _write(summary: dict[str, Any], n_calls: int) -> None:
    summary["n_calls"] = n_calls
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
