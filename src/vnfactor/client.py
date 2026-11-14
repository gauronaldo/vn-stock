"""The only module allowed to import ``vnstock`` (AGENTS.md rule 4).

``vnstock`` is imported inside each call so that tests and CI run without it.
Every call names its ``source`` explicitly and asks for v6 column names.

Two layers:

* raw wrappers (``ohlcv``, ``volume_history``, ...) return the source DataFrame
  unchanged, ``df.attrs`` included. The data audit script uses these.
* ``fetch_*`` functions return frames in the internal schema (AGENTS.md 7.2):
  prices in thousand VND, values in VND, timezone-naive dates.

Source errors are never swallowed: vnstock exceptions propagate unchanged, and a
response that cannot be mapped to the internal schema raises ``DataError``.

Signatures checked against ``vnstock doc`` for vnstock 6.0.0 on 2026-11-13.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

COMPAT = "v6"
VN_TZ = "Asia/Ho_Chi_Minh"

BAR_COLUMNS = ["date", "symbol", "open", "high", "low", "close_adj", "volume"]
TRADED_VALUE_COLUMNS = ["date", "symbol", "matched_volume", "matched_value"]
STATEMENT_COLUMNS = ["symbol", "report_period", "item_id", "period_end", "value", "company_type"]
SHARES_COLUMNS = ["symbol", "outstanding_shares"]
COMPANY_TYPES = ("regular", "bank", "securities", "insurance")

# Source item code -> internal item_id, per statement. All four statement templates carry
# these codes in the reduced (level-1) item set (docs/data_audit.md, A3).
STATEMENT_ITEMS: dict[str, dict[str, str]] = {
    "balance_sheet": {"BS_EQUITY": "equity"},  # total equity, minority interest included
    "income_statement": {
        "IS_PROFIT_AFTER_TAX_FOR_SHAREHOLDERS_OF_PARENT_COMPANY": "npat_parent",
        "IS_NET_PROFIT_AFTER_TAX": "npat_total",
    },
}
_QUARTER_LABEL = re.compile(r"^(\d{4})-Q([1-4])$")


class DataError(ValueError):
    """A source response that cannot be mapped to the internal schema."""


def _vnstock() -> Any:
    import vnstock

    return vnstock


def describe_error(exc: BaseException) -> dict[str, Any] | None:
    """Classify a vnstock error; return None for anything that is not one.

    ``retryable`` is True only for RateLimitError and NetworkError (AGENTS.md 3.3).
    ``credential`` is True when the keys or session must be fixed by the owner.
    """
    try:
        errors = _vnstock().errors
    except ImportError:
        return None
    if not isinstance(exc, errors.VnstockError):
        return None
    credential = (
        errors.CredentialError,
        errors.AuthError,
        errors.LoginRequiredError,
        errors.OtpRequiredError,
    )
    return {
        "error_class": type(exc).__name__,
        "code": getattr(exc, "code", None),
        "retryable": isinstance(exc, errors.RateLimitError | errors.NetworkError),
        "credential": isinstance(exc, credential),
        "message": str(exc),
    }


# --- raw wrappers: source frames unchanged ------------------------------------------------


def list_equities(exchange: str, source: str) -> pd.DataFrame:
    """Currently listed common stocks on ``exchange`` (not point-in-time)."""
    reference = _vnstock().Reference(source)
    return reference.equity.list(exchange=exchange, type="stock", compat=COMPAT)


def index_members(index: str, source: str) -> pd.DataFrame:
    """Current members of ``index``; there is no membership history."""
    return _vnstock().Reference(source).index(index).members(compat=COMPAT)


def list_indices(source: str) -> pd.DataFrame:
    """Indices the source knows about."""
    return _vnstock().Reference(source).index.list(compat=COMPAT)


def index_constituents(index: str, source: str) -> pd.DataFrame:
    """Current constituents of ``index`` with exchange and weight; no history."""
    return _vnstock().Reference(source).index(index).constituents(compat=COMPAT)


def company_listing(symbol: str, source: str) -> pd.DataFrame:
    """Current share counts (listed, outstanding, treasury, free float) for ``symbol``."""
    return _vnstock().Reference(source).company(symbol).listing(compat=COMPAT)


def ohlcv(
    symbol: str,
    start: str,
    end: str,
    source: str,
    interval: str = "1D",
    columns: str | None = None,
) -> pd.DataFrame:
    """Daily bars adjusted for corporate actions; prices in thousand VND, volume in shares."""
    equity = _vnstock().Market(source).equity(symbol)
    kwargs: dict[str, Any] = {"start": start, "end": end, "interval": interval, "compat": COMPAT}
    if columns is not None:
        kwargs["columns"] = columns
    return equity.ohlcv(**kwargs)


def trade_history(
    symbol: str, start: str, end: str, source: str, columns: str | None = None
) -> pd.DataFrame:
    """Per-session statistics: prices, matched volume (shares) and matched value (VND)."""
    equity = _vnstock().Market(source).equity(symbol)
    kwargs: dict[str, Any] = {"start": start, "end": end, "compat": COMPAT}
    if columns is not None:
        kwargs["columns"] = columns
    return equity.trade_history(**kwargs)


def volume_history(
    symbol: str, start: str, end: str, source: str, columns: str | None = None
) -> pd.DataFrame:
    """Per-session matched and block volume (shares) and value (VND)."""
    equity = _vnstock().Market(source).equity(symbol)
    kwargs: dict[str, Any] = {"start": start, "end": end, "compat": COMPAT}
    if columns is not None:
        kwargs["columns"] = columns
    return equity.volume_history(**kwargs)


def trading_stats(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
    """Trading statistics for ``symbol`` over a date range."""
    company = _vnstock().Reference(source).company(symbol)
    return company.trading_stats(start=start, end=end, compat=COMPAT)


def income_statement(symbol: str, period: str, source: str) -> pd.DataFrame:
    """Income statement, long format; quarterly figures are standalone, values in VND."""
    equity = _vnstock().Fundamental(source).equity(symbol)
    return equity.income_statement(period=period, format="long", compat=COMPAT)


def balance_sheet(symbol: str, period: str, source: str) -> pd.DataFrame:
    """Balance sheet, long format, values in VND; BS_EQUITY includes minority interest."""
    equity = _vnstock().Fundamental(source).equity(symbol)
    return equity.balance_sheet(period=period, format="long", compat=COMPAT)


# --- internal schema ----------------------------------------------------------------------


def _naive_day(values: pd.Series) -> pd.Series:
    t = pd.to_datetime(values)
    if t.dt.tz is not None:
        t = t.dt.tz_convert(VN_TZ).dt.tz_localize(None)
    return t.dt.normalize()


def _require(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"{what}: source response lacks columns {missing}")


def _numeric(series: pd.Series, what: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        raise DataError(f"{what}: {int(out.isna().sum())} missing or non-numeric values")
    return out.astype(float)


def fetch_candidates(index: str, exchange: str, source: str) -> pd.DataFrame:
    """Current constituents of ``index`` listed on ``exchange``: columns symbol, exchange.

    This is today's listing. Delisted symbols are absent (survivorship limitation).
    """
    df = index_constituents(index, source)
    _require(df, ["symbol", "exchange"], f"{index} constituents")
    out = df.loc[df["exchange"] == exchange, ["symbol", "exchange"]].astype(str)
    if out["symbol"].duplicated().any():
        raise DataError(f"{index} constituents: duplicate symbols")
    return out.sort_values("symbol", ignore_index=True)


def fetch_prices(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
    """Daily bars: date, symbol, open, high, low, close_adj (thousand VND), volume (shares).

    Prices are back-adjusted by the source; volume is unadjusted. An unfinished last bar
    (``attrs["last_bar_partial"]``) is dropped.
    """
    df = ohlcv(symbol, start, end, source)
    if df.attrs.get("last_bar_partial"):
        df = df.iloc[:-1]
    if len(df) == 0:
        return pd.DataFrame(columns=BAR_COLUMNS)
    _require(df, ["time", "open", "high", "low", "close", "volume"], f"{symbol} ohlcv")
    out = pd.DataFrame(
        {
            "date": _naive_day(df["time"]).to_numpy(),
            "symbol": symbol,
            "open": _numeric(df["open"], f"{symbol} open").to_numpy(),
            "high": _numeric(df["high"], f"{symbol} high").to_numpy(),
            "low": _numeric(df["low"], f"{symbol} low").to_numpy(),
            "close_adj": _numeric(df["close"], f"{symbol} close").to_numpy(),
            "volume": _numeric(df["volume"], f"{symbol} volume").to_numpy(),
        }
    )
    if out["date"].duplicated().any():
        raise DataError(f"{symbol} ohlcv: duplicate dates")
    return out.sort_values("date", ignore_index=True)[BAR_COLUMNS]


def fetch_traded_value(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
    """Per-session matched volume (shares) and matched value (VND, unadjusted).

    The source rejects ranges longer than five years; the caller splits the range.
    """
    df = volume_history(symbol, start, end, source)
    if len(df) == 0:
        return pd.DataFrame(columns=TRADED_VALUE_COLUMNS)
    _require(df, ["time", "matched_volume", "matched_value"], f"{symbol} volume_history")
    what = f"{symbol} volume_history"
    out = pd.DataFrame(
        {
            "date": _naive_day(df["time"]).to_numpy(),
            "symbol": symbol,
            "matched_volume": _numeric(df["matched_volume"], what).to_numpy(),
            "matched_value": _numeric(df["matched_value"], what).to_numpy(),
        }
    )
    if out["date"].duplicated().any():
        raise DataError(f"{what}: duplicate dates")
    return out.sort_values("date", ignore_index=True)[TRADED_VALUE_COLUMNS]


def quarter_end(report_period: str) -> pd.Timestamp:
    """``"2025-Q4"`` -> 2025-12-31. Labels are calendar quarters (docs/data_audit.md)."""
    match = _QUARTER_LABEL.match(str(report_period))
    if match is None:
        raise DataError(f"unexpected quarter label: {report_period!r}")
    return pd.Period(f"{match[1]}Q{match[2]}", freq="Q").end_time.normalize()


def fetch_statement(symbol: str, statement: str, source: str) -> pd.DataFrame:
    """Quarterly items of one statement in long format, values in VND.

    ``statement`` is ``"balance_sheet"`` or ``"income_statement"``. Quarterly income figures
    are standalone quarters. ``available_date`` is not set here: it depends on the reporting
    lag and is added when the point-in-time table is built.
    """
    wanted = STATEMENT_ITEMS[statement]
    fetch = balance_sheet if statement == "balance_sheet" else income_statement
    raw = fetch(symbol, "quarter", source)
    what = f"{symbol} {statement}"
    company_type = raw.attrs.get("company_type")
    if company_type not in COMPANY_TYPES:
        raise DataError(f"{what}: unknown company_type {company_type!r}")
    _require(raw, ["report_period", "item_id", "item_value"], what)
    rows = raw[raw["item_id"].isin(list(wanted))]
    missing = sorted(set(wanted) - set(rows["item_id"]))
    if missing:
        raise DataError(f"{what}: items not returned: {missing}")
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "report_period": rows["report_period"].astype(str).to_numpy(),
            "item_id": rows["item_id"].map(wanted).to_numpy(),
            "period_end": [quarter_end(p) for p in rows["report_period"]],
            "value": pd.to_numeric(rows["item_value"], errors="coerce").to_numpy(dtype=float),
            "company_type": company_type,
        }
    )
    if out.duplicated(["report_period", "item_id"]).any():
        raise DataError(f"{what}: duplicate (report_period, item_id) rows")
    return out.sort_values(["period_end", "item_id"], ignore_index=True)[STATEMENT_COLUMNS]


def fetch_shares(symbol: str, source: str) -> pd.DataFrame:
    """Current outstanding shares: columns symbol, outstanding_shares. No history."""
    df = company_listing(symbol, source)
    _require(df, ["outstanding_shares"], f"{symbol} listing")
    if len(df) != 1:
        raise DataError(f"{symbol} listing: expected 1 row, got {len(df)}")
    shares = float(_numeric(df["outstanding_shares"], f"{symbol} outstanding_shares").iloc[0])
    if shares <= 0:
        raise DataError(f"{symbol} listing: outstanding_shares is {shares}")
    return pd.DataFrame({"symbol": [symbol], "outstanding_shares": [shares]})
