"""Ingest logic against a fake data source: pacing, retries, cache, traded value proxy."""

from __future__ import annotations

import copy
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tests.conftest import (
    LATE_LISTER,
    PRICE_COLUMNS,
    STOCK_DIVIDEND_SYMBOL,
    make_fundamentals,
    make_prices,
)
from vnfactor import client, ingest
from vnfactor.cli import main

TRUE_VALUE_START = "2020-10-07"  # session 200 of the synthetic calendar
INSIDE_WINDOW = datetime(2026, 11, 14, 16, 0, tzinfo=ingest.VN_TIME)
OUTSIDE_WINDOW = datetime(2026, 11, 14, 10, 0, tzinfo=ingest.VN_TIME)


class RetryableError(Exception):
    pass


class FatalSourceError(Exception):
    pass


class KeyProblem(Exception):
    pass


def describe_error(exc: BaseException) -> dict | None:
    if not isinstance(exc, RetryableError | FatalSourceError | KeyProblem):
        return None
    return {
        "error_class": type(exc).__name__,
        "code": None,
        "retryable": isinstance(exc, RetryableError),
        "credential": isinstance(exc, KeyProblem),
        "message": str(exc),
    }


class FakeApi:
    """Serves the synthetic dataset through the ``fetch_*`` interface of client.py."""

    def __init__(self) -> None:
        self.prices = make_prices()
        symbols = sorted(self.prices["symbol"].unique())
        self.statements = make_fundamentals(symbols).drop(columns="available_date")
        self.calls: list[tuple] = []
        self.broken: dict[str, Exception] = {}  # symbol -> error raised by fetch_prices
        self.describe_error = describe_error

    def fetch_candidates(self, index: str, exchange: str, source: str) -> pd.DataFrame:
        self.calls.append(("candidates", index))
        symbols = sorted(self.prices["symbol"].unique())
        return pd.DataFrame({"symbol": symbols, "exchange": exchange})

    def fetch_prices(self, symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
        self.calls.append(("prices", symbol))
        if symbol in self.broken:
            raise self.broken[symbol]
        rows = self.prices[self.prices["symbol"] == symbol]
        return rows[client.BAR_COLUMNS].reset_index(drop=True)

    def fetch_traded_value(self, symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
        self.calls.append(("traded_value", symbol, start, end))
        rows = self.prices[self.prices["symbol"] == symbol]
        rows = rows[rows["date"].between(start, end) & (rows["date"] >= TRUE_VALUE_START)]
        out = pd.DataFrame(
            {
                "date": rows["date"].to_numpy(),
                "symbol": symbol,
                "matched_volume": rows["volume"].to_numpy(dtype=float),
                "matched_value": rows["value"].to_numpy(),
            }
        )
        return out[client.TRADED_VALUE_COLUMNS]

    def fetch_statement(self, symbol: str, statement: str, source: str) -> pd.DataFrame:
        self.calls.append(("statement", symbol, statement))
        wanted = set(client.STATEMENT_ITEMS[statement].values())
        rows = self.statements[
            (self.statements["symbol"] == symbol) & self.statements["item_id"].isin(wanted)
        ]
        return rows[client.STATEMENT_COLUMNS].reset_index(drop=True)

    def fetch_shares(self, symbol: str, source: str) -> pd.DataFrame:
        self.calls.append(("shares", symbol))
        return pd.DataFrame({"symbol": [symbol], "outstanding_shares": [2.5e8]})


@pytest.fixture
def cfg(base_config, tmp_path) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["paths"]["raw_dir"] = str(tmp_path / "raw")
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["ingest"].update(
        start_date="2020-01-01", end_date="2022-04-19", max_range_years=1, fundamentals_top_n=5
    )
    cfg["traded_value"].update(true_value_start=TRUE_VALUE_START, anchor_sessions=20)
    cfg["calendar"].update(first_rebalance="2020-12-31", last_rebalance="2022-03-31")
    return cfg


def run(cfg: dict, api: FakeApi, now: datetime = INSIDE_WINDOW, refresh: bool = False):
    lines: list[str] = []
    code = ingest.run(
        cfg, refresh=refresh, api=api, sleep=lambda _: None, now=lambda: now, log=lines.append
    )
    return code, lines


def settings(**overrides) -> dict:
    base = {
        "request_interval_seconds": 1.0,
        "max_retries": 3,
        "backoff_base_seconds": 2.0,
        "earliest_local_hour": 15,
        "latest_local_hour": 21,
    }
    return {**base, **overrides}


# --- date chunks --------------------------------------------------------------------------


def test_date_chunks_cover_the_range_without_overlap():
    chunks = ingest.date_chunks("2021-08-02", "2026-10-30", 5)
    assert chunks == [("2021-08-02", "2026-08-01"), ("2026-08-02", "2026-10-30")]


def test_date_chunks_short_range_is_one_piece():
    assert ingest.date_chunks("2025-01-15", "2026-10-30", 5) == [("2025-01-15", "2026-10-30")]
    assert ingest.date_chunks("2026-10-30", "2026-10-30", 5) == [("2026-10-30", "2026-10-30")]
    assert ingest.date_chunks("2026-10-31", "2026-10-30", 5) == []


def test_date_chunks_never_reach_the_source_limit():
    for start, end in ingest.date_chunks("2017-01-03", "2026-10-30", 5):
        assert pd.Timestamp(end) < pd.Timestamp(start) + pd.DateOffset(years=5)


# --- traded value -------------------------------------------------------------------------


def _one_symbol(n: int = 60, matched_from: int = 30, factor: float = 0.5):
    """Bars adjusted by ``factor`` for an event after the anchor; matched value is the truth."""
    dates = pd.bdate_range("2021-06-01", periods=n)
    raw_close = np.linspace(40.0, 50.0, n)
    volume = np.arange(1, n + 1) * 1000.0
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA",
            "open": raw_close * factor,
            "high": raw_close * factor,
            "low": raw_close * factor,
            "close_adj": raw_close * factor,
            "volume": volume,
        }
    )
    true_value = raw_close * 1000.0 * volume
    traded = pd.DataFrame(
        {
            "date": dates[matched_from:],
            "symbol": "AAA",
            "matched_volume": volume[matched_from:],
            "matched_value": true_value[matched_from:],
        }
    )
    return bars, traded, true_value


def test_build_prices_uses_matched_value_and_anchored_proxy():
    bars, traded, true_value = _one_symbol()
    prices = ingest.build_prices(bars, traded, anchor_sessions=10)

    assert list(prices.columns) == PRICE_COLUMNS
    assert list(prices["value_source"]) == ["proxy"] * 30 + ["matched"] * 30
    # k = 0.5 at the anchor, so the proxy undoes the later adjustment exactly
    np.testing.assert_allclose(prices["value"], true_value)
    assert ingest.anchor_factors(bars, traded, anchor_sessions=10)["AAA"] == pytest.approx(0.5)


def test_anchor_uses_only_the_first_sessions_with_matched_value():
    bars, traded, true_value = _one_symbol()
    # A second event after the anchor window: sessions 45+ are quoted at the raw price.
    bars.loc[45:, "close_adj"] = bars.loc[45:, "close_adj"] * 2.0
    prices = ingest.build_prices(bars, traded, anchor_sessions=10)
    np.testing.assert_allclose(prices["value"].iloc[:30], true_value[:30])


def test_symbol_listed_after_true_value_start_needs_no_proxy():
    bars, traded, true_value = _one_symbol(matched_from=0)
    prices = ingest.build_prices(bars, traded, anchor_sessions=10)
    assert set(prices["value_source"]) == {"matched"}
    np.testing.assert_allclose(prices["value"], true_value)


def test_missing_traded_value_stays_missing():
    bars, traded, _ = _one_symbol()
    no_anchor = ingest.build_prices(bars, traded.iloc[0:0], anchor_sessions=10)
    assert no_anchor["value"].isna().all()
    assert set(no_anchor["value_source"]) == {"proxy"}

    gap = traded.drop(traded.index[5])  # the source lacks one session after the anchor
    prices = ingest.build_prices(bars, gap, anchor_sessions=10)
    assert prices["value"].isna().sum() == 1
    assert prices.loc[prices["value"].isna(), "value_source"].item() == "matched"


def test_zero_volume_sessions_do_not_enter_the_anchor():
    bars, traded, true_value = _one_symbol()
    bars.loc[30:32, "volume"] = 0.0
    traded.loc[traded.index[:3], "matched_value"] = 0.0
    k = ingest.anchor_factors(bars, traded, anchor_sessions=10)["AAA"]
    assert k == pytest.approx(0.5)
    prices = ingest.build_prices(bars, traded, anchor_sessions=10)
    assert list(prices["value"].iloc[30:33]) == [0.0, 0.0, 0.0]


# --- which symbols get statements ---------------------------------------------------------


def test_symbols_ever_in_top_n():
    dates = pd.bdate_range("2021-01-01", periods=120)
    values = {
        "TOP": np.full(120, 100.0),
        "RISER": np.r_[np.full(60, 1.0), np.full(60, 500.0)],
        "NEVER": np.full(120, 10.0),
        "SHORT": np.r_[np.full(115, np.nan), np.full(5, 9999.0)],  # too few sessions
    }
    prices = pd.concat(
        pd.DataFrame({"date": dates, "symbol": s, "value": v}) for s, v in values.items()
    ).dropna()
    selected = ingest.symbols_ever_in_top_n(
        prices, top_n=1, window=20, min_periods=16, first_date="2021-01-01", last_date="2021-12-31"
    )
    assert selected == ["RISER", "TOP"]

    early_only = ingest.symbols_ever_in_top_n(
        prices, top_n=1, window=20, min_periods=16, first_date="2021-01-01", last_date="2021-03-31"
    )
    assert early_only == ["TOP"]


# --- summary ------------------------------------------------------------------------------


def test_summarize_prices_counts_gaps_and_limit_breaches():
    dates = pd.bdate_range("2022-01-03", periods=10)
    frames = []
    for symbol, close in {
        "CALM": np.full(10, 20.0),
        "JUMP": np.r_[np.full(5, 20.0), np.full(5, 22.0)],  # one +10% day
        "EDGE": np.r_[np.full(5, 20.0), np.full(5, 21.4)],  # +7%, inside the limit
    }.items():
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "close_adj": close,
                    "volume": 1000.0,
                    "value": close * 1e6,
                    "value_source": "matched",
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.drop(prices.index[(prices["symbol"] == "CALM") & (prices["date"] == dates[4])])
    prices.loc[prices.index[0], "value"] = np.nan

    s = ingest.summarize_prices(prices, daily_limit=0.07, tolerance=0.005)
    assert (s["symbols"], s["rows"], s["sessions"]) == (3, 29, 10)
    assert (s["first_date"], s["last_date"]) == ("2022-01-03", "2022-01-14")
    assert s["missing_bar_ratio"] == pytest.approx(1 / 30)
    assert s["missing_value_ratio"] == pytest.approx(1 / 29)
    assert s["limit_breaches"] == {"JUMP": 1}
    assert s["limit_breach_rows"] == 1


# --- pacing and retries -------------------------------------------------------------------


def test_fetcher_spaces_calls_and_backs_off_on_retryable_errors():
    sleeps: list[float] = []
    outcomes = [RetryableError("slow down"), RetryableError("slow down"), "data"]

    def fn(symbol):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return f"{outcome} for {symbol}"

    fetcher = ingest.Fetcher(
        settings(), describe_error, sleep=sleeps.append, now=lambda: INSIDE_WINDOW
    )
    assert fetcher.call(fn, "FPT") == "data for FPT"
    assert fetcher.calls == 3
    assert sleeps == [1.0, 2.0, 1.0, 4.0, 1.0]


def test_fetcher_stops_after_max_retries():
    def fn():
        raise RetryableError("still limited")

    fetcher = ingest.Fetcher(
        settings(max_retries=3), describe_error, sleep=lambda _: None, now=lambda: INSIDE_WINDOW
    )
    with pytest.raises(ingest.IngestStop, match="after 3 retries"):
        fetcher.call(fn)
    assert fetcher.calls == 4


@pytest.mark.parametrize("error", [FatalSourceError("no such symbol"), ValueError("a bug")])
def test_fetcher_does_not_retry_other_errors(error):
    def fn():
        raise error

    fetcher = ingest.Fetcher(
        settings(), describe_error, sleep=lambda _: None, now=lambda: INSIDE_WINDOW
    )
    with pytest.raises(type(error)) as caught:
        fetcher.call(fn)
    assert caught.value is error
    assert fetcher.calls == 1


@pytest.mark.parametrize("hour", [0, 9, 14, 21, 23])
def test_fetcher_refuses_to_call_outside_the_window(hour):
    called = []
    now = datetime(2026, 11, 14, hour, 30, tzinfo=ingest.VN_TIME)
    fetcher = ingest.Fetcher(settings(), describe_error, sleep=lambda _: None, now=lambda: now)
    with pytest.raises(ingest.IngestStop, match="download window"):
        fetcher.call(called.append, 1)
    assert called == []
    assert fetcher.calls == 0


@pytest.mark.parametrize("hour", [15, 18, 20])
def test_fetcher_calls_inside_the_window(hour):
    now = datetime(2026, 11, 14, hour, 59, tzinfo=ingest.VN_TIME)
    fetcher = ingest.Fetcher(settings(), describe_error, sleep=lambda _: None, now=lambda: now)
    assert fetcher.call(lambda: "ok") == "ok"


# --- the whole step -----------------------------------------------------------------------


def test_run_writes_tables_in_the_internal_schema(cfg, tmp_path):
    api = FakeApi()
    code, lines = run(cfg, api)
    assert code == 0, lines

    prices = pd.read_parquet(tmp_path / "processed" / "prices_daily.parquet")
    assert list(prices.columns) == PRICE_COLUMNS
    assert not prices.duplicated(["date", "symbol"]).any()
    expected = api.prices
    assert len(prices) == len(expected)
    # The stock dividend falls after the anchor, so the proxy recovers the true value.
    np.testing.assert_allclose(prices["value"], expected["value"])
    before = prices["date"] < TRUE_VALUE_START
    assert set(prices.loc[before, "value_source"]) == {"proxy"}
    assert set(prices.loc[~before, "value_source"]) == {"matched"}
    assert set(prices.loc[prices["symbol"] == LATE_LISTER, "value_source"]) == {"matched"}
    dividend = prices[(prices["symbol"] == STOCK_DIVIDEND_SYMBOL) & before]
    raw_over_adj = dividend["value"] / (dividend["close_adj"] * 1000.0 * dividend["volume"])
    np.testing.assert_allclose(raw_over_adj, 1.3)

    statements = pd.read_parquet(tmp_path / "processed" / "statements_q.parquet")
    shares = pd.read_parquet(tmp_path / "processed" / "shares_current.parquet")
    assert list(statements.columns) == client.STATEMENT_COLUMNS
    assert set(statements["item_id"]) == {"equity", "npat_parent", "npat_total"}
    assert list(shares.columns) == client.SHARES_COLUMNS
    selected = set(shares["symbol"])
    assert set(statements["symbol"]) == selected
    assert 5 <= len(selected) < prices["symbol"].nunique()
    assert any(line.startswith("summary") for line in lines)


def test_run_chunks_traded_value_from_the_first_session_that_can_have_it(cfg):
    api = FakeApi()
    run(cfg, api)
    ranges = {
        symbol: [(c[2], c[3]) for c in api.calls if c[0] == "traded_value" and c[1] == symbol]
        for symbol in ("S00", LATE_LISTER)
    }
    assert ranges["S00"] == [("2020-10-07", "2021-10-06"), ("2021-10-07", "2022-04-19")]
    assert ranges[LATE_LISTER] == [("2021-07-14", "2022-04-19")]


def test_second_run_makes_no_network_call(cfg):
    api = FakeApi()
    assert run(cfg, api)[0] == 0
    n_calls = len(api.calls)
    assert n_calls > 0

    # Even outside the download window: nothing has to be requested.
    code, lines = run(cfg, api, now=OUTSIDE_WINDOW)
    assert code == 0
    assert len(api.calls) == n_calls
    assert lines[-1] == "network calls: 0"


def test_refresh_downloads_everything_again(cfg):
    api = FakeApi()
    run(cfg, api)
    n_calls = len(api.calls)
    assert run(cfg, api, refresh=True)[0] == 0
    assert len(api.calls) == 2 * n_calls


def test_run_outside_the_window_stops_before_any_call(cfg, tmp_path):
    api = FakeApi()
    code, lines = run(cfg, api, now=OUTSIDE_WINDOW)
    assert code == 1
    assert api.calls == []
    assert any("download window" in line for line in lines)
    assert not (tmp_path / "processed").exists()


def test_failed_symbol_blocks_the_processed_tables_and_is_retried(cfg, tmp_path):
    api = FakeApi()
    api.broken["S03"] = client.DataError("S03 ohlcv: duplicate dates")
    code, lines = run(cfg, api)
    assert code == 1
    assert any("S03 [prices] DataError" in line for line in lines)
    assert not (tmp_path / "processed" / "prices_daily.parquet").exists()
    assert (tmp_path / "raw" / "bars" / "S04.parquet").exists()  # the others are cached
    assert not (tmp_path / "raw" / "bars" / "S03.parquet").exists()

    api.broken.clear()
    api.calls.clear()
    assert run(cfg, api)[0] == 0
    assert {c[1] for c in api.calls if c[0] in ("prices", "traded_value")} == {"S03"}


def test_source_error_for_one_symbol_is_reported(cfg):
    api = FakeApi()
    api.broken["S07"] = FatalSourceError("symbol not served")
    code, lines = run(cfg, api)
    assert code == 1
    assert any("S07 [prices] FatalSourceError: symbol not served" in line for line in lines)


def test_skip_symbols_leaves_a_symbol_out(cfg, tmp_path):
    api = FakeApi()
    api.broken["S07"] = FatalSourceError("symbol not served")
    cfg["ingest"]["skip_symbols"] = ["S07"]
    assert run(cfg, api)[0] == 0
    prices = pd.read_parquet(tmp_path / "processed" / "prices_daily.parquet")
    assert "S07" not in set(prices["symbol"])
    assert ("prices", "S07") not in api.calls


def test_credential_error_stops_the_run_at_once(cfg):
    api = FakeApi()
    api.broken["S00"] = KeyProblem("key rejected")
    code, lines = run(cfg, api)
    assert code == 1
    assert [c for c in api.calls if c[0] == "prices"] == [("prices", "S00")]
    assert any("credential problem" in line for line in lines)


def test_many_failures_in_a_row_stop_the_run(cfg):
    api = FakeApi()
    for i in range(12):
        api.broken[f"S{i:02d}"] = FatalSourceError("source is down")
    cfg["ingest"]["max_consecutive_failures"] = 3
    code, lines = run(cfg, api)
    assert code == 1
    assert len([c for c in api.calls if c[0] == "prices"]) == 3
    assert any("3 symbols failed in a row" in line for line in lines)


def test_unexpected_errors_are_not_swallowed(cfg):
    api = FakeApi()
    api.broken["S02"] = KeyError("a bug, not a source problem")
    with pytest.raises(KeyError):
        run(cfg, api)


def test_cli_ingest_passes_the_refresh_flag(config_path, monkeypatch):
    seen = {}

    def fake_run(cfg, refresh=False):
        seen["refresh"] = refresh
        seen["source"] = cfg["sources"]["prices"]
        return 0

    monkeypatch.setattr(ingest, "run", fake_run)
    assert main(["--config", str(config_path), "ingest", "--refresh"]) == 0
    assert seen == {"refresh": True, "source": "fhsc"}
    assert main(["--config", str(config_path), "ingest"]) == 0
    assert seen["refresh"] is False
