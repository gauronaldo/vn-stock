"""client.py against a fake ``vnstock`` module: no network, no real package needed."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from vnfactor import client

SRC = Path(client.__file__).resolve().parent


class FakeVnstockError(Exception):
    code = "VNSTOCK_FAKE"


class FakeRateLimitError(FakeVnstockError):
    pass


class FakeNetworkError(FakeVnstockError):
    pass


class FakeCredentialError(FakeVnstockError):
    pass


class FakeAuthError(FakeVnstockError):
    pass


class FakeLoginRequiredError(FakeVnstockError):
    pass


class FakeOtpRequiredError(FakeVnstockError):
    pass


class FakeSourceError(FakeVnstockError):
    pass


class FakeSource:
    """Stands in for the vnstock module; answers come from ``responses[method]``."""

    def __init__(self) -> None:
        self.responses: dict[str, object] = {}
        self.calls: list[tuple[str, str, str | None, dict]] = []
        self.errors = types.SimpleNamespace(
            VnstockError=FakeVnstockError,
            RateLimitError=FakeRateLimitError,
            NetworkError=FakeNetworkError,
            CredentialError=FakeCredentialError,
            AuthError=FakeAuthError,
            LoginRequiredError=FakeLoginRequiredError,
            OtpRequiredError=FakeOtpRequiredError,
            SourceError=FakeSourceError,
        )

    def answer(self, method: str, source: str, target: str | None, kwargs: dict):
        self.calls.append((method, source, target, kwargs))
        response = self.responses[method]
        if isinstance(response, Exception):
            raise response
        return response

    def _node(self, source: str, target: str | None, methods: list[str]):
        def bind(method: str):
            return lambda **kwargs: self.answer(method, source, target, kwargs)

        return types.SimpleNamespace(**{m: bind(m) for m in methods})

    def Market(self, source: str):  # noqa: N802 - mirrors the vnstock name
        methods = ["ohlcv", "volume_history", "trade_history"]
        return types.SimpleNamespace(equity=lambda symbol: self._node(source, symbol, methods))

    def Reference(self, source: str):  # noqa: N802
        return types.SimpleNamespace(
            index=lambda index: self._node(source, index, ["constituents", "members"]),
            company=lambda symbol: self._node(source, symbol, ["listing", "trading_stats"]),
        )

    def Fundamental(self, source: str):  # noqa: N802
        methods = ["income_statement", "balance_sheet"]
        return types.SimpleNamespace(equity=lambda symbol: self._node(source, symbol, methods))


@pytest.fixture
def fake(monkeypatch) -> FakeSource:
    source = FakeSource()
    monkeypatch.setitem(sys.modules, "vnstock", source)
    return source


def source_bars(n: int = 5, partial: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-10-26", periods=n, freq="D", tz="Asia/Ho_Chi_Minh"),
            "open": [10.0 + i for i in range(n)],
            "high": [11.0 + i for i in range(n)],
            "low": [9.0 + i for i in range(n)],
            "close": [10.5 + i for i in range(n)],
            "volume": pd.array([1000 * (i + 1) for i in range(n)], dtype="Int64"),
        }
    )
    df.attrs["last_bar_partial"] = partial
    return df


def source_statement(items: dict[str, list[float]], company_type: str = "regular") -> pd.DataFrame:
    periods = ["2025-Q3", "2025-Q4"]
    rows = [
        {"report_period": p, "item_id": item, "item_name": item.lower(), "item_value": v}
        for item, values in items.items()
        for p, v in zip(periods, values, strict=True)
    ]
    df = pd.DataFrame(rows)
    df.attrs["company_type"] = company_type
    return df


INCOME_ITEMS = {
    "IS_NET_REVENUE": [100.0, 110.0],
    "IS_NET_PROFIT_AFTER_TAX": [10.0, 12.0],
    "IS_PROFIT_AFTER_TAX_FOR_SHAREHOLDERS_OF_PARENT_COMPANY": [9.0, 11.0],
}


# --- column mapping to the internal schema ------------------------------------------------


def test_fetch_prices_maps_to_internal_schema(fake):
    fake.responses["ohlcv"] = source_bars()
    out = client.fetch_prices("FPT", "2026-10-26", "2026-10-30", "fhsc")

    assert list(out.columns) == client.BAR_COLUMNS
    assert out["date"].dt.tz is None
    assert list(out["date"]) == list(pd.date_range("2026-10-26", periods=5, freq="D"))
    assert set(out["symbol"]) == {"FPT"}
    assert list(out["close_adj"]) == [10.5, 11.5, 12.5, 13.5, 14.5]
    assert out["volume"].dtype == float

    method, source, symbol, kwargs = fake.calls[0]
    assert (method, source, symbol) == ("ohlcv", "fhsc", "FPT")
    assert kwargs == {
        "start": "2026-10-26",
        "end": "2026-10-30",
        "interval": "1D",
        "compat": "v6",
    }


def test_fetch_prices_uses_the_vietnam_calendar_day(fake):
    bars = source_bars(2)
    bars["time"] = pd.to_datetime(["2026-10-26 17:30", "2026-10-27 17:30"], utc=True)
    fake.responses["ohlcv"] = bars
    out = client.fetch_prices("FPT", "2026-10-26", "2026-10-30", "fhsc")
    assert list(out["date"]) == [pd.Timestamp("2026-10-27"), pd.Timestamp("2026-10-28")]


@pytest.mark.parametrize(("partial", "expected_rows"), [(True, 4), (False, 5)])
def test_fetch_prices_drops_an_unfinished_last_bar(fake, partial, expected_rows):
    fake.responses["ohlcv"] = source_bars(partial=partial)
    out = client.fetch_prices("FPT", "2026-10-26", "2026-10-30", "fhsc")
    assert len(out) == expected_rows
    assert out["date"].max() == pd.Timestamp("2026-10-26") + pd.Timedelta(days=expected_rows - 1)


def test_fetch_prices_with_no_bars_returns_the_empty_schema(fake):
    fake.responses["ohlcv"] = source_bars(0)
    out = client.fetch_prices("NEW", "2026-10-26", "2026-10-30", "fhsc")
    assert len(out) == 0
    assert list(out.columns) == client.BAR_COLUMNS


def test_fetch_traded_value_maps_to_internal_schema(fake):
    fake.responses["volume_history"] = pd.DataFrame(
        {
            "time": pd.date_range("2021-08-02", periods=3, freq="D", tz="Asia/Ho_Chi_Minh"),
            "matched_volume": [100, 200, 300],
            "matched_value": [1.0e9, 2.0e9, 3.0e9],
            "block_volume": [0, 0, 5],
            "total_value": [1.0e9, 2.0e9, 3.5e9],
        }
    )
    out = client.fetch_traded_value("FPT", "2021-08-02", "2026-08-01", "fhsc")
    assert list(out.columns) == client.TRADED_VALUE_COLUMNS
    assert out["date"].dt.tz is None
    assert list(out["matched_value"]) == [1.0e9, 2.0e9, 3.0e9]
    assert fake.calls[0][3] == {"start": "2021-08-02", "end": "2026-08-01", "compat": "v6"}


def test_fetch_statement_maps_items_periods_and_company_type(fake):
    fake.responses["income_statement"] = source_statement(INCOME_ITEMS, company_type="bank")
    out = client.fetch_statement("VCB", "income_statement", "fhsc")

    assert list(out.columns) == client.STATEMENT_COLUMNS
    assert set(out["item_id"]) == {"npat_parent", "npat_total"}
    assert set(out["company_type"]) == {"bank"}
    assert set(out["period_end"]) == {pd.Timestamp("2025-09-30"), pd.Timestamp("2025-12-31")}
    q4_parent = out[(out["report_period"] == "2025-Q4") & (out["item_id"] == "npat_parent")]
    assert list(q4_parent["value"]) == [11.0]
    assert fake.calls[0][3] == {"period": "quarter", "format": "long", "compat": "v6"}


def test_fetch_statement_reads_total_equity_from_the_balance_sheet(fake):
    fake.responses["balance_sheet"] = source_statement(
        {"BS_EQUITY": [500.0, 520.0], "BS_TOTAL_ASSETS": [900.0, 950.0]}
    )
    out = client.fetch_statement("FPT", "balance_sheet", "fhsc")
    assert set(out["item_id"]) == {"equity"}
    assert list(out.sort_values("period_end")["value"]) == [500.0, 520.0]


@pytest.mark.parametrize(
    ("label", "expected"),
    [("2025-Q1", "2025-03-31"), ("2025-Q2", "2025-06-30"), ("2024-Q4", "2024-12-31")],
)
def test_quarter_end(label, expected):
    assert client.quarter_end(label) == pd.Timestamp(expected)


@pytest.mark.parametrize("label", ["2025", "2025-Q5", "Q1-2025", "2025Q1"])
def test_quarter_end_rejects_other_labels(label):
    with pytest.raises(client.DataError):
        client.quarter_end(label)


def test_fetch_shares_returns_one_row(fake):
    fake.responses["listing"] = pd.DataFrame(
        {"listed_shares": [1.5e9], "outstanding_shares": [1.47e9]}
    )
    out = client.fetch_shares("FPT", "fhsc")
    assert list(out.columns) == client.SHARES_COLUMNS
    assert out.to_dict("records") == [{"symbol": "FPT", "outstanding_shares": 1.47e9}]


def test_fetch_candidates_keeps_one_exchange(fake):
    fake.responses["constituents"] = pd.DataFrame(
        {
            "symbol": ["HPG", "FPT", "SHS"],
            "name": ["a", "b", "c"],
            "exchange": ["HOSE", "HOSE", "HNX"],
        }
    )
    out = client.fetch_candidates("VNINDEX", "HOSE", "fhsc")
    assert list(out.columns) == ["symbol", "exchange"]
    assert list(out["symbol"]) == ["FPT", "HPG"]
    assert fake.calls[0][:3] == ("constituents", "fhsc", "VNINDEX")


# --- responses that cannot be mapped raise instead of becoming empty tables ---------------


def test_missing_source_column_raises(fake):
    fake.responses["ohlcv"] = source_bars().drop(columns="close")
    with pytest.raises(client.DataError, match="close"):
        client.fetch_prices("FPT", "2026-10-26", "2026-10-30", "fhsc")


def test_missing_price_values_raise(fake):
    bars = source_bars()
    bars.loc[2, "close"] = None
    fake.responses["ohlcv"] = bars
    with pytest.raises(client.DataError, match="close"):
        client.fetch_prices("FPT", "2026-10-26", "2026-10-30", "fhsc")


def test_duplicate_dates_raise(fake):
    bars = pd.concat([source_bars(), source_bars().tail(1)], ignore_index=True)
    fake.responses["ohlcv"] = bars
    with pytest.raises(client.DataError, match="duplicate"):
        client.fetch_prices("FPT", "2026-10-26", "2026-10-30", "fhsc")


def test_statement_without_a_required_item_raises(fake):
    items = {k: v for k, v in INCOME_ITEMS.items() if k != "IS_NET_PROFIT_AFTER_TAX"}
    fake.responses["income_statement"] = source_statement(items)
    with pytest.raises(client.DataError, match="IS_NET_PROFIT_AFTER_TAX"):
        client.fetch_statement("FPT", "income_statement", "fhsc")


def test_statement_with_unknown_company_type_raises(fake):
    fake.responses["income_statement"] = source_statement(INCOME_ITEMS, company_type="fund")
    with pytest.raises(client.DataError, match="company_type"):
        client.fetch_statement("FPT", "income_statement", "fhsc")


@pytest.mark.parametrize("shares", [0.0, -5.0, None])
def test_shares_must_be_positive(fake, shares):
    fake.responses["listing"] = pd.DataFrame({"outstanding_shares": [shares]})
    with pytest.raises(client.DataError):
        client.fetch_shares("FPT", "fhsc")


# --- source errors propagate unchanged ----------------------------------------------------

FETCHES = {
    "ohlcv": lambda: client.fetch_prices("FPT", "2026-10-26", "2026-10-30", "fhsc"),
    "volume_history": lambda: client.fetch_traded_value("FPT", "2021-08-02", "2026-08-01", "fhsc"),
    "balance_sheet": lambda: client.fetch_statement("FPT", "balance_sheet", "fhsc"),
    "income_statement": lambda: client.fetch_statement("FPT", "income_statement", "fhsc"),
    "listing": lambda: client.fetch_shares("FPT", "fhsc"),
    "constituents": lambda: client.fetch_candidates("VNINDEX", "HOSE", "fhsc"),
}


@pytest.mark.parametrize("method", sorted(FETCHES))
@pytest.mark.parametrize("error", [FakeSourceError, FakeRateLimitError, FakeCredentialError])
def test_source_errors_are_reraised(fake, method, error):
    raised = error("source says no")
    fake.responses[method] = raised
    with pytest.raises(error) as caught:
        FETCHES[method]()
    assert caught.value is raised


@pytest.mark.parametrize(
    ("error", "retryable", "credential"),
    [
        (FakeRateLimitError, True, False),
        (FakeNetworkError, True, False),
        (FakeSourceError, False, False),
        (FakeCredentialError, False, True),
        (FakeAuthError, False, True),
        (FakeLoginRequiredError, False, True),
        (FakeOtpRequiredError, False, True),
    ],
)
def test_describe_error_classifies_vnstock_errors(fake, error, retryable, credential):
    info = client.describe_error(error("boom"))
    assert info["error_class"] == error.__name__
    assert info["code"] == "VNSTOCK_FAKE"
    assert info["retryable"] is retryable
    assert info["credential"] is credential


def test_describe_error_ignores_other_exceptions(fake):
    assert client.describe_error(ValueError("not a source error")) is None


def test_describe_error_without_vnstock_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "vnstock", None)  # makes ``import vnstock`` fail
    assert client.describe_error(RuntimeError("boom")) is None


# --- AGENTS.md rule 4 ---------------------------------------------------------------------


def _vnstock_imports(tree: ast.AST) -> list[ast.AST]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(name.split(".")[0].startswith("vnstock") for name in names):
            found.append(node)
    return found


def test_only_client_imports_vnstock():
    offenders = [
        path.name
        for path in sorted(SRC.glob("*.py"))
        if path.name != "client.py" and _vnstock_imports(ast.parse(path.read_text("utf-8")))
    ]
    assert offenders == []


def test_client_imports_vnstock_only_inside_functions():
    tree = ast.parse((SRC / "client.py").read_text("utf-8"))
    assert _vnstock_imports(tree), "client.py is expected to import vnstock somewhere"
    top_level = [node for node in tree.body if _vnstock_imports(node)]
    assert all(isinstance(node, ast.FunctionDef | ast.ClassDef) for node in top_level)
