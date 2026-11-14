"""Guard the synthetic fixtures: later tests rely on their schema and edge cases."""

import numpy as np
import pandas as pd

from tests.conftest import (
    FUNDAMENTAL_COLUMNS,
    ITEM_IDS,
    LATE_LISTER,
    LOSS_MAKER,
    MISSING_QUARTER,
    PRICE_COLUMNS,
    SPARSE_TRADER,
    STOCK_DIVIDEND_SYMBOL,
    make_prices,
)


def test_prices_follow_internal_schema(synthetic_prices):
    p = synthetic_prices
    assert list(p.columns) == PRICE_COLUMNS
    assert not p.duplicated(["date", "symbol"]).any()
    assert not p.isna().any().any()
    assert p["date"].dt.tz is None
    assert (p[["open", "high", "low", "close_adj", "value"]] > 0).all().all()
    assert (p["high"] >= p[["open", "close_adj"]].max(axis=1)).all()
    assert (p["low"] <= p[["open", "close_adj"]].min(axis=1)).all()
    assert set(p["value_source"]) == {"matched"}


def test_prices_are_deterministic():
    pd.testing.assert_frame_equal(make_prices(), make_prices())


def test_price_edge_cases_present(synthetic_prices):
    sessions = synthetic_prices.groupby("symbol").size()
    n_dates = synthetic_prices["date"].nunique()
    assert sessions[LATE_LISTER] < 252
    assert sessions[SPARSE_TRADER] / n_dates < 0.8
    div = synthetic_prices[synthetic_prices["symbol"] == STOCK_DIVIDEND_SYMBOL]
    raw_over_adj = div["value"] / (div["close_adj"] * 1000.0 * div["volume"])
    np.testing.assert_allclose(raw_over_adj.iloc[:300], 1.3)
    np.testing.assert_allclose(raw_over_adj.iloc[300:], 1.0)


def test_fundamentals_follow_internal_schema(synthetic_fundamentals):
    f = synthetic_fundamentals
    assert list(f.columns) == FUNDAMENTAL_COLUMNS
    assert not f.duplicated(["symbol", "report_period", "item_id"]).any()
    assert set(f["item_id"]) == set(ITEM_IDS)
    assert set(f["company_type"]) == {"regular", "bank", "securities", "insurance"}
    assert ((f["available_date"] - f["period_end"]) == pd.Timedelta(days=45)).all()
    assert f["period_end"].dt.is_quarter_end.all()


def test_fundamental_edge_cases_present(synthetic_fundamentals):
    f = synthetic_fundamentals
    symbol, quarter = MISSING_QUARTER
    assert f[(f["symbol"] == symbol) & (f["report_period"] == quarter)].empty
    loss = f[(f["symbol"] == LOSS_MAKER) & (f["item_id"] == "npat_parent")]
    assert (loss["value"] < 0).all()
