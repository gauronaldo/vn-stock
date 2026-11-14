"""The most important test: signals at t do not change when data after t is removed."""

import copy

import pandas as pd
import pytest

from tests.conftest import make_fundamentals, make_prices
from vnfactor import pipeline

DATES = ["2021-01-29", "2021-04-30", "2021-07-30", "2021-10-29", "2022-01-31", "2022-03-31"]


@pytest.fixture(scope="module")
def data():
    prices = make_prices()
    symbols = sorted(prices["symbol"].unique())
    statements = make_fundamentals(symbols).drop(columns="available_date")
    shares = pd.DataFrame({"symbol": symbols, "outstanding_shares": 3e8})
    return prices, statements, shares


@pytest.fixture(scope="module")
def cfg():
    from vnfactor.config import load_config

    cfg = copy.deepcopy(load_config("configs/base.yaml"))
    cfg["calendar"].update(first_rebalance="2021-01-01", last_rebalance="2022-03-31")
    cfg["universe"]["top_n"] = 8
    return cfg


@pytest.fixture(scope="module")
def full(data, cfg):
    return pipeline.build_signals(*data, cfg)


@pytest.mark.parametrize("t", DATES)
def test_signals_at_t_ignore_data_after_t(data, cfg, full, t):
    prices, statements, shares = data
    t = pd.Timestamp(t)
    lag = pd.Timedelta(days=int(cfg["fundamentals"]["lag_days"]))
    cut = pipeline.build_signals(
        prices[prices["date"] <= t],
        statements[statements["period_end"] + lag <= t],
        shares,
        cfg,
        last_rebalance=str(t.date()),
    )
    for name in ("universe", "factors", "scores"):
        a = full[name][full[name]["rebalance_date"] == t].reset_index(drop=True)
        b = cut[name][cut[name]["rebalance_date"] == t].reset_index(drop=True)
        assert len(a) > 0, name
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, obj=name)


def test_every_test_date_is_a_rebalance_date(full):
    assert set(pd.to_datetime(DATES)) <= set(full["scores"]["rebalance_date"])


def test_fundamental_factors_are_present(full):
    f = full["factors"]
    for factor in ("ep", "bp", "roe"):
        assert f.loc[f["factor"] == factor, "z"].notna().any(), factor
