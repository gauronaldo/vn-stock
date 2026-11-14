import numpy as np
import pandas as pd
import pytest

from vnfactor import calendar as cal


def _prices(dates, closes: dict[str, list[float]]) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"date": dates[: len(c)], "symbol": s, "close_adj": c})
        for s, c in closes.items()
    ]
    return pd.concat(frames, ignore_index=True)


def test_rebalance_dates_are_last_sessions_of_each_month(synthetic_prices):
    days = cal.trading_calendar(synthetic_prices)
    rebal = cal.rebalance_dates(days, "2020-03-01", "2021-06-30")
    assert len(rebal) == 16
    for d in rebal:
        same_month = days[days.to_period("M") == d.to_period("M")]
        assert d == same_month.max()


def test_holding_periods_chain_without_gaps(synthetic_prices):
    days = cal.trading_calendar(synthetic_prices)
    rebal = cal.rebalance_dates(days, "2020-03-01", "2021-06-30")
    periods = cal.holding_periods(days, rebal, lag=1)
    assert (periods["entry_date"].iloc[1:].to_numpy() == periods["exit_date"].iloc[:-1]).all()
    for _, row in periods.iterrows():
        assert days.get_loc(row["entry_date"]) == days.get_loc(row["rebalance_date"]) + 1


def test_forward_return_from_t_plus_1_to_next_rebalance_plus_1():
    dates = pd.bdate_range("2021-01-25", "2021-03-10")
    closes = {"A": list(np.arange(1.0, len(dates) + 1.0))}
    prices = _prices(dates, closes)
    days = cal.trading_calendar(prices)
    rebal = pd.DatetimeIndex(["2021-01-29"])
    fwd = cal.forward_returns(prices, days, rebal, lag=1)
    entry = days[days.get_loc(pd.Timestamp("2021-01-29")) + 1]  # 2021-02-01
    exit_ = days[days.get_loc(pd.Timestamp("2021-02-26")) + 1]  # 2021-03-01
    p = dict(zip(dates, closes["A"], strict=True))
    assert fwd["fwd_ret_1m"].item() == pytest.approx(p[exit_] / p[entry] - 1)


def test_missing_bar_at_exit_uses_the_last_close():
    dates = pd.bdate_range("2021-01-25", "2021-03-10")
    prices = _prices(dates, {"A": [10.0] * len(dates), "B": [10.0] * len(dates)})
    prices.loc[(prices["symbol"] == "B") & (prices["date"] == "2021-02-26"), "close_adj"] = 12.0
    prices = prices[~((prices["symbol"] == "B") & (prices["date"] == "2021-03-01"))]
    days = cal.trading_calendar(prices)
    fwd = cal.forward_returns(prices, days, pd.DatetimeIndex(["2021-01-29"]))
    assert fwd.set_index("symbol").loc["B", "fwd_ret_1m"] == pytest.approx(0.2)


def test_calendar_must_reach_the_last_exit(synthetic_prices):
    days = cal.trading_calendar(synthetic_prices)
    last_end = cal.month_ends(days)[-2]
    with pytest.raises(ValueError):
        cal.holding_periods(days, pd.DatetimeIndex([last_end]), lag=1)
