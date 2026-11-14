import numpy as np
import pandas as pd
import pytest

from vnfactor import backtest as bt


def _data(n_months=12, n_names=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    universe = pd.DataFrame(
        [(d, f"S{i}") for d in dates for i in range(n_names)], columns=["rebalance_date", "symbol"]
    )
    fwd = universe.assign(fwd_ret_1m=rng.normal(0.01, 0.06, len(universe)))
    scores = fwd.rename(columns={"fwd_ret_1m": "score"}).assign(model="m")
    scores["score"] = rng.normal(size=len(scores))
    scores["quintile"] = scores.groupby("rebalance_date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
    )
    return universe, fwd, scores


def test_weights_sum_to_one():
    universe, _, scores = _data()
    for w in (bt.top_quantile_weights(scores, 5), bt.equal_weights(universe)):
        np.testing.assert_allclose(w.groupby("rebalance_date")["weight"].sum(), 1.0)
    assert bt.top_quantile_weights(scores, 5).groupby("rebalance_date").size().eq(2).all()


def test_holding_the_whole_universe_without_costs_matches_the_benchmark():
    universe, fwd, _ = _data()
    result = bt.run_backtest(bt.equal_weights(universe), fwd, 0.0, 0.0)
    expected = fwd.groupby("rebalance_date")["fwd_ret_1m"].mean().to_numpy()
    np.testing.assert_allclose(result["gross"], expected)
    np.testing.assert_allclose(result["net"], expected)


def test_costs_reduce_returns_by_turnover_times_fee():
    universe, fwd, scores = _data()
    w = bt.top_quantile_weights(scores, 5)
    free = bt.run_backtest(w, fwd, 0.0, 0.0)
    paid = bt.run_backtest(w, fwd, 0.0015, 0.0)
    np.testing.assert_allclose(free["gross"] - paid["net"], paid["turnover"] * 0.0015)
    assert paid["turnover"].iloc[0] == pytest.approx(1.0)  # first month buys from cash


def test_sell_tax_applies_to_sells_only():
    universe, fwd, scores = _data()
    w = bt.top_quantile_weights(scores, 5)
    res = bt.run_backtest(w, fwd, 0.0, 0.001)
    assert res["cost"].iloc[0] == 0.0  # nothing to sell in the first month
    # turnover = buys + sells and buys = sells after the first month (fully invested)
    np.testing.assert_allclose(res["cost"].iloc[1:], res["turnover"].iloc[1:] / 2 * 0.001)


def test_unchanged_weights_only_trade_the_drift():
    dates = pd.date_range("2022-01-31", periods=3, freq="ME")
    w = pd.DataFrame(
        [(d, s, 0.5) for d in dates for s in ("A", "B")],
        columns=["rebalance_date", "symbol", "weight"],
    )
    fwd = w.drop(columns="weight").assign(fwd_ret_1m=[0.1, 0.1, 0.2, 0.0, 0.0, 0.0])
    res = bt.run_backtest(w, fwd, 0.01, 0.0)
    assert res["turnover"].iloc[1] == pytest.approx(0.0)  # equal returns: no drift
    drift = 0.5 * 1.2 / 1.1 - 0.5
    assert res["turnover"].iloc[2] == pytest.approx(2 * drift)


def test_missing_return_of_a_held_name_raises():
    universe, fwd, _ = _data()
    with pytest.raises(ValueError, match="no forward return"):
        bt.run_backtest(bt.equal_weights(universe), fwd.iloc[1:], 0.0, 0.0)


def test_performance_metrics_by_hand():
    r = pd.Series([0.1, -0.2, 0.1, 0.05])
    perf = bt.performance(r, 12)
    wealth = np.cumprod(1 + r)
    assert perf["cagr"] == pytest.approx(wealth.iloc[-1] ** 3 - 1)
    assert perf["max_drawdown"] == pytest.approx(0.88 / 1.1 - 1)
    assert perf["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(12))
