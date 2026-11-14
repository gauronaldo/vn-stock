import numpy as np
import pandas as pd
import pytest

from vnfactor import evaluation as ev
from vnfactor.factors import assign_quantiles


def _panel(n_months=6, n_names=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    rows = []
    for d in dates:
        ret = rng.normal(0, 0.05, n_names)
        for i in range(n_names):
            rows.append({"rebalance_date": d, "symbol": f"S{i:02d}", "fwd_ret_1m": ret[i]})
    return pd.DataFrame(rows)


def _scores(fwd, transform, model="m"):
    s = fwd.rename(columns={"fwd_ret_1m": "score"}).copy()
    s["score"] = transform(s["score"])
    s["model"] = model
    s["quintile"] = assign_quantiles(s, 5)
    return s[["rebalance_date", "symbol", "model", "score", "quintile"]]


def test_perfect_ordering_gives_ic_of_one():
    fwd = _panel()
    ic = ev.rank_ic(_scores(fwd, lambda x: 3 * x + 1), fwd)
    np.testing.assert_allclose(ic["ic"], 1.0)


def test_reversed_ordering_gives_ic_of_minus_one():
    fwd = _panel()
    ic = ev.rank_ic(_scores(fwd, lambda x: -np.exp(x)), fwd)
    np.testing.assert_allclose(ic["ic"], -1.0)


def test_ic_summary_statistics():
    ic = pd.DataFrame(
        {
            "model": "m",
            "rebalance_date": pd.date_range("2022-01-31", periods=4, freq="ME"),
            "ic": [0.1, 0.2, -0.1, 0.2],
        }
    )
    s = ev.ic_summary(ic).loc["m"]
    assert s["months"] == 4
    assert s["mean_ic"] == pytest.approx(0.1)
    std = np.std([0.1, 0.2, -0.1, 0.2], ddof=1)
    assert s["icir"] == pytest.approx(0.1 / std)
    assert s["t_stat"] == pytest.approx(0.1 / std * 2)
    assert s["pct_positive"] == pytest.approx(0.75)


def test_quintile_returns_order_with_a_perfect_signal():
    fwd = _panel()
    q = ev.quantile_summary(ev.quantile_returns(_scores(fwd, lambda x: x), fwd)).loc["m"]
    means = q[["q1", "q2", "q3", "q4", "q5"]].to_numpy()
    assert (np.diff(means) > 0).all()
    assert q["spread"] == pytest.approx(q["q5"] - q["q1"])


def test_months_with_too_few_names_have_no_ic():
    fwd = _panel(n_names=8)
    ic = ev.rank_ic(_scores(fwd, lambda x: x), fwd)
    assert ic["ic"].isna().all()


def test_signal_table_has_one_row_per_model():
    fwd = _panel()
    scores = pd.concat([_scores(fwd, lambda x: x, "a"), _scores(fwd, lambda x: -x, "b")])
    table = ev.signal_table(scores, fwd)
    assert list(table.index) == ["a", "b"]
    assert table.loc["a", "mean_ic"] == pytest.approx(1.0)
    assert table.loc["b", "spread"] < 0
