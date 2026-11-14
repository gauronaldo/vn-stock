"""Walk-forward never trains on a label that is unknown at the prediction date."""

import copy

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_fundamentals, make_prices
from vnfactor import calendar as cal
from vnfactor import ml, pipeline


@pytest.mark.parametrize(("min_train", "every", "gap"), [(36, 12, 1), (6, 4, 1), (3, 1, 2)])
def test_schedule_respects_the_label_gap(min_train, every, gap):
    plan = ml.schedule(104, min_train, every, gap)
    assert min(plan) == min_train + gap
    for k, p in plan.items():
        assert len(p["train"]) >= min_train
        assert max(p["train"]) <= k - 1 - gap
        assert p["train"] == list(range(len(p["train"])))  # expanding window from month 0
        assert (p["refit"] - min(plan)) % every == 0 and p["refit"] <= k


def test_schedule_matches_the_methodology():
    plan = ml.schedule(104, 36, 12, 1)
    assert min(plan) == 37
    assert sorted({p["refit"] for p in plan.values()}) == [37, 49, 61, 73, 85, 97]
    assert plan[37]["train"] == list(range(36))
    assert plan[48]["train"] == list(range(36))


@pytest.fixture(scope="module")
def setup():
    prices = make_prices(n_sessions=1100)
    symbols = sorted(prices["symbol"].unique())
    statements = make_fundamentals(symbols, last_quarter="2024Q1").drop(columns="available_date")
    shares = pd.DataFrame({"symbol": symbols, "outstanding_shares": 3e8})
    from vnfactor.config import load_config

    cfg = copy.deepcopy(load_config("configs/base.yaml"))
    cfg["calendar"].update(first_rebalance="2021-01-01", last_rebalance="2024-01-31")
    cfg["universe"]["top_n"] = 10
    cfg["ml"].update(min_train_months=6, retrain_every_months=4)
    cfg["ml"]["lightgbm"].update(n_estimators=20, min_child_samples=5)
    signals = pipeline.build_signals(prices, statements, shares, cfg)
    fwd = pipeline.build_returns(prices, cfg)
    return prices, cfg, signals, fwd


def test_training_labels_are_known_at_every_prediction_date(setup):
    prices, cfg, signals, fwd = setup
    months = sorted(signals["universe"]["rebalance_date"].unique())
    predict = months[7:]
    _, log = ml.walk_forward(signals["factors"], signals["universe"], fwd, predict, cfg)
    days = cal.trading_calendar(prices)
    periods = cal.holding_periods(days, pd.DatetimeIndex(months)).set_index("rebalance_date")
    for _, row in log.iterrows():
        label_known = periods.loc[row["train_last"], "exit_date"]
        assert label_known <= row["predict_date"]
        assert row["train_last"] < row["predict_date"]
        assert row["refit_date"] <= row["predict_date"]


def test_predictions_ignore_labels_after_the_prediction_date(setup):
    _, cfg, signals, fwd = setup
    months = sorted(signals["universe"]["rebalance_date"].unique())
    t = months[12]
    scrambled = fwd.copy()
    later = scrambled["rebalance_date"] >= months[11]  # month 11's label is unknown at t
    rng = np.random.default_rng(0)
    scrambled.loc[later, "fwd_ret_1m"] = rng.normal(size=int(later.sum()))
    a, _ = ml.walk_forward(signals["factors"], signals["universe"], fwd, [t], cfg)
    b, _ = ml.walk_forward(signals["factors"], signals["universe"], scrambled, [t], cfg)
    pd.testing.assert_frame_equal(a, b)


def test_scores_follow_the_schema(setup):
    _, cfg, signals, fwd = setup
    months = sorted(signals["universe"]["rebalance_date"].unique())
    scores, log = ml.walk_forward(signals["factors"], signals["universe"], fwd, months[7:9], cfg)
    assert list(scores.columns) == ["rebalance_date", "symbol", "model", "score", "quintile"]
    assert set(scores["model"]) == {"ridge", "lightgbm"}
    assert scores["score"].notna().all()
    assert set(log.columns) >= set(cfg["factors"]["definitions"])


def test_too_early_prediction_date_raises(setup):
    _, cfg, signals, fwd = setup
    months = sorted(signals["universe"]["rebalance_date"].unique())
    with pytest.raises(ValueError, match="months of known labels"):
        ml.walk_forward(signals["factors"], signals["universe"], fwd, [months[3]], cfg)
