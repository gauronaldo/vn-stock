import numpy as np
import pandas as pd
import pytest

from vnfactor import calendar as cal
from vnfactor import factors as fac

DEFS = {
    "mom_12_1": {"sign": 1, "lookback": 252, "skip": 21},
    "rev_1m": {"sign": -1, "lookback": 21},
    "vol_126": {"sign": -1, "window": 126},
    "adv_63": {"sign": -1, "window": 63},
}
PRE = {
    "winsor_lower": 0.01,
    "winsor_upper": 0.99,
    "group_zscore_factors": ["ep"],
}


def _one_symbol(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    value = rng.uniform(1e9, 5e9, n)
    prices = pd.DataFrame(
        {"date": dates, "symbol": "A", "close_adj": close, "volume": 1000.0, "value": value}
    )
    return prices, dates, close, value


def _price_factors(prices, t):
    days = cal.trading_calendar(prices)
    return fac.price_factors(prices, days, pd.DatetimeIndex([t]), DEFS, 0.2).iloc[0]


# --- formulas -----------------------------------------------------------------------------


def test_price_factor_formulas_by_hand():
    prices, dates, close, value = _one_symbol()
    i = 280
    row = _price_factors(prices, dates[i])
    assert row["mom_12_1"] == pytest.approx(close[i - 21] / close[i - 252] - 1)
    assert row["rev_1m"] == pytest.approx(close[i] / close[i - 21] - 1)
    log_ret = np.diff(np.log(close))[i - 126 : i]
    assert row["vol_126"] == pytest.approx(np.std(log_ret, ddof=1))
    assert row["adv_63"] == pytest.approx(np.log(np.median(value[i - 62 : i + 1])))


def test_too_short_history_gives_nan():
    prices, dates, _, _ = _one_symbol()
    row = _price_factors(prices, dates[200])
    assert np.isnan(row["mom_12_1"])
    assert np.isfinite(row["rev_1m"]) and np.isfinite(row["vol_126"])


def test_window_missing_more_than_20_percent_gives_nan():
    prices, dates, _, _ = _one_symbol()
    i = 280
    other = prices.assign(symbol="B")
    gap = other["date"].isin(dates[i - 40 : i - 10])  # 30 sessions without a bar
    both = pd.concat([prices, other[~gap]])
    days = cal.trading_calendar(both)
    out = fac.price_factors(both, days, pd.DatetimeIndex([dates[i]]), DEFS, 0.2)
    a = out.xs("A", level="symbol").iloc[0]
    b = out.xs("B", level="symbol").iloc[0]
    assert a.notna().all()
    assert np.isnan(b["adv_63"])  # 30 of 63 missing
    assert np.isnan(b["rev_1m"])  # 11 of 22 missing
    assert np.isnan(b["vol_126"])  # 30 of 126 missing (23.8%)
    assert np.isfinite(b["mom_12_1"])  # 30 of 232 missing (12.9%)


def test_fundamental_factor_formulas_by_hand():
    pit = pd.DataFrame(
        {
            "rebalance_date": pd.Timestamp("2024-01-31"),
            "symbol": ["A", "B", "C"],
            "npat_parent_ttm": [100e9, -50e9, 10e9],
            "npat_total_ttm": [110e9, -40e9, 12e9],
            "equity": [1000e9, 500e9, -5e9],
            "equity_avg": [900e9, 400e9, 5e9],
        }
    )
    close = pd.Series([20.0, 10.0, 5.0])  # thousand VND
    shares = pd.Series({"A": 1e8, "B": 2e8, "C": 1e7})
    out = fac.fundamental_factors(pit, close, shares).droplevel("rebalance_date")
    mcap_a = 20.0 * 1000 * 1e8
    assert out.loc["A", "ep"] == pytest.approx(100e9 / mcap_a)
    assert out.loc["A", "bp"] == pytest.approx(1000e9 / mcap_a)
    assert out.loc["A", "roe"] == pytest.approx(110e9 / 900e9)
    assert out.loc["B", "ep"] < 0  # losses keep a meaningful, rankable E/P
    assert np.isnan(out.loc["C", "bp"]) and np.isnan(out.loc["C", "roe"])
    assert np.isfinite(out.loc["C", "ep"])


def test_missing_shares_give_nan_valuation():
    pit = pd.DataFrame(
        {
            "rebalance_date": pd.Timestamp("2024-01-31"),
            "symbol": ["A"],
            "npat_parent_ttm": [1.0],
            "npat_total_ttm": [1.0],
            "equity": [1.0],
            "equity_avg": [1.0],
        }
    )
    out = fac.fundamental_factors(pit, pd.Series([1.0]), pd.Series(dtype=float))
    assert out[["ep", "bp"]].isna().all().all()
    assert np.isfinite(out["roe"].iloc[0])


# --- preprocessing ------------------------------------------------------------------------


def _raw(n=60, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    symbols = [f"X{i:02d}" for i in range(n)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["rebalance_date", "symbol"])
    raw = pd.DataFrame(
        {"mom_12_1": rng.normal(0, 1, len(index)), "ep": rng.normal(0, 1, len(index))},
        index=index,
    )
    raw.iloc[3, 0] = 50.0  # outlier
    raw.iloc[5, 1] = np.nan
    universe = index.to_frame(index=False)
    universe["sector_group"] = np.where(np.arange(len(index)) % 3 == 0, "financial", "other")
    universe.loc[universe.index % 3 == 0, "symbol"] = universe["symbol"]  # keep keys
    return raw, universe


def test_zscore_is_standardised_within_each_group():
    raw, universe = _raw()
    defs = {"mom_12_1": {"sign": 1}, "ep": {"sign": 1}}
    f = fac.preprocess(raw, universe, defs, PRE).merge(universe, on=["rebalance_date", "symbol"])
    mom = f[f["factor"] == "mom_12_1"].groupby("rebalance_date")["z"]
    np.testing.assert_allclose(mom.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(mom.std(ddof=0), 1.0)
    ep = f[f["factor"] == "ep"].groupby(["rebalance_date", "sector_group"])["z"]
    np.testing.assert_allclose(ep.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(ep.std(ddof=0), 1.0)


def test_nan_stays_nan_and_outliers_are_capped():
    raw, universe = _raw()
    defs = {"mom_12_1": {"sign": 1}, "ep": {"sign": 1}}
    f = fac.preprocess(raw, universe, defs, PRE).set_index(["rebalance_date", "symbol", "factor"])
    first = raw.index[5]
    assert np.isnan(f.loc[(*first, "ep"), "z"])
    assert np.isnan(f.loc[(*first, "ep"), "raw"])
    outlier = raw.index[3]
    z = f.xs("mom_12_1", level="factor").loc[outlier[0], "z"]
    assert z.loc[outlier[1]] == pytest.approx(z.max())
    col = raw.loc[outlier[0], "mom_12_1"]
    unclipped = (50.0 - col.mean()) / col.std(ddof=0)
    assert z.max() < unclipped  # the outlier is pulled in to the 99th percentile


def test_sign_flips_the_z_score():
    raw, universe = _raw()
    plus = fac.preprocess(raw[["mom_12_1"]], universe, {"mom_12_1": {"sign": 1}}, PRE)
    minus = fac.preprocess(raw[["mom_12_1"]], universe, {"mom_12_1": {"sign": -1}}, PRE)
    np.testing.assert_allclose(plus["z"], -minus["z"])
    np.testing.assert_allclose(plus["raw"], minus["raw"])


# --- composite ----------------------------------------------------------------------------


def test_composite_is_mean_of_group_means_and_needs_two_groups():
    t = pd.Timestamp("2024-01-31")
    rows = {
        "A": {"mom_12_1": 1.0, "vol_126": 0.0, "ep": 2.0, "bp": 0.0, "roe": np.nan},
        "B": {"mom_12_1": np.nan, "vol_126": np.nan, "ep": 1.0, "bp": np.nan, "roe": np.nan},
        "C": {"mom_12_1": -1.0, "vol_126": np.nan, "ep": np.nan, "bp": np.nan, "roe": 1.0},
    }
    long = [
        {"rebalance_date": t, "symbol": s, "factor": f, "raw": z, "z": z}
        for s, d in rows.items()
        for f, z in d.items()
    ]
    groups = {
        "momentum": ["mom_12_1"],
        "volatility": ["vol_126"],
        "value": ["ep", "bp"],
        "quality": ["roe"],
    }
    scores = fac.composite_scores(pd.DataFrame(long), groups, 2, 5).set_index("symbol")
    assert scores.loc["A", "score"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert "B" not in scores.index  # only one group available
    assert scores.loc["C", "score"] == pytest.approx(0.0)
    assert set(scores["model"]) == {"composite"}


def test_quantiles_split_evenly_with_highest_score_in_top():
    scores = pd.DataFrame({"rebalance_date": pd.Timestamp("2024-01-31"), "score": np.arange(10.0)})
    q = fac.assign_quantiles(scores, 5)
    assert q.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
