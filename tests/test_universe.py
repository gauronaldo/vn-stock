import pandas as pd

from tests.conftest import LATE_LISTER, SPARSE_TRADER
from vnfactor import calendar as cal
from vnfactor import universe as uni

SETTINGS = {
    "min_history_sessions": 252,
    "liquidity_window": 126,
    "min_trading_fraction": 0.8,
    "top_n": 5,
}


def _build(prices, settings=SETTINGS, members=None, first="2021-01-01", last="2022-03-31"):
    days = cal.trading_calendar(prices)
    rebal = cal.rebalance_dates(days, first, last)
    groups = pd.Series({"S01": uni.FINANCIAL, "S00": uni.NON_FINANCIAL})
    return uni.build_universe(prices, days, rebal, settings, groups, members=members)


def test_size_never_exceeds_top_n(synthetic_prices):
    out = _build(synthetic_prices)
    assert out.groupby("rebalance_date").size().max() == SETTINGS["top_n"]
    assert list(out.columns) == ["rebalance_date", "symbol", "adv_126", "sector_group"]


def test_short_history_is_excluded(synthetic_prices):
    out = _build(synthetic_prices, {**SETTINGS, "top_n": 12})
    days = cal.trading_calendar(synthetic_prices)
    assert LATE_LISTER not in set(out["symbol"])  # 200 sessions of history at the end

    days = cal.trading_calendar(synthetic_prices)
    s00 = synthetic_prices[synthetic_prices["symbol"] == "S00"]
    later = s00[s00["date"] >= days[250]].assign(symbol="LATE2", value=s00["value"] * 100)
    out = _build(pd.concat([synthetic_prices, later]), {**SETTINGS, "top_n": 12})
    eligible_from = days[250 + 251]
    late = out[out["symbol"] == "LATE2"]
    assert len(late) > 0
    assert (late["rebalance_date"] >= eligible_from).all()


def test_sparse_trader_is_excluded(synthetic_prices):
    out = _build(synthetic_prices, {**SETTINGS, "top_n": 12})
    assert SPARSE_TRADER not in set(out["symbol"])


def test_ranked_by_median_traded_value(synthetic_prices):
    out = _build(synthetic_prices)
    days = cal.trading_calendar(synthetic_prices)
    t = out["rebalance_date"].max()
    window = days[days <= t][-126:]
    p = synthetic_prices[synthetic_prices["date"].isin(window)]
    med = p.groupby("symbol")["value"].median().drop([LATE_LISTER, SPARSE_TRADER], errors="ignore")
    expected = set(med.sort_values(ascending=False).index[:5])
    assert set(out.loc[out["rebalance_date"] == t, "symbol"]) == expected


def test_universe_at_t_ignores_later_data(synthetic_prices):
    full = _build(synthetic_prices)
    for t in full["rebalance_date"].unique()[::3]:
        cut = _build(synthetic_prices[synthetic_prices["date"] <= t], last=str(t.date()))
        pd.testing.assert_frame_equal(
            full[full["rebalance_date"] == t].reset_index(drop=True),
            cut[cut["rebalance_date"] == t].reset_index(drop=True),
        )


def test_sector_groups_and_members_filter(synthetic_prices):
    out = _build(synthetic_prices, {**SETTINGS, "top_n": 12}, members=["S00", "S01", "S02"])
    assert set(out["symbol"]) <= {"S00", "S01", "S02"}
    groups = out.drop_duplicates("symbol").set_index("symbol")["sector_group"]
    assert groups["S01"] == uni.FINANCIAL
    assert groups["S00"] == uni.NON_FINANCIAL
    assert groups["S02"] == uni.UNKNOWN


def test_sector_groups_mapping():
    types = pd.Series({"A": "bank", "B": "regular", "C": "insurance", "D": "securities"})
    groups = uni.sector_groups(types, ["bank", "securities", "insurance"])
    assert groups.to_dict() == {
        "A": uni.FINANCIAL,
        "B": uni.NON_FINANCIAL,
        "C": uni.FINANCIAL,
        "D": uni.FINANCIAL,
    }
