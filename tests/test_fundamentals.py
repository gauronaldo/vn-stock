import numpy as np
import pandas as pd
import pytest

from tests.conftest import MISSING_QUARTER
from vnfactor import fundamentals as fund


def _quarters(symbol, values: dict[str, list[float]], start="2020Q1", lag=45):
    n = len(next(iter(values.values())))
    periods = pd.period_range(start, periods=n, freq="Q")
    rows = []
    for item, vals in values.items():
        for p, v in zip(periods, vals, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "report_period": f"{p.year}-Q{p.quarter}",
                    "item_id": item,
                    "period_end": p.end_time.normalize(),
                    "value": v,
                    "company_type": "regular",
                }
            )
    return fund.add_available_date(pd.DataFrame(rows), lag)


def test_available_date_is_period_end_plus_lag():
    f = _quarters("A", {"equity": [1.0, 2.0]}, lag=45)
    assert (f["available_date"] - f["period_end"] == pd.Timedelta(days=45)).all()


def test_ttm_is_sum_of_four_consecutive_quarters():
    f = _quarters(
        "A",
        {
            "npat_parent": [1.0, 2.0, 3.0, 4.0, 5.0],
            "npat_total": [2.0, 2.0, 2.0, 2.0, 2.0],
            "equity": [10.0, 11.0, 12.0, 13.0, 14.0],
        },
    )
    t = fund.ttm_table(f, 4).set_index("period_end")
    assert t["npat_parent_ttm"].isna().sum() == 3
    assert t["npat_parent_ttm"].iloc[3] == 10.0
    assert t["npat_parent_ttm"].iloc[4] == 14.0
    assert t["npat_total_ttm"].iloc[4] == 8.0
    assert t["equity_avg"].iloc[4] == pytest.approx((14.0 + 10.0) / 2)
    assert np.isnan(t["equity_avg"].iloc[3])


def test_ttm_is_nan_when_a_quarter_is_missing(synthetic_fundamentals):
    t = fund.ttm_table(synthetic_fundamentals, 4)
    symbol, missing = MISSING_QUARTER
    s = t[t["symbol"] == symbol].set_index("period_end")["npat_parent_ttm"]
    gap = pd.Period(missing.replace("-", ""), freq="Q")
    affected = [(gap + k).end_time.normalize() for k in (1, 2, 3)]
    assert s.loc[affected].isna().all()
    recovered = (gap + 4).end_time.normalize()
    assert np.isfinite(s.loc[recovered])
    assert pd.Timestamp(gap.end_time.normalize()) not in s.index


def test_ttm_is_nan_when_a_value_is_missing():
    f = _quarters("A", {"npat_parent": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]})
    s = fund.ttm_table(f, 4)["npat_parent_ttm"]
    assert s.iloc[:5].isna().all()
    assert s.iloc[5] == 18.0


def test_asof_join_uses_reports_available_on_t_but_not_after():
    f = _quarters("A", {"npat_parent": [1.0, 2.0, 3.0, 4.0, 5.0]})
    table = fund.ttm_table(f, 4)
    last_available = table["available_date"].iloc[4]
    keys = pd.DataFrame(
        {
            "rebalance_date": [last_available - pd.Timedelta(days=1), last_available],
            "symbol": ["A", "A"],
        }
    )
    out = fund.asof_join(table, keys)
    assert out["npat_parent_ttm"].tolist() == [10.0, 14.0]


def test_asof_join_before_any_report_is_nan():
    table = fund.ttm_table(_quarters("A", {"npat_parent": [1.0, 2.0, 3.0, 4.0]}), 4)
    keys = pd.DataFrame({"rebalance_date": [pd.Timestamp("2019-12-31")], "symbol": ["A"]})
    assert fund.asof_join(table, keys)["npat_parent_ttm"].isna().all()


def test_company_types_takes_latest():
    f = _quarters("A", {"equity": [1.0, 2.0]})
    f.loc[f["period_end"] == f["period_end"].max(), "company_type"] = "bank"
    assert fund.company_types(f).to_dict() == {"A": "bank"}
