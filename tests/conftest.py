"""Small synthetic datasets in the internal schema (AGENTS.md section 7.2).

Nothing here touches the network. Prices are in thousand VND, traded value and
financial statement items in VND, dates are timezone-naive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vnfactor.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260101

PRICE_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close_adj",
    "volume",
    "value",
    "value_source",
]
FUNDAMENTAL_COLUMNS = [
    "symbol",
    "report_period",
    "item_id",
    "period_end",
    "available_date",
    "value",
    "company_type",
]
# equity is total equity (incl. minority interest); npat_total includes the minority share.
ITEM_IDS = ("equity", "npat_parent", "npat_total")

# Edge cases baked into the default synthetic data.
LATE_LISTER = "S10"  # listed late: fewer than 252 sessions of history
SPARSE_TRADER = "S11"  # trades on roughly 60% of sessions
STOCK_DIVIDEND_SYMBOL = "S05"  # 30% stock dividend: raw price is 1.3x adjusted close before it
MISSING_QUARTER = ("S04", "2021-Q2")  # one quarter absent, so TTM must be NaN around it
LOSS_MAKER = "S06"  # negative quarterly earnings
COMPANY_TYPES = {"S01": "bank", "S02": "securities", "S03": "insurance"}


def make_prices(
    n_symbols: int = 12,
    n_sessions: int = 600,
    start: str = "2020-01-01",
    seed: int = SEED,
) -> pd.DataFrame:
    """Daily bars for ``n_symbols`` on a business-day calendar, sorted by date, symbol."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_sessions)
    frames = []
    for i in range(n_symbols):
        symbol = f"S{i:02d}"
        log_ret = rng.normal(0.0003, 0.02, n_sessions)
        close_adj = 20.0 * np.exp(np.cumsum(log_ret))
        open_ = np.r_[close_adj[0], close_adj[:-1]] * (1 + rng.normal(0, 0.005, n_sessions))
        high = np.maximum(open_, close_adj) * (1 + np.abs(rng.normal(0, 0.005, n_sessions)))
        low = np.minimum(open_, close_adj) * (1 - np.abs(rng.normal(0, 0.005, n_sessions)))
        base_volume = 10 ** rng.uniform(4.5, 6.5)
        volume = np.round(base_volume * rng.lognormal(0.0, 0.5, n_sessions)).astype("int64")

        close_raw = close_adj.copy()
        if symbol == STOCK_DIVIDEND_SYMBOL:
            close_raw[:300] = close_adj[:300] * 1.3

        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close_adj": close_adj,
                "volume": volume,
                "value": close_raw * 1000.0 * volume,  # traded at the unadjusted price
                "value_source": "matched",
            }
        )
        if symbol == LATE_LISTER:
            frame = frame.iloc[400:]
        elif symbol == SPARSE_TRADER:
            frame = frame[rng.random(n_sessions) < 0.6]
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["date", "symbol"], ignore_index=True)[PRICE_COLUMNS]


def make_fundamentals(
    symbols: list[str],
    first_quarter: str = "2019Q1",
    last_quarter: str = "2022Q2",
    lag_days: int = 45,
    seed: int = SEED,
) -> pd.DataFrame:
    """Quarterly statement items in long format with ``available_date = period_end + lag``."""
    rng = np.random.default_rng(seed + 1)
    quarters = pd.period_range(first_quarter, last_quarter, freq="Q")
    rows = []
    for symbol in symbols:
        company_type = COMPANY_TYPES.get(symbol, "regular")
        shares = float(rng.integers(100, 1000)) * 1e6
        paid_in_capital = shares * 10_000.0
        equity_parent = paid_in_capital * rng.uniform(1.0, 2.5)
        minority = equity_parent * rng.uniform(0.0, 0.05)
        quarterly_roe = rng.uniform(0.01, 0.05)
        for quarter in quarters:
            report_period = f"{quarter.year}-Q{quarter.quarter}"  # same label style as the source
            if (symbol, report_period) == MISSING_QUARTER:
                continue
            npat_parent = equity_parent * quarterly_roe * rng.normal(1.0, 0.3)
            if symbol == LOSS_MAKER:
                npat_parent = -abs(npat_parent)
            equity_parent += npat_parent
            period_end = quarter.end_time.normalize()
            items = {
                "equity": equity_parent + minority,
                "npat_parent": npat_parent,
                "npat_total": npat_parent * 1.03,
            }
            for item_id, value in items.items():
                rows.append(
                    {
                        "symbol": symbol,
                        "report_period": report_period,
                        "item_id": item_id,
                        "period_end": period_end,
                        "available_date": period_end + pd.Timedelta(days=lag_days),
                        "value": value,
                        "company_type": company_type,
                    }
                )
    return pd.DataFrame(rows, columns=FUNDAMENTAL_COLUMNS)


@pytest.fixture(scope="session")
def config_path() -> Path:
    return REPO_ROOT / "configs" / "base.yaml"


@pytest.fixture
def base_config(config_path: Path) -> dict:
    return load_config(config_path)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def synthetic_fundamentals(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    return make_fundamentals(sorted(synthetic_prices["symbol"].unique()))
