"""Walk-forward Ridge and LightGBM on factor z-scores (AGENTS.md section 5.8).

Features: signed z of every factor. Label: percentile rank of the forward return within
the month. Months are indexed 0..K-1 by rebalance date. The model that predicts month
``k`` is the one refitted at the latest refit month ``r <= k`` and is trained on months
``0 .. r - 1 - label_gap``: the label of month ``j`` is known one session after
``t_{j+1}``, so at ``t_k`` the labels of months up to ``k - 2`` are known.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vnfactor.factors import SCORE_COLUMNS, assign_quantiles

MODELS = ("ridge", "lightgbm")


def feature_matrix(factors: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Signed z per factor, indexed by ``(rebalance_date, symbol)``, columns ``names``."""
    wide = factors.set_index(["rebalance_date", "symbol", "factor"])["z"].unstack("factor")
    return wide.reindex(columns=names)


def labels(fwd: pd.DataFrame, keys: pd.MultiIndex) -> pd.Series:
    """Percentile rank (0, 1] of the forward return among ``keys`` in each month."""
    r = fwd.set_index(["rebalance_date", "symbol"])["fwd_ret_1m"].reindex(keys)
    return r.groupby(level="rebalance_date").rank(pct=True).rename("label")


def schedule(
    n_months: int, min_train: int, retrain_every: int, label_gap: int
) -> dict[int, dict[str, Any]]:
    """Map each predictable month ``k`` to its refit month and training months."""
    first = min_train + label_gap
    plan = {}
    for k in range(first, n_months):
        refit = first + ((k - first) // retrain_every) * retrain_every
        plan[k] = {"refit": refit, "train": list(range(0, refit - label_gap))}
    return plan


def _fit(model: str, x_train, y_train, params: dict[str, Any], seed: int):
    """Fit one model; return ``(predict, info)`` with per-feature coefficients or gain."""
    if model == "ridge":
        from sklearn.linear_model import Ridge

        est = Ridge(alpha=float(params["alpha"]))
        est.fit(np.nan_to_num(x_train, nan=0.0), y_train)  # NaN z -> 0, the cross-sectional mean
        info = dict(zip(params["_features"], est.coef_, strict=True))
        return (lambda x: est.predict(np.nan_to_num(x, nan=0.0))), info
    if model == "lightgbm":
        import lightgbm as lgb

        kwargs = {k: v for k, v in params.items() if not k.startswith("_")}
        est = lgb.LGBMRegressor(random_state=seed, verbose=-1, **kwargs)
        est.fit(x_train, y_train)  # NaN handled natively
        gain = est.booster_.feature_importance(importance_type="gain")
        total = gain.sum() or 1.0
        info = dict(zip(params["_features"], gain / total, strict=True))
        return est.predict, info
    raise ValueError(f"unknown model {model!r}")


def walk_forward(
    factors: pd.DataFrame,
    universe: pd.DataFrame,
    fwd: pd.DataFrame,
    predict_dates: list,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample scores for ``predict_dates`` and a log of every refit.

    Returns ``(scores, log)``: scores in the ``scores`` schema with ``model`` in ``MODELS``;
    log rows ``model, refit_date, predict_date, train_first, train_last, train_rows`` plus
    one column per feature (Ridge coefficient or LightGBM gain share).
    """
    ml = cfg["ml"]
    names = list(cfg["factors"]["definitions"])
    keys = pd.MultiIndex.from_frame(universe[["rebalance_date", "symbol"]])
    x_all = feature_matrix(factors, names).reindex(keys)
    y_all = labels(fwd, keys)
    months = sorted(universe["rebalance_date"].unique())
    month_idx = {pd.Timestamp(d): i for i, d in enumerate(months)}
    plan = schedule(
        len(months),
        int(ml["min_train_months"]),
        int(ml["retrain_every_months"]),
        int(ml["label_gap_months"]),
    )
    date_level = x_all.index.get_level_values("rebalance_date")
    n_q = int(cfg["evaluation"]["n_quantiles"])
    seed = int(ml["seed"])

    scores, log, fitted = [], [], {}
    for date in sorted(pd.Timestamp(d) for d in predict_dates):
        k = month_idx[date]
        if k not in plan:
            raise ValueError(
                f"{date:%Y-%m-%d} has fewer than {ml['min_train_months']} months of known labels"
            )
        refit, train = plan[k]["refit"], plan[k]["train"]
        train_dates = [months[j] for j in train]
        for model in MODELS:
            if (model, refit) not in fitted:
                mask = np.isin(date_level, train_dates) & y_all.notna().to_numpy()
                params = {**ml[model], "_features": names}
                predict, info = _fit(
                    model, x_all[mask].to_numpy(), y_all[mask].to_numpy(), params, seed
                )
                fitted[(model, refit)] = (predict, info, int(mask.sum()))
            predict, info, train_rows = fitted[(model, refit)]
            pred_mask = date_level == date
            pred = predict(x_all[pred_mask].to_numpy())
            out = pd.DataFrame(
                {
                    "rebalance_date": date,
                    "symbol": x_all.index[pred_mask].get_level_values("symbol"),
                    "model": model,
                    "score": pred,
                }
            )
            out["quintile"] = assign_quantiles(out, n_q)
            scores.append(out[SCORE_COLUMNS])
            log.append(
                {
                    "model": model,
                    "refit_date": months[refit],
                    "predict_date": date,
                    "train_first": train_dates[0],
                    "train_last": train_dates[-1],
                    "train_rows": train_rows,
                    **info,
                }
            )
    return pd.concat(scores, ignore_index=True), pd.DataFrame(log)
