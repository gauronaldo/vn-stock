"""Load the YAML config into a plain dict and run minimal sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")

REQUIRED_SECTIONS = (
    "paths",
    "sources",
    "ingest",
    "calendar",
    "split",
    "universe",
    "traded_value",
    "fundamentals",
    "factors",
    "preprocess",
    "composite",
    "evaluation",
    "backtest",
    "ml",
    "robustness",
)

MARKET_CAP_METHODS = (None, "paid_in_capital", "current_shares")


class ConfigError(ValueError):
    """Raised when the config file is missing a section or holds an invalid value."""


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read ``path`` as YAML, validate it and return it as a nested dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ``ConfigError`` on the first structural or range problem found."""
    missing = [s for s in REQUIRED_SECTIONS if not isinstance(cfg.get(s), dict)]
    if missing:
        raise ConfigError(f"missing config sections: {', '.join(missing)}")

    _check_fraction(cfg["split"], "research_fraction", open_interval=True)
    _check_fraction(cfg["universe"], "min_trading_fraction")
    _check_fraction(cfg["factors"], "max_missing_fraction")

    for section, key in (
        ("ingest", "max_range_years"),
        ("ingest", "fundamentals_top_n"),
        ("ingest", "max_consecutive_failures"),
        ("traded_value", "anchor_sessions"),
        ("universe", "top_n"),
        ("universe", "min_history_sessions"),
        ("universe", "liquidity_window"),
        ("fundamentals", "lag_days"),
        ("fundamentals", "ttm_quarters"),
        ("evaluation", "n_quantiles"),
        ("ml", "min_train_months"),
        ("ml", "retrain_every_months"),
    ):
        value = cfg[section].get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{section}.{key} must be a positive integer, got {value!r}")

    ing = cfg["ingest"]
    for key in ("request_interval_seconds", "backoff_base_seconds", "max_retries"):
        value = ing.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"ingest.{key} must be a non-negative number, got {value!r}")
    lo_hour, hi_hour = ing.get("earliest_local_hour"), ing.get("latest_local_hour")
    if not (isinstance(lo_hour, int) and isinstance(hi_hour, int) and 0 <= lo_hour < hi_hour <= 24):
        raise ConfigError("need 0 <= ingest.earliest_local_hour < latest_local_hour <= 24")
    if ing["fundamentals_top_n"] < cfg["universe"]["top_n"]:
        raise ConfigError("ingest.fundamentals_top_n must be at least universe.top_n")

    pre = cfg["preprocess"]
    lo, hi = pre.get("winsor_lower"), pre.get("winsor_upper")
    if not (isinstance(lo, float) and isinstance(hi, float) and 0.0 <= lo < hi <= 1.0):
        raise ConfigError(f"need 0 <= winsor_lower < winsor_upper <= 1, got {lo!r}, {hi!r}")

    for key in ("fee_rate", "sell_tax_rate"):
        rate = cfg["backtest"].get(key)
        if not isinstance(rate, int | float) or rate < 0:
            raise ConfigError(f"backtest.{key} must be a non-negative number, got {rate!r}")

    method = cfg["fundamentals"].get("market_cap_method")
    if method not in MARKET_CAP_METHODS:
        raise ConfigError(f"fundamentals.market_cap_method must be one of {MARKET_CAP_METHODS}")

    definitions = cfg["factors"].get("definitions")
    if not isinstance(definitions, dict) or not definitions:
        raise ConfigError("factors.definitions must be a non-empty mapping")
    for name, spec in definitions.items():
        if not isinstance(spec, dict) or spec.get("sign") not in (1, -1):
            raise ConfigError(f"factor {name!r} needs sign 1 or -1")

    unknown = [f for f in pre.get("group_zscore_factors", []) if f not in definitions]
    if unknown:
        raise ConfigError(f"preprocess.group_zscore_factors not defined: {unknown}")

    comp = cfg["composite"]
    groups = comp.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ConfigError("composite.groups must be a non-empty mapping")
    for group, members in groups.items():
        undefined = [f for f in members if f not in definitions]
        if not members or undefined:
            raise ConfigError(f"composite group {group!r} has undefined or no factors")
    min_groups = comp.get("min_groups")
    if not isinstance(min_groups, int) or not 1 <= min_groups <= len(groups):
        raise ConfigError(f"composite.min_groups must be in [1, {len(groups)}]")


def _check_fraction(section: dict[str, Any], key: str, open_interval: bool = False) -> None:
    value = section.get(key)
    ok = isinstance(value, int | float) and not isinstance(value, bool)
    if ok:
        ok = 0.0 < value < 1.0 if open_interval else 0.0 <= value <= 1.0
    if not ok:
        bounds = "(0, 1)" if open_interval else "[0, 1]"
        raise ConfigError(f"{key} must be in {bounds}, got {value!r}")
