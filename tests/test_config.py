import copy

import pytest

from vnfactor.config import REQUIRED_SECTIONS, ConfigError, load_config, validate_config

# Hypothesised signs from AGENTS.md section 5.4, fixed before looking at data.
EXPECTED_SIGNS = {
    "mom_12_1": 1,
    "rev_1m": -1,
    "vol_126": -1,
    "adv_63": -1,
    "ep": 1,
    "bp": 1,
    "roe": 1,
}


def test_base_config_loads(base_config):
    for section in REQUIRED_SECTIONS:
        assert isinstance(base_config[section], dict)


def test_base_config_signs_match_preregistered_hypotheses(base_config):
    signs = {name: spec["sign"] for name, spec in base_config["factors"]["definitions"].items()}
    assert signs == EXPECTED_SIGNS


def test_audit_decisions_are_frozen(base_config):
    """Approved after the data audit on 2026-11-13; changing them needs a new decision."""
    assert base_config["calendar"]["first_rebalance"] == "2018-01-31"
    assert base_config["calendar"]["last_rebalance"] == "2026-08-31"
    assert base_config["split"]["holdout_start"] == "2024-02-01"
    assert base_config["fundamentals"]["market_cap_method"] == "current_shares"
    assert base_config["fundamentals"]["equity_basis"] == "total"
    assert base_config["traded_value"]["method"] == "anchored_proxy"
    assert set(base_config["sources"].values()) == {"fhsc"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_root_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def _set(cfg, dotted, value):
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        node = node[key]
    node[leaf] = value


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("split.research_fraction", 1.0),
        ("split.research_fraction", 0),
        ("universe.top_n", 0),
        ("universe.min_trading_fraction", 1.5),
        ("fundamentals.lag_days", -45),
        ("fundamentals.market_cap_method", "free_float"),
        ("preprocess.winsor_lower", 0.99),
        ("backtest.fee_rate", -0.001),
        ("factors.definitions.ep.sign", 0),
        ("composite.groups.value", ["ep", "not_a_factor"]),
        ("composite.min_groups", 5),
        ("preprocess.group_zscore_factors", ["ep", "pe"]),
    ],
)
def test_invalid_values_raise(base_config, dotted, value):
    cfg = copy.deepcopy(base_config)
    _set(cfg, dotted, value)
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_missing_section_raises(base_config):
    cfg = copy.deepcopy(base_config)
    del cfg["backtest"]
    with pytest.raises(ConfigError, match="backtest"):
        validate_config(cfg)
