import subprocess
import sys

import pytest

from vnfactor.cli import HANDLERS, STEPS, main


def test_help_lists_every_step(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for step in STEPS:
        assert step in out


@pytest.mark.parametrize("argv", [[], ["not-a-step"]])
def test_bad_usage_exits_with_code_2(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_every_step_has_a_handler():
    assert set(HANDLERS) == set(STEPS)


def test_missing_config_returns_nonzero(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml"), "build"]) == 1
    assert "config file not found" in capsys.readouterr().err


def test_invalid_config_returns_nonzero(tmp_path, config_path, capsys):
    bad = tmp_path / "bad.yaml"
    text = config_path.read_text(encoding="utf-8")
    bad.write_text(text.replace("research_fraction: 0.7", "research_fraction: 1.7"), "utf-8")
    assert main(["--config", str(bad), "build"]) == 1
    assert "research_fraction" in capsys.readouterr().err


def test_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "vnfactor", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout


# --- end to end on synthetic data -----------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_project(tmp_path_factory):
    import copy

    import pandas as pd
    import yaml

    from tests.conftest import REPO_ROOT, make_fundamentals, make_prices
    from vnfactor.config import load_config

    root = tmp_path_factory.mktemp("project")
    cfg = copy.deepcopy(load_config(REPO_ROOT / "configs" / "base.yaml"))
    cfg["paths"] = {
        "raw_dir": str(root / "raw"),
        "processed_dir": str(root / "processed"),
        "reports_dir": str(root / "reports"),
    }
    cfg["calendar"].update(first_rebalance="2021-01-01", last_rebalance="2024-01-31")
    cfg["split"]["holdout_start"] = "2023-06-01"
    cfg["universe"]["top_n"] = 10
    cfg["ingest"]["fundamentals_top_n"] = 10
    cfg["ml"].update(min_train_months=6, retrain_every_months=4)
    cfg["ml"]["lightgbm"].update(n_estimators=20, min_child_samples=5)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    prices = make_prices(n_sessions=1100)
    symbols = sorted(prices["symbol"].unique())
    statements = make_fundamentals(symbols, last_quarter="2024Q1").drop(columns="available_date")
    processed = root / "processed"
    processed.mkdir()
    prices.to_parquet(processed / "prices_daily.parquet", index=False)
    statements.to_parquet(processed / "statements_q.parquet", index=False)
    shares = pd.DataFrame({"symbol": symbols, "outstanding_shares": 3e8})
    shares.to_parquet(processed / "shares_current.parquet", index=False)
    return root, path


def test_pipeline_runs_end_to_end(synthetic_project):
    root, path = synthetic_project
    readme = root / "README.md"
    lines = ["intro", "<!-- results:start -->", "PLACEHOLDER", "<!-- results:end -->", "end"]
    readme.write_text("\n".join(lines) + "\n", "utf-8")
    for step in ("build", "evaluate", "ml", "robustness", "holdout"):
        assert main(["--config", str(path), step]) == 0, step
    assert main(["--config", str(path), "report", "--readme", str(readme)]) == 0
    text = readme.read_text("utf-8")
    assert "PLACEHOLDER" not in text and "intro" in text and "end" in text
    assert "rank IC" in text
    reports = root / "reports"
    assert (reports / "report.md").exists()
    for fig in ("equity_curve", "ic_by_signal", "quintile_returns"):
        assert (reports / "figures" / f"{fig}.png").exists(), fig
    tables = root / "reports" / "tables"
    for name in (
        "research_signals",
        "research_backtest",
        "research_ml_signals",
        "research_robustness",
        "holdout_signals",
        "holdout_ml_backtest",
    ):
        assert (tables / f"{name}.csv").exists(), name


def test_holdout_runs_only_once(synthetic_project, capsys):
    root, path = synthetic_project
    marker = root / "reports" / "tables" / "holdout_run.txt"
    if not marker.exists():
        assert main(["--config", str(path), "build"]) == 0
        assert main(["--config", str(path), "holdout"]) == 0
    capsys.readouterr()
    assert main(["--config", str(path), "holdout"]) == 1
    assert "already been run" in capsys.readouterr().err


def test_update_readme_requires_markers():
    from vnfactor import report

    with pytest.raises(ValueError, match="markers"):
        report.update_readme("no markers here", "block")
    block = "\n".join([report.README_START, "new", report.README_END])
    text = "\n".join(["a", report.README_START, "old", report.README_END, "b"])
    assert report.update_readme(text, block) == "\n".join(["a", block, "b"])
