import subprocess
import sys

import pytest

from vnfactor.cli import HANDLERS, STEPS, main

UNIMPLEMENTED = [step for step in STEPS if step not in HANDLERS]


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


@pytest.mark.parametrize("step", UNIMPLEMENTED)
def test_unimplemented_step_fails_loudly(step, config_path, capsys):
    assert main(["--config", str(config_path), step]) != 0
    assert "not implemented" in capsys.readouterr().err


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
