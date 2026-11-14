"""Command line entry point: one subcommand per pipeline step."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vnfactor import ingest, ml, pipeline, report
from vnfactor.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from vnfactor.ingest import VN_TIME

STEPS: dict[str, str] = {
    "ingest": "download daily bars and financial statements (resumable)",
    "build": "build universe, factors, scores and forward returns",
    "evaluate": "IC, quintile returns and backtest on the research period",
    "ml": "walk-forward Ridge and LightGBM on the research period",
    "robustness": "one-change-at-a-time robustness checks on the research period",
    "holdout": "evaluate the holdout period (run exactly once)",
    "report": "write figures, tables and reports/report.md",
}

Handler = Callable[[dict[str, Any], argparse.Namespace], int]


def _ingest(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    return ingest.run(cfg, refresh=args.refresh)


def _processed(cfg: dict[str, Any], name: str) -> Path:
    return Path(cfg["paths"]["processed_dir"]) / f"{name}.parquet"


def read_table(cfg: dict[str, Any], name: str) -> pd.DataFrame:
    """Read ``data/processed/<name>.parquet`` and print its row count."""
    path = _processed(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the earlier steps first")
    df = pd.read_parquet(path)
    print(f"in  {name}: {len(df)} rows")
    return df


def write_table(cfg: dict[str, Any], df: pd.DataFrame, name: str) -> None:
    """Write ``data/processed/<name>.parquet`` and print its row count."""
    path = _processed(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"out {name}: {len(df)} rows")


def _build(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    prices = read_table(cfg, "prices_daily")
    statements = read_table(cfg, "statements_q")
    shares = read_table(cfg, "shares_current")
    tables = pipeline.build_signals(prices, statements, shares, cfg)
    tables["fwd_returns"] = pipeline.build_returns(prices, cfg)
    for name, df in tables.items():
        write_table(cfg, df, name)
    return 0


def write_report_table(cfg: dict[str, Any], df: pd.DataFrame, name: str) -> None:
    """Write an aggregate table to ``reports/tables/<name>.csv``."""
    path = Path(cfg["paths"]["reports_dir"]) / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"out tables/{name}.csv: {len(df)} rows")


def _evaluate(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    scores = read_table(cfg, "scores")
    factors = read_table(cfg, "factors")
    universe = read_table(cfg, "universe")
    fwd = read_table(cfg, "fwd_returns")
    dates = pipeline.period_dates(scores["rebalance_date"], cfg, "research")
    print(
        f"research period: {len(dates)} rebalance dates, {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}"
    )
    result = pipeline.evaluate_period(scores, factors, universe, fwd, dates, cfg)
    for name, df in result.items():
        write_report_table(cfg, df, f"research_{name}")
    return 0


def _model_comparison(
    cfg: dict[str, Any], period: str, predictions: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Composite, Ridge and LightGBM on the same out-of-sample months."""
    dates = sorted(predictions["rebalance_date"].unique())
    composite = tables["scores"][tables["scores"]["rebalance_date"].isin(dates)]
    scores = pd.concat([composite, predictions], ignore_index=True)
    print(
        f"{period} model comparison: {len(dates)} months, {pd.Timestamp(dates[0]):%Y-%m-%d}.."
        f"{pd.Timestamp(dates[-1]):%Y-%m-%d}"
    )
    return pipeline.evaluate_period(
        scores,
        tables["factors"],
        tables["universe"],
        tables["fwd_returns"],
        dates,
        cfg,
        include_factors=False,
    )


def _predict(
    cfg: dict[str, Any], period: str, tables: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(tables["universe"]["rebalance_date"].unique())
    first = int(cfg["ml"]["min_train_months"]) + int(cfg["ml"]["label_gap_months"])
    dates = [d for d in pipeline.period_dates(months, cfg, period) if months.index(d) >= first]
    return ml.walk_forward(tables["factors"], tables["universe"], tables["fwd_returns"], dates, cfg)


def _ml(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    tables = {n: read_table(cfg, n) for n in ("universe", "factors", "scores", "fwd_returns")}
    predictions, log = _predict(cfg, "research", tables)
    write_table(cfg, predictions, "predictions_research")
    write_report_table(cfg, log, "research_ml_training_log")
    for name, df in _model_comparison(cfg, "research", predictions, tables).items():
        write_report_table(cfg, df, f"research_ml_{name}")
    return 0


def _robustness(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    prices = read_table(cfg, "prices_daily")
    statements = read_table(cfg, "statements_q")
    shares = read_table(cfg, "shares_current")
    base = {n: read_table(cfg, n) for n in ("universe", "factors", "scores", "fwd_returns")}
    fwd = base["fwd_returns"]
    dates = pipeline.period_dates(base["scores"]["rebalance_date"], cfg, "research")
    rob = cfg["robustness"]

    def run(name: str, tables: dict[str, pd.DataFrame], **kwargs: Any) -> dict[str, Any]:
        result = pipeline.evaluate_period(
            tables["scores"],
            tables["factors"],
            tables["universe"],
            fwd,
            dates,
            cfg,
            include_factors=False,
            **kwargs,
        )
        universe = tables["universe"][tables["universe"]["rebalance_date"].isin(dates)]
        base_universe = base["universe"][base["universe"]["rebalance_date"].isin(dates)]
        print(f"robustness: {name} done")
        return pipeline.robustness_row(name, result, universe, base_universe)

    rows = [run("baseline", base)]
    lag = int(rob["lag_days_alt"])
    rows.append(
        run(
            f"lag_{lag}_days", pipeline.build_signals(prices, statements, shares, cfg, lag_days=lag)
        )
    )
    mult = float(rob["cost_multiplier"])
    rows.append(run(f"costs_x{mult:g}", base, cost_multiplier=mult))
    as_of, basket = pipeline.current_liquid_basket(prices, statements, cfg)
    print(f"current liquid basket: {len(basket)} symbols as of {as_of:%Y-%m-%d}")
    rows.append(
        run(
            "current_liquid_basket",
            pipeline.build_signals(prices, statements, shares, cfg, members=basket),
        )
    )
    write_report_table(cfg, pd.DataFrame(rows), "research_robustness")
    return 0


def _git_state() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}{' with uncommitted changes' if dirty else ''}"


def _holdout(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    marker = Path(cfg["paths"]["reports_dir"]) / "tables" / "holdout_run.txt"
    if marker.exists():
        print(
            f"error: the holdout has already been run ({marker}). It is evaluated exactly once;"
            " delete that file only if you deliberately start a new, documented experiment.",
            file=sys.stderr,
        )
        return 1
    tables = {n: read_table(cfg, n) for n in ("universe", "factors", "scores", "fwd_returns")}
    dates = pipeline.period_dates(tables["scores"]["rebalance_date"], cfg, "holdout")
    span = f"{len(dates)} rebalance dates, {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}"
    print(f"holdout period: {span}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"holdout run at {datetime.now(VN_TIME):%Y-%m-%d %H:%M:%S} Vietnam time",
        f"code: {_git_state()}",
        f"period: {span}",
    ]
    # Written before any result: a crash half-way still counts as the one run.
    marker.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = pipeline.evaluate_period(
        tables["scores"], tables["factors"], tables["universe"], tables["fwd_returns"], dates, cfg
    )
    for name, df in result.items():
        write_report_table(cfg, df, f"holdout_{name}")
    predictions, log = _predict(cfg, "holdout", tables)
    write_table(cfg, predictions, "predictions_holdout")
    write_report_table(cfg, log, "holdout_ml_training_log")
    for name, df in _model_comparison(cfg, "holdout", predictions, tables).items():
        write_report_table(cfg, df, f"holdout_ml_{name}")
    return 0


def _report(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    reports = Path(cfg["paths"]["reports_dir"])
    tables = report.load_tables(reports / "tables")
    print(f"in  tables: {len(tables)} files")
    universe = read_table(cfg, "universe")
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    holdout_start = min(tables["holdout_returns"]["rebalance_date"])
    titles = {
        "equity": report.equity_figure(tables, figures / "equity_curve.png", holdout_start),
        "ic": report.ic_figure(tables, figures / "ic_by_signal.png"),
        "quintile": report.quintile_figure(tables, figures / "quintile_returns.png"),
    }
    print(f"out figures: {len(titles)} files")
    marker = reports / "tables" / "holdout_run.txt"
    record = marker.read_text(encoding="utf-8").strip()
    note = "Run record:\n\n```text\n" + record + "\n```"
    facts = report.facts(tables, universe, cfg)
    text = report.build_report(tables, facts, titles, cfg, note)
    (reports / "report.md").write_text(text, encoding="utf-8")
    print(f"out report.md: {len(text.splitlines())} lines")
    readme = Path(args.readme)
    if readme.exists():
        updated = report.update_readme(
            readme.read_text(encoding="utf-8"), report.results_block(tables, facts)
        )
        readme.write_text(updated, encoding="utf-8")
        print(f"out {readme}: results block updated")
    return 0


# Filled in as each phase lands; a step without a handler fails loudly.
HANDLERS: dict[str, Handler] = {
    "ingest": _ingest,
    "build": _build,
    "evaluate": _evaluate,
    "ml": _ml,
    "robustness": _robustness,
    "holdout": _holdout,
    "report": _report,
}


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser with one subparser per step."""
    parser = argparse.ArgumentParser(
        prog="python -m vnfactor",
        description="Point-in-time factor ranking pipeline for liquid HOSE stocks.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the YAML config (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="step", required=True, metavar="STEP")
    steps = {
        name: sub.add_parser(name, help=help_text, description=help_text)
        for name, help_text in STEPS.items()
    }
    steps["ingest"].add_argument(
        "--refresh",
        action="store_true",
        help="download everything again instead of reusing cached files",
    )
    steps["report"].add_argument(
        "--readme",
        default="README.md",
        help="README whose results block is regenerated (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv``, load the config and run one step. Returns the exit code."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    handler = HANDLERS.get(args.step)
    if handler is None:
        print(f"error: step '{args.step}' is not implemented yet", file=sys.stderr)
        return 1
    try:
        return handler(cfg, args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
