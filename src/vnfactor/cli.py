"""Command line entry point: one subcommand per pipeline step."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from vnfactor import ingest
from vnfactor.config import DEFAULT_CONFIG_PATH, ConfigError, load_config

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


# Filled in as each phase lands; a step without a handler fails loudly.
HANDLERS: dict[str, Handler] = {"ingest": _ingest}


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
    return handler(cfg, args)
