#!/usr/bin/env python3
"""Evaluate per-trial baseline score files with AMEC-compatible metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from external_common import (
    DEFAULT_CONFIG,
    git_commit,
    load_config,
    make_run_id,
    metric_rows,
    read_table,
    resolve_path,
    utc_timestamp,
    write_metric_report,
    write_table,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--llr-column")
    parser.add_argument("--scores-are-llr", action="store_true")
    parser.add_argument("--source-system")
    parser.add_argument("--score-kind", default="raw_score")
    parser.add_argument("--split")
    parser.add_argument("--no-breakdowns", action="store_true")
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    load_config(args.config)
    frame = read_table(args.scores)
    if args.split and "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == args.split].reset_index(drop=True)
    llr_column = args.llr_column
    if args.scores_are_llr and llr_column is None:
        llr_column = args.score_column
    run_id = ""
    if "run_id" not in frame.columns:
        run_id = make_run_id(args.source_system or "evaluate_baseline")
        frame["run_id"] = run_id
        frame["timestamp_utc"] = utc_timestamp()
        frame["git_commit"] = git_commit()

    rows = metric_rows(
        frame,
        score_column=args.score_column,
        llr_column=llr_column,
        source_system=args.source_system or "",
        run_id=run_id,
        score_kind=args.score_kind,
        include_breakdowns=not args.no_breakdowns,
    )
    metrics = pd.DataFrame(rows)
    write_table(metrics, args.metrics_output)
    if args.report_output:
        write_metric_report(metrics, args.report_output, "External Baseline Metrics")
    print(f"wrote {resolve_path(args.metrics_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
