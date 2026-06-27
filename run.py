from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"

TASKS = {
    "c0": EXPERIMENTS / "c0.py",
    "c1": EXPERIMENTS / "c1.py",
    "c2": EXPERIMENTS / "c2.py",
    "c3": EXPERIMENTS / "c3.py",
    "c4": EXPERIMENTS / "c4.py",
    "c5": EXPERIMENTS / "c5.py",
    "c8": EXPERIMENTS / "c8.py",
    "c9": EXPERIMENTS / "c9.py",
    "c10": EXPERIMENTS / "c10.py",
    "c10_ablation": EXPERIMENTS / "c10_ablation.py",
    "c8_coverage": EXPERIMENTS / "c8_coverage_curve.py",
    "c10_coverage": EXPERIMENTS / "c10_coverage_curve.py",
    "metadata_controls": EXPERIMENTS / "c10_metadata_controls.py",
    "condition_analysis": EXPERIMENTS / "c13.py",
    "similarity_fusion": EXPERIMENTS / "c12.py",
}

MAIN_TASKS = (
    "c0",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c8",
    "c10",
    "c10_ablation",
)

FOLLOWUP_TASKS = (
    "c8_coverage",
    "c10_coverage",
    "metadata_controls",
    "condition_analysis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AMECxSV paper experiments."
    )
    parser.add_argument("tasks", nargs="*", choices=sorted(TASKS))
    parser.add_argument(
        "--main",
        action="store_true",
        help="Run the C0--C5, C8, C10, and C10-ablation experiments.",
    )
    parser.add_argument(
        "--followup",
        action="store_true",
        help="Run coverage, metadata-control, and condition analyses.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List task names and exit."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("\n".join(TASKS))
        return

    selected = list(args.tasks)
    if args.main:
        selected.extend(MAIN_TASKS)
    if args.followup:
        selected.extend(FOLLOWUP_TASKS)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise SystemExit(
            "Select tasks explicitly, or use --main or --followup."
        )

    for task in selected:
        script = TASKS[task]
        print(f"\n=== Running {task} ===", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
