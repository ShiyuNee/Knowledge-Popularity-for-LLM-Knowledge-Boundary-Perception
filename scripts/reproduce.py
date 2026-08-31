#!/usr/bin/env python3
"""Run the paper analyses in a documented, reproducible order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "reproduction_logs"

STAGES = {
    "prepare": [
        ("materialize_compact_data", ["scripts/materialize_compact_data.py"]),
    ],
    "core": [
        ("verify_core_artifacts", ["scripts/verify_artifacts.py", "--level", "core"]),
        ("rq1_rq2_statistics", ["code/analysis_correlation/verify_per_model.py"]),
        ("bootstrap_intervals", ["code/analysis_correlation/bootstrap_ci.py"]),
    ],
    "calibration": [
        (
            "verify_calibration_artifacts",
            ["scripts/verify_artifacts.py", "--level", "calibration"],
        ),
        (
            "correctness_detection",
            ["code/analysis_correlation/calibration_experiment.py"],
        ),
    ],
    "distribution": [
        (
            "balanced_train_natural_test",
            ["code/analysis_correlation/setting2_natural_test.py"],
        ),
        (
            "natural_train_natural_test",
            ["code/analysis_correlation/setting3_natural_train_test.py"],
        ),
    ],
    "transfer": [
        (
            "balanced_and_shifted_transfer",
            ["code/analysis_correlation/transfer_all_methods_v3.py"],
        ),
        (
            "natural_distribution_transfer",
            ["code/analysis_correlation/transfer_all_methods_setting3.py"],
        ),
    ],
    "figures": [
        ("paper_figures", ["code/analysis_correlation/plot_paper_figures.py"]),
        (
            "relation_specificity_figure",
            ["code/analysis_correlation/plot/plot_relation_specificity.py"],
        ),
        (
            "confidence_bar_figure",
            ["code/analysis_correlation/plot_confidence_bar_final.py"],
        ),
    ],
    "check": [
        ("compare_expected_results", ["scripts/check_expected_results.py"]),
    ],
}


def run_one(name: str, arguments: list[str], dry_run: bool) -> int:
    command = [sys.executable, *arguments]
    printable = " ".join(command)
    print(f"\n[{name}]\n$ {printable}", flush=True)
    if dry_run:
        return 0

    LOG_DIR.mkdir(exist_ok=True)
    plot_cache = ROOT / "reproduction_outputs" / "matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "42")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(plot_cache))
    log_path = LOG_DIR / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

    if return_code:
        print(f"[{name}] failed with exit code {return_code}. Log: {log_path}")
    else:
        print(f"[{name}] completed. Log: {log_path}")
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        action="append",
        choices=(*STAGES.keys(), "all"),
        help="stage to run; repeat the flag for multiple stages (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing them",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue to later stages when a command fails",
    )
    args = parser.parse_args()

    requested = args.stage or ["all"]
    if "all" in requested:
        requested = [
            "prepare",
            "core",
            "calibration",
            "distribution",
            "transfer",
            "figures",
            "check",
        ]

    print("Reproduction stages:", ", ".join(requested))
    for stage in requested:
        for name, arguments in STAGES[stage]:
            return_code = run_one(name, arguments, args.dry_run)
            if return_code and not args.continue_on_error:
                return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
