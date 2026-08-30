#!/usr/bin/env python3
"""Validate the artifact tree required by the reproduction scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROWS = {
    "movies": 10_964,
    "songs": 2_157,
    "basketball": 13_309,
}
RES_MODELS = (
    "llama8b",
    "qwen2",
    "chatgpt",
    "Qwen2.5-7B",
    "Qwen2.5-14B",
    "Qwen2.5-32B",
)
BASELINE_MODELS = (
    "llama3-8b",
    "qwen2",
    "chatgpt",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen2.5-32b",
)
LLM_POP_MODELS = (
    "llama8b",
    "qwen2",
    "chatgpt",
    "qwen25-7b",
    "qwen25-14b",
    "qwen25-32b",
)
LLM_POP_TYPES = ("question_pop", "gene_pop", "coo_pop")


def jsonl_rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {error}") from error
            count += 1
    return count


def require_jsonl(
    path: Path,
    expected_rows: int,
    failures: list[str],
    check_rows: bool,
) -> None:
    if not path.is_file():
        failures.append(f"missing: {path.relative_to(ROOT)}")
        return
    if check_rows:
        try:
            actual = jsonl_rows(path)
        except ValueError as error:
            failures.append(str(error))
            return
        if actual != expected_rows:
            failures.append(
                f"row mismatch: {path.relative_to(ROOT)} "
                f"(expected {expected_rows}, found {actual})"
            )


def require_json(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing: {path.relative_to(ROOT)}")
        return
    try:
        with path.open(encoding="utf-8") as stream:
            json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"invalid JSON: {path.relative_to(ROOT)} ({error})")


def core_artifacts(failures: list[str], check_rows: bool) -> None:
    for dataset, expected_rows in DATASET_ROWS.items():
        require_jsonl(
            ROOT / "data" / f"{dataset}.jsonl",
            expected_rows,
            failures,
            check_rows,
        )
        for model in RES_MODELS:
            require_jsonl(
                ROOT
                / "res"
                / dataset
                / f"{dataset}_{model}_temperature1.jsonl",
                expected_rows,
                failures,
                check_rows,
            )

    require_jsonl(
        ROOT
        / "res"
        / "gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl",
        expected_rows=0,
        failures=failures,
        check_rows=False,
    )
    require_json(
        ROOT
        / "res"
        / "cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json",
        failures,
    )
    require_json(
        ROOT
        / "res"
        / "single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json",
        failures,
    )


def calibration_artifacts(failures: list[str], check_rows: bool) -> None:
    for dataset, expected_rows in DATASET_ROWS.items():
        for model in BASELINE_MODELS:
            require_jsonl(
                ROOT
                / "baselines"
                / "self_consistency"
                / "consis_judge_res"
                / f"{dataset}_{model}_sc.jsonl",
                expected_rows,
                failures,
                check_rows,
            )
            require_jsonl(
                ROOT
                / "baselines"
                / "verbalized_confidence"
                / f"{dataset}_{model}_vc.jsonl",
                expected_rows,
                failures,
                check_rows,
            )

        for model in LLM_POP_MODELS:
            for pop_type in LLM_POP_TYPES:
                require_jsonl(
                    ROOT
                    / "llm_pop_generation"
                    / dataset
                    / f"{dataset}_{model}_{pop_type}_0.jsonl",
                    expected_rows,
                    failures,
                    check_rows,
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        choices=("core", "calibration", "full"),
        default="full",
        help="core checks RQ1/RQ2 inputs; calibration/full also check confidence artifacts",
    )
    parser.add_argument(
        "--skip-row-counts",
        action="store_true",
        help="only check file presence and JSON validity",
    )
    args = parser.parse_args()

    failures: list[str] = []
    check_rows = not args.skip_row_counts
    core_artifacts(failures, check_rows)
    if args.level in {"calibration", "full"}:
        calibration_artifacts(failures, check_rows)

    if failures:
        print(f"Artifact verification failed with {len(failures)} issue(s):")
        for failure in failures:
            print(f"  - {failure}")
        print("\nSee docs/REPRODUCE.md for the expected layout and acquisition steps.")
        return 1

    print(f"Artifact verification passed for level={args.level}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

