#!/usr/bin/env python3
"""Restore the legacy analysis layout from compact public model-output files."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "llama3-8b": ("llama8b", "llama3-8b", "llama8b"),
    "qwen2-7b": ("qwen2", "qwen2", "qwen2"),
    "gpt-3.5-turbo": ("chatgpt", "chatgpt", "chatgpt"),
    "qwen2.5-7b": ("Qwen2.5-7B", "qwen2.5-7b", "qwen25-7b"),
    "qwen2.5-14b": ("Qwen2.5-14B", "qwen2.5-14b", "qwen25-14b"),
    "qwen2.5-32b": ("Qwen2.5-32B", "qwen2.5-32b", "qwen25-32b"),
}
PATTERN = {
    "movies": "Who is the director of the movie ",
    "songs": "Who is the performer of the song ",
    "basketball": "Where is the birthplace of the basketball player ",
}
POP_FILE = (
    "gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl"
)
COO_FILE = "cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json"
SINGLE_FILE = "single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json"


def read_compact(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def entity_names(row: dict) -> tuple[str, str, str]:
    dataset = row["dataset"]
    question_entity = row["question"].replace(PATTERN[dataset], "").lower()
    # The compact builder has already computed features with the paper's entity
    # normalization. Reusing values is sufficient; keys only need to be stable.
    reference = row["reference"][0] if row["reference"] else ""
    response = row["response"] or ""
    import re

    def clean(value: str) -> str:
        value = value.replace("\n", "").split("(")[0].strip()
        if dataset == "basketball" or len(value) <= 20:
            value = value.split(",")[0].strip()
        return re.sub(r"^[^\w]+|[^\w]+$", "", value).strip()

    return question_entity, clean(reference), clean(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SOURCE_ROOT / "data" / "model_outputs",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=SOURCE_ROOT,
        help="repository in which the legacy res/baseline directories are created",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing materialized response tree",
    )
    args = parser.parse_args()
    target = args.target_root.resolve()
    marker = target / "res" / "movies" / "movies_llama8b_temperature1.jsonl"
    if marker.exists() and not args.force:
        print(f"Materialized data already exists at {target}; nothing to do.")
        return 0

    popularity: dict[str, dict] = {}
    cooccurrence: dict[str, dict[str, int]] = {}
    occurrence: dict[str, int] = {}

    for public_slug, (res_model, baseline_slug, llm_pop_slug) in MODELS.items():
        rows = read_compact(args.input_dir / f"{public_slug}.jsonl.gz")
        for dataset in ("movies", "songs", "basketball"):
            subset = sorted(
                (row for row in rows if row["dataset"] == dataset),
                key=lambda row: row["index"],
            )
            response_rows = []
            sc_rows = []
            vc_rows = []
            llm_rows = {"question_pop": [], "gene_pop": [], "coo_pop": []}

            for row in subset:
                conf = row["confidence"]
                if "gpt" in res_model.lower():
                    log_p = {
                        "token_logprobs": [math.log(max(conf, 1e-300))]
                        if conf is not None
                        else []
                    }
                else:
                    log_p = {"token_probs": [conf] if conf is not None else []}
                response_rows.append(
                    {
                        "Res": row["response"],
                        "Log_p": log_p,
                        "question": row["question"],
                        "has_answer": row["correct"],
                        "reference": row["reference"],
                        "popularity": row["qpop"],
                    }
                )
                sc_rows.append(
                    {
                        "question": row["question"],
                        "sc_confidence": row["sc_confidence"],
                    }
                )
                vc_rows.append(
                    {
                        "question": row["question"],
                        "vc_confidence": row["vc_confidence"],
                    }
                )
                llm_rows["question_pop"].append({"Res": row["llm_qpop"]})
                llm_rows["gene_pop"].append({"Res": row["llm_gene_pop"]})
                llm_rows["coo_pop"].append({"Res": row["llm_coo"]})

                question_entity, gt_entity, gene_entity = entity_names(row)
                if row["gt_pop_found"]:
                    popularity[gt_entity] = {"popularity": row["gt_pop"]}
                if row["gene_pop_found"]:
                    popularity[gene_entity] = {"popularity": row["gene_pop"]}
                if row["gt_coo"] is not None:
                    cooccurrence.setdefault(question_entity, {})[
                        gt_entity.lower()
                    ] = row["gt_coo"]
                if row["gene_coo"] is not None:
                    cooccurrence.setdefault(question_entity, {})[
                        gene_entity.lower()
                    ] = row["gene_coo"]
                for entity, key in (
                    (question_entity, "q_occ"),
                    (gt_entity.lower(), "gt_occ"),
                    (gene_entity.lower(), "gene_occ"),
                ):
                    if row[key] is not None:
                        occurrence[entity] = row[key]

            write_jsonl(
                target
                / "res"
                / dataset
                / f"{dataset}_{res_model}_temperature1.jsonl",
                response_rows,
            )
            write_jsonl(
                target
                / "baselines"
                / "self_consistency"
                / "consis_judge_res"
                / f"{dataset}_{baseline_slug}_sc.jsonl",
                sc_rows,
            )
            write_jsonl(
                target
                / "baselines"
                / "verbalized_confidence"
                / f"{dataset}_{baseline_slug}_vc.jsonl",
                vc_rows,
            )
            for feature, values in llm_rows.items():
                write_jsonl(
                    target
                    / "llm_pop_generation"
                    / dataset
                    / f"{dataset}_{llm_pop_slug}_{feature}_0.jsonl",
                    values,
                )

    write_jsonl(target / "res" / POP_FILE, [{key: value} for key, value in popularity.items()])
    (target / "res").mkdir(parents=True, exist_ok=True)
    (target / "res" / COO_FILE).write_text(
        json.dumps(cooccurrence, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (target / "res" / SINGLE_FILE).write_text(
        json.dumps(occurrence, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Materialized compact data under {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
