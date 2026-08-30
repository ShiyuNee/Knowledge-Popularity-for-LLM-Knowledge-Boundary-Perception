#!/usr/bin/env python3
"""Merge paper features into six compact, deterministic model-output archives."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("movies", "songs", "basketball")
MODELS = {
    "llama8b": ("llama3-8b", "llama3-8b", "llama8b"),
    "qwen2": ("qwen2-7b", "qwen2", "qwen2"),
    "chatgpt": ("gpt-3.5-turbo", "chatgpt", "chatgpt"),
    "Qwen2.5-7B": ("qwen2.5-7b", "qwen2.5-7b", "qwen25-7b"),
    "Qwen2.5-14B": ("qwen2.5-14b", "qwen2.5-14b", "qwen25-14b"),
    "Qwen2.5-32B": ("qwen2.5-32b", "qwen2.5-32b", "qwen25-32b"),
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


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean_entity(value: str, dataset: str) -> str:
    value = value.replace("\n", "").split("(")[0].strip()
    if dataset == "basketball" or len(value) <= 20:
        value = value.split(",")[0].strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", value).strip()


def load_popularity() -> dict:
    result = {}
    for item in read_jsonl(ROOT / "res" / POP_FILE):
        result.update(item)
    return result


def pop_value(popularity: dict, entity: str) -> tuple[int, bool]:
    if entity not in popularity:
        return 0, False
    info = popularity[entity]
    value = info.get("popularity", 0) if isinstance(info, dict) else info
    if value in {"No", None}:
        return 0, False
    try:
        return int(value), True
    except (TypeError, ValueError):
        return 0, False


def confidence(item: dict, model: str) -> float | None:
    log_p = item.get("Log_p") or {}
    if "gpt" in model.lower():
        values = [math.exp(float(value)) for value in log_p.get("token_logprobs", [])]
    else:
        values = [float(value) for value in log_p.get("token_probs", [])]
    return sum(values) / len(values) if values else None


def keyed_confidence(path: Path, field: str) -> dict:
    if not path.is_file():
        return {}
    return {
        item["question"]: item.get(field)
        for item in read_jsonl(path)
        if item.get("question")
    }


def llm_pop_values(dataset: str, model_slug: str, feature: str) -> list[int | None]:
    path = (
        ROOT
        / "llm_pop_generation"
        / dataset
        / f"{dataset}_{model_slug}_{feature}_0.jsonl"
    )
    if not path.is_file():
        return []
    values = []
    for item in read_jsonl(path):
        try:
            values.append(int(item["Res"]))
        except (KeyError, TypeError, ValueError):
            values.append(None)
    return values


def compact_rows(model: str, baseline_slug: str, llm_pop_slug: str) -> list[dict]:
    popularity = load_popularity()
    cooccurrence = json.loads((ROOT / "res" / COO_FILE).read_text(encoding="utf-8"))
    occurrence = json.loads((ROOT / "res" / SINGLE_FILE).read_text(encoding="utf-8"))
    rows = []

    for dataset in DATASETS:
        source = ROOT / "res" / dataset / f"{dataset}_{model}_temperature1.jsonl"
        responses = read_jsonl(source)
        sc = keyed_confidence(
            ROOT
            / "baselines"
            / "self_consistency"
            / "consis_judge_res"
            / f"{dataset}_{baseline_slug}_sc.jsonl",
            "sc_confidence",
        )
        vc = keyed_confidence(
            ROOT
            / "baselines"
            / "verbalized_confidence"
            / f"{dataset}_{baseline_slug}_vc.jsonl",
            "vc_confidence",
        )
        llm_qpop = llm_pop_values(dataset, llm_pop_slug, "question_pop")
        llm_gpop = llm_pop_values(dataset, llm_pop_slug, "gene_pop")
        llm_coo = llm_pop_values(dataset, llm_pop_slug, "coo_pop")

        for index, item in enumerate(responses):
            question = item["question"]
            reference = item.get("reference") or []
            response = item.get("Res")
            q_entity = question.replace(PATTERN[dataset], "").lower()
            gt_entity = clean_entity(reference[0], dataset) if reference else ""
            gene_entity = clean_entity(response, dataset) if response else ""
            gt_pop, gt_pop_found = pop_value(popularity, gt_entity)
            gene_pop, gene_pop_found = pop_value(popularity, gene_entity)
            q_coo = cooccurrence.get(q_entity)

            rows.append(
                {
                    "dataset": dataset,
                    "index": index,
                    "question": question,
                    "reference": reference,
                    "response": response,
                    "correct": item.get("has_answer"),
                    "confidence": confidence(item, model),
                    "qpop": item.get("popularity"),
                    "gt_pop": gt_pop,
                    "gt_pop_found": gt_pop_found,
                    "gene_pop": gene_pop,
                    "gene_pop_found": gene_pop_found,
                    "gt_coo": q_coo.get(gt_entity.lower(), 0) if q_coo else None,
                    "gene_coo": q_coo.get(gene_entity.lower(), 0) if q_coo else None,
                    "q_occ": occurrence.get(q_entity),
                    "gt_occ": occurrence.get(gt_entity.lower()),
                    "gene_occ": occurrence.get(gene_entity.lower()),
                    "sc_confidence": sc.get(question),
                    "vc_confidence": vc.get(question),
                    "llm_qpop": llm_qpop[index] if index < len(llm_qpop) else None,
                    "llm_gene_pop": llm_gpop[index] if index < len(llm_gpop) else None,
                    "llm_coo": llm_coo[index] if index < len(llm_coo) else None,
                }
            )
    return rows


def write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                line = json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                zipped.write(line.encode("utf-8") + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "model_outputs",
    )
    args = parser.parse_args()

    for model, (public_slug, baseline_slug, llm_pop_slug) in MODELS.items():
        rows = compact_rows(model, baseline_slug, llm_pop_slug)
        output = args.output_dir / f"{public_slug}.jsonl.gz"
        write_gzip_jsonl(output, rows)
        print(f"{output}: {len(rows):,} rows, {output.stat().st_size / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

