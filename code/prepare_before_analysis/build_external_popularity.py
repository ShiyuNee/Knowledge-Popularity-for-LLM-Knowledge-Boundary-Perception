#!/usr/bin/env python3
"""Build Wikidata entity popularity and Wikipedia co-occurrence resources.

This is the public, parameterized replacement for the original one-off
preprocessing scripts. Run ``python ... --help`` or see
``docs/BUILD_POPULARITY_FEATURES.md`` for the complete workflow.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable


QUESTION_PREFIXES = {
    "movies": "Who is the director of the movie ",
    "songs": "Who is the performer of the song ",
    "basketball": "Where is the birthplace of the basketball player ",
}


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[object]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def normalize_surface(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    text = text.split("(", 1)[0].strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", text, flags=re.UNICODE).strip()


def response_text(row: dict) -> str:
    value = row.get("response", row.get("Res", ""))
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def references(row: dict) -> list[str]:
    value = row.get("reference", row.get("answer", []))
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value else []


def question_entity(row: dict) -> str:
    if row.get("question_entity"):
        return normalize_surface(row["question_entity"])
    question = str(row.get("question", ""))
    dataset = str(row.get("dataset", ""))
    prefix = QUESTION_PREFIXES.get(dataset)
    if prefix and question.startswith(prefix):
        return normalize_surface(question[len(prefix):])
    for candidate in QUESTION_PREFIXES.values():
        if question.startswith(candidate):
            return normalize_surface(question[len(candidate):])
    raise ValueError(f"cannot extract question entity from: {question!r}")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def command_prepare(args: argparse.Namespace) -> None:
    surfaces: dict[str, set[str]] = defaultdict(set)
    roles: dict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str]] = set()
    row_count = 0

    def add(surface: str, role: str) -> str:
        cleaned = normalize_surface(surface)
        key = cleaned.casefold()
        if cleaned and key:
            surfaces[key].add(cleaned)
            roles[key].add(role)
        return key

    for input_name in args.inputs:
        path = Path(input_name)
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: expected JSON objects")
            q_key = add(question_entity(row), "question")
            answers = [(item, "ground_truth") for item in references(row)]
            generated = response_text(row)
            if generated:
                answers.append((generated, "generated"))
            for answer, role in answers:
                answer_key = add(answer, role)
                if q_key and answer_key:
                    pairs.add((q_key, answer_key))
            row_count += 1

    entity_rows = [
        {
            "entity": key,
            "surface_forms": sorted(surfaces[key], key=lambda value: (len(value), value)),
            "roles": sorted(roles[key]),
        }
        for key in sorted(surfaces)
    ]
    pair_rows = [
        {"question_entity": question, "answer_entity": answer}
        for question, answer in sorted(pairs)
    ]
    write_jsonl(Path(args.entities_output), entity_rows)
    write_jsonl(Path(args.pairs_output), pair_rows)
    print(f"Read {row_count:,} response rows")
    print(f"Wrote {len(entity_rows):,} entities to {args.entities_output}")
    print(f"Wrote {len(pair_rows):,} unique pairs to {args.pairs_output}")


def sparql_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def load_existing_wikidata(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        row["entity"]: row
        for row in read_jsonl(path)
        if isinstance(row, dict) and row.get("entity")
    }


def query_wikidata(endpoint: str, user_agent: str, surface: str, timeout: float) -> tuple[str | None, int | None]:
    import requests

    query = f'''SELECT ?entity ?sitelinks WHERE {{
      ?entity rdfs:label "{sparql_escape(surface)}"@en ;
              wikibase:sitelinks ?sitelinks .
    }} ORDER BY DESC(?sitelinks) LIMIT 1'''
    response = requests.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    response.raise_for_status()
    bindings = response.json()["results"]["bindings"]
    if not bindings:
        return None, None
    binding = bindings[0]
    return binding["entity"]["value"].rsplit("/", 1)[-1], int(binding["sitelinks"]["value"])


def command_wikidata(args: argparse.Namespace) -> None:
    entities = [row for row in read_jsonl(Path(args.entities)) if isinstance(row, dict)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_wikidata(output) if args.resume else {}
    mode = "a" if args.resume and output.exists() else "w"
    completed = 0
    with output.open(mode, encoding="utf-8") as handle:
        for row in entities:
            key = row["entity"]
            if key in existing:
                continue
            result = None
            error = None
            for surface in row.get("surface_forms", [key]):
                for attempt in range(args.retries + 1):
                    try:
                        item_id, count = query_wikidata(args.endpoint, args.user_agent, surface, args.timeout)
                        result = {
                            "entity": key,
                            "matched_surface": surface if item_id else None,
                            "wikidata_id": item_id,
                            "sitelinks": count,
                            "found": item_id is not None,
                        }
                        break
                    except Exception as exc:  # network failures are recorded after retries
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt < args.retries:
                            time.sleep(args.retry_delay * (attempt + 1))
                if result and result["found"]:
                    break
            if result is None or not result["found"]:
                result = result or {
                    "entity": key,
                    "matched_surface": None,
                    "wikidata_id": None,
                    "sitelinks": None,
                    "found": False,
                }
                if error:
                    result["error"] = error
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed += 1
            if args.delay:
                time.sleep(args.delay)
    print(f"Queried {completed:,} new entities; output: {output}")


def resolve_parquet_files(values: list[str]) -> list[Path]:
    files: set[Path] = set()
    for value in values:
        path = Path(value)
        if path.is_dir():
            files.update(path.rglob("*.parquet"))
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("no Wikipedia parquet files found")
    return sorted(files)


def command_wikipedia_index(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    entities = sorted({row["entity"] for row in read_jsonl(Path(args.entities)) if isinstance(row, dict)})
    try:
        import ahocorasick

        automaton = ahocorasick.Automaton()
        for entity in entities:
            if entity:
                automaton.add_word(entity, entity)
        automaton.make_automaton()

        def find_entities(text: str) -> set[str]:
            return {entity for _, entity in automaton.iter(text)}

    except ImportError:
        if len(entities) > 1_000:
            raise RuntimeError(
                "pyahocorasick is required for a full Wikipedia scan; "
                "install requirements-data.txt"
            ) from None
        print("pyahocorasick is unavailable; using the small-fixture fallback matcher")

        def find_entities(text: str) -> set[str]:
            return {entity for entity in entities if entity and entity in text}

    entity_docs: dict[str, set[str]] = defaultdict(set)
    parquet_files = resolve_parquet_files(args.wikipedia)

    for number, path in enumerate(parquet_files, 1):
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema.names)
        if args.id_column not in columns or args.text_column not in columns:
            raise ValueError(
                f"{path}: expected columns {args.id_column!r} and {args.text_column!r}; "
                f"found {sorted(columns)}"
            )
        print(f"[{number}/{len(parquet_files)}] {path}")
        for batch in parquet.iter_batches(
            batch_size=args.batch_size,
            columns=[args.id_column, args.text_column],
        ):
            ids = batch.column(0).to_pylist()
            texts = batch.column(1).to_pylist()
            for doc_id, text in zip(ids, texts):
                if text is None:
                    continue
                matched = find_entities(str(text).casefold())
                for entity in matched:
                    entity_docs[entity].add(str(doc_id))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "entity": entities,
            "doc_ids": [sorted(entity_docs.get(entity, set())) for entity in entities],
        }
    )
    pq.write_table(table, output, compression="zstd")
    print(f"Wrote {len(entities):,} entity-document rows to {output}")


def command_aggregate(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(args.index, columns=["entity", "doc_ids"])
    entity_docs = {
        entity: set(doc_ids or [])
        for entity, doc_ids in zip(table["entity"].to_pylist(), table["doc_ids"].to_pylist())
    }
    single = {entity: len(docs) for entity, docs in sorted(entity_docs.items())}
    cooccurrence: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(Path(args.pairs)):
        question = row["question_entity"]
        answer = row["answer_entity"]
        cooccurrence[question][answer] = len(
            entity_docs.get(question, set()) & entity_docs.get(answer, set())
        )

    single_path = Path(args.single_output)
    pair_path = Path(args.cooccurrence_output)
    single_path.parent.mkdir(parents=True, exist_ok=True)
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    single_path.write_text(json.dumps(single, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    pair_path.write_text(
        json.dumps({key: value for key, value in sorted(cooccurrence.items())}, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote single-entity counts to {single_path}")
    print(f"Wrote pairwise co-occurrence counts to {pair_path}")


def command_annotate(args: argparse.Namespace) -> None:
    wikidata = {
        row["entity"]: row
        for row in read_jsonl(Path(args.wikidata))
        if isinstance(row, dict) and row.get("entity")
    }
    occurrence = json.loads(Path(args.single).read_text(encoding="utf-8"))
    cooccurrence = json.loads(Path(args.cooccurrence).read_text(encoding="utf-8"))

    def sitelinks(entity: str) -> tuple[int, bool]:
        item = wikidata.get(entity)
        found = bool(item and item.get("found") and item.get("sitelinks") is not None)
        return (int(item["sitelinks"]) if found else 0, found)

    def annotated_rows() -> Iterable[dict]:
        for original in read_jsonl(Path(args.input)):
            if not isinstance(original, dict):
                raise ValueError("response input must contain JSON objects")
            row = dict(original)
            question = question_entity(row).casefold()
            reference_values = references(row)
            ground_truth = normalize_surface(reference_values[0]).casefold() if reference_values else ""
            generated = normalize_surface(response_text(row)).casefold()
            qpop, q_found = sitelinks(question)
            gt_pop, gt_found = sitelinks(ground_truth)
            gene_pop, gene_found = sitelinks(generated)
            question_pairs = cooccurrence.get(question, {})
            row.update(
                {
                    "qpop": qpop,
                    "qpop_found": q_found,
                    "gt_pop": gt_pop,
                    "gt_pop_found": gt_found,
                    "gene_pop": gene_pop,
                    "gene_pop_found": gene_found,
                    "q_occ": int(occurrence.get(question, 0)),
                    "gt_occ": int(occurrence.get(ground_truth, 0)),
                    "gene_occ": int(occurrence.get(generated, 0)),
                    "gt_coo": int(question_pairs.get(ground_truth, 0)),
                    "gene_coo": int(question_pairs.get(generated, 0)),
                }
            )
            yield row

    write_jsonl(Path(args.output), annotated_rows())
    print(f"Wrote annotated responses to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="extract unique entities and question-answer pairs")
    prepare.add_argument("--inputs", nargs="+", required=True, help="response JSONL/JSONL.GZ files")
    prepare.add_argument("--entities-output", required=True)
    prepare.add_argument("--pairs-output", required=True)
    prepare.set_defaults(func=command_prepare)

    wikidata = subparsers.add_parser("wikidata", help="query exact English labels and sitelink counts")
    wikidata.add_argument("--entities", required=True)
    wikidata.add_argument("--output", required=True)
    wikidata.add_argument("--endpoint", default="https://query.wikidata.org/sparql")
    wikidata.add_argument("--user-agent", default=os.environ.get("WIKIDATA_USER_AGENT", "knowledge-popularity-reproduction/1.0"))
    wikidata.add_argument("--timeout", type=float, default=30)
    wikidata.add_argument("--retries", type=int, default=3)
    wikidata.add_argument("--retry-delay", type=float, default=5)
    wikidata.add_argument("--delay", type=float, default=0.1)
    wikidata.add_argument("--resume", action="store_true")
    wikidata.set_defaults(func=command_wikidata)

    index = subparsers.add_parser("wikipedia-index", help="build an entity-to-document index")
    index.add_argument("--entities", required=True)
    index.add_argument("--wikipedia", nargs="+", required=True, help="parquet file(s) or directories")
    index.add_argument("--id-column", default="id")
    index.add_argument("--text-column", default="text")
    index.add_argument("--batch-size", type=int, default=2048)
    index.add_argument("--output", required=True)
    index.set_defaults(func=command_wikipedia_index)

    aggregate = subparsers.add_parser("aggregate", help="derive occurrence and co-occurrence counts")
    aggregate.add_argument("--index", required=True)
    aggregate.add_argument("--pairs", required=True)
    aggregate.add_argument("--single-output", required=True)
    aggregate.add_argument("--cooccurrence-output", required=True)
    aggregate.set_defaults(func=command_aggregate)

    annotate = subparsers.add_parser("annotate", help="attach the external fields to response rows")
    annotate.add_argument("--input", required=True, help="one response JSONL/JSONL.GZ file")
    annotate.add_argument("--wikidata", required=True)
    annotate.add_argument("--single", required=True)
    annotate.add_argument("--cooccurrence", required=True)
    annotate.add_argument("--output", required=True)
    annotate.set_defaults(func=command_annotate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
