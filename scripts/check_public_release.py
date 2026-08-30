#!/usr/bin/env python3
"""Fail fast when a proposed public commit contains private or large artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 20 * 1024 * 1024
FORBIDDEN_PREFIXES = (
    "res/",
    "llm_pop_generation/",
    "baselines/self_consistency/sampling_res/",
    "baselines/self_consistency/consis_judge_res/",
    "pop_generation/data/",
    "artifacts/",
    "downloads/",
    "models/",
    "checkpoints/",
)
SENSITIVE_NAMES = {
    ".env",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
}
SENSITIVE_SUFFIXES = {".pem", ".key"}
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".tex",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
}
SENSITIVE_CONTENT = (
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "possible API key"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"/" + r"Users/[^/\s]+/"), "private macOS home path"),
    (re.compile(r"/" + r"data/(?:users|upfs)/"), "private server path"),
)


def repository_files(
    include_untracked: bool = False, standalone: bool = False
) -> list[str]:
    if standalone:
        return [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]

    command = ["git", "ls-files", "-z"]
    if include_untracked:
        command = [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working-tree",
        action="store_true",
        help="also check untracked, non-ignored files intended for a fresh public export",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="check every file in a folder that has no Git repository",
    )
    args = parser.parse_args()

    failures: list[str] = []
    files = repository_files(
        include_untracked=args.working_tree,
        standalone=args.standalone,
    )

    for relative in files:
        normalized = relative.replace("\\", "/")
        path = ROOT / relative

        is_verbalized_output = (
            normalized.startswith("baselines/verbalized_confidence/")
            and normalized.endswith(".jsonl")
        )
        if normalized.startswith(FORBIDDEN_PREFIXES) or is_verbalized_output:
            failures.append(f"generated artifact is tracked: {relative}")

        name = path.name.lower()
        if name in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
            failures.append(f"possible credential file is tracked: {relative}")

        if path.is_file():
            size = path.stat().st_size
            if size > MAX_TRACKED_BYTES:
                failures.append(
                    f"tracked file exceeds 20 MB ({size / 1024 / 1024:.1f} MB): "
                    f"{relative}"
                )
            if path.suffix.lower() in TEXT_SUFFIXES and size <= MAX_TRACKED_BYTES:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pattern, description in SENSITIVE_CONTENT:
                    if pattern.search(content):
                        failures.append(f"{description} found in: {relative}")

    if failures:
        print("Public-release check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nKeep local files in ignored directories and remove already tracked "
            "artifacts from the Git index before publishing.",
            file=sys.stderr,
        )
        return 1

    existing_count = sum((ROOT / item).is_file() for item in files)
    scope = (
        "standalone"
        if args.standalone
        else "candidate working-tree"
        if args.working_tree
        else "tracked"
    )
    print(
        f"Public-release check passed: {existing_count} {scope} files, "
        "no forbidden artifact paths, credentials, or files over 20 MB."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
