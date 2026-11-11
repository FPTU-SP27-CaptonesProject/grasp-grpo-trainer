"""Fixed, seeded eval set (spec §4): N questions per dataset that have gold evidence."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import read_examples, write_json


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_eval_ids(rows: list[dict[str, Any]], *, per_dataset: int, seed: int, exclude: set[str] | frozenset = frozenset()) -> list[str]:
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("gold_doc_ids") and str(row["id"]) not in exclude:
            by_dataset[str(row["dataset"])].append(str(row["id"]))
    rng = random.Random(seed)
    chosen: list[str] = []
    for name in sorted(by_dataset):
        ids = sorted(by_dataset[name])
        if len(ids) < per_dataset:
            raise ValueError(f"dataset {name} has only {len(ids)} eligible questions, need {per_dataset}")
        chosen.extend(sorted(rng.sample(ids, per_dataset)))
    return chosen


def filter_examples(examples: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(ids)
    kept = [example for example in examples if example["id"] in wanted]
    missing = wanted - {example["id"] for example in kept}
    if missing:
        raise ValueError(f"{len(missing)} eval ids missing from the QA file, e.g. {sorted(missing)[:3]}")
    return kept


def load_eval_examples(qa_path: str, ids_path: str | None) -> list[dict[str, Any]]:
    examples = read_examples(qa_path)
    if ids_path:
        examples = filter_examples(examples, json.loads(Path(ids_path).read_text(encoding="utf-8"))["ids"])
    return examples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select the fixed GRPO eval question ids")
    parser.add_argument("--dev", required=True, help="grasp-gold-evidence dev.jsonl")
    parser.add_argument("--exclude", help="JSONL whose ids must never be selected (the training file)")
    parser.add_argument("--exclude-ids", action="append", default=[], help="ids JSON (like the output) whose ids must not be selected; repeatable. "
                        "The report test set excludes config/eval_ids.json, which picked checkpoints and early-stopped runs")
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    exclude = {str(row["id"]) for row in _jsonl(args.exclude)} if args.exclude else set()
    for path in args.exclude_ids:
        exclude |= set(json.loads(Path(path).read_text(encoding="utf-8"))["ids"])
    ids = select_eval_ids(_jsonl(args.dev), per_dataset=args.per_dataset, seed=args.seed, exclude=exclude)
    write_json(args.output, {"source": Path(args.dev).name, "seed": args.seed, "per_dataset": args.per_dataset,
                             "excluded": [Path(x).name for x in ([args.exclude] if args.exclude else []) + args.exclude_ids], "ids": ids})
    print(json.dumps({"ids": len(ids), "output": args.output}))


if __name__ == "__main__":
    main()
