"""Greedy single-path evaluation of a proxy policy, plus a learning-signal probe (spec §4, §5 stage 0)."""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .batcher import GenerationBatcher, role_generate_fn
from .contracts import ROLES
from .data import write_json
from .environment import BinaryTreeEnvironment
from .eval_ids import load_eval_examples
from .evaluate import exact, f1
from .infer import run_example
from .model import generate_batch, load_policy, load_tokenizer
from .policy_server import VLLMPolicy, add_policy_arguments
from .reader import add_reader_arguments, reader_from_args
from .retrieval import (
    BoundRetriever,
    add_retrieval_arguments,
    extraction_stats,
    retriever_from_args,
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(examples: list[dict[str, Any]], predictions: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    by_id = {prediction["id"]: prediction for prediction in predictions}
    rows, recalls, retrievals = [], [], []
    formats: dict[str, list[bool]] = {role: [] for role in ROLES}
    for example in examples:
        prediction = by_id[example["id"]]
        rows.append({"id": example["id"], "f1": f1(prediction["prediction"], example["gold_answers"]), "em": exact(prediction["prediction"], example["gold_answers"])})
        gold = set(example.get("gold_doc_ids", []))
        if gold:
            recalls.append(len(gold & set(prediction["context_doc_ids"])) / len(gold))
        for action in prediction["actions"]:
            formats[action["role"]].append(bool(action["valid_format"]))
        retrievals.append(sum(action["role"] == "qf" for action in prediction["actions"]))
    return {
        "questions": len(rows),
        "f1": _mean([row["f1"] for row in rows]),
        "em": _mean([row["em"] for row in rows]),
        "context_recall": _mean(recalls),
        "format_rate": {role: _mean([float(x) for x in values]) for role, values in formats.items()},
        "mean_retrievals": _mean([float(x) for x in retrievals]),
        "seconds_per_question": elapsed_seconds / len(rows),
        "by_dataset": _by_dataset(examples, rows, by_id),
        "per_question": rows,
    }


def _by_dataset(examples: list[dict[str, Any]], rows: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, list[float]]] = {}
    for example, row in zip(examples, rows, strict=True):
        dataset = str(example.get("metadata", {}).get("dataset") or example.get("dataset") or str(example["id"]).split("::", 1)[0])
        group = groups.setdefault(dataset, {"f1": [], "em": [], "recall": []})
        group["f1"].append(row["f1"])
        group["em"].append(row["em"])
        gold = set(example.get("gold_doc_ids", []))
        if gold:
            group["recall"].append(len(gold & set(by_id[example["id"]]["context_doc_ids"])) / len(gold))
    return {name: {"questions": len(g["f1"]), "f1": _mean(g["f1"]), "em": _mean(g["em"]), "context_recall": _mean(g["recall"])} for name, g in sorted(groups.items())}


def evaluate_policy(generate_fn, reader, retriever, examples: list[dict[str, Any]], *, max_steps: int, workers: int) -> dict[str, Any]:
    lock = threading.Lock()
    started = time.monotonic()

    def one(example: dict[str, Any]) -> dict[str, Any]:
        return run_example(generate_fn, reader, BoundRetriever(retriever, example, lock), example, max_steps)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(pool.map(one, examples))
    return summarize(examples, predictions, time.monotonic() - started)


def interleave_datasets(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin over datasets (order of first appearance), keeping each dataset's own order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example.get("dataset") or str(example["id"]).split("::", 1)[0]), []).append(example)
    queues = list(groups.values())
    return [queue[i] for i in range(max(map(len, queues), default=0)) for queue in queues if i < len(queue)]


def signal_probe(make_env, examples: list[dict[str, Any]], *, workers: int) -> dict[str, Any]:
    """Share of sampled trees whose leaves all score the same (zero advantage, no gradient)."""
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        trees = list(pool.map(lambda example: make_env(example).run(example), examples))
    elapsed = time.monotonic() - started
    action_trees = [tree for tree in trees if any(leaf["actions"] for leaf in tree["leaves"])]
    zero = sum(bool(tree["zero_signal"]) for tree in action_trees)
    return {"trees": len(trees), "action_trees": len(action_trees), "zero_signal_fraction": zero / len(action_trees) if action_trees else 1.0, "seconds_per_tree": elapsed / len(trees) if trees else None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Greedy eval of a proxy adapter on the fixed eval set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--eval-ids")
    add_retrieval_arguments(parser)
    add_reader_arguments(parser)
    add_policy_arguments(parser)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--signal-probe", type=int, default=0, help="Also sample N training-style trees and report the zero-signal share")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-leaves", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--shaping-scale", type=float, default=0.1)
    parser.add_argument("--load-in-4bit", action="store_true", help="nf4 base (slower generation, less memory); default bf16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    # The eval set is grouped by dataset; interleave so --limit and the signal probe see every dataset.
    examples = interleave_datasets(load_eval_examples(args.qa, args.eval_ids))
    if args.limit is not None:
        examples = examples[: args.limit]
    retriever = retriever_from_args(args)
    batcher = None
    if args.policy_url:  # the server already serves --checkpoint as --policy-model
        actor, tokenizer = None, load_tokenizer(args.checkpoint)
        generate = VLLMPolicy(args.policy_url, args.policy_api_key, tokenizer, model=args.policy_model).generate
    else:
        actor, tokenizer, _ = load_policy(args.checkpoint, trainable=False, load_in_4bit=args.load_in_4bit)
        batcher = GenerationBatcher(lambda prompts, **kwargs: generate_batch(actor, tokenizer, prompts, **kwargs), max_batch=args.max_batch)
        generate = batcher.generate
    reader = reader_from_args(args)
    report = evaluate_policy(role_generate_fn(generate, temperature=0.0, top_p=1.0), reader, retriever, examples, max_steps=args.max_steps, workers=args.workers)
    if args.signal_probe:
        lock = threading.Lock()
        sample_fn = role_generate_fn(generate, temperature=args.temperature, top_p=args.top_p)

        def make_env(example):
            return BinaryTreeEnvironment(actor, tokenizer, reader, BoundRetriever(retriever, example, lock), branch_factor=2, max_steps=args.max_steps, max_leaves=args.max_leaves, temperature=args.temperature, top_p=args.top_p, alpha=args.alpha, retrieval_cost=0.0, behavior_checkpoint=str(Path(args.checkpoint).resolve()), shaping_scale=args.shaping_scale, generate_fn=sample_fn)

        report["signal_probe"] = signal_probe(make_env, examples[: args.signal_probe], workers=args.workers)
    report.update(checkpoint=str(Path(args.checkpoint).resolve()), reader_protocol=args.reader_protocol, max_steps=args.max_steps, generation="vllm" if args.policy_url else ("hf-nf4" if args.load_in_4bit else "hf-bf16"))
    # Retrieved documents left without triples give KI nothing to choose from; with --extract-triples the
    # missing ones are extracted on the fly (cache_misses counts them before extraction).
    report["triple_coverage"] = getattr(retriever, "stats", None)
    report["triple_extraction"] = extraction_stats(retriever)
    if batcher is not None:
        batcher.close()
    retriever.close()
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_question"}))


if __name__ == "__main__":
    main()
