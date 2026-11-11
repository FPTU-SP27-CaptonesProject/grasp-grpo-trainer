"""Which part of the reward drives the GRPO advantage? Roll out trees exactly as training does, keep every leaf's
reward components, and split each tree's advantage into its answer-F1, context-recall and shaping parts.

GRPO learns only from differences between the leaves of one tree. After four evals (2026-11-08..11) the
rollout reward kept rising or moving while eval F1 stayed within noise of SFT (0.541-0.573 vs 0.549); the
training log only keeps per-step means, which mix questions and say nothing about within-tree differences.
This probe answers: of the advantage the loss sees, how much comes from answer F1, and how often does the
leaf with the better answer still get the lower advantage?

Runs beside a live training run on the same vLLM servers (adds load to them for a few minutes):

    grasp-reward-probe --qa train.jsonl --question-offset 3750 --questions 48 --adapter <dir> ... --output probe.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .batcher import role_generate_fn
from .data import read_examples
from .environment import BinaryTreeEnvironment
from .policy_server import VLLMPolicy, add_policy_arguments
from .reader import add_reader_arguments, reader_from_args
from .retrieval import BoundRetriever, add_retrieval_arguments, extraction_stats, retriever_from_args


def leaf_parts(leaf: dict[str, Any], alpha: float) -> dict[str, Any]:
    r = leaf["reward"]
    f1 = float(r["answer_f1"])
    recall = r["context_recall"]
    by_kind: dict[str, float] = {}
    for action in leaf["actions"]:
        for key, value in (action.get("reward_shaping") or {}).items():
            if key != "total":
                by_kind[f"{action['role']}:{key}"] = by_kind.get(f"{action['role']}:{key}", 0.0) + float(value)
    return {
        "f1_part": (1 - alpha) * f1 if recall is not None else f1,
        "recall_part": alpha * float(recall) if recall is not None else 0.0,
        "shaping_part": float(r["action_shaping"]) - float(r["retrieval_cost"]),
        "answer_f1": f1,
        "context_recall": recall,
        "total": float(r["total"]),
        "advantage": float(leaf["advantage"]),
        "retrievals": int(r["retrieval_count"]),
        "terminal": leaf["terminal"],
        "answer": leaf["final_answer"],
        "shaping_raw": by_kind,  # before shaping_scale
    }


def analyse(trees: list[dict[str, Any]]) -> dict[str, Any]:
    """Advantage = centered F1 part + centered recall part + centered shaping part (exactly, per tree).
    share_X = sum over leaves of (centered X x advantage) / sum of advantage^2: the parts' shares of the
    advantage variance; they add up to 1 and a part can be negative when it works against the rest."""
    parts = ("f1_part", "recall_part", "shaping_part")
    cov = dict.fromkeys(parts, 0.0)
    total_sq = 0.0
    pairs = flips = f1_ties_with_signal = 0
    signal_trees = 0
    for tree in trees:
        leaves = tree["leaves"]
        if len(leaves) < 2 or tree["zero_signal"]:
            continue
        signal_trees += 1
        means = {p: statistics.fmean(x[p] for x in leaves) for p in parts}
        for leaf in leaves:
            adv = leaf["advantage"]
            total_sq += adv * adv
            for p in parts:
                cov[p] += (leaf[p] - means[p]) * adv
        for i, a in enumerate(leaves):
            for b in leaves[i + 1:]:
                if a["answer_f1"] == b["answer_f1"]:
                    f1_ties_with_signal += abs(a["advantage"] - b["advantage"]) > 1e-6
                    continue
                pairs += 1
                better, worse = (a, b) if a["answer_f1"] > b["answer_f1"] else (b, a)
                flips += better["advantage"] < worse["advantage"] - 1e-9
    all_leaves = [x for t in trees for x in t["leaves"]]
    f1_constant = sum(1 for t in trees if len({x["answer_f1"] for x in t["leaves"]}) == 1)
    return {
        "trees": len(trees),
        "zero_signal_trees": sum(t["zero_signal"] for t in trees),
        "trees_with_same_f1_on_every_leaf": f1_constant,
        "signal_trees": signal_trees,
        "advantage_variance_share": {p: round(cov[p] / total_sq, 3) if total_sq else None for p in parts},
        "leaf_pairs_with_different_f1": pairs,
        "pairs_where_better_answer_gets_lower_advantage": flips,
        "leaf_pairs_with_equal_f1_but_different_advantage": f1_ties_with_signal,
        "mean_leaf": {p: round(statistics.fmean(x[p] for x in all_leaves), 4) for p in (*parts, "answer_f1")} if all_leaves else {},
        "terminals": {k: sum(x["terminal"] == k for x in all_leaves) for k in sorted({x["terminal"] for x in all_leaves})},
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft-checkpoint", required=True, help="tokenizer source")
    p.add_argument("--adapter", help="adapter dir to probe (loaded on the policy server as probe-<time>); default --policy-model")
    p.add_argument("--qa", required=True)
    p.add_argument("--question-offset", type=int, default=0)
    p.add_argument("--questions", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", required=True)
    add_retrieval_arguments(p)
    add_reader_arguments(p)
    add_policy_arguments(p)
    for name, kind, default in (("max-steps", int, 5), ("max-leaves", int, 4), ("temperature", float, 0.7), ("top-p", float, 0.95), ("alpha", float, 0.3),
                                ("retrieval-cost", float, 0.0), ("rd-gain-weight", float, 0.3), ("qf-gain-weight", float, 0.7), ("ki-hit-weight", float, 0.3),
                                ("ki-miss-penalty", float, 0.1), ("shaping-scale", float, 0.1)):
        p.add_argument(f"--{name}", type=kind, default=default)
    args = p.parse_args(argv)
    if not args.policy_url:
        p.error("--policy-url is required (the probe never loads the model itself)")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, use_fast=True)
    examples = read_examples(args.qa)
    picked = [examples[(args.question_offset + i) % len(examples)] for i in range(args.questions)]
    policy = VLLMPolicy(args.policy_url, args.policy_api_key, tokenizer, model=args.policy_model)
    probe_name = None
    if args.adapter:  # not policy.update(): that deletes older adapter directories
        probe_name = f"probe-{time.strftime('%m%d%H%M%S')}"
        policy._admin.post("load_lora_adapter", {"lora_name": probe_name, "lora_path": str(Path(args.adapter).resolve())})
        policy.model = probe_name
    retriever = retriever_from_args(args)
    reader = reader_from_args(args)
    lock = threading.Lock()
    generate = role_generate_fn(policy.generate, temperature=args.temperature, top_p=args.top_p)

    def one(example):
        env = BinaryTreeEnvironment(None, tokenizer, reader, BoundRetriever(retriever, example, lock), branch_factor=2, max_steps=args.max_steps,
                                    max_leaves=args.max_leaves, temperature=args.temperature, top_p=args.top_p, alpha=args.alpha,
                                    retrieval_cost=args.retrieval_cost, behavior_checkpoint=policy.model, rd_gain_weight=args.rd_gain_weight,
                                    qf_gain_weight=args.qf_gain_weight, ki_hit_weight=args.ki_hit_weight, ki_miss_penalty=args.ki_miss_penalty,
                                    shaping_scale=args.shaping_scale, generate_fn=generate)
        tree = env.run(example)
        return {"question_id": example["id"], "dataset": example.get("metadata", {}).get("dataset"), "question": example["question"],
                "gold_answers": example["gold_answers"], "zero_signal": tree["zero_signal"], "reward_std": tree["reward_std"],
                "leaves": [leaf_parts(leaf, args.alpha) for leaf in tree["leaves"]]}

    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            trees = []
            for i, tree in enumerate(pool.map(one, picked), 1):
                trees.append(tree)
                if i % 8 == 0:
                    print(json.dumps({"trees": i, "of": len(picked), "minutes": round((time.time() - started) / 60, 1)}), flush=True)
    finally:
        if probe_name:
            policy._admin.post("unload_lora_adapter", {"lora_name": probe_name})
        retriever.close()
    report = {"adapter": args.adapter or args.policy_model, "question_offset": args.question_offset, "summary": analyse(trees),
              "extraction": extraction_stats(retriever), "minutes": round((time.time() - started) / 60, 1), "trees": trees}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
