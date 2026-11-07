"""On-policy binary-tree GRPO training with periodic greedy eval and early stopping.

Protocol, defaults and thresholds: docs/superpowers/specs/2026-11-04-grpo-training-protocol-design.md
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .batcher import GenerationBatcher, role_generate_fn
from .contracts import ROLES
from .data import read_examples, write_json
from .environment import BinaryTreeEnvironment
from .eval_ids import load_eval_examples
from .grpo import grpo_backward
from .model import generate_batch, load_policy
from .policy_eval import evaluate_policy
from .policy_server import VLLMPolicy, add_policy_arguments
from .reader import add_reader_arguments, reader_from_args
from .retrieval import (
    BoundRetriever,
    add_retrieval_arguments,
    extraction_stats,
    retriever_from_args,
)
from .stopping import EarlyStopper, StopConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the GRASP Qwen 3B LoRA proxy with on-policy binary-tree GRPO")
    p.add_argument("--sft-checkpoint", required=True, help="Project A PEFT adapter; frozen KL reference")
    p.add_argument("--actor-checkpoint", help="Optional GRPO adapter to continue from; SFT stays the KL reference")
    p.add_argument("--qa", required=True)
    add_retrieval_arguments(p)
    add_reader_arguments(p)
    add_policy_arguments(p)
    p.add_argument("--output-dir", required=True)
    # batch and schedule
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--questions-per-iteration", type=int, default=16)
    p.add_argument("--max-refill-factor", type=float, default=2.0, help="Extra questions tried per step to replace zero-signal trees")
    p.add_argument("--workers", type=int, default=16, help="Rollout trees run in parallel")
    p.add_argument("--max-batch", type=int, default=32, help="Prompts per batched generate call")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument("--clip-high", type=float, default=0.28)
    p.add_argument("--kl-beta", type=float, default=0.002)
    p.add_argument("--backward-batch-tokens", type=int, default=12288, help="Padded tokens per backward pass (rows x longest row); lower it on out-of-memory")
    p.add_argument("--async-rollout", action="store_true", help="Sample step t+1 on the vLLM server while step t trains (one step off-policy; old log-probs = the sampler's). Needs --policy-url")
    p.add_argument("--rollout-oversample", type=float, default=1.6, help="Questions started at once per step = questions-per-iteration x this, so zero-signal trees rarely need a second wave")
    # tree and reward
    p.add_argument("--branch-factor", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--max-leaves", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--retrieval-cost", type=float, default=0.0)
    p.add_argument("--rd-gain-weight", type=float, default=0.3)
    p.add_argument("--qf-gain-weight", type=float, default=0.7)
    p.add_argument("--ki-hit-weight", type=float, default=0.3)
    p.add_argument("--ki-miss-penalty", type=float, default=0.1)
    p.add_argument("--shaping-scale", type=float, default=0.1)
    p.add_argument("--advantage-std-norm", action="store_true", help="Divide advantages by the tree std (off = Dr. GRPO)")
    # eval and early stopping
    p.add_argument("--eval-qa")
    p.add_argument("--eval-ids")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-workers", type=int, default=16)
    p.add_argument("--sft-f1", type=float, help="SFT F1 on the eval set from stage 0; measured at step 0 when omitted")
    p.add_argument("--stop-min-improvement", type=float, default=0.02)
    p.add_argument("--stop-patience-evals", type=int, default=3)
    p.add_argument("--stop-sft-drop", type=float, default=0.03)
    p.add_argument("--stop-sft-drop-evals", type=int, default=2)
    p.add_argument("--stop-min-format-rate", type=float, default=0.90)
    p.add_argument("--stop-max-kl", type=float, default=0.3)
    p.add_argument("--stop-min-entropy-ratio", type=float, default=0.30)
    p.add_argument("--stop-max-zero-signal", type=float, default=0.60)
    p.add_argument("--stop-window", type=int, default=20)
    p.add_argument("--max-hours", type=float)
    # misc
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-in-4bit", action="store_true", help="nf4 base (slower generation, less memory); default bf16")
    p.add_argument("--no-initial-retrieval", action="store_true")
    return p.parse_args(argv)


def lr_at(step: int, base: float, warmup: int) -> float:
    return base if warmup <= 0 else base * min(1.0, step / warmup)


def public_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if "api_key" not in key}


def tree_stats(trees: list[dict[str, Any]]) -> dict[str, Any]:
    leaves = [leaf for tree in trees for leaf in tree["leaves"]]
    seen: set[int] = set()
    counts = {role: [0, 0] for role in ROLES}
    for leaf in leaves:
        for action in leaf["actions"]:
            if id(action) in seen:
                continue
            seen.add(id(action))
            counts[action["role"]][0] += bool(action["valid_format"])
            counts[action["role"]][1] += 1
    valid, total = sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values())

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    stats: dict[str, Any] = {"format_rate": valid / total if total else 1.0}
    for role, (ok, seen_count) in counts.items():
        stats[f"format_rate_{role}"] = ok / seen_count if seen_count else None
    stats["mean_reward"] = mean([float(leaf["reward"]["total"]) for leaf in leaves])
    stats["mean_answer_f1"] = mean([float(leaf["reward"]["answer_f1"]) for leaf in leaves])
    stats["mean_context_recall"] = mean([float(leaf["reward"]["context_recall"] or 0.0) for leaf in leaves])
    stats["leaves"] = len(leaves)
    return stats


def _now() -> str:
    return datetime.now(UTC).isoformat()


def save_adapter(actor, tokenizer, path: Path, metadata: dict[str, Any]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    actor.save_pretrained(path, selected_adapters=["default"])  # not the frozen reference copy
    tokenizer.save_pretrained(path)
    write_json(path / "grpo_config.json", {**metadata, "created_at_utc": _now()})
    return path



def adapter_name(run_tag: str, step: int) -> str:
    return f"policy-{run_tag}-{step:06d}"

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if args.branch_factor != 2:
        raise ValueError("branch-factor must be 2")
    if args.iterations < 1 or args.questions_per_iteration < 1 or args.workers < 1:
        raise ValueError("iterations, questions-per-iteration and workers must be positive")
    if args.async_rollout and not args.policy_url:
        raise ValueError("--async-rollout needs --policy-url (the HF generator shares the training GPU)")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples = read_examples(args.qa)
    if not examples:
        raise ValueError("no valid training examples")
    eval_examples = load_eval_examples(args.eval_qa, args.eval_ids) if args.eval_qa else None
    retriever = retriever_from_args(args)
    actor_source = args.actor_checkpoint or args.sft_checkpoint
    # One base model: trainable adapter "default" + the frozen SFT adapter as the KL reference.
    actor, tokenizer, _ = load_policy(actor_source, trainable=True, load_in_4bit=args.load_in_4bit, reference_checkpoint=args.sft_checkpoint)
    params = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    reader = reader_from_args(args)
    lock = threading.Lock()
    policy, batcher = None, None
    if args.policy_url:
        policy = VLLMPolicy(args.policy_url, args.policy_api_key, tokenizer, model=args.policy_model)
        # The server outlives a run (stage 1 -> 2 -> 3 share it) and keeps each run's last adapter loaded, so
        # names carry a per-run tag: a bare "policy-000020" collided with stage 1's final adapter (2026-11-07).
        run_tag = datetime.now(UTC).strftime("%m%d%H%M%S")
        if args.actor_checkpoint:  # the server starts from the SFT adapter; continue from this one instead
            policy.update(save_adapter(actor, tokenizer, output / "live" / "step-000000", {"step": 0}), adapter_name(run_tag, 0))
        generate = policy.generate
    else:
        batcher = GenerationBatcher(lambda prompts, **kwargs: generate_batch(actor, tokenizer, prompts, **kwargs), max_batch=args.max_batch)
        generate = batcher.generate
    sample_fn = role_generate_fn(generate, temperature=args.temperature, top_p=args.top_p)
    greedy_fn = role_generate_fn(generate, temperature=0.0, top_p=1.0)
    stopper = EarlyStopper(StopConfig(min_improvement=args.stop_min_improvement, patience_evals=args.stop_patience_evals, sft_drop=args.stop_sft_drop, sft_drop_evals=args.stop_sft_drop_evals, min_format_rate=args.stop_min_format_rate, max_kl=args.stop_max_kl, min_entropy_ratio=args.stop_min_entropy_ratio, max_zero_signal=args.stop_max_zero_signal, window=args.stop_window, max_hours=args.max_hours), args.sft_f1)
    config = public_config(args)

    env_actor = None if policy is not None else actor  # a server-side policy never touches the training model

    def make_env(example: dict[str, Any], step: int, generate_fn=sample_fn) -> BinaryTreeEnvironment:
        return BinaryTreeEnvironment(env_actor, tokenizer, reader, BoundRetriever(retriever, example, lock), branch_factor=2, max_steps=args.max_steps, max_leaves=args.max_leaves, temperature=args.temperature, top_p=args.top_p, alpha=args.alpha, retrieval_cost=args.retrieval_cost, behavior_checkpoint=f"in_memory_actor_before_step_{step}", initial_retrieval=not args.no_initial_retrieval, rd_gain_weight=args.rd_gain_weight, qf_gain_weight=args.qf_gain_weight, ki_hit_weight=args.ki_hit_weight, ki_miss_penalty=args.ki_miss_penalty, shaping_scale=args.shaping_scale, advantage_std_norm=args.advantage_std_norm, generate_fn=generate_fn)

    pool = ThreadPoolExecutor(max_workers=args.workers)
    prefetcher = ThreadPoolExecutor(max_workers=1) if args.async_rollout else None
    max_attempts = math.ceil(args.questions_per_iteration * (1 + args.max_refill_factor))
    cursor = 0

    def collect(step: int, generate_fn) -> dict[str, Any]:
        """Streamed dynamic sampling: start questions x oversample trees at once; whenever one finishes without
        signal, start a replacement while fewer than questions-per-iteration are usable or pending; queued trees
        are cancelled once the batch is full."""
        nonlocal cursor
        collected: list[dict[str, Any]] = []
        zero_trees: list[dict[str, Any]] = []
        direct, attempts, extra_trees = 0, 0, 0
        started_at = time.monotonic()
        pending: set = set()
        want = math.ceil(args.questions_per_iteration * max(1.0, args.rollout_oversample))
        while True:
            for _ in range(max(0, min(want, max_attempts - attempts))):
                example = examples[cursor % len(examples)]
                pending.add(pool.submit(make_env(example, step, generate_fn).run, example))
                cursor += 1
                attempts += 1
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                tree = future.result()
                if not any(leaf["actions"] for leaf in tree["leaves"]):
                    direct += 1  # reader answered directly: no proxy tokens to train
                elif tree["zero_signal"]:
                    zero_trees.append(tree)  # all leaves tie: zero advantage, dropped (DAPO dynamic sampling)
                elif len(collected) < args.questions_per_iteration:
                    collected.append(tree)
                else:
                    extra_trees += 1  # usable but over the batch size: same batch size every step
            if len(collected) >= args.questions_per_iteration:
                cancelled = {future for future in pending if future.cancel()}
                pending -= cancelled
                attempts -= len(cancelled)
            want = args.questions_per_iteration - len(collected) - len(pending)
        return {"collected": collected, "zero_trees": zero_trees, "direct": direct, "attempts": attempts, "extra_trees": extra_trees, "seconds": time.monotonic() - started_at}

    def launch(step: int):
        """Start the rollouts for `step` in the background, pinned to the adapter the server serves now."""
        pinned = role_generate_fn(policy.pinned(policy.current()), temperature=args.temperature, top_p=args.top_p)
        return prefetcher.submit(collect, step, pinned)

    eval_history: list[dict[str, Any]] = []

    def run_eval(step: int) -> dict[str, Any]:
        actor.eval()
        report = evaluate_policy(greedy_fn, reader, retriever, eval_examples, max_steps=args.max_steps, workers=args.eval_workers)
        summary = {key: value for key, value in report.items() if key != "per_question"}
        summary["step"] = step
        eval_history.append(summary)
        write_json(output / "eval_history.json", eval_history)
        tqdm.write(json.dumps({"eval": summary}))
        return summary

    if eval_examples is not None and args.sft_f1 is None:
        stopper.set_baseline(run_eval(0)["f1"])

    metrics: list[dict[str, Any]] = []
    live_log = output / "metrics.jsonl"  # one line per step, for the live dashboard (metrics.json is only rewritten at saves)
    live_log.write_text("", encoding="utf-8")
    stop_reason, step = None, 0
    started = time.monotonic()
    upcoming = launch(1) if prefetcher is not None else None
    progress = tqdm(total=args.iterations, desc="GRPO", unit="step", dynamic_ncols=True)
    for step in range(1, args.iterations + 1):
        actor.eval()  # rollouts and ratios use the same dropout-free policy
        waited = time.monotonic()
        batch = upcoming.result() if prefetcher is not None else collect(step, sample_fn)
        wait_seconds = time.monotonic() - waited
        if prefetcher is not None and step < args.iterations:
            upcoming = launch(step + 1)  # samples with the adapter of step - 1 while this step trains
        collected, zero_trees, direct = batch["collected"], batch["zero_trees"], batch["direct"]
        attempts, extra_trees, rollout_seconds = batch["attempts"], batch["extra_trees"], batch["seconds"]
        signal_trees = len(collected) + len(zero_trees)
        zero_fraction = len(zero_trees) / signal_trees if signal_trees else 1.0
        info: dict[str, Any] = {"step": step, **tree_stats(collected + zero_trees), "zero_signal_groups": len(zero_trees), "zero_signal_fraction": zero_fraction, "direct_answer_trees": direct, "refilled_questions": max(0, attempts - args.questions_per_iteration), "extra_trees": extra_trees, "rollout_seconds": rollout_seconds, "rollout_wait_seconds": wait_seconds, "seconds_per_question": rollout_seconds / max(attempts, 1)}
        if collected:
            lr = lr_at(step, args.learning_rate, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            leaves = [leaf for tree in collected for leaf in tree["leaves"]]
            backward_started = time.monotonic()
            info.update(grpo_backward(actor, None, tokenizer, leaves, clip_low=args.clip_low, clip_high=args.clip_high, kl_beta=args.kl_beta, max_batch_tokens=args.backward_batch_tokens, sampler_old=args.async_rollout))
            info["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            optimizer.step()
            info.update(lr=lr, updated=True, backward_seconds=time.monotonic() - backward_started)
            if policy is not None:  # the next rollouts must sample from the updated adapter
                synced = time.monotonic()
                policy.update(save_adapter(actor, tokenizer, output / "live" / f"step-{step:06d}", {"step": step}), adapter_name(run_tag, step), keep_loaded=2 if args.async_rollout else 1, keep_files=3 if args.async_rollout else 2)
                info["sync_seconds"] = time.monotonic() - synced
        else:
            info["updated"] = False
        extraction = extraction_stats(retriever)
        if extraction is not None:
            info.update(extract_usd=extraction["usd"], extract_requested=extraction["requested"], extract_failed=extraction["failed"])
        elapsed_hours = (time.monotonic() - started) / 3600
        info.update(elapsed_hours=elapsed_hours, eta_hours=elapsed_hours / step * (args.iterations - step))
        metrics.append(info)
        with live_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(info) + "\n")
        tqdm.write(json.dumps(info))
        progress.set_postfix(reward=f"{info['mean_reward'] or 0.0:.3f}", kl=f"{info.get('kl', 0.0):.4f}", zero=f"{zero_fraction:.2f}")
        progress.update(1)
        reason = stopper.after_step(format_rate=info["format_rate"], kl=info.get("kl", 0.0), entropy=info.get("entropy"), zero_signal=zero_fraction, elapsed_hours=elapsed_hours)
        if eval_examples is not None and step % args.eval_every == 0:
            summary = run_eval(step)
            eval_reason, is_best = stopper.after_eval(step=step, f1=summary["f1"])
            if is_best:
                save_adapter(actor, tokenizer, output / "best", {"step": step, "eval": summary, "sft_checkpoint": str(Path(args.sft_checkpoint).resolve()), "config": config})
            reason = reason or eval_reason
        if reason or step % args.save_every == 0:
            save_adapter(actor, tokenizer, output / f"checkpoint-grpo-step-{step:06d}", {"step": step, "sft_checkpoint": str(Path(args.sft_checkpoint).resolve()), "config": config, "metrics": metrics[-50:]})
            write_json(output / "metrics.json", metrics)
        if reason:
            stop_reason = reason
            tqdm.write(json.dumps({"early_stop": reason, "step": step}))
            break
    progress.close()
    if prefetcher is not None:
        if step < args.iterations:  # stopped early: let the prefetched rollout finish before closing clients
            upcoming.cancel() or upcoming.exception()
        prefetcher.shutdown()
    pool.shutdown()
    final = output / f"checkpoint-grpo-step-{step:06d}"
    if not final.exists():
        save_adapter(actor, tokenizer, final, {"step": step, "sft_checkpoint": str(Path(args.sft_checkpoint).resolve()), "config": config, "metrics": metrics[-50:]})
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_manifest.json", {"schema_version": "grasp-grpo-tree-v2", "stop_reason": stop_reason, "stop_step": step if stop_reason else None, "completed_steps": step, "sft_f1": stopper.sft_f1, "best_step": stopper.best_step, "best_f1": stopper.best_f1, "best_checkpoint": str(output / "best") if stopper.best_step is not None else None, "final_checkpoint": str(final), "sft_checkpoint": str(Path(args.sft_checkpoint).resolve()), "actor_initialization": str(Path(actor_source).resolve()), "qa": str(Path(args.qa).resolve()), "config": config, "created_at_utc": _now()})
    if batcher is not None:
        batcher.close()
    retriever.close()
    print(json.dumps({"stop_reason": stop_reason, "completed_steps": step, "best_step": stopper.best_step, "best_f1": stopper.best_f1, "final_checkpoint": str(final)}))


if __name__ == "__main__":
    main()
