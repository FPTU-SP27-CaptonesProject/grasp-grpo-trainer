"""On-policy binary-tree GRPO training entry point."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch

from .data import read_examples, write_json
from .environment import BinaryTreeEnvironment
from .grpo import grpo_loss
from .model import load_policy
from .reader import OpenAIReader
from .retrieval import CorpusRouter, DenseRetriever


def parse_args():
    p = argparse.ArgumentParser(description="Train GRASP Qwen 3B LoRA with on-policy binary-tree GRPO")
    p.add_argument("--sft-checkpoint", required=True, help="Local Project A PEFT adapter; frozen KL reference")
    p.add_argument("--actor-checkpoint", help="Optional GRPO adapter to continue training; SFT remains the KL reference")
    p.add_argument("--qa", required=True); p.add_argument("--index-dir"); p.add_argument("--corpus-config"); p.add_argument("--triple-cache")
    p.add_argument("--reader-url", default="http://localhost:8000/v1"); p.add_argument("--reader-api-key", default="EMPTY"); p.add_argument("--reader-model", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--encoder", default="intfloat/e5-large-v2")
    p.add_argument("--iterations", type=int, default=100); p.add_argument("--questions-per-iteration", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-7); p.add_argument("--clip-epsilon", type=float, default=0.2); p.add_argument("--kl-beta", type=float, default=0.002)
    p.add_argument("--branch-factor", type=int, default=2); p.add_argument("--max-steps", type=int, default=3); p.add_argument("--max-leaves", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7); p.add_argument("--top-p", type=float, default=0.95); p.add_argument("--top-k", type=int, default=10); p.add_argument("--max-candidate-triples", type=int, default=20); p.add_argument("--alpha", type=float, default=0.3); p.add_argument("--retrieval-cost", type=float, default=0.0)
    p.add_argument("--save-every", type=int, default=10); p.add_argument("--seed", type=int, default=42); p.add_argument("--load-in-4bit", action="store_true"); p.add_argument("--no-initial-retrieval", action="store_true")
    return p.parse_args()


def save_checkpoint(actor, tokenizer, output: Path, step: int, args, metrics: list[dict]) -> Path:
    path = output / f"checkpoint-grpo-step-{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    actor.save_pretrained(path); tokenizer.save_pretrained(path)
    write_json(path / "grpo_config.json", {"sft_checkpoint": str(Path(args.sft_checkpoint).resolve()), "created_at_utc": datetime.now(UTC).isoformat(), "tree": {"branch_factor": args.branch_factor, "max_steps": args.max_steps, "max_leaves": args.max_leaves}, "grpo": {"learning_rate": args.learning_rate, "clip_epsilon": args.clip_epsilon, "kl_beta": args.kl_beta}, "metrics": metrics[-50:]})
    return path


def main() -> None:
    args = parse_args(); torch.manual_seed(args.seed)
    if args.branch_factor != 2 or args.questions_per_iteration < 1:
        raise ValueError("branch-factor must be 2 and questions-per-iteration positive")
    if bool(args.index_dir) == bool(args.corpus_config):
        raise ValueError("provide exactly one of --index-dir or --corpus-config")
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    examples = read_examples(args.qa)
    if not examples: raise ValueError("no valid training examples")
    actor_source = args.actor_checkpoint or args.sft_checkpoint
    actor, tokenizer, _ = load_policy(actor_source, trainable=True, load_in_4bit=args.load_in_4bit)
    reference, _, _ = load_policy(args.sft_checkpoint, trainable=False, load_in_4bit=args.load_in_4bit)
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad), lr=args.learning_rate)
    reader = OpenAIReader(args.reader_url, args.reader_api_key, args.reader_model)
    retriever = CorpusRouter(args.corpus_config, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples) if args.corpus_config else DenseRetriever(args.index_dir, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples, triple_cache=args.triple_cache)
    metrics = []
    for step in range(1, args.iterations + 1):
        batch = [examples[(step - 1 + i) % len(examples)] for i in range(args.questions_per_iteration)]
        # The actor itself generates every tree: never reuse rollout data after an update.
        environment = BinaryTreeEnvironment(actor, tokenizer, reader, retriever, branch_factor=2, max_steps=args.max_steps, max_leaves=args.max_leaves, temperature=args.temperature, top_p=args.top_p, alpha=args.alpha, retrieval_cost=args.retrieval_cost, behavior_checkpoint=f"in_memory_actor_before_step_{step}", initial_retrieval=not args.no_initial_retrieval)
        trees = [environment.run(example) for example in batch]
        leaves = [leaf for tree in trees for leaf in tree["leaves"]]
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        loss, info = grpo_loss(actor, reference, tokenizer, leaves, clip_epsilon=args.clip_epsilon, kl_beta=args.kl_beta)
        loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in actor.parameters() if p.requires_grad), 1.0); optimizer.step()
        info.update({"step": step, "mean_reward": sum(float(x["reward"]["total"]) for x in leaves) / len(leaves), "leaves": len(leaves)})
        metrics.append(info); print(json.dumps(info))
        if step % args.save_every == 0 or step == args.iterations:
            save_checkpoint(actor, tokenizer, output, step, args, metrics)
            write_json(output / "metrics.json", metrics)
    final = save_checkpoint(actor, tokenizer, output, args.iterations, args, metrics)
    write_json(output / "run_manifest.json", {"schema_version": "grasp-grpo-tree-v1", "final_checkpoint": str(final), "sft_checkpoint": str(Path(args.sft_checkpoint).resolve()), "actor_initialization": str(Path(actor_source).resolve()), "qa": str(Path(args.qa).resolve()), "created_at_utc": datetime.now(UTC).isoformat()})
    retriever.close(); print(f"Final GRPO adapter: {final}")


if __name__ == "__main__": main()
