from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import read_examples, write_json
from .environment import BinaryTreeEnvironment
from .model import load_policy
from .reader import OpenAIReader
from .retrieval import CorpusRouter, DenseRetriever


def arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--index-dir")
    parser.add_argument("--corpus-config", help="JSON mapping metadata.dataset to index/triple artifacts")
    parser.add_argument("--triple-cache")
    parser.add_argument("--reader-url", default="http://localhost:8000/v1")
    parser.add_argument("--reader-api-key", default="EMPTY")
    parser.add_argument("--reader-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-candidate-triples", type=int, default=20)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-leaves", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--retrieval-cost", type=float, default=0.0)
    parser.add_argument("--rd-gain-weight", type=float, default=0.3)
    parser.add_argument("--qf-gain-weight", type=float, default=0.7)
    parser.add_argument("--ki-hit-weight", type=float, default=0.3)
    parser.add_argument("--ki-miss-penalty", type=float, default=0.1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-proxy-actions", action="store_true", help="Skip direct reader answers until collecting requested action-bearing trees")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-initial-retrieval", action="store_true")


def collect(args) -> list[dict]:
    import torch
    random.seed(args.seed); torch.manual_seed(args.seed)
    actor, tokenizer, _ = load_policy(args.checkpoint, trainable=False, load_in_4bit=args.load_in_4bit)
    reader = OpenAIReader(args.reader_url, args.reader_api_key, args.reader_model)
    if bool(args.index_dir) == bool(args.corpus_config):
        raise ValueError("provide exactly one of --index-dir or --corpus-config")
    retriever = CorpusRouter(args.corpus_config, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples) if args.corpus_config else DenseRetriever(args.index_dir, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples, triple_cache=args.triple_cache)
    env = BinaryTreeEnvironment(actor, tokenizer, reader, retriever, branch_factor=args.branch_factor, max_steps=args.max_steps, max_leaves=args.max_leaves, temperature=args.temperature, top_p=args.top_p, alpha=args.alpha, retrieval_cost=args.retrieval_cost, behavior_checkpoint=str(Path(args.checkpoint).resolve()), initial_retrieval=not args.no_initial_retrieval, rd_gain_weight=args.rd_gain_weight, qf_gain_weight=args.qf_gain_weight, ki_hit_weight=args.ki_hit_weight, ki_miss_penalty=args.ki_miss_penalty)
    examples = read_examples(args.qa)
    trees = []
    skipped_direct_answers = 0
    for example in examples:
        tree = env.run(example)
        has_actions = any(leaf["actions"] for leaf in tree["leaves"])
        if args.require_proxy_actions and not has_actions:
            skipped_direct_answers += 1
            continue
        trees.append(tree)
        if args.limit is not None and len(trees) >= args.limit:
            break
    if not trees:
        raise RuntimeError("No rollout tree matched the requested criteria")
    retriever.close()
    return trees


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fresh on-policy binary-tree GRPO rollouts")
    arguments(parser); args = parser.parse_args()
    trees = collect(args)
    write_json(args.output, {"schema_version": "grasp-grpo-tree-v1", "behavior_checkpoint": str(Path(args.checkpoint).resolve()), "trees": trees})
    print(f"Wrote {len(trees)} binary trees to {args.output}")


if __name__ == "__main__":
    main()
