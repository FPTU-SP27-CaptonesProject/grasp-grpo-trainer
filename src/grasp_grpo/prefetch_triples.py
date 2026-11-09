"""Extract triples ahead of a run for the documents its questions are likely to retrieve.

During the stage 3 pilot (2026-11-09) every step met ~350 documents missing from the triple cache, so
rollouts waited on Bedrock (steps of 600-1000 s instead of ~330 s) and the $10 budget ran out at step
137, after which new documents got no triples. The policy's queries are not known in advance, but they
paraphrase the question, so the top `depth` documents for the question itself cover most of what a rollout
retrieves. This fills them first; the run's live extraction then only handles the remainder.

`--dry-run` only counts the missing documents and estimates the cost. Reruns skip cached documents, so an
interrupted prefetch resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .data import read_examples
from .eval_ids import load_eval_examples
from .retrieval import add_retrieval_arguments, extraction_stats, retriever_from_args

# Pilot average: $10.00 for 55,554 documents.
USD_PER_DOC = 10.0 / 55_554


def _children(retriever) -> dict[str, Any]:
    return dict(retriever.retrievers) if hasattr(retriever, "retrievers") else {"": retriever}


def _dataset(example: dict[str, Any]) -> str:
    return str(example.get("metadata", {}).get("dataset", ""))


def _uncached(child, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Passages with no cache row at all (a row with an empty list means "extracted, no triples")."""
    if not passages or child.triple_db is None:
        return []
    ids = [p["doc_id"] for p in passages]
    marks = ",".join("?" for _ in ids)
    cached = {str(row[0]) for row in child.triple_db.execute(f"SELECT DISTINCT doc_id FROM triples WHERE doc_id IN ({marks})", ids)}
    return [p for p in passages if p["doc_id"] not in cached]


def prefetch(retriever, examples: list[dict[str, Any]], *, depth: int, dry_run: bool, workers: int = 8, log_every: int = 200) -> dict[str, Any]:
    """Search each question (serially: the retrievers are not thread-safe) and extract the missing
    documents in the background. Returns document counts and extraction stats."""
    children = _children(retriever)
    for child in children.values():
        child.top_k = depth
    seen: set[str] = set()
    counts = {"questions": 0, "documents": 0, "missing": 0}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prefetch")
    pending = []
    started = time.time()
    try:
        for example in examples:
            child = children.get(_dataset(example)) or children.get("")
            if child is None:
                raise KeyError(f"example {example.get('id')} has dataset={_dataset(example)!r}; corpus config has {', '.join(children)}")
            result = child.search(example["question"], fill=False)
            fresh = [p for p in result if p["doc_id"] not in seen]
            seen.update(p["doc_id"] for p in fresh)
            missing = _uncached(child, fresh)
            for passage in missing:
                passage["needs_triples"] = True
            counts["questions"] += 1
            counts["documents"] += len(fresh)
            counts["missing"] += len(missing)
            if missing and not dry_run and getattr(child, "extractor", None) is not None:
                pending.append(pool.submit(child.extractor.fill, missing))
            if counts["questions"] % log_every == 0:
                done = sum(f.done() for f in pending)
                print(json.dumps({**counts, "fills_done": f"{done}/{len(pending)}", "elapsed_min": round((time.time() - started) / 60, 1), "extraction": extraction_stats(retriever)}), flush=True)
        for future in pending:
            future.result()
    finally:
        pool.shutdown(wait=True)
    return {**counts, "estimated_usd": round(counts["missing"] * USD_PER_DOC, 2), "extraction": extraction_stats(retriever), "minutes": round((time.time() - started) / 60, 1)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fill the triple cache for the top documents of every question before training")
    parser.add_argument("--qa", action="append", required=True, help="training QA JSONL (repeatable)")
    parser.add_argument("--eval-qa", help="eval QA JSONL; with --eval-ids only the fixed eval questions")
    parser.add_argument("--eval-ids")
    parser.add_argument("--depth", type=int, default=20, help="documents kept per question (training retrieves 10 per query)")
    parser.add_argument("--dry-run", action="store_true", help="count missing documents and estimate the cost; extract nothing")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="first N questions only (a timing probe)")
    add_retrieval_arguments(parser)
    args = parser.parse_args(argv)
    args.extract_triples = not args.dry_run
    examples = [x for path in args.qa for x in read_examples(path)]
    if args.eval_qa:
        examples += load_eval_examples(args.eval_qa, args.eval_ids)
    if args.limit:
        examples = examples[: args.limit]
    retriever = retriever_from_args(args)
    try:
        report = prefetch(retriever, examples, depth=args.depth, dry_run=args.dry_run, workers=args.workers)
    finally:
        retriever.close()  # also waits for and closes the extractors
    print(json.dumps({"prefetch": report, "dry_run": args.dry_run}, indent=2), flush=True)


if __name__ == "__main__":
    main()
