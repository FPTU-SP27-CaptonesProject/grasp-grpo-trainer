"""Report baselines on the same retriever, reader prompt and metrics as the GRASP eval (policy_eval).

- llm:           reader only, no retrieval (context "None").
- single:        one search with the question (top-k), then the reader answers in one completion.
- single-bedrock: as `single`, answered by a hosted chat model (default Qwen3-235B on Bedrock) with the
                 same reader prompt.
- ircot:         Trivedi et al. 2023: start from the question's top-k; each step the reader writes one
                 thought, that thought is the next query, and `docs_per_step` new documents join the context;
                 stop at <answer> or after `max_steps` (then the reader answers from scratch, as GRASP does).

Every run function returns the `run_example` record (id, prediction, context_doc_ids, actions) plus
`retrievals`; `emit(kind, data)` reports each step for the demo.

    grasp-eval-baseline --method ircot --qa dev.jsonl --eval-ids config/test_ids.json --corpus-config ... \\
        --reader-model Qwen/Qwen2.5-7B-Instruct --output ircot.json
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .contracts import READER, reader_context
from .data import read_examples, write_json
from .eval_ids import filter_examples
from .policy_eval import summarize
from .reader import add_reader_arguments, extract_answer, reader_from_args
from .retrieval import BoundRetriever, add_retrieval_arguments, retriever_from_args

Emit = Callable[[str, dict[str, Any]], None] | None
METHODS = ("llm", "single", "single-bedrock", "ircot")
BEDROCK_URL = "https://bedrock-mantle.ap-southeast-2.api.aws/v1"


def _emit(emit: Emit, kind: str, data: dict[str, Any]) -> None:
    if emit is not None:
        emit(kind, data)


def _select(retriever, example: dict[str, Any]) -> None:
    select = getattr(retriever, "select", None)
    if select is not None:
        select(example)


def _record(example: dict[str, Any], answer: str, context: list[dict[str, Any]], retrievals: int) -> dict[str, Any]:
    return {"id": example["id"], "prediction": answer, "context_doc_ids": [d["doc_id"] for d in context], "actions": [], "retrievals": retrievals}


def run_llm_only(reader, retriever, example: dict[str, Any], *, emit: Emit = None) -> dict[str, Any]:
    answer = reader.final(example["question"], [], [], "")
    _emit(emit, "answer", {"text": answer})
    return _record(example, answer, [], 0)


def run_single_step(reader, retriever, example: dict[str, Any], *, emit: Emit = None) -> dict[str, Any]:
    _select(retriever, example)
    question = example["question"]
    context = retriever.search(question)
    _emit(emit, "retrieve", {"query": question, "docs": context})
    answer = reader.final(question, context, [], "")
    _emit(emit, "answer", {"text": answer})
    return _record(example, answer, context, 1)


def run_ircot(reader, retriever, example: dict[str, Any], *, max_steps: int = 5, docs_per_step: int = 3, emit: Emit = None) -> dict[str, Any]:
    _select(retriever, example)
    question = example["question"]
    context = list(retriever.search(question))
    _emit(emit, "retrieve", {"query": question, "docs": context})
    seen = {d["doc_id"] for d in context}
    thoughts: list[str] = []
    transcript, answer, retrievals = "", None, 1
    for step in range(1, max_steps + 1):
        result = reader.step(question, context, thoughts, transcript)
        transcript += result.piece
        if result.answer is not None:
            answer = result.answer
            break
        if result.thought is None:
            break
        thoughts.append(result.thought)
        _emit(emit, "thought", {"step": step, "text": result.thought})
        new = [d for d in retriever.search(result.thought) if d["doc_id"] not in seen][:docs_per_step]
        retrievals += 1
        seen.update(d["doc_id"] for d in new)
        context.extend(new)
        _emit(emit, "retrieve", {"step": step, "query": result.thought, "docs": new})
    if answer is None:
        answer = reader.final(question, context, thoughts, transcript)
    _emit(emit, "answer", {"text": answer})
    return _record(example, answer, context, retrievals)


def answer_or_last_line(text: str) -> str:
    """<answer> if present; a chat model that skips the tag usually ends with the bare answer."""
    answer = extract_answer(text)
    if answer is not None:
        return answer
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


class ChatReader:
    """The reader prompt (reader.txt) sent as one chat message to a hosted model; answers only (`final`)."""

    def __init__(self, client, model: str, *, max_tokens: int = 1024):
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def final(self, question: str, context: list[dict[str, Any]], thoughts: list[str], transcript: str) -> str:
        prompt = READER.format(context=reader_context(context) or "None", question=question)
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=self.max_tokens)
        return answer_or_last_line(response.choices[0].message.content or "")

    def step(self, *args, **kwargs):
        raise NotImplementedError("ChatReader answers in one call (single-bedrock); use the local reader for IRCoT")


def bedrock_reader(model: str) -> ChatReader:
    from openai import OpenAI

    key = os.environ.get("BEDROCK_API_KEY")
    if not key:
        raise RuntimeError("BEDROCK_API_KEY is not set (needed for --method single-bedrock)")
    return ChatReader(OpenAI(base_url=os.environ.get("BEDROCK_BASE_URL", BEDROCK_URL), api_key=key, timeout=180, max_retries=8), model)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--qa", required=True)
    p.add_argument("--eval-ids", help="JSON with an `ids` list (config/eval_ids.json, config/test_ids.json)")
    p.add_argument("--bedrock-model", default="qwen.qwen3-235b-a22b-2507")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--docs-per-step", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", required=True)
    add_retrieval_arguments(p)
    add_reader_arguments(p)
    args = p.parse_args(argv)

    examples = read_examples(args.qa)
    if args.eval_ids:
        with open(args.eval_ids, encoding="utf-8") as handle:
            examples = filter_examples(examples, json.load(handle)["ids"])
    if args.limit is not None:
        examples = examples[: args.limit]
    retriever = None if args.method == "llm" else retriever_from_args(args)
    reader = bedrock_reader(args.bedrock_model) if args.method == "single-bedrock" else reader_from_args(args)
    lock = threading.Lock()

    def one(example: dict[str, Any]) -> dict[str, Any]:
        bound = BoundRetriever(retriever, example, lock) if retriever is not None else None
        if args.method == "llm":
            return run_llm_only(reader, bound, example)
        if args.method == "ircot":
            return run_ircot(reader, bound, example, max_steps=args.max_steps, docs_per_step=args.docs_per_step)
        return run_single_step(reader, bound, example)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        predictions = list(pool.map(one, examples))
    report = summarize(examples, predictions, time.monotonic() - started)
    if retriever is not None:
        retriever.close()
    report.update(method=args.method, reader=args.bedrock_model if args.method == "single-bedrock" else args.reader_model,
                  eval_ids=args.eval_ids, top_k=args.top_k, max_steps=args.max_steps, docs_per_step=args.docs_per_step,
                  predictions=[{k: p[k] for k in ("id", "prediction", "context_doc_ids", "retrievals")} for p in predictions])
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_question", "predictions")}), flush=True)


if __name__ == "__main__":
    main()
