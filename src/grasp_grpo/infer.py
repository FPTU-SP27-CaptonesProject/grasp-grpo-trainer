"""Deterministic sequential GRASP inference for checkpoint evaluation."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from .batcher import role_generate_fn
from .contracts import candidate_index, parse, prompt
from .data import read_examples, write_json
from .model import generate, load_policy
from .reader import add_reader_arguments, reader_from_args
from .retrieval import add_retrieval_arguments, retriever_from_args


def act(generate_fn, role: str, state: dict) -> tuple[object, bool, str | None, str]:
    action = generate_fn(role, prompt(role, state), 1)[0]
    parsed, valid, error = parse(role, action.text)
    return parsed, valid, error, action.text


Emit = Callable[[str, dict[str, Any]], None] | None


def run_example(generate_fn, reader, retriever, example: dict, max_steps: int, *, emit: Emit = None) -> dict:
    """`emit(kind, data)` reports each step for the demo (retrieve, thought, rd, qf, ki, answer); it never
    changes the result."""
    def report(kind: str, data: dict[str, Any]) -> None:
        if emit is not None:
            emit(kind, data)

    select = getattr(retriever, "select", None)
    if select is not None:
        select(example)
    question = example["question"]
    context, thoughts, actions, transcript, answer = retriever.search(question), [], [], "", None
    report("retrieve", {"step": 0, "query": question, "docs": list(context)})
    for index in range(1, max_steps + 1):
        step = reader.step(question, context, thoughts, transcript)
        transcript += step.piece
        if step.answer is not None:
            answer = step.answer
            report("answer", {"text": answer, "raw": step.piece})
            break
        if step.thought is None:
            break
        thought = step.thought
        thoughts.append(thought)
        report("thought", {"step": index, "text": thought})
        state = {"question": question, "context": context, "thoughts": thoughts, "current_thought": thought}
        retrieve, valid, error, text = act(generate_fn, "rd", state)
        actions.append({"role": "rd", "response": text, "valid_format": valid, "validation_error": error})
        if not valid:
            retrieve = True
        report("rd", {"step": index, "retrieve": bool(retrieve), "raw": text, "valid": valid})
        if not retrieve:
            continue
        qf, valid_qf, error, text = act(generate_fn, "qf", state)
        actions.append({"role": "qf", "response": text, "valid_format": valid_qf, "validation_error": error})
        query = str(qf.get("query") or question) if valid_qf else question
        report("qf", {"step": index, "query": query, "chain": qf.get("chain", []) if valid_qf else [], "raw": text, "valid": valid_qf})
        passages = retriever.search(query)
        report("retrieve", {"step": index, "query": query, "docs": passages})
        candidates = retriever.candidate_triples(query, passages, question=question)
        if not candidates:
            report("ki", {"step": index, "candidates": [], "chosen": None, "raw": None, "valid": None, "added_doc": None})
            continue
        ki_state = {"question": question, "chain": qf.get("chain", []) if valid_qf else [], "query": query, "candidates": candidates}
        selected, valid_ki, error, text = act(generate_fn, "ki", ki_state)
        actions.append({"role": "ki", "response": text, "valid_format": valid_ki, "validation_error": error})
        chosen = candidate_index(selected if valid_ki else None, candidates)
        evidence = next((item for item in passages if item["doc_id"] == candidates[chosen if chosen is not None else 0]["doc_id"]), passages[0])
        added = all(item["doc_id"] != evidence["doc_id"] for item in context)
        if added:
            context.append(evidence)
        report("ki", {"step": index, "candidates": candidates, "chosen": chosen if chosen is not None else 0, "raw": text, "valid": valid_ki,
                      "added_doc": evidence["doc_id"] if added else None})
    if answer is None:
        answer = reader.final(question, context, thoughts, transcript)
        report("answer", {"text": answer, "raw": None})
    return {"id": example["id"], "prediction": answer, "context_doc_ids": [item["doc_id"] for item in context], "actions": actions,
            "thoughts": thoughts, "transcript": transcript}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic GRASP predictions without gold-evidence access")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qa", required=True)
    add_retrieval_arguments(parser)
    add_reader_arguments(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    retriever = retriever_from_args(args)
    proxy, tokenizer, _ = load_policy(args.checkpoint, trainable=False, load_in_4bit=args.load_in_4bit)
    generate_fn = role_generate_fn(lambda text, **kwargs: generate(proxy, tokenizer, text, **kwargs), temperature=0.0, top_p=1.0)
    reader = reader_from_args(args)
    examples = read_examples(args.qa)
    if args.limit is not None:
        examples = examples[: args.limit]
    predictions = [run_example(generate_fn, reader, retriever, example, args.max_steps) for example in examples]
    retriever.close()
    write_json(args.output, predictions)
    print(json.dumps({"predictions": len(predictions), "output": args.output}))


if __name__ == "__main__":
    main()
