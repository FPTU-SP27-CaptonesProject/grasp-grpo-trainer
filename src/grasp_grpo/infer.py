"""Deterministic sequential GRASP inference for checkpoint evaluation."""
from __future__ import annotations

import argparse
import json

from .contracts import ROLE_MAX_TOKENS, candidate_index, parse, prompt
from .data import read_examples, write_json
from .environment import ANSWER_RE, THOUGHT_RE, reader_prompt
from .model import generate, load_policy
from .reader import OpenAIReader
from .retrieval import CorpusRouter, DenseRetriever


def act(model, tokenizer, role: str, state: dict) -> tuple[object, bool, str | None, str]:
    action = generate(model, tokenizer, prompt(role, state), samples=1, max_new_tokens=ROLE_MAX_TOKENS[role], temperature=0.0, top_p=1.0)[0]
    parsed, valid, error = parse(role, action.text)
    return parsed, valid, error, action.text


def run_example(proxy, tokenizer, reader, retriever, example: dict, max_steps: int) -> dict:
    select = getattr(retriever, "select", None)
    if select is not None:
        select(example)
    question = example["question"]
    context, thoughts, actions, answer = retriever.search(question), [], [], None
    for _ in range(max_steps):
        response = reader.generate(reader_prompt(question, context, thoughts, final_only=False))
        matched = ANSWER_RE.search(response)
        if matched:
            answer = matched.group(1).strip()
            break
        thought_match = THOUGHT_RE.search(response)
        thought = thought_match.group(1).strip() if thought_match else response.strip()
        thoughts.append(thought)
        state = {"question": question, "context": context, "thoughts": thoughts, "current_thought": thought}
        retrieve, valid, error, text = act(proxy, tokenizer, "rd", state)
        actions.append({"role": "rd", "response": text, "valid_format": valid, "validation_error": error})
        if not valid:
            retrieve = True
        if not retrieve:
            continue
        qf, valid, error, text = act(proxy, tokenizer, "qf", state)
        actions.append({"role": "qf", "response": text, "valid_format": valid, "validation_error": error})
        query = str(qf.get("query") or question) if valid else question
        passages = retriever.search(query)
        candidates = retriever.candidate_triples(query, passages)
        if not candidates:
            continue
        selected, valid, error, text = act(proxy, tokenizer, "ki", {"question": question, "chain": qf.get("chain", []) if valid else [], "query": query, "candidates": candidates})
        actions.append({"role": "ki", "response": text, "valid_format": valid, "validation_error": error})
        chosen = candidate_index(selected if valid else None, candidates)
        evidence = next((item for item in passages if item["doc_id"] == candidates[chosen if chosen is not None else 0]["doc_id"]), passages[0])
        if all(item["doc_id"] != evidence["doc_id"] for item in context):
            context.append(evidence)
    if answer is None:
        response = reader.generate(reader_prompt(question, context, thoughts, final_only=True))
        matched = ANSWER_RE.search(response)
        answer = matched.group(1).strip() if matched else response.strip()
    return {"id": example["id"], "prediction": answer, "context_doc_ids": [item["doc_id"] for item in context], "actions": actions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic GRASP predictions without gold-evidence access")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--qa", required=True)
    parser.add_argument("--index-dir"); parser.add_argument("--corpus-config"); parser.add_argument("--triple-cache")
    parser.add_argument("--reader-url", default="http://localhost:8000/v1"); parser.add_argument("--reader-api-key", default="EMPTY"); parser.add_argument("--reader-model", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--top-k", type=int, default=10); parser.add_argument("--max-candidate-triples", type=int, default=20); parser.add_argument("--max-steps", type=int, default=3); parser.add_argument("--limit", type=int); parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    if bool(args.index_dir) == bool(args.corpus_config):
        raise ValueError("provide exactly one of --index-dir or --corpus-config")
    proxy, tokenizer, _ = load_policy(args.checkpoint, trainable=False, load_in_4bit=args.load_in_4bit)
    reader = OpenAIReader(args.reader_url, args.reader_api_key, args.reader_model)
    retriever = CorpusRouter(args.corpus_config, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples) if args.corpus_config else DenseRetriever(args.index_dir, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples, triple_cache=args.triple_cache)
    examples = read_examples(args.qa)
    if args.limit is not None:
        examples = examples[:args.limit]
    predictions = [run_example(proxy, tokenizer, reader, retriever, example, args.max_steps) for example in examples]
    retriever.close()
    write_json(args.output, predictions)
    print(json.dumps({"predictions": len(predictions), "output": args.output}))


if __name__ == "__main__":
    main()
