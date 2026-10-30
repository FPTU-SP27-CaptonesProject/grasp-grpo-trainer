"""Sequential deployment runtime using the final GRPO proxy adapter."""
from __future__ import annotations

import argparse

from .contracts import candidate_index, parse, prompt
from .environment import ANSWER_RE, THOUGHT_RE, reader_prompt
from .model import generate, load_policy
from .reader import OpenAIReader
from .retrieval import DenseRetriever


def act(model, tokenizer, role: str, state: dict):
    action = generate(model, tokenizer, prompt(role, state), samples=1, max_new_tokens={"rd": 16, "qf": 256, "ki": 128}[role], temperature=0.0, top_p=1.0)[0]
    return parse(role, action.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive sequential GRASP chatbot")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-dir", required=True); parser.add_argument("--triple-cache")
    parser.add_argument("--reader-url", default="http://localhost:8000/v1"); parser.add_argument("--reader-api-key", default="EMPTY"); parser.add_argument("--reader-model", required=True)
    parser.add_argument("--encoder", default="intfloat/e5-large-v2"); parser.add_argument("--top-k", type=int, default=10); parser.add_argument("--max-steps", type=int, default=3); parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    proxy, tokenizer, _ = load_policy(args.checkpoint, trainable=False, load_in_4bit=args.load_in_4bit)
    reader = OpenAIReader(args.reader_url, args.reader_api_key, args.reader_model)
    retriever = DenseRetriever(args.index_dir, args.encoder, top_k=args.top_k, max_candidate_triples=20, triple_cache=args.triple_cache)
    print("GRASP chat ready. Type /quit to exit.")
    while (question := input("You> ").strip()) not in {"/quit", "/exit"}:
        if not question:
            continue
        context, thoughts, answer = retriever.search(question), [], None
        for _ in range(args.max_steps):
            response = reader.generate(reader_prompt(question, context, thoughts, final_only=False))
            matched = ANSWER_RE.search(response)
            if matched:
                answer = matched.group(1).strip()
                break
            thought_match = THOUGHT_RE.search(response)
            thought = thought_match.group(1).strip() if thought_match else response
            thoughts.append(thought)
            state = {"question": question, "context": context, "thoughts": thoughts, "current_thought": thought}
            retrieve, valid, _ = act(proxy, tokenizer, "rd", state)
            if not valid:
                retrieve = True
            if not retrieve:
                continue
            qf, valid, _ = act(proxy, tokenizer, "qf", state)
            query = str(qf.get("query") or question) if valid else question
            passages = retriever.search(query)
            candidates = retriever.candidate_triples(query, passages)
            if not candidates:
                continue
            selected, valid, _ = act(proxy, tokenizer, "ki", {"question": question, "chain": qf.get("chain", []) if valid else [], "query": query, "candidates": candidates})
            chosen = candidate_index(selected if valid else None, candidates)
            evidence = next((x for x in passages if x["doc_id"] == candidates[chosen if chosen is not None else 0]["doc_id"]), passages[0])
            if all(x["doc_id"] != evidence["doc_id"] for x in context):
                context.append(evidence)
        if answer is None:
            response = reader.generate(reader_prompt(question, context, thoughts, final_only=True))
            matched = ANSWER_RE.search(response)
            answer = matched.group(1).strip() if matched else response
        print(f"GRASP> {answer}")
        print("Sources:", ", ".join(f"[{x['doc_id']}] {x.get('title') or ''}" for x in context))
    retriever.close()


if __name__ == "__main__":
    main()
