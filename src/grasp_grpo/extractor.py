"""Lazy knowledge-triple extraction for retrieved documents missing from the Project A triple cache.

Reproduces Project A grasp.kg.extract.TripleExtractor (prompts/extract.txt, Qwen3-235B on Bedrock,
temperature 0, max 1024 tokens, one user message) and writes to the same `triples(doc_id, model, triples)`
table, keyed by the same model id, so the cache stays shareable with Project A.

Extraction runs outside the retriever lock, in parallel, and each document is extracted at most once
(concurrent requests for the same document wait for the first one). Spending is capped by `budget_usd`:
past it, missing documents simply get no triples (and are not cached, so a later run can fill them).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Project A src/grasp/agents/prompts/extract.txt, verbatim.
EXTRACT_PROMPT = """Extract knowledge triples from the passage below. Each triple states one fact as <head; relation; tail>.

Rules:
- Cover every factual statement in the passage, including dates, numbers, locations, roles and relations between entities.
- Use full entity names: replace pronouns and vague references ("he", "the film", "the company") with the entity they refer to. The passage title is usually the main subject.
- Keep relations short and explicit (e.g. "directed by", "born in", "capital of", "released in").
- Do not add facts that are not stated in the passage.

Output one triple per line and nothing else.

Title: {title}
Passage: {text}

Triples:"""

# Project A grasp.kg.triples._TRIPLE
_TRIPLE = re.compile(r"<\s*([^<>;]+?)\s*;\s*([^<>;]+?)\s*;\s*([^<>]+?)\s*>")

DEFAULT_BASE_URL = "https://bedrock-mantle.ap-southeast-2.api.aws/v1"
DEFAULT_MODEL = "qwen.qwen3-235b-a22b-2507"
# USD per million tokens, Sydney Standard tier (Project A configs/teacher/qwen3_235b_bedrock.yaml).
DEFAULT_PRICE = (0.2266, 0.9064)


def parse_triples(text: str) -> list[list[str]]:
    return [[h.strip(), r.strip(), t.strip()] for h, r, t in _TRIPLE.findall(text or "")]


class TripleExtractor:
    def __init__(self, cache_path: str, *, chat=None, model: str = DEFAULT_MODEL, budget_usd: float = 10.0, concurrency: int = 24, max_tokens: int = 1024, price: tuple[float, float] = DEFAULT_PRICE):
        """`chat(prompt, max_tokens) -> (text, prompt_tokens, completion_tokens)`; defaults to Bedrock."""
        self.model, self.budget_usd, self.max_tokens, self.price = model, budget_usd, max_tokens, price
        self._chat = chat or _bedrock_chat(model)
        self._db = sqlite3.connect(str(Path(cache_path).resolve()), check_same_thread=False, timeout=60)
        self._db.execute("CREATE TABLE IF NOT EXISTS triples (doc_id TEXT, model TEXT, triples TEXT, PRIMARY KEY (doc_id, model))")
        self._db.commit()
        self._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="triple-extract")
        self._lock = threading.Lock()
        self._inflight: dict[str, Future] = {}
        self.stats = {"requested": 0, "extracted": 0, "empty": 0, "failed": 0, "skipped_budget": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
        self._budget_warned = False

    def fill(self, passages: list[dict[str, Any]]) -> None:
        """Extract triples for every passage flagged `needs_triples` (in place) and clear the flag."""
        todo = [p for p in passages if p.pop("needs_triples", False)]
        if not todo:
            return
        futures = [(p, self._submit(p)) for p in todo]
        for passage, future in futures:
            try:
                triples = future.result()
            except Exception:  # noqa: BLE001 - counted in stats; the rollout continues without triples
                triples = []
            passage["triples"] = [{"doc_id": passage["doc_id"], "head": h, "relation": r, "tail": t} for h, r, t in triples]

    def _submit(self, passage: dict[str, Any]) -> Future:
        doc_id = passage["doc_id"]
        with self._lock:
            future = self._inflight.get(doc_id)
            if future is None:
                self.stats["requested"] += 1
                future = self._pool.submit(self._extract, doc_id, passage.get("title") or "", passage["text"])
                self._inflight[doc_id] = future
                future.add_done_callback(lambda _, key=doc_id: self._forget(key))
            return future

    def _forget(self, doc_id: str) -> None:
        with self._lock:
            self._inflight.pop(doc_id, None)

    def _extract(self, doc_id: str, title: str, text: str) -> list[list[str]]:
        with self._lock:
            row = self._db.execute("SELECT triples FROM triples WHERE doc_id = ? AND model = ?", (doc_id, self.model)).fetchone()
            if row is not None:  # extracted by another thread or process since the retriever looked
                return json.loads(row[0])
            if self.stats["usd"] >= self.budget_usd:
                self.stats["skipped_budget"] += 1
                if not self._budget_warned:
                    self._budget_warned = True
                    print(json.dumps({"triple_extraction": "budget reached", "usd": self.stats["usd"], "budget_usd": self.budget_usd}), flush=True)
                return []
        try:
            content, prompt_tokens, completion_tokens = self._chat(EXTRACT_PROMPT.format(title=title, text=text), self.max_tokens)
        except _BadRequest:
            content, prompt_tokens, completion_tokens = "", 0, 0  # Project A caches such documents as empty
        except Exception:
            with self._lock:
                self.stats["failed"] += 1
            raise
        triples = parse_triples(content)
        with self._lock:
            self.stats["prompt_tokens"] += prompt_tokens
            self.stats["completion_tokens"] += completion_tokens
            self.stats["usd"] += (prompt_tokens * self.price[0] + completion_tokens * self.price[1]) / 1e6
            self.stats["extracted" if triples else "empty"] += 1
            self._db.execute("INSERT OR REPLACE INTO triples VALUES (?, ?, ?)", (doc_id, self.model, json.dumps(triples, ensure_ascii=False)))
            self._db.commit()
        return triples

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._db.close()


class _BadRequest(Exception):
    pass


def _bedrock_chat(model: str):
    from openai import BadRequestError, OpenAI

    api_key = os.environ.get("BEDROCK_API_KEY")
    if not api_key:
        raise RuntimeError("BEDROCK_API_KEY is not set (needed for --extract-triples)")
    client = OpenAI(base_url=os.environ.get("BEDROCK_BASE_URL", DEFAULT_BASE_URL), api_key=api_key, timeout=180, max_retries=8)

    def chat(prompt: str, max_tokens: int) -> tuple[str, int, int]:
        try:
            response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens)
        except BadRequestError as error:  # e.g. a document over the context limit
            raise _BadRequest(str(error)) from error
        usage = response.usage
        return response.choices[0].message.content or "", getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0

    return chat


def add_extraction_arguments(parser) -> None:
    parser.add_argument("--extract-triples", action="store_true", help="Extract triples for retrieved documents missing from the cache (Bedrock, BEDROCK_API_KEY)")
    parser.add_argument("--extract-budget-usd", type=float, default=10.0, help="Stop extracting once this much was spent in this process")
    parser.add_argument("--extract-concurrency", type=int, default=24)
    parser.add_argument("--extract-model", default=os.environ.get("BEDROCK_TEACHER_MODEL", DEFAULT_MODEL))
