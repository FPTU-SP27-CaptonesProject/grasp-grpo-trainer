# GRPO Training Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `grasp-grpo-trainer` able to actually learn, cheaply prove that it does, and stop itself when it does not: Project A reader protocol, batched bf16 rollouts, a DAPO/Dr.GRPO-style objective, periodic greedy eval, early stopping, and a 4-stage gated run protocol.

**Architecture:** Rollout trees run in a thread pool; every proxy generation request goes through one `GenerationBatcher` thread that owns the model, and every retriever call goes through a lock (`BoundRetriever`). The trainer loop collects 16 trees with learning signal per step (refilling zero-variance trees), applies a token-level clipped objective with k3 KL, evaluates greedily on a fixed 300-question set every 50 steps, and consults a pure `EarlyStopper`. Stage scripts wrap the trainer and a pure `stage_gate` module decides pass/fail between stages.

**Tech Stack:** Python 3.11–3.12, PyTorch, Transformers, PEFT, OpenAI client (vLLM reader), FAISS + sentence-transformers, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-grpo-training-protocol-design.md`

## Global Constraints

- `requires-python = ">=3.11,<3.13"`; no new runtime dependencies (`safetensors` already comes with Transformers).
- Proxy actor and reference load in **bf16** by default; `--load-in-4bit` stays available but no script passes it.
- Reader prompt must match `grasp/src/grasp/agents/prompts/reader.txt` byte for byte (sha256 `63d28c7d8045eb3ce2760504d4341893f0b3bc888b3752cfc2b51a29e62e54aa`), context formatted `Title: …\nText: …`, default `--max-steps 5`.
- Pilot defaults (spec §3.4): lr `1e-5`, 16 questions/iteration, `max_leaves` 4, `max_steps` 5, temperature 0.7, top-p 0.95, α 0.3, shaping scale 0.1, KL β 0.002, warmup 20, clip 0.2/0.28, eval every 50 steps on 300 fixed questions.
- Early-stopping defaults (spec §6): improvement ≥ 0.02 F1, patience 3 evals, worse-than-SFT 0.03 for 2 evals, format rate < 0.90, token KL > 0.3, entropy < 30% of first window, zero-signal > 60%, window 20 steps.
- `chat.py` must keep importing `ANSWER_RE`, `THOUGHT_RE`, `reader_prompt` from `grasp_grpo.environment`.
- Work on branch `feat/grpo-training-protocol`; commit locally, **never push** without the user's explicit OK.
- Commands below assume Git Bash on Windows from `D:\DoAn\grasp-grpo-trainer` with `PY=.venv/Scripts/python.exe` (on Linux use `.venv/bin/python`).

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/grasp_grpo/contracts.py` | modify | add `READER` template and `reader_context()` |
| `src/grasp_grpo/reader.py` | rewrite | `ReaderStep`, `GraspReader` (Project A protocol), `LegacyReader`, CLI helpers |
| `src/grasp_grpo/environment.py` | rewrite | tree rollout on the reader protocol; per-role sample counts; shaping scale; centered advantage; `zero_signal`; injectable `generate_fn` |
| `src/grasp_grpo/retrieval.py` | modify | `check_same_thread=False`; `BoundRetriever`; `add_retrieval_arguments`/`retriever_from_args` |
| `src/grasp_grpo/model.py` | rewrite | float32 token log-probs + entropy; `generate_batch` with left padding |
| `src/grasp_grpo/batcher.py` | create | `GenerationBatcher`, `role_generate_fn` |
| `src/grasp_grpo/grpo.py` | rewrite | `merge_actions`, `token_loss`, `grpo_backward` |
| `src/grasp_grpo/stopping.py` | create | `StopConfig`, `EarlyStopper` |
| `src/grasp_grpo/eval_ids.py` | create | fixed eval-set selection + `grasp-make-eval-ids` CLI |
| `src/grasp_grpo/infer.py` | rewrite | `run_example(generate_fn, reader, retriever, example, max_steps)` |
| `src/grasp_grpo/policy_eval.py` | create | `summarize`, `evaluate_policy`, `signal_probe`, `grasp-eval-policy` CLI |
| `src/grasp_grpo/train.py` | rewrite | training loop with refill, warmup, eval, early stopping, `best/` |
| `src/grasp_grpo/stage_gate.py` | create | stage 0–3 gates + `grasp-stage-gate` CLI |
| `src/grasp_grpo/rollout.py` | modify | use reader/retrieval helpers, new defaults |
| `config/eval_ids.json` | create | 300 fixed eval ids |
| `config/vast.env.example`, `scripts/*.sh` | modify/create | stage scripts and new settings |
| `README.md` | modify | protocol section |
| `tests/…` | create/modify | one test file per module above |

---

### Task 0: Branch and local test environment

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Create the branch and commit the spec and plan**

```bash
cd /d/DoAn/grasp-grpo-trainer
git checkout -b feat/grpo-training-protocol
printf '.venv/\n' >> .gitignore
git add .gitignore docs/superpowers/specs/2026-11-04-grpo-training-protocol-design.md docs/superpowers/plans/2026-11-04-grpo-training-protocol.md
git commit -m "docs: GRPO training protocol spec and plan"
```

- [ ] **Step 2: Create a Python 3.12 venv with the dev extra**

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```
Expected: install finishes; `torch`, `transformers`, `peft`, `pytest` present.

- [ ] **Step 3: Run the existing suite as a baseline**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all existing tests pass (`test_build_gold_qa.py`, `test_evaluate_recall.py`, `test_reward_shaping.py`). If any fail before changes, stop and report.

---

### Task 1: Project A reader protocol

**Files:**
- Modify: `src/grasp_grpo/contracts.py` (append after the `KI = '''…'''` block)
- Rewrite: `src/grasp_grpo/reader.py`
- Test: `tests/test_reader.py`

**Interfaces:**
- Produces: `contracts.READER: str`, `contracts.reader_context(items) -> str`; `reader.ReaderStep(piece, thought, answer)`; `reader.extract_answer(text) -> str | None`; `GraspReader(complete, render, *, max_tokens=512, max_tokens_answer=1024)` and `LegacyReader(chat)`, both with `.step(question, context, thoughts, transcript) -> ReaderStep` and `.final(question, context, thoughts, transcript) -> str`; `legacy_reader_prompt(question, context, thoughts, final_only) -> str`; `ANSWER_RE`, `THOUGHT_RE`; `add_reader_arguments(parser)`, `reader_from_args(args)`; `OpenAIReader` unchanged.

- [ ] **Step 1: Write the failing tests** — `tests/test_reader.py`

```python
import hashlib

from grasp_grpo.contracts import READER, reader_context
from grasp_grpo.reader import GraspReader, LegacyReader, ReaderStep, extract_answer

GRASP_READER_SHA256 = "63d28c7d8045eb3ce2760504d4341893f0b3bc888b3752cfc2b51a29e62e54aa"


def test_reader_template_matches_grasp_byte_for_byte():
    assert hashlib.sha256(READER.encode("utf-8")).hexdigest() == GRASP_READER_SHA256


def test_reader_context_uses_title_text_blocks():
    items = [{"title": "A", "text": "x"}, {"title": None, "text": "y"}]
    assert reader_context(items) == "Title: A\nText: x\n\nTitle: \nText: y"
    assert reader_context([]) == ""


class ScriptedCompletion:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def __call__(self, prompt, *, stop, max_tokens):
        self.calls.append({"prompt": prompt, "stop": stop, "max_tokens": max_tokens})
        return self.outputs.pop(0)


def render(user):
    return f"<user>{user}</user><assistant>"


def test_grasp_reader_continues_transcript_and_stops_at_think():
    complete = ScriptedCompletion(["<think> Who directed it?", " <answer> Ann </answer>"])
    reader = GraspReader(complete, render)
    context = [{"title": "Film", "text": "Directed by Ann."}]
    first = reader.step("Q?", context, [], "")
    assert first == ReaderStep("<think> Who directed it? </think>\n", "Who directed it?", None)
    second = reader.step("Q?", context, ["Who directed it?"], first.piece)
    assert second.answer == "Ann" and second.thought is None
    assert complete.calls[0]["stop"] == ["</think>"]
    assert complete.calls[1]["prompt"].endswith("<assistant>" + first.piece)
    assert "Title: Film\nText: Directed by Ann." in complete.calls[0]["prompt"]


def test_grasp_reader_wraps_bare_thought_and_closes_open_answer():
    reader = GraspReader(ScriptedCompletion(["Bare thought  ", "<answer> Bob"]), render)
    step = reader.step("Q?", [], [], "")
    assert step.piece == "<think> Bare thought </think>\n" and step.thought == "Bare thought"
    answer = reader.step("Q?", [], ["Bare thought"], step.piece)
    assert answer.piece == "<answer> Bob </answer>" and answer.answer == "Bob"


def test_grasp_reader_empty_output_and_final_answer():
    complete = ScriptedCompletion(["   ", "<think> a </think>\n<answer> Carol </answer>"])
    reader = GraspReader(complete, render)
    assert reader.step("Q?", [], [], "") == ReaderStep("", None, None)
    assert reader.final("Q?", [], [], "ignored") == "Carol"
    assert complete.calls[1]["stop"] is None and complete.calls[1]["max_tokens"] == 1024
    assert complete.calls[1]["prompt"] == render(READER.format(context="None", question="Q?"))


class FakeChat:
    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def test_legacy_reader_keeps_first_run_behaviour():
    chat = FakeChat(["<think>Need X</think>", "no tags"])
    reader = LegacyReader(chat)
    step = reader.step("Q?", [], [], "")
    assert step.thought == "Need X" and step.answer is None
    assert reader.final("Q?", [], ["Need X"], step.piece) == "no tags"
    assert "Return the final short answer now" in chat.prompts[1]


def test_extract_answer_normalises_whitespace():
    assert extract_answer("x <answer>  a   b </answer>") == "a b"
    assert extract_answer("<answer> open") == "open"
    assert extract_answer("none") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_reader.py -v`
Expected: FAIL — `ImportError: cannot import name 'READER'`.

- [ ] **Step 3: Append to `src/grasp_grpo/contracts.py`** (after the `KI = '''…'''` block, before `def format_triple`)

```python
# Project A (grasp) reader template, byte for byte: grasp/src/grasp/agents/prompts/reader.txt
READER = '''Please answer the following question. Think step by step to solve it.

First write out your reasoning one thought at a time. Put each thought inside <think> and </think> tags. After reasoning, put your final answer inside <answer> and </answer>. The final answer must be a short phrase or keyword only, without explanation.

Your output should look like:
<think> [your first thought] </think>
<think> [your second thought] </think>
...
<think> [your final thought] </think>
<answer> [short final answer] </answer>

---
Context:
{context}

Question: {question}
'''


def reader_context(items: list[dict[str, Any]] | None) -> str:
    """Project A reader context: one `Title:`/`Text:` block per passage."""
    return "\n\n".join(f"Title: {x.get('title') or ''}\nText: {x.get('text', '')}" for x in (items or []))
```

- [ ] **Step 4: Replace `src/grasp_grpo/reader.py` entirely**

```python
"""Frozen reader protocols.

`GraspReader` reproduces the Project A (grasp) reader that generated the SFT data: text completion
over the chat-rendered reader.txt with the current context, continuing the transcript so far and
stopping at </think>. `LegacyReader` keeps the chat prompt used by the first GRPO run.
"""
from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .contracts import READER, reader_context

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
THOUGHT_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.I | re.S)
_OPEN_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)


@dataclass
class ReaderStep:
    piece: str  # text appended to the transcript
    thought: str | None  # set when the reader produced a thought
    answer: str | None  # set when the reader produced <answer>


def extract_answer(text: str) -> str | None:
    match = _OPEN_ANSWER_RE.search(text or "")
    answer = " ".join(match.group(1).split()).strip() if match else ""
    return answer or None


class OpenAIReader:
    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 512, temperature: float = 0.0):
        from openai import OpenAI
        self.client, self.model, self.max_tokens, self.temperature = OpenAI(base_url=base_url, api_key=api_key, max_retries=5), model, max_tokens, temperature

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=self.temperature, max_tokens=self.max_tokens)
        return (response.choices[0].message.content or "").strip()


def openai_completion(base_url: str, api_key: str, model: str, *, temperature: float) -> Callable[..., str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)

    def complete(prompt: str, *, stop: list[str] | None, max_tokens: int) -> str:
        response = client.completions.create(model=model, prompt=prompt, stop=stop, temperature=temperature, max_tokens=max_tokens)
        return response.choices[0].text or ""

    return complete


def chat_renderer(tokenizer_name: str) -> Callable[[str], str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def render(user: str) -> str:
        return tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

    return render


class GraspReader:
    """One thought per call: prompt = chat(reader.txt with the CURRENT context) + transcript so far."""

    def __init__(self, complete: Callable[..., str], render: Callable[[str], str], *, max_tokens: int = 512, max_tokens_answer: int = 1024):
        self.complete, self.render = complete, render
        self.max_tokens, self.max_tokens_answer = max_tokens, max_tokens_answer

    def _prompt(self, question: str, context: list[dict[str, Any]]) -> str:
        return self.render(READER.format(context=reader_context(context) or "None", question=question))

    def step(self, question: str, context: list[dict[str, Any]], thoughts: list[str], transcript: str) -> ReaderStep:
        text = self.complete(self._prompt(question, context) + transcript, stop=["</think>"], max_tokens=self.max_tokens)
        if "<answer>" in text:
            piece = text if "</answer>" in text else text.rstrip() + " </answer>"
            return ReaderStep(piece, None, extract_answer(piece) or "")
        text = text.rstrip()
        if not text:
            return ReaderStep("", None, None)
        if "<think>" not in text:
            text = "<think> " + text.lstrip()
        piece = text + " </think>\n"
        found = THOUGHT_RE.findall(piece)
        thought = " ".join(found[-1].split()) if found else " ".join(text.split())
        return ReaderStep(piece, thought, None)

    def final(self, question: str, context: list[dict[str, Any]], thoughts: list[str], transcript: str) -> str:
        # Out of steps without an answer: regenerate from scratch with the final context (as SPARKLE).
        text = self.complete(self._prompt(question, context), stop=None, max_tokens=self.max_tokens_answer)
        return extract_answer(text) or ""


def legacy_reader_prompt(question: str, context: list[dict[str, Any]], thoughts: list[str], final_only: bool) -> str:
    ctx = "\n\n".join(f"[{x['doc_id']}] {x.get('title') or ''}\n{x['text']}" for x in context) or "(empty)"
    instruction = "Return the final short answer now as <answer>...</answer>." if final_only else "Return one next thought as <think>...</think>, or the final answer as <answer>...</answer>."
    thought_text = "\n".join(thoughts) or "(none)"
    return f"You are the frozen reader in an adaptive RAG system.\nUse the supplied context when external evidence is required. {instruction}\n\nQuestion: {question}\nContext:\n{ctx}\nReasoning so far:\n{thought_text}\n"


class LegacyReader:
    """The chat prompt of the first GRPO run, kept for the stage 0 comparison."""

    def __init__(self, chat):
        self.chat = chat

    def step(self, question: str, context: list[dict[str, Any]], thoughts: list[str], transcript: str) -> ReaderStep:
        response = self.chat.generate(legacy_reader_prompt(question, context, thoughts, final_only=False))
        answer = ANSWER_RE.search(response)
        if answer:
            return ReaderStep(response, None, answer.group(1).strip())
        match = THOUGHT_RE.search(response)
        return ReaderStep(response, match.group(1).strip() if match else response.strip(), None)

    def final(self, question: str, context: list[dict[str, Any]], thoughts: list[str], transcript: str) -> str:
        response = self.chat.generate(legacy_reader_prompt(question, context, thoughts, final_only=True))
        match = ANSWER_RE.search(response)
        return match.group(1).strip() if match else response.strip()


def add_reader_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reader-url", default="http://localhost:8000/v1")
    parser.add_argument("--reader-api-key", default="EMPTY")
    parser.add_argument("--reader-model", required=True)
    parser.add_argument("--reader-protocol", choices=("grasp", "legacy"), default="grasp", help="grasp = Project A reader that generated the SFT data")
    parser.add_argument("--reader-tokenizer", help="Chat template for the grasp protocol (default: --reader-model)")
    parser.add_argument("--reader-temperature", type=float, default=0.0)


def reader_from_args(args) -> GraspReader | LegacyReader:
    if args.reader_protocol == "grasp":
        complete = openai_completion(args.reader_url, args.reader_api_key, args.reader_model, temperature=args.reader_temperature)
        return GraspReader(complete, chat_renderer(args.reader_tokenizer or args.reader_model))
    if args.reader_protocol == "legacy":
        return LegacyReader(OpenAIReader(args.reader_url, args.reader_api_key, args.reader_model, temperature=args.reader_temperature))
    raise ValueError(f"unknown reader protocol {args.reader_protocol!r}")
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_reader.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/grasp_grpo/contracts.py src/grasp_grpo/reader.py tests/test_reader.py
git commit -m "feat(reader): Project A reader protocol with byte-exact template"
```

---

### Task 2: Environment on the reader protocol, scaled shaping, centered advantage

**Files:**
- Rewrite: `src/grasp_grpo/environment.py`
- Modify: `tests/test_reward_shaping.py`
- Modify: `src/grasp_grpo/rollout.py` (Task 3 finishes the retrieval part; here only the reader and env args)

**Interfaces:**
- Consumes: `reader.ReaderStep`, reader objects with `.step/.final` (Task 1).
- Produces: `environment.GenerateFn = Callable[[str, str, int], list[Action]]` (role, prompt, samples); `BinaryTreeEnvironment(..., shaping_scale=1.0, advantage_std_norm=False, generate_fn=None)`; `run(example)` returns the old dict plus `"zero_signal": bool`; each leaf has `"advantage"` = `R − mean` (0 when the tree std < 1e-6).

- [ ] **Step 1: Update and extend `tests/test_reward_shaping.py`** (replace the whole file)

```python
import pytest

from grasp_grpo.environment import BinaryTreeEnvironment
from grasp_grpo.model import Action
from grasp_grpo.reader import ReaderStep

EXAMPLE = {"id": "q", "question": "Question", "gold_answers": ["answer"], "gold_doc_ids": ["gold"]}
QF_TEXT = "Reasoning Chain: None\nKnowledge Gap: need evidence\nQuery: gold query"


class FakeActor:
    def eval(self):
        return self


class FakeReader:
    def __init__(self):
        self.calls = 0

    def step(self, question, context, thoughts, transcript):
        self.calls += 1
        if self.calls == 1:
            return ReaderStep("<think> Need evidence. </think>\n", "Need evidence.", None)
        return ReaderStep("<answer>answer</answer>", None, "answer")

    def final(self, question, context, thoughts, transcript):
        return "answer"


class FakeRetriever:
    def search(self, query):
        if query == "gold query":
            return [{"doc_id": "gold", "title": "Gold", "text": "evidence", "triples": [{"doc_id": "gold", "head": "Gold", "relation": "answers", "tail": "answer"}]}]
        return [{"doc_id": "other", "title": "Other", "text": "noise", "triples": []}]

    def candidate_triples(self, _query, passages):
        return [triple for passage in passages for triple in passage["triples"]]


def make_environment(monkeypatch, rd_texts=("Yes (retrieval is required)",) * 2, **overrides):
    def fake_generate(_actor, _tokenizer, prompt, **_kwargs):
        if "retrieval decision assistant" in prompt:
            return [Action(text, [1], [-0.1]) for text in rd_texts]
        text = QF_TEXT if "query formulation assistant" in prompt else "<Gold; answers; answer>"
        return [Action(text, [1], [-0.1]), Action(text, [1], [-0.1])]

    monkeypatch.setattr("grasp_grpo.environment.generate", fake_generate)
    settings = {"branch_factor": 2, "max_steps": 1, "max_leaves": 2, "temperature": 0.7, "top_p": 0.95, "alpha": 0.7, "retrieval_cost": 0.0, "behavior_checkpoint": "test", **overrides}
    return BinaryTreeEnvironment(FakeActor(), object(), FakeReader(), FakeRetriever(), **settings)


def test_gold_evidence_adds_auditable_rd_qf_and_ki_rewards(monkeypatch):
    tree = make_environment(monkeypatch).run(EXAMPLE)
    leaf = tree["leaves"][0]
    assert leaf["reward"]["answer_f1"] == 1.0
    assert leaf["reward"]["context_recall"] == 1.0
    assert leaf["reward"]["action_shaping"] == pytest.approx(1.3)
    assert leaf["reward"]["total"] == pytest.approx(2.3)
    shaped = [action["reward_shaping"]["total"] for action in leaf["actions"]]
    assert shaped == [0.3, 0.7, 0.3]


def test_shaping_scale_and_identical_branches_give_no_signal(monkeypatch):
    tree = make_environment(monkeypatch, shaping_scale=0.1).run(EXAMPLE)
    assert tree["leaves"][0]["reward"]["total"] == pytest.approx(1.0 + 0.13)
    assert tree["zero_signal"] is True
    assert all(leaf["advantage"] == 0.0 for leaf in tree["leaves"])


def test_advantage_is_reward_minus_tree_mean(monkeypatch):
    rd = ("Yes (retrieval is required)", "No (retrieval is not required)")
    tree = make_environment(monkeypatch, rd_texts=rd).run(EXAMPLE)
    by_choice = {leaf["actions"][0]["response"]: leaf["advantage"] for leaf in tree["leaves"]}
    assert by_choice == {rd[0]: pytest.approx(1.0), rd[1]: pytest.approx(-1.0)}
    assert tree["zero_signal"] is False


def test_std_normalised_advantage_is_optional(monkeypatch):
    rd = ("Yes (retrieval is required)", "No (retrieval is not required)")
    tree = make_environment(monkeypatch, rd_texts=rd, advantage_std_norm=True).run(EXAMPLE)
    assert sorted(round(leaf["advantage"], 4) for leaf in tree["leaves"]) == [-1.0, 1.0]


def test_qf_and_ki_draw_one_sample_through_generate_fn(monkeypatch):
    calls = []

    def generate_fn(role, prompt, samples):
        calls.append((role, samples))
        text = {"rd": "Yes (retrieval is required)", "qf": QF_TEXT, "ki": "<Gold; answers; answer>"}[role]
        return [Action(text, [1], [-0.1]) for _ in range(samples)]

    make_environment(monkeypatch, generate_fn=generate_fn).run(EXAMPLE)
    assert calls == [("rd", 2), ("qf", 1), ("ki", 1), ("qf", 1), ("ki", 1)]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_reward_shaping.py -v`
Expected: FAIL — `AttributeError: 'FakeReader' object has no attribute 'generate'` and unexpected keyword `shaping_scale`.

- [ ] **Step 3: Replace `src/grasp_grpo/environment.py` entirely**

```python
"""Project A-compatible RAG state transitions with bounded binary-tree expansion."""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .contracts import ROLE_MAX_TOKENS, candidate_index, parse, prompt
from .model import Action, generate
from .reader import ANSWER_RE, THOUGHT_RE  # noqa: F401  (re-exported for chat.py)
from .reader import legacy_reader_prompt as reader_prompt  # noqa: F401  (re-exported for chat.py)

# (role, prompt, samples) -> sampled proxy actions
GenerateFn = Callable[[str, str, int], list[Action]]


def _f1(prediction: str, gold: list[str]) -> float:
    import string
    from collections import Counter

    def tokens(text: str) -> list[str]:
        # Mirrors grasp-proxy-rollout.metrics.normalize_answer exactly.
        text = text.lower()
        text = "".join(char for char in text if char not in string.punctuation)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return text.split()
    p = tokens(prediction)
    scores = []
    for answer in gold:
        g = tokens(answer)
        common = sum((Counter(p) & Counter(g)).values())
        scores.append(0.0 if not common else 2 * common / (len(p) + len(g)))
    return max(scores, default=0.0)


def reward(answer: str, example: dict[str, Any], context: list[dict[str, Any]], retrieval_count: int, alpha: float, retrieval_cost: float, action_shaping: float = 0.0) -> dict[str, float | int | None]:
    f1 = _f1(answer, example["gold_answers"])
    gold_ids = set(example.get("gold_doc_ids", []))
    recall = len(gold_ids & {x["doc_id"] for x in context}) / len(gold_ids) if gold_ids else None
    total = f1 if recall is None else (1 - alpha) * f1 + alpha * recall
    sequence_reward = total - retrieval_cost * retrieval_count
    return {"answer_f1": f1, "context_recall": recall, "retrieval_count": retrieval_count, "retrieval_cost": retrieval_cost * retrieval_count, "sequence_reward": sequence_reward, "action_shaping": action_shaping, "total": sequence_reward + action_shaping}


@dataclass
class TreeNode:
    node_id: str
    parent_id: str | None
    depth: int
    question: str
    context: list[dict[str, Any]]
    thoughts: list[str]
    actions: list[dict[str, Any]] = field(default_factory=list)
    reader_response: str | None = None
    answer: str | None = None
    terminal: str | None = None
    children: list[str] = field(default_factory=list)
    transcript: str = ""


class BinaryTreeEnvironment:
    def __init__(self, actor, tokenizer, reader, retriever, *, branch_factor: int, max_steps: int, max_leaves: int, temperature: float, top_p: float, alpha: float, retrieval_cost: float, behavior_checkpoint: str, initial_retrieval: bool = True, rd_gain_weight: float = 0.3, qf_gain_weight: float = 0.7, ki_hit_weight: float = 0.3, ki_miss_penalty: float = 0.1, shaping_scale: float = 1.0, advantage_std_norm: bool = False, generate_fn: GenerateFn | None = None):
        if branch_factor != 2:
            raise ValueError("This implementation intentionally fixes binary-tree branch_factor to 2")
        self.actor, self.tokenizer, self.reader, self.retriever = actor, tokenizer, reader, retriever
        self.branch_factor, self.max_steps, self.max_leaves = branch_factor, max_steps, max_leaves
        self.temperature, self.top_p, self.alpha, self.retrieval_cost = temperature, top_p, alpha, retrieval_cost
        self.behavior_checkpoint, self.initial_retrieval = behavior_checkpoint, initial_retrieval
        self.rd_gain_weight, self.qf_gain_weight = rd_gain_weight, qf_gain_weight
        self.ki_hit_weight, self.ki_miss_penalty = ki_hit_weight, ki_miss_penalty
        self.shaping_scale, self.advantage_std_norm, self.generate_fn = shaping_scale, advantage_std_norm, generate_fn

    @staticmethod
    def _recall(gold_ids: set[str], passages: list[dict[str, Any]]) -> float | None:
        if not gold_ids:
            return None
        return len(gold_ids & {str(item["doc_id"]) for item in passages}) / len(gold_ids)

    @staticmethod
    def _shape(action: dict[str, Any], **values: float) -> None:
        action["reward_shaping"] = {key: float(value) for key, value in values.items()}
        action["reward_shaping"]["total"] = float(sum(values.values()))

    def _sample(self, role: str, state: dict[str, Any], samples: int) -> list[dict[str, Any]]:
        text_prompt = prompt(role, state)
        if self.generate_fn is not None:
            actions: list[Action] = self.generate_fn(role, text_prompt, samples)
        else:
            actions = generate(self.actor, self.tokenizer, text_prompt, samples=samples, max_new_tokens=ROLE_MAX_TOKENS[role], temperature=self.temperature, top_p=self.top_p)
        result = []
        for action in actions:
            parsed, valid, error = parse(role, action.text)
            result.append({"role": role, "prompt": text_prompt, "response": action.text, "parsed_action": parsed, "valid_format": valid, "validation_error": error, "response_token_ids": action.token_ids, "old_token_logprobs": action.token_logprobs, "old_logprob_sum": float(sum(action.token_logprobs)), "behavior_policy_checkpoint": self.behavior_checkpoint})
        return result

    def _leaf(self, node: TreeNode, reason: str) -> TreeNode:
        node.answer = self.reader.final(node.question, node.context, node.thoughts, node.transcript)
        node.terminal = reason
        return node

    def _next_nodes(self, node: TreeNode, node_number: list[int], gold_ids: set[str], question_baseline: list[dict[str, Any]]) -> list[TreeNode]:
        """One reader thought plus RD -> optional QF -> KI, branching only at RD.

        Keeping QF/KI to one sample per selected RD branch caps the binary tree while all
        proxy actions remain on-policy and receive the descendant leaf advantage.
        """
        step = self.reader.step(node.question, node.context, node.thoughts, node.transcript)
        if step.answer is not None:
            node.reader_response, node.answer, node.terminal = step.piece, step.answer, "reader_answer"
            node.transcript += step.piece
            return []
        if step.thought is None:
            self._leaf(node, "reader_empty")
            return []
        thought = step.thought
        state = {"question": node.question, "context": node.context, "thoughts": node.thoughts + [thought], "current_thought": thought}
        children = []
        for rd in self._sample("rd", state, self.branch_factor):
            child = copy.deepcopy(node)
            child.node_id, child.parent_id, child.depth = f"n{node_number[0]}", node.node_id, node.depth + 1
            node_number[0] += 1
            child.thoughts.append(thought)
            child.transcript = node.transcript + step.piece
            child.reader_response = step.piece
            child.actions = [rd]
            retrieve = rd["parsed_action"] if rd["valid_format"] else True
            if not rd["valid_format"]:
                rd["environment_fallback"] = "invalid RD defaults to retrieval"
            if retrieve:
                qf = self._sample("qf", state, 1)[0]
                child.actions.append(qf)
                parsed_qf = qf["parsed_action"] if qf["valid_format"] else None
                query = str(parsed_qf.get("query") or node.question) if parsed_qf else node.question
                passages = self.retriever.search(query)
                remaining_gold = gold_ids - {str(item["doc_id"]) for item in node.context}
                if remaining_gold:
                    baseline_recall = self._recall(remaining_gold, question_baseline) or 0.0
                    query_recall = self._recall(remaining_gold, passages) or 0.0
                    qf_gain = self.qf_gain_weight * max(0.0, query_recall - baseline_recall) ** 0.5
                else:
                    qf_gain = 0.0
                self._shape(qf, query_recall_gain=qf_gain)
                candidates = self.retriever.candidate_triples(query, passages)
                if candidates:
                    ki_state = {"question": node.question, "chain": parsed_qf.get("chain", []) if parsed_qf else [], "query": query, "candidates": candidates}
                    ki = self._sample("ki", ki_state, 1)[0]
                    child.actions.append(ki)
                    chosen = candidate_index(ki["parsed_action"] if ki["valid_format"] else None, candidates)
                    if chosen is None:
                        chosen, ki["environment_fallback"] = 0, "invalid KI selected candidate 0"
                    selected = candidates[chosen]
                    evidence = next((x for x in passages if x["doc_id"] == selected["doc_id"]), passages[0])
                    if ki["valid_format"] and str(evidence["doc_id"]) in gold_ids:
                        rank = next((index + 1 for index, item in enumerate(passages) if item["doc_id"] == evidence["doc_id"]), 1)
                        ki_reward = self.ki_hit_weight / math.log2(rank + 1)
                    elif gold_ids:
                        ki_reward = -self.ki_miss_penalty
                    else:
                        ki_reward = 0.0
                    self._shape(ki, evidence_selection=ki_reward)
                    if all(x["doc_id"] != evidence["doc_id"] for x in child.context):
                        child.context.append(evidence)
                else:
                    self._shape(qf, query_recall_gain=qf_gain)
            before = self._recall(gold_ids, node.context)
            after = self._recall(gold_ids, child.context)
            rd_gain = self.rd_gain_weight * ((after or 0.0) - (before or 0.0)) if before is not None else 0.0
            self._shape(rd, coverage_gain=rd_gain)
            node.children.append(child.node_id)
            children.append(child)
        return children

    def run(self, example: dict[str, Any]) -> dict[str, Any]:
        self.actor.eval()
        select = getattr(self.retriever, "select", None)
        if select is not None:
            select(example)
        initial = self.retriever.search(example["question"]) if self.initial_retrieval else []
        question_baseline = list(initial)
        gold_ids = {str(value) for value in example.get("gold_doc_ids", [])}
        root = TreeNode("n0", None, 0, example["question"], initial, [])
        nodes: dict[str, TreeNode] = {root.node_id: root}
        frontier, leaves, counter = [root], [], [1]
        while frontier and len(leaves) < self.max_leaves:
            node = frontier.pop(0)
            if node.depth >= self.max_steps:
                leaves.append(self._leaf(node, "max_steps")); continue
            children = self._next_nodes(node, counter, gold_ids, question_baseline)
            if node.terminal:
                leaves.append(node); continue
            for child in children:
                nodes[child.node_id] = child
                if len(frontier) + len(leaves) < self.max_leaves:
                    frontier.append(child)
        for node in frontier:
            leaves.append(self._leaf(node, "leaf_cap"))
        records = []
        for leaf in leaves[: self.max_leaves]:
            chain, current = [], leaf
            while current:
                chain.append(current.node_id)
                current = nodes.get(current.parent_id) if current.parent_id else None
            chain.reverse()
            retrievals = sum(action["role"] == "rd" and bool(action.get("parsed_action")) for nid in chain for action in nodes[nid].actions)
            actions = [action for nid in chain for action in nodes[nid].actions]
            shaping = self.shaping_scale * sum(float(action.get("reward_shaping", {}).get("total", 0.0)) for action in actions)
            records.append({"leaf_id": leaf.node_id, "path": chain, "question_id": example["id"], "question": example["question"], "gold_answers": example["gold_answers"], "final_answer": leaf.answer, "terminal": leaf.terminal, "reward": reward(leaf.answer or "", example, leaf.context, retrievals, self.alpha, self.retrieval_cost, shaping), "actions": actions})
        values = [float(x["reward"]["total"]) for x in records]
        mean = sum(values) / len(values) if values else 0.0
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5 if values else 0.0
        for record in records:
            centered = float(record["reward"]["total"]) - mean
            if std < 1e-6:
                record["advantage"] = 0.0
            elif self.advantage_std_norm:
                record["advantage"] = centered / (std + 1e-6)
            else:
                record["advantage"] = centered  # Dr. GRPO: no std division
        return {"question_id": example["id"], "question": example["question"], "tree": {node_id: {"parent_id": node.parent_id, "depth": node.depth, "children": node.children, "terminal": node.terminal} for node_id, node in nodes.items()}, "leaves": records, "reward_mean": mean, "reward_std": std, "zero_signal": std < 1e-6, "behavior_policy_checkpoint": self.behavior_checkpoint}
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_reward_shaping.py -v`
Expected: 5 passed.

- [ ] **Step 5: Point `src/grasp_grpo/rollout.py` at the reader helpers and new defaults**

In `rollout.py` replace the import line `from .reader import OpenAIReader` with:

```python
from .reader import add_reader_arguments, reader_from_args
```

In `arguments()` delete the three lines that add `--reader-url`, `--reader-api-key`, `--reader-model`, and add right after `parser.add_argument("--triple-cache")`:

```python
    add_reader_arguments(parser)
```

Change the defaults `--max-steps` to `5` and `--alpha` to `0.3`, and add after the `--ki-miss-penalty` line:

```python
    parser.add_argument("--shaping-scale", type=float, default=0.1)
```

In `collect()` replace `reader = OpenAIReader(args.reader_url, args.reader_api_key, args.reader_model)` with `reader = reader_from_args(args)` and append `, shaping_scale=args.shaping_scale` to the `BinaryTreeEnvironment(...)` call arguments.

- [ ] **Step 6: Run the whole suite and commit**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all pass.

```bash
git add src/grasp_grpo/environment.py src/grasp_grpo/rollout.py tests/test_reward_shaping.py
git commit -m "feat(env): reader protocol, shaping scale, centered advantage, zero-signal flag"
```

---

### Task 3: Thread-safe retrieval binding and shared CLI helpers

**Files:**
- Modify: `src/grasp_grpo/retrieval.py`
- Modify: `src/grasp_grpo/rollout.py`
- Test: `tests/test_retrieval_bound.py`

**Interfaces:**
- Produces: `BoundRetriever(retriever, example, lock)` with `.select(example)` (no-op), `.search(query)`, `.candidate_triples(query, passages)`; `add_retrieval_arguments(parser)` adding `--index-dir --corpus-config --triple-cache --encoder --top-k --max-candidate-triples`; `retriever_from_args(args)`.

- [ ] **Step 1: Write the failing test** — `tests/test_retrieval_bound.py`

```python
import argparse
import threading

import pytest

from grasp_grpo.retrieval import BoundRetriever, add_retrieval_arguments, retriever_from_args


class FakeRouter:
    def __init__(self):
        self.current = None

    def select(self, example):
        self.current = example["metadata"]["dataset"]

    def search(self, query):
        return [{"doc_id": f"{self.current}:{query}"}]

    def candidate_triples(self, query, passages):
        return [{"corpus": self.current}]


def test_each_bound_view_searches_its_own_corpus():
    router, lock = FakeRouter(), threading.Lock()
    a = BoundRetriever(router, {"metadata": {"dataset": "wiki18"}}, lock)
    b = BoundRetriever(router, {"metadata": {"dataset": "2wiki"}}, lock)
    assert a.search("q") == [{"doc_id": "wiki18:q"}]
    assert b.search("q") == [{"doc_id": "2wiki:q"}]
    assert a.candidate_triples("q", []) == [{"corpus": "wiki18"}]
    a.select({"metadata": {"dataset": "2wiki"}})  # binding is fixed per tree
    assert a.search("q") == [{"doc_id": "wiki18:q"}]


def test_retriever_from_args_needs_exactly_one_source():
    parser = argparse.ArgumentParser()
    add_retrieval_arguments(parser)
    with pytest.raises(ValueError, match="exactly one"):
        retriever_from_args(parser.parse_args([]))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval_bound.py -v`
Expected: FAIL — `ImportError: cannot import name 'BoundRetriever'`.

- [ ] **Step 3: Edit `src/grasp_grpo/retrieval.py`**

Add `import threading` to the imports. In `DenseRetriever.__init__` make both SQLite connections usable from rollout threads (calls are serialised by `BoundRetriever`'s lock):

```python
        self.db = sqlite3.connect(f"file:{self.passage_path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self.triple_db = sqlite3.connect(f"file:{Path(triple_cache).resolve().as_posix()}?mode=ro", uri=True, check_same_thread=False) if triple_cache else None
```

Append at the end of the file:

```python
class BoundRetriever:
    """Pins a shared retriever to one example's corpus and serialises calls across rollout threads."""

    def __init__(self, retriever, example: dict[str, Any], lock: threading.Lock):
        self.retriever, self.example, self.lock = retriever, example, lock

    def select(self, example: dict[str, Any]) -> None:
        return None  # bound at construction

    def _bind(self) -> None:
        select = getattr(self.retriever, "select", None)
        if select is not None:
            select(self.example)

    def search(self, query: str) -> list[dict[str, Any]]:
        with self.lock:
            self._bind()
            return self.retriever.search(query)

    def candidate_triples(self, query: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self.lock:
            self._bind()
            return self.retriever.candidate_triples(query, passages)


def add_retrieval_arguments(parser) -> None:
    parser.add_argument("--index-dir")
    parser.add_argument("--corpus-config", help="JSON mapping metadata.dataset to index/triple artifacts")
    parser.add_argument("--triple-cache")
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-candidate-triples", type=int, default=20)


def retriever_from_args(args) -> DenseRetriever | CorpusRouter:
    if bool(args.index_dir) == bool(args.corpus_config):
        raise ValueError("provide exactly one of --index-dir or --corpus-config")
    if args.corpus_config:
        return CorpusRouter(args.corpus_config, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples)
    return DenseRetriever(args.index_dir, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples, triple_cache=args.triple_cache)
```

- [ ] **Step 4: Use the helpers in `src/grasp_grpo/rollout.py`**

Replace `from .retrieval import CorpusRouter, DenseRetriever` with `from .retrieval import add_retrieval_arguments, retriever_from_args`. In `arguments()` delete the lines adding `--index-dir`, `--corpus-config`, `--triple-cache`, `--encoder`, `--top-k`, `--max-candidate-triples`, and put `add_retrieval_arguments(parser)` right after `parser.add_argument("--qa", required=True)`. In `collect()` delete the `if bool(args.index_dir) == bool(args.corpus_config): raise …` lines and replace the `retriever = CorpusRouter(...) if args.corpus_config else DenseRetriever(...)` line with `retriever = retriever_from_args(args)`.

- [ ] **Step 5: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all pass.

```bash
git add src/grasp_grpo/retrieval.py src/grasp_grpo/rollout.py tests/test_retrieval_bound.py
git commit -m "feat(retrieval): thread-safe per-tree binding and shared CLI helpers"
```

---

### Task 4: Batched bf16 generation and the generation batcher

**Files:**
- Rewrite: `src/grasp_grpo/model.py`
- Create: `src/grasp_grpo/batcher.py`
- Test: `tests/test_batcher.py`, `tests/test_generate_batch_gpu.py`

**Interfaces:**
- Produces (`model.py`): `Action(text, token_ids, token_logprobs)` unchanged; `generate_batch(model, tokenizer, prompts, *, samples, max_new_tokens, temperature, top_p) -> list[list[Action]]`; `generate(...)` = single-prompt wrapper (same signature as before); `response_logprob(model, tokenizer, prompt, response_ids) -> (sum, per_token)`; `response_logprob_entropy(model, tokenizer, prompt, response_ids) -> (per_token_logprobs, per_token_entropy)` — both float32; `load_policy` unchanged.
- Produces (`batcher.py`): `GenerationBatcher(generate_batch, *, max_batch=32, window_seconds=0.02)` with `.generate(prompt, *, samples, max_new_tokens, temperature, top_p) -> list[Action]` and `.close()`; `role_generate_fn(generate, *, temperature, top_p) -> GenerateFn`.

- [ ] **Step 1: Write the failing tests** — `tests/test_batcher.py`

```python
import threading

import pytest

from grasp_grpo.batcher import GenerationBatcher, role_generate_fn
from grasp_grpo.model import Action


def echo_batch(calls):
    def run(prompts, *, samples, max_new_tokens, temperature, top_p):
        calls.append({"prompts": list(prompts), "samples": samples, "max_new_tokens": max_new_tokens})
        return [[Action(f"{prompt}#{i}", [i], [0.0]) for i in range(samples)] for prompt in prompts]
    return run


def run_threads(targets):
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_concurrent_requests_are_batched_and_routed_back():
    calls, results = [], {}
    batcher = GenerationBatcher(echo_batch(calls), max_batch=8, window_seconds=0.2)

    def worker(n):
        return lambda: results.__setitem__(n, batcher.generate(f"p{n}", samples=2, max_new_tokens=16, temperature=0.7, top_p=0.95))

    run_threads([worker(n) for n in range(6)])
    batcher.close()
    assert {n: [a.text for a in r] for n, r in results.items()} == {n: [f"p{n}#0", f"p{n}#1"] for n in range(6)}
    assert len(calls) < 6 and sum(len(c["prompts"]) for c in calls) == 6


def test_requests_with_different_settings_are_not_mixed():
    calls, out = [], {}
    batcher = GenerationBatcher(echo_batch(calls), max_batch=8, window_seconds=0.2)
    run_threads([
        lambda: out.__setitem__("a", batcher.generate("x", samples=2, max_new_tokens=16, temperature=0.7, top_p=0.95)),
        lambda: out.__setitem__("b", batcher.generate("y", samples=1, max_new_tokens=256, temperature=0.7, top_p=0.95)),
    ])
    batcher.close()
    assert len(out["a"]) == 2 and len(out["b"]) == 1
    assert sorted((c["samples"], c["max_new_tokens"]) for c in calls) == [(1, 256), (2, 16)]


def test_greedy_batched_matches_sequential():
    fn = echo_batch([])
    batcher = GenerationBatcher(fn, max_batch=4, window_seconds=0.05)
    sequential = [fn([f"q{n}"], samples=1, max_new_tokens=8, temperature=0.0, top_p=1.0)[0] for n in range(5)]
    batched = [batcher.generate(f"q{n}", samples=1, max_new_tokens=8, temperature=0.0, top_p=1.0) for n in range(5)]
    batcher.close()
    assert batched == sequential


def test_errors_reach_the_caller_and_batcher_keeps_running():
    attempts = []

    def flaky(prompts, **_):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("cuda oom")
        return [[Action("ok", [1], [0.0])] for _ in prompts]

    batcher = GenerationBatcher(flaky, max_batch=2, window_seconds=0.01)
    with pytest.raises(RuntimeError, match="cuda oom"):
        batcher.generate("x", samples=1, max_new_tokens=4, temperature=0.0, top_p=1.0)
    assert batcher.generate("y", samples=1, max_new_tokens=4, temperature=0.0, top_p=1.0)[0].text == "ok"
    batcher.close()


def test_role_generate_fn_uses_role_token_budget():
    seen = {}

    def gen(prompt, **kwargs):
        seen.update(kwargs)
        return [Action(prompt, [1], [0.0])]

    role_generate_fn(gen, temperature=0.7, top_p=0.9)("qf", "prompt", 1)
    assert seen == {"samples": 1, "max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_batcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grasp_grpo.batcher'`.

- [ ] **Step 3: Replace `src/grasp_grpo/model.py` entirely**

```python
"""Trainable PEFT actor, frozen reference, and token-level policy utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Action:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]


def _token_logprobs_entropy(model, prefix: list[int], response_ids: list[int]):
    """Float32 log P(token) and entropy of each response position, under raw logits (no sampling warpers)."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    ids = torch.tensor([prefix + response_ids], device=device)
    logits = model(input_ids=ids).logits[0, :-1]
    start = len(prefix) - 1
    window = F.log_softmax(logits[start : start + len(response_ids)].float(), dim=-1)
    targets = ids[0, start + 1 : start + 1 + len(response_ids)]
    selected = window.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(window.exp() * window).sum(-1)
    return selected, entropy


def load_policy(checkpoint: str, *, trainable: bool, load_in_4bit: bool = False):
    import torch
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir() or not (root / "adapter_config.json").is_file():
        raise ValueError("checkpoint must be a local PEFT adapter directory from Project A")
    config = PeftConfig.from_pretrained(root)
    kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit loading requires CUDA")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, **kwargs)
    if trainable and load_in_4bit:
        base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, root, is_trainable=trainable)
    tokenizer = AutoTokenizer.from_pretrained(root if (root / "tokenizer_config.json").exists() else config.base_model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available() and not load_in_4bit:
        model.to("cuda")
    model.train(trainable)
    if not trainable:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, tokenizer, str(config.base_model_name_or_path)


def _prompt_ids(tokenizer, prompt: str, max_input_tokens: int = 3072, prompt_head_tokens: int = 768) -> list[int]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(ids) <= max_input_tokens:
        return ids
    head = min(prompt_head_tokens, max_input_tokens // 2)
    return ids[:head] + ids[-(max_input_tokens - head):]


def generate_batch(model, tokenizer, prompts: list[str], *, samples: int, max_new_tokens: int, temperature: float, top_p: float) -> list[list[Action]]:
    """Sample `samples` actions for every prompt in one left-padded `generate` call."""
    import torch

    if samples > 1 and temperature <= 0:
        raise ValueError("multiple tree branches require temperature > 0")
    prefixes = [_prompt_ids(tokenizer, prompt) for prompt in prompts]
    width = max(len(prefix) for prefix in prefixes)
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([[pad] * (width - len(p)) + p for p in prefixes], device=device)
    attention = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prefixes], device=device, dtype=torch.long)
    generation = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens, "num_return_sequences": samples, "pad_token_id": pad, "eos_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        generation.update(temperature=temperature, top_p=top_p)
    with torch.no_grad():
        sequences = model.generate(input_ids=input_ids, attention_mask=attention, **generation)[:, width:]
    results: list[list[Action]] = [[] for _ in prompts]
    eos = tokenizer.eos_token_id
    for row, ids in enumerate(sequences):
        index = row // samples  # generate() repeats each prompt `samples` times in a row
        token_ids = ids.detach().cpu().tolist()
        if eos in token_ids:
            token_ids = token_ids[: token_ids.index(eos) + 1]
        # Raw behavior-policy probabilities, recomputed without padding or sampling warpers,
        # with the same float32 path the loss uses so the importance ratio starts at 1.
        with torch.no_grad():
            raw, _ = _token_logprobs_entropy(model, prefixes[index], token_ids)
        results[index].append(Action(tokenizer.decode(token_ids, skip_special_tokens=True).strip(), token_ids, raw.cpu().tolist()))
    return results


def generate(model, tokenizer, prompt: str, *, samples: int, max_new_tokens: int, temperature: float, top_p: float) -> list[Action]:
    return generate_batch(model, tokenizer, [prompt], samples=samples, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)[0]


def response_logprob_entropy(model, tokenizer, prompt: str, response_ids: list[int]):
    """Differentiable per-token log P(response) and entropy under the exact Project A chat prompt."""
    return _token_logprobs_entropy(model, _prompt_ids(tokenizer, prompt), response_ids)


def response_logprob(model, tokenizer, prompt: str, response_ids: list[int]):
    """Differentiable sum log P(response tokens | exact Project A chat prompt)."""
    selected, _ = response_logprob_entropy(model, tokenizer, prompt, response_ids)
    return selected.sum(), selected
```

- [ ] **Step 4: Create `src/grasp_grpo/batcher.py`**

```python
"""Micro-batching of proxy generation requests coming from concurrent rollout threads."""
from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future

from .contracts import ROLE_MAX_TOKENS
from .model import Action

# (prompts, *, samples, max_new_tokens, temperature, top_p) -> one action list per prompt
BatchGenerate = Callable[..., list[list[Action]]]


class GenerationBatcher:
    """Collects `generate` calls from many threads and runs them as one padded batch per setting.

    Only the batcher thread touches the model, so rollout threads never run CUDA work concurrently.
    """

    def __init__(self, generate_batch: BatchGenerate, *, max_batch: int = 32, window_seconds: float = 0.02):
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self._generate_batch, self._max_batch, self._window = generate_batch, max_batch, window_seconds
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="generation-batcher", daemon=True)
        self._thread.start()

    def generate(self, prompt: str, *, samples: int, max_new_tokens: int, temperature: float, top_p: float) -> list[Action]:
        future: Future = Future()
        self._queue.put(((samples, max_new_tokens, temperature, top_p), prompt, future))
        return future.result()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _collect(self, first) -> list:
        pending = [first]
        deadline = time.monotonic() + self._window
        while len(pending) < self._max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                self._queue.put(None)  # shut down after this batch
                break
            pending.append(item)
        return pending

    def _loop(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            groups: dict[tuple, list] = defaultdict(list)
            for key, prompt, future in self._collect(first):
                groups[key].append((prompt, future))
            for (samples, max_new_tokens, temperature, top_p), items in groups.items():
                try:
                    outputs = self._generate_batch([prompt for prompt, _ in items], samples=samples, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
                    if len(outputs) != len(items):
                        raise RuntimeError(f"generate_batch returned {len(outputs)} results for {len(items)} prompts")
                except BaseException as error:  # surface to every waiting caller, keep serving
                    for _, future in items:
                        future.set_exception(error)
                    continue
                for (_, future), output in zip(items, outputs, strict=True):
                    future.set_result(output)


def role_generate_fn(generate: Callable[..., list[Action]], *, temperature: float, top_p: float):
    """Adapts a `generate(prompt, *, samples, max_new_tokens, temperature, top_p)` callable to `(role, prompt, samples)`."""

    def run(role: str, prompt: str, samples: int) -> list[Action]:
        return generate(prompt, samples=samples, max_new_tokens=ROLE_MAX_TOKENS[role], temperature=temperature, top_p=top_p)

    return run
```

- [ ] **Step 5: Create the opt-in GPU check** — `tests/test_generate_batch_gpu.py`

```python
"""Real-model check that left padding does not change greedy output. Opt-in: needs CUDA + an adapter."""
import os

import pytest

CHECKPOINT = os.environ.get("GRASP_TEST_CHECKPOINT")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="set GRASP_TEST_CHECKPOINT to a local SFT adapter (needs CUDA)")


def test_batched_greedy_generation_matches_single_prompt():
    from grasp_grpo.model import generate, generate_batch, load_policy

    model, tokenizer, _ = load_policy(CHECKPOINT, trainable=False)
    prompts = ["Say yes.", "Name a colour.", "Count to three, separated by commas."]
    single = [generate(model, tokenizer, p, samples=1, max_new_tokens=16, temperature=0.0, top_p=1.0)[0].text for p in prompts]
    batched = [r[0].text for r in generate_batch(model, tokenizer, prompts, samples=1, max_new_tokens=16, temperature=0.0, top_p=1.0)]
    assert batched == single
```

- [ ] **Step 6: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all pass; `test_generate_batch_gpu.py` reports 1 skipped.

- [ ] **Step 7: Commit**

```bash
git add src/grasp_grpo/model.py src/grasp_grpo/batcher.py tests/test_batcher.py tests/test_generate_batch_gpu.py
git commit -m "feat(model): batched generation with float32 logprobs and a generation batcher"
```

---

### Task 5: Token-level GRPO objective

**Files:**
- Rewrite: `src/grasp_grpo/grpo.py`
- Test: `tests/test_grpo_objective.py`

**Interfaces:**
- Consumes: `model.response_logprob_entropy` (Task 4); leaves with `"advantage"` and `"actions"` whose items carry `"prompt"`, `"response_token_ids"`, `"old_token_logprobs"` (Task 2).
- Produces: `merge_actions(leaves) -> list[tuple[dict, float]]`; `token_loss(logp, old, ref, advantage, *, clip_low, clip_high, kl_beta) -> (loss_sum: Tensor, {"ratio_sum", "clipped", "kl_sum"})`; `grpo_backward(actor, reference, tokenizer, leaves, *, clip_low, clip_high, kl_beta) -> {"loss", "ratio", "clip_frac", "kl", "entropy", "tokens", "action_count"}` (per-token means except `tokens`/`action_count`).

- [ ] **Step 1: Write the failing tests** — `tests/test_grpo_objective.py`

```python
import pytest
import torch

from grasp_grpo.grpo import merge_actions, token_loss


def test_on_policy_gradient_is_minus_advantage_per_token():
    old = torch.tensor([-0.5, -1.0, -2.0])
    logp = old.clone().requires_grad_(True)
    loss, stats = token_loss(logp, old, old.clone(), torch.tensor(1.5), clip_low=0.2, clip_high=0.28, kl_beta=0.002)
    loss.backward()
    assert torch.allclose(logp.grad, torch.full((3,), -1.5))
    assert stats["kl_sum"] == pytest.approx(0.0, abs=1e-7)
    assert stats["clipped"] == 0


def test_k3_kl_is_positive_when_policy_moves_away():
    old = torch.tensor([-1.0, -1.0])
    _, stats = token_loss(torch.tensor([-0.5, -1.5]), old, old, torch.tensor(0.0), clip_low=0.2, clip_high=0.28, kl_beta=0.002)
    assert stats["kl_sum"] > 0


def test_clip_higher_counts_tokens_outside_asymmetric_range():
    old = torch.zeros(3)
    logp = torch.log(torch.tensor([0.7, 1.25, 1.35]))  # ratios 0.7, 1.25, 1.35
    _, stats = token_loss(logp, old, logp, torch.tensor(1.0), clip_low=0.2, clip_high=0.28, kl_beta=0.0)
    assert stats["clipped"] == 2  # 0.7 < 0.8 and 1.35 > 1.28; 1.25 is inside
    assert stats["ratio_sum"] == pytest.approx(3.3, rel=1e-5)


def test_merge_actions_sums_advantages_of_shared_ancestors():
    shared, a, b = {"id": "rd0"}, {"id": "qf1"}, {"id": "qf2"}
    leaves = [{"advantage": 0.5, "actions": [shared, a]}, {"advantage": -0.25, "actions": [shared, b]}]
    merged = merge_actions(leaves)
    assert [(action["id"], advantage) for action, advantage in merged] == [("rd0", 0.25), ("qf1", 0.5), ("qf2", -0.25)]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_grpo_objective.py -v`
Expected: FAIL — `ImportError: cannot import name 'merge_actions'`.

- [ ] **Step 3: Replace `src/grasp_grpo/grpo.py` entirely**

```python
"""Token-level clipped GRPO objective (DAPO aggregation, clip-higher, k3 KL); no critic."""
from __future__ import annotations

import torch

from .model import response_logprob_entropy


def merge_actions(leaves: list[dict]) -> list[tuple[dict, float]]:
    """One entry per distinct action; an action shared by several leaves gets their summed advantage.

    Every action is scored once per rollout batch and followed by a single optimizer step, so all
    ratios start at 1 and summing the advantages equals scoring the action once per leaf.
    """
    merged: dict[int, list] = {}
    for leaf in leaves:
        for action in leaf["actions"]:
            entry = merged.setdefault(id(action), [action, 0.0])
            entry[1] += float(leaf["advantage"])
    return [(action, advantage) for action, advantage in merged.values()]


def token_loss(logp: torch.Tensor, old: torch.Tensor, ref: torch.Tensor, advantage: torch.Tensor, *, clip_low: float, clip_high: float, kl_beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Summed (not averaged) per-token loss of one action; the caller divides by the batch token count."""
    ratio = torch.exp(torch.clamp(logp - old, -20, 20))
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * advantage
    surrogate = torch.minimum(unclipped, clipped)
    log_ratio_ref = ref - logp
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1  # k3 estimator, >= 0
    loss = (-surrogate + kl_beta * kl).sum()
    with torch.no_grad():
        outside = ((ratio < 1 - clip_low) | (ratio > 1 + clip_high)).sum()
    return loss, {"ratio_sum": float(ratio.detach().sum()), "clipped": float(outside), "kl_sum": float(kl.detach().sum())}


def grpo_backward(actor, reference, tokenizer, leaves: list[dict], *, clip_low: float, clip_high: float, kl_beta: float) -> dict[str, float]:
    """Backpropagate one action at a time (bounded activation memory); loss = sum over tokens / batch tokens."""
    entries = merge_actions(leaves)
    if not entries:
        raise ValueError("tree batch has no proxy actions")
    total_tokens = sum(len(action["response_token_ids"]) for action, _ in entries)
    if total_tokens == 0:
        raise ValueError("tree batch has no response tokens")
    device = next(actor.parameters()).device
    sums = {"loss": 0.0, "ratio": 0.0, "clipped": 0.0, "kl": 0.0, "entropy": 0.0}
    for action, advantage in entries:
        logp, entropy = response_logprob_entropy(actor, tokenizer, action["prompt"], action["response_token_ids"])
        with torch.no_grad():
            ref, _ = response_logprob_entropy(reference, tokenizer, action["prompt"], action["response_token_ids"])
        old = torch.tensor(action["old_token_logprobs"], device=device, dtype=logp.dtype)
        loss, stats = token_loss(logp, old, ref, torch.tensor(advantage, device=device, dtype=logp.dtype), clip_low=clip_low, clip_high=clip_high, kl_beta=kl_beta)
        (loss / total_tokens).backward()
        sums["loss"] += float(loss.detach())
        sums["ratio"] += stats["ratio_sum"]
        sums["clipped"] += stats["clipped"]
        sums["kl"] += stats["kl_sum"]
        sums["entropy"] += float(entropy.detach().sum())
    return {"loss": sums["loss"] / total_tokens, "ratio": sums["ratio"] / total_tokens, "clip_frac": sums["clipped"] / total_tokens, "kl": sums["kl"] / total_tokens, "entropy": sums["entropy"] / total_tokens, "tokens": float(total_tokens), "action_count": float(len(entries))}
```

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_grpo_objective.py -v`
Expected: 4 passed.

```bash
git add src/grasp_grpo/grpo.py tests/test_grpo_objective.py
git commit -m "feat(grpo): token-level clipped objective with clip-higher and k3 KL"
```

---

### Task 6: Early stopping rules

**Files:**
- Create: `src/grasp_grpo/stopping.py`
- Test: `tests/test_stopping.py`

**Interfaces:**
- Produces: `StopConfig(min_improvement=0.02, patience_evals=3, sft_drop=0.03, sft_drop_evals=2, min_format_rate=0.90, max_kl=0.3, min_entropy_ratio=0.30, max_zero_signal=0.60, window=20, max_hours=None)`; `EarlyStopper(config, sft_f1: float | None)` with `.set_baseline(f1)`, `.after_step(*, format_rate, kl, entropy: float | None, zero_signal, elapsed_hours) -> str | None`, `.after_eval(*, step, f1) -> (str | None, is_best: bool)`; attributes `sft_f1`, `best_f1`, `best_step`. Stop reasons: `"max_hours"`, `"format_collapse"`, `"kl_drift"`, `"entropy_collapse"`, `"no_signal"`, `"worse_than_sft"`, `"no_improvement"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_stopping.py`

```python
from grasp_grpo.stopping import EarlyStopper, StopConfig


def run_steps(stopper, n, **overrides):
    values = {"format_rate": 1.0, "kl": 0.01, "entropy": 1.0, "zero_signal": 0.1, "elapsed_hours": 0.0, **overrides}
    for _ in range(n):
        reason = stopper.after_step(**values)
        if reason:
            return reason
    return None


def test_healthy_training_never_stops():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 50) is None


def test_rules_wait_for_a_full_window():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 4, format_rate=0.0) is None


def test_format_collapse():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 5, format_rate=0.8) == "format_collapse"


def test_kl_drift():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 5, kl=0.5) == "kl_drift"


def test_entropy_collapse_is_relative_to_first_window():
    stopper = EarlyStopper(StopConfig(window=5), sft_f1=0.4)
    assert run_steps(stopper, 5, entropy=2.0) is None
    assert run_steps(stopper, 5, entropy=0.5) == "entropy_collapse"


def test_steps_without_update_do_not_feed_entropy():
    assert run_steps(EarlyStopper(StopConfig(window=3), sft_f1=0.4), 6, entropy=None) is None


def test_no_signal():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 5, zero_signal=0.9) == "no_signal"


def test_budget():
    stopper = EarlyStopper(StopConfig(window=5, max_hours=2.0), sft_f1=0.4)
    assert stopper.after_step(format_rate=1.0, kl=0.0, entropy=1.0, zero_signal=0.0, elapsed_hours=2.5) == "max_hours"


def test_no_improvement_after_patience():
    stopper = EarlyStopper(StopConfig(), sft_f1=0.40)
    assert stopper.after_eval(step=50, f1=0.41) == (None, True)
    assert stopper.after_eval(step=100, f1=0.415) == (None, True)
    assert stopper.after_eval(step=150, f1=0.40) == ("no_improvement", False)


def test_real_gain_resets_patience():
    stopper = EarlyStopper(StopConfig(), sft_f1=0.40)
    stopper.after_eval(step=50, f1=0.41)
    stopper.after_eval(step=100, f1=0.41)
    assert stopper.after_eval(step=150, f1=0.43) == (None, True)
    assert stopper.after_eval(step=200, f1=0.43) == (None, False)
    assert stopper.after_eval(step=250, f1=0.44) == (None, True)
    assert stopper.after_eval(step=300, f1=0.44)[0] == "no_improvement"
    assert (stopper.best_f1, stopper.best_step) == (0.44, 250)


def test_worse_than_sft_needs_consecutive_evals():
    stopper = EarlyStopper(StopConfig(patience_evals=10), sft_f1=0.40)
    assert stopper.after_eval(step=50, f1=0.36)[0] is None
    assert stopper.after_eval(step=100, f1=0.39)[0] is None
    assert stopper.after_eval(step=150, f1=0.365)[0] is None
    assert stopper.after_eval(step=200, f1=0.36)[0] == "worse_than_sft"


def test_baseline_can_be_set_after_construction():
    stopper = EarlyStopper(StopConfig(), sft_f1=None)
    stopper.set_baseline(0.30)
    assert stopper.after_eval(step=50, f1=0.26)[0] is None
    assert stopper.after_eval(step=100, f1=0.25)[0] == "worse_than_sft"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stopping.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grasp_grpo.stopping'`.

- [ ] **Step 3: Create `src/grasp_grpo/stopping.py`**

```python
"""Early-stopping rules of the GRPO protocol (spec §6). Pure logic: no I/O, no torch."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class StopConfig:
    min_improvement: float = 0.02  # eval F1 gain that counts as real (~2 paired standard errors on 300 questions)
    patience_evals: int = 3
    sft_drop: float = 0.03
    sft_drop_evals: int = 2
    min_format_rate: float = 0.90
    max_kl: float = 0.3
    min_entropy_ratio: float = 0.30
    max_zero_signal: float = 0.60
    window: int = 20
    max_hours: float | None = None


class EarlyStopper:
    def __init__(self, config: StopConfig, sft_f1: float | None):
        self.config = config
        self.sft_f1 = sft_f1
        self.reference_f1 = sft_f1  # the bar a real improvement must clear
        self.history: deque = deque(maxlen=config.window)
        self.initial_entropy: list[float] = []
        self.evals_without_gain = 0
        self.evals_below_sft = 0
        self.best_f1: float | None = None
        self.best_step: int | None = None

    def set_baseline(self, f1: float) -> None:
        self.sft_f1 = self.reference_f1 = f1

    def after_step(self, *, format_rate: float, kl: float, entropy: float | None, zero_signal: float, elapsed_hours: float) -> str | None:
        c = self.config
        if c.max_hours is not None and elapsed_hours > c.max_hours:
            return "max_hours"
        if entropy is not None and len(self.initial_entropy) < c.window:
            self.initial_entropy.append(entropy)
        self.history.append((format_rate, kl, entropy, zero_signal))
        if len(self.history) < c.window:
            return None

        def mean(index: int) -> float | None:
            values = [row[index] for row in self.history if row[index] is not None]
            return sum(values) / len(values) if values else None

        if mean(0) < c.min_format_rate:
            return "format_collapse"
        if mean(1) > c.max_kl:
            return "kl_drift"
        entropy_now = mean(2)
        if entropy_now is not None and len(self.initial_entropy) >= c.window:
            baseline = sum(self.initial_entropy) / len(self.initial_entropy)
            if baseline > 0 and entropy_now < c.min_entropy_ratio * baseline:
                return "entropy_collapse"
        if mean(3) > c.max_zero_signal:
            return "no_signal"
        return None

    def after_eval(self, *, step: int, f1: float) -> tuple[str | None, bool]:
        c = self.config
        is_best = self.best_f1 is None or f1 > self.best_f1
        if is_best:
            self.best_f1, self.best_step = f1, step
        if self.reference_f1 is None:
            self.reference_f1 = f1  # no baseline: the first eval sets the bar
        elif f1 >= self.reference_f1 + c.min_improvement:
            self.reference_f1, self.evals_without_gain = f1, 0
        else:
            self.evals_without_gain += 1
        if self.sft_f1 is not None and f1 <= self.sft_f1 - c.sft_drop:
            self.evals_below_sft += 1
        else:
            self.evals_below_sft = 0
        if self.evals_below_sft >= c.sft_drop_evals:
            return "worse_than_sft", is_best
        if self.evals_without_gain >= c.patience_evals:
            return "no_improvement", is_best
        return None, is_best
```

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stopping.py -v`
Expected: 12 passed.

```bash
git add src/grasp_grpo/stopping.py tests/test_stopping.py
git commit -m "feat(train): early-stopping rules"
```

---

### Task 7: Fixed eval set

**Files:**
- Create: `src/grasp_grpo/eval_ids.py`
- Create: `config/eval_ids.json` (generated)
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_eval_ids.py`

**Interfaces:**
- Produces: `select_eval_ids(rows, *, per_dataset, seed, exclude=frozenset()) -> list[str]`; `filter_examples(examples, ids) -> list[dict]` (keeps file order, `ValueError` naming missing ids); `load_eval_examples(qa_path, ids_path | None) -> list[dict]`; CLI `grasp-make-eval-ids --dev --output [--exclude] [--per-dataset 100] [--seed 42]` writing `{"source", "seed", "per_dataset", "ids"}`.

- [ ] **Step 1: Write the failing tests** — `tests/test_eval_ids.py`

```python
import pytest

from grasp_grpo.eval_ids import filter_examples, select_eval_ids


def rows():
    return [{"id": f"{ds}{i}", "dataset": ds, "gold_doc_ids": ["g"] if i < 8 else []} for ds in ("a", "b") for i in range(10)]


def test_selection_is_per_dataset_deterministic_and_gold_only():
    first = select_eval_ids(rows(), per_dataset=3, seed=42)
    assert first == select_eval_ids(rows(), per_dataset=3, seed=42)
    assert len(first) == 6 and sum(x.startswith("a") for x in first) == 3
    assert not set(first) & {"a8", "a9", "b8", "b9"}


def test_excluded_ids_are_never_selected():
    chosen = select_eval_ids(rows(), per_dataset=7, seed=1, exclude={"a0"})
    assert "a0" not in chosen and len(chosen) == 14


def test_too_few_questions_raises():
    with pytest.raises(ValueError, match="only 8"):
        select_eval_ids(rows(), per_dataset=9, seed=0)


def test_filter_examples_keeps_order_and_reports_missing():
    examples = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    assert [e["id"] for e in filter_examples(examples, ["z", "x"])] == ["x", "z"]
    with pytest.raises(ValueError, match="missing"):
        filter_examples(examples, ["x", "nope"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_ids.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grasp_grpo.eval_ids'`.

- [ ] **Step 3: Create `src/grasp_grpo/eval_ids.py`**

```python
"""Fixed, seeded eval set (spec §4): N questions per dataset that have gold evidence."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import read_examples, write_json


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_eval_ids(rows: list[dict[str, Any]], *, per_dataset: int, seed: int, exclude: set[str] | frozenset = frozenset()) -> list[str]:
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("gold_doc_ids") and str(row["id"]) not in exclude:
            by_dataset[str(row["dataset"])].append(str(row["id"]))
    rng = random.Random(seed)
    chosen: list[str] = []
    for name in sorted(by_dataset):
        ids = sorted(by_dataset[name])
        if len(ids) < per_dataset:
            raise ValueError(f"dataset {name} has only {len(ids)} eligible questions, need {per_dataset}")
        chosen.extend(sorted(rng.sample(ids, per_dataset)))
    return chosen


def filter_examples(examples: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(ids)
    kept = [example for example in examples if example["id"] in wanted]
    missing = wanted - {example["id"] for example in kept}
    if missing:
        raise ValueError(f"{len(missing)} eval ids missing from the QA file, e.g. {sorted(missing)[:3]}")
    return kept


def load_eval_examples(qa_path: str, ids_path: str | None) -> list[dict[str, Any]]:
    examples = read_examples(qa_path)
    if ids_path:
        examples = filter_examples(examples, json.loads(Path(ids_path).read_text(encoding="utf-8"))["ids"])
    return examples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select the fixed GRPO eval question ids")
    parser.add_argument("--dev", required=True, help="grasp-gold-evidence dev.jsonl")
    parser.add_argument("--exclude", help="JSONL whose ids must never be selected (the training file)")
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    exclude = {str(row["id"]) for row in _jsonl(args.exclude)} if args.exclude else set()
    ids = select_eval_ids(_jsonl(args.dev), per_dataset=args.per_dataset, seed=args.seed, exclude=exclude)
    write_json(args.output, {"source": Path(args.dev).name, "seed": args.seed, "per_dataset": args.per_dataset, "ids": ids})
    print(json.dumps({"ids": len(ids), "output": args.output}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register the CLI** — in `pyproject.toml` under `[project.scripts]` add:

```toml
grasp-make-eval-ids = "grasp_grpo.eval_ids:main"
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_ids.py -v`
Expected: 4 passed.

- [ ] **Step 6: Generate the committed id list**

```bash
GE=../grasp-gold-evidence-builder/outputs/grasp-gold-evidence
.venv/Scripts/python.exe -m grasp_grpo.eval_ids --dev $GE/dev.jsonl --exclude $GE/train.jsonl --output config/eval_ids.json
.venv/Scripts/python.exe -c "import json,collections; d=json.load(open('config/eval_ids.json')); print(len(d['ids']), collections.Counter(i.split('::')[0] for i in d['ids']))"
```
Expected: `{"ids": 300, ...}` then `300 Counter({'2wiki_ircot': 100, 'hotpotqa_abstracts_2017': 100, 'wiki18': 100})`.

- [ ] **Step 7: Commit**

```bash
git add src/grasp_grpo/eval_ids.py tests/test_eval_ids.py pyproject.toml config/eval_ids.json
git commit -m "feat(eval): fixed seeded 300-question eval set"
```

---

### Task 8: Greedy policy evaluation and signal probe

**Files:**
- Rewrite: `src/grasp_grpo/infer.py`
- Create: `src/grasp_grpo/policy_eval.py`
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_policy_eval.py`

**Interfaces:**
- Consumes: reader protocol (Task 1), `BoundRetriever`, `add_retrieval_arguments`, `retriever_from_args` (Task 3), `generate_batch`, `GenerationBatcher`, `role_generate_fn` (Task 4), `BinaryTreeEnvironment` (Task 2), `load_eval_examples` (Task 7), `evaluate.f1/exact`.
- Produces: `infer.run_example(generate_fn, reader, retriever, example, max_steps) -> {"id", "prediction", "context_doc_ids", "actions"}`; `policy_eval.summarize(examples, predictions, elapsed_seconds) -> dict` with keys `questions, f1, em, context_recall, format_rate{rd,qf,ki}, mean_retrievals, seconds_per_question, per_question`; `evaluate_policy(generate_fn, reader, retriever, examples, *, max_steps, workers) -> dict`; `signal_probe(make_env, examples, *, workers) -> {"trees", "action_trees", "zero_signal_fraction", "seconds_per_tree"}`; CLI `grasp-eval-policy`.

- [ ] **Step 1: Write the failing tests** — `tests/test_policy_eval.py`

```python
from grasp_grpo.infer import run_example
from grasp_grpo.model import Action
from grasp_grpo.policy_eval import signal_probe, summarize
from grasp_grpo.reader import ReaderStep


class Reader:
    def __init__(self):
        self.calls = 0

    def step(self, question, context, thoughts, transcript):
        self.calls += 1
        if self.calls == 1:
            return ReaderStep("<think> need </think>\n", "need", None)
        return ReaderStep("<answer> answer </answer>", None, "answer")

    def final(self, *args):
        raise AssertionError("the reader answered before max_steps")


class Retriever:
    def search(self, query):
        return [{"doc_id": "gold" if query == "gold query" else "q", "title": "", "text": ""}]

    def candidate_triples(self, query, passages):
        return [{"doc_id": p["doc_id"], "head": "H", "relation": "r", "tail": "T"} for p in passages]


def generate_fn(role, prompt, samples):
    text = {"rd": "Yes (retrieval is required)", "qf": "Reasoning Chain: None\nKnowledge Gap: need\nQuery: gold query", "ki": "<H; r; T>"}[role]
    return [Action(text, [1], [0.0])]


def test_run_example_follows_reader_and_proxy():
    example = {"id": "q1", "question": "Q?", "gold_answers": ["answer"], "gold_doc_ids": ["gold"], "metadata": {}}
    result = run_example(generate_fn, Reader(), Retriever(), example, max_steps=5)
    assert result["prediction"] == "answer"
    assert result["context_doc_ids"] == ["q", "gold"]
    assert [a["role"] for a in result["actions"]] == ["rd", "qf", "ki"]


def test_summarize_reports_quality_format_and_speed():
    examples = [{"id": "a", "gold_answers": ["x"], "gold_doc_ids": ["d1", "d2"]}, {"id": "b", "gold_answers": ["y"], "gold_doc_ids": []}]
    predictions = [
        {"id": "a", "prediction": "x", "context_doc_ids": ["d1"], "actions": [{"role": "rd", "valid_format": True}, {"role": "qf", "valid_format": False}]},
        {"id": "b", "prediction": "z", "context_doc_ids": [], "actions": [{"role": "rd", "valid_format": True}]},
    ]
    report = summarize(examples, predictions, elapsed_seconds=10.0)
    assert report["f1"] == 0.5 and report["em"] == 0.5
    assert report["context_recall"] == 0.5
    assert report["format_rate"] == {"rd": 1.0, "qf": 0.0, "ki": None}
    assert report["mean_retrievals"] == 0.5
    assert report["seconds_per_question"] == 5.0


def test_signal_probe_counts_zero_signal_trees():
    trees = iter([
        {"zero_signal": True, "leaves": [{"actions": [1]}]},
        {"zero_signal": False, "leaves": [{"actions": [1]}]},
        {"zero_signal": True, "leaves": [{"actions": []}]},  # reader answered directly: ignored
    ])

    class Env:
        def __init__(self, tree):
            self.tree = tree

        def run(self, example):
            return self.tree

    probe = signal_probe(lambda example: Env(next(trees)), [{}, {}, {}], workers=1)
    assert probe["trees"] == 3 and probe["action_trees"] == 2
    assert probe["zero_signal_fraction"] == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_policy_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grasp_grpo.policy_eval'`.

- [ ] **Step 3: Replace `src/grasp_grpo/infer.py` entirely**

```python
"""Deterministic sequential GRASP inference for checkpoint evaluation."""
from __future__ import annotations

import argparse
import json

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


def run_example(generate_fn, reader, retriever, example: dict, max_steps: int) -> dict:
    select = getattr(retriever, "select", None)
    if select is not None:
        select(example)
    question = example["question"]
    context, thoughts, actions, transcript, answer = retriever.search(question), [], [], "", None
    for _ in range(max_steps):
        step = reader.step(question, context, thoughts, transcript)
        transcript += step.piece
        if step.answer is not None:
            answer = step.answer
            break
        if step.thought is None:
            break
        thought = step.thought
        thoughts.append(thought)
        state = {"question": question, "context": context, "thoughts": thoughts, "current_thought": thought}
        retrieve, valid, error, text = act(generate_fn, "rd", state)
        actions.append({"role": "rd", "response": text, "valid_format": valid, "validation_error": error})
        if not valid:
            retrieve = True
        if not retrieve:
            continue
        qf, valid_qf, error, text = act(generate_fn, "qf", state)
        actions.append({"role": "qf", "response": text, "valid_format": valid_qf, "validation_error": error})
        query = str(qf.get("query") or question) if valid_qf else question
        passages = retriever.search(query)
        candidates = retriever.candidate_triples(query, passages)
        if not candidates:
            continue
        ki_state = {"question": question, "chain": qf.get("chain", []) if valid_qf else [], "query": query, "candidates": candidates}
        selected, valid_ki, error, text = act(generate_fn, "ki", ki_state)
        actions.append({"role": "ki", "response": text, "valid_format": valid_ki, "validation_error": error})
        chosen = candidate_index(selected if valid_ki else None, candidates)
        evidence = next((item for item in passages if item["doc_id"] == candidates[chosen if chosen is not None else 0]["doc_id"]), passages[0])
        if all(item["doc_id"] != evidence["doc_id"] for item in context):
            context.append(evidence)
    if answer is None:
        answer = reader.final(question, context, thoughts, transcript)
    return {"id": example["id"], "prediction": answer, "context_doc_ids": [item["doc_id"] for item in context], "actions": actions}


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
```

- [ ] **Step 4: Create `src/grasp_grpo/policy_eval.py`**

```python
"""Greedy single-path evaluation of a proxy policy, plus a learning-signal probe (spec §4, §5 stage 0)."""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .batcher import GenerationBatcher, role_generate_fn
from .contracts import ROLES
from .data import write_json
from .environment import BinaryTreeEnvironment
from .eval_ids import load_eval_examples
from .evaluate import exact, f1
from .infer import run_example
from .model import generate_batch, load_policy
from .reader import add_reader_arguments, reader_from_args
from .retrieval import BoundRetriever, add_retrieval_arguments, retriever_from_args


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(examples: list[dict[str, Any]], predictions: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    by_id = {prediction["id"]: prediction for prediction in predictions}
    rows, recalls, retrievals = [], [], []
    formats: dict[str, list[bool]] = {role: [] for role in ROLES}
    for example in examples:
        prediction = by_id[example["id"]]
        rows.append({"id": example["id"], "f1": f1(prediction["prediction"], example["gold_answers"]), "em": exact(prediction["prediction"], example["gold_answers"])})
        gold = set(example.get("gold_doc_ids", []))
        if gold:
            recalls.append(len(gold & set(prediction["context_doc_ids"])) / len(gold))
        for action in prediction["actions"]:
            formats[action["role"]].append(bool(action["valid_format"]))
        retrievals.append(sum(action["role"] == "qf" for action in prediction["actions"]))
    return {
        "questions": len(rows),
        "f1": _mean([row["f1"] for row in rows]),
        "em": _mean([row["em"] for row in rows]),
        "context_recall": _mean(recalls),
        "format_rate": {role: _mean([float(x) for x in values]) for role, values in formats.items()},
        "mean_retrievals": _mean([float(x) for x in retrievals]),
        "seconds_per_question": elapsed_seconds / len(rows),
        "per_question": rows,
    }


def evaluate_policy(generate_fn, reader, retriever, examples: list[dict[str, Any]], *, max_steps: int, workers: int) -> dict[str, Any]:
    lock = threading.Lock()
    started = time.monotonic()

    def one(example: dict[str, Any]) -> dict[str, Any]:
        return run_example(generate_fn, reader, BoundRetriever(retriever, example, lock), example, max_steps)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(pool.map(one, examples))
    return summarize(examples, predictions, time.monotonic() - started)


def signal_probe(make_env, examples: list[dict[str, Any]], *, workers: int) -> dict[str, Any]:
    """Share of sampled trees whose leaves all score the same (zero advantage, no gradient)."""
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        trees = list(pool.map(lambda example: make_env(example).run(example), examples))
    elapsed = time.monotonic() - started
    action_trees = [tree for tree in trees if any(leaf["actions"] for leaf in tree["leaves"])]
    zero = sum(bool(tree["zero_signal"]) for tree in action_trees)
    return {"trees": len(trees), "action_trees": len(action_trees), "zero_signal_fraction": zero / len(action_trees) if action_trees else 1.0, "seconds_per_tree": elapsed / len(trees) if trees else None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Greedy eval of a proxy adapter on the fixed eval set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--eval-ids")
    add_retrieval_arguments(parser)
    add_reader_arguments(parser)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--signal-probe", type=int, default=0, help="Also sample N training-style trees and report the zero-signal share")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-leaves", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--shaping-scale", type=float, default=0.1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    examples = load_eval_examples(args.qa, args.eval_ids)
    if args.limit is not None:
        examples = examples[: args.limit]
    retriever = retriever_from_args(args)
    actor, tokenizer, _ = load_policy(args.checkpoint, trainable=False, load_in_4bit=args.load_in_4bit)
    reader = reader_from_args(args)
    batcher = GenerationBatcher(lambda prompts, **kwargs: generate_batch(actor, tokenizer, prompts, **kwargs), max_batch=args.max_batch)
    report = evaluate_policy(role_generate_fn(batcher.generate, temperature=0.0, top_p=1.0), reader, retriever, examples, max_steps=args.max_steps, workers=args.workers)
    if args.signal_probe:
        lock = threading.Lock()
        sample_fn = role_generate_fn(batcher.generate, temperature=args.temperature, top_p=args.top_p)

        def make_env(example):
            return BinaryTreeEnvironment(actor, tokenizer, reader, BoundRetriever(retriever, example, lock), branch_factor=2, max_steps=args.max_steps, max_leaves=args.max_leaves, temperature=args.temperature, top_p=args.top_p, alpha=args.alpha, retrieval_cost=0.0, behavior_checkpoint=str(Path(args.checkpoint).resolve()), shaping_scale=args.shaping_scale, generate_fn=sample_fn)

        report["signal_probe"] = signal_probe(make_env, examples[: args.signal_probe], workers=args.workers)
    report.update(checkpoint=str(Path(args.checkpoint).resolve()), reader_protocol=args.reader_protocol, max_steps=args.max_steps)
    batcher.close()
    retriever.close()
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_question"}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Register the CLI** — in `pyproject.toml` under `[project.scripts]` add:

```toml
grasp-eval-policy = "grasp_grpo.policy_eval:main"
```

- [ ] **Step 6: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all pass.

```bash
git add src/grasp_grpo/infer.py src/grasp_grpo/policy_eval.py tests/test_policy_eval.py pyproject.toml
git commit -m "feat(eval): greedy policy eval with format/recall/speed and signal probe"
```

---

### Task 9: Training loop with refill, warmup, eval, early stopping

**Files:**
- Rewrite: `src/grasp_grpo/train.py`
- Test: `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces: CLI `grasp-train-grpo` (flags below); helpers `lr_at(step, base, warmup) -> float`, `public_config(args) -> dict`, `tree_stats(trees) -> dict` (`format_rate`, `format_rate_{rd,qf,ki}`, `mean_reward`, `mean_answer_f1`, `mean_context_recall`, `leaves`). Output dir contents: `metrics.json` (one dict per step: `step, updated, loss, ratio, clip_frac, kl, entropy, tokens, grad_norm, lr, format_rate*, mean_reward, zero_signal_groups, zero_signal_fraction, direct_answer_trees, refilled_questions, seconds_per_question, elapsed_hours, eta_hours`), `eval_history.json`, `best/`, `checkpoint-grpo-step-NNNNNN/`, `run_manifest.json` (`stop_reason, stop_step, completed_steps, sft_f1, best_step, best_f1, best_checkpoint, final_checkpoint, …`).

- [ ] **Step 1: Write the failing tests** — `tests/test_train_helpers.py`

```python
import argparse

import pytest

from grasp_grpo.train import lr_at, parse_args, public_config, tree_stats


def test_warmup_is_linear_then_constant():
    assert lr_at(1, 1e-5, 20) == pytest.approx(5e-7)
    assert lr_at(20, 1e-5, 20) == pytest.approx(1e-5)
    assert lr_at(50, 1e-5, 20) == pytest.approx(1e-5)
    assert lr_at(3, 1e-5, 0) == pytest.approx(1e-5)


def test_tree_stats_counts_each_shared_action_once():
    shared = {"role": "rd", "valid_format": True}
    leaves = [
        {"actions": [shared, {"role": "qf", "valid_format": False}], "reward": {"total": 1.0, "answer_f1": 1.0, "context_recall": None}},
        {"actions": [shared], "reward": {"total": 0.0, "answer_f1": 0.0, "context_recall": 1.0}},
    ]
    stats = tree_stats([{"leaves": leaves}])
    assert stats["format_rate"] == 0.5
    assert (stats["format_rate_rd"], stats["format_rate_qf"], stats["format_rate_ki"]) == (1.0, 0.0, None)
    assert stats["mean_reward"] == 0.5 and stats["mean_context_recall"] == 0.5 and stats["leaves"] == 2


def test_tree_stats_without_trees():
    stats = tree_stats([])
    assert stats["format_rate"] == 1.0 and stats["mean_reward"] is None


def test_public_config_drops_secrets():
    assert public_config(argparse.Namespace(reader_api_key="secret", learning_rate=1e-5)) == {"learning_rate": 1e-5}


def test_pilot_defaults_match_the_spec():
    args = parse_args(["--sft-checkpoint", "s", "--qa", "q", "--corpus-config", "c", "--reader-model", "m", "--output-dir", "o"])
    assert (args.learning_rate, args.questions_per_iteration, args.max_leaves, args.max_steps) == (1e-5, 16, 4, 5)
    assert (args.alpha, args.shaping_scale, args.kl_beta, args.warmup_steps) == (0.3, 0.1, 0.002, 20)
    assert (args.clip_low, args.clip_high, args.eval_every, args.reader_protocol) == (0.2, 0.28, 50, "grasp")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'lr_at'`.

- [ ] **Step 3: Replace `src/grasp_grpo/train.py` entirely**

```python
"""On-policy binary-tree GRPO training with periodic greedy eval and early stopping.

Protocol, defaults and thresholds: docs/superpowers/specs/2026-11-04-grpo-training-protocol-design.md
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from .reader import add_reader_arguments, reader_from_args
from .retrieval import BoundRetriever, add_retrieval_arguments, retriever_from_args
from .stopping import EarlyStopper, StopConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the GRASP Qwen 3B LoRA proxy with on-policy binary-tree GRPO")
    p.add_argument("--sft-checkpoint", required=True, help="Project A PEFT adapter; frozen KL reference")
    p.add_argument("--actor-checkpoint", help="Optional GRPO adapter to continue from; SFT stays the KL reference")
    p.add_argument("--qa", required=True)
    add_retrieval_arguments(p)
    add_reader_arguments(p)
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
    p.add_argument("--load-in-4bit", action="store_true")
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
    actor.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_json(path / "grpo_config.json", {**metadata, "created_at_utc": _now()})
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if args.branch_factor != 2:
        raise ValueError("branch-factor must be 2")
    if args.iterations < 1 or args.questions_per_iteration < 1 or args.workers < 1:
        raise ValueError("iterations, questions-per-iteration and workers must be positive")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples = read_examples(args.qa)
    if not examples:
        raise ValueError("no valid training examples")
    eval_examples = load_eval_examples(args.eval_qa, args.eval_ids) if args.eval_qa else None
    retriever = retriever_from_args(args)
    actor_source = args.actor_checkpoint or args.sft_checkpoint
    actor, tokenizer, _ = load_policy(actor_source, trainable=True, load_in_4bit=args.load_in_4bit)
    reference, _, _ = load_policy(args.sft_checkpoint, trainable=False, load_in_4bit=args.load_in_4bit)
    params = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    reader = reader_from_args(args)
    lock = threading.Lock()
    batcher = GenerationBatcher(lambda prompts, **kwargs: generate_batch(actor, tokenizer, prompts, **kwargs), max_batch=args.max_batch)
    sample_fn = role_generate_fn(batcher.generate, temperature=args.temperature, top_p=args.top_p)
    greedy_fn = role_generate_fn(batcher.generate, temperature=0.0, top_p=1.0)
    stopper = EarlyStopper(StopConfig(min_improvement=args.stop_min_improvement, patience_evals=args.stop_patience_evals, sft_drop=args.stop_sft_drop, sft_drop_evals=args.stop_sft_drop_evals, min_format_rate=args.stop_min_format_rate, max_kl=args.stop_max_kl, min_entropy_ratio=args.stop_min_entropy_ratio, max_zero_signal=args.stop_max_zero_signal, window=args.stop_window, max_hours=args.max_hours), args.sft_f1)
    config = public_config(args)

    def make_env(example: dict[str, Any], step: int) -> BinaryTreeEnvironment:
        return BinaryTreeEnvironment(actor, tokenizer, reader, BoundRetriever(retriever, example, lock), branch_factor=2, max_steps=args.max_steps, max_leaves=args.max_leaves, temperature=args.temperature, top_p=args.top_p, alpha=args.alpha, retrieval_cost=args.retrieval_cost, behavior_checkpoint=f"in_memory_actor_before_step_{step}", initial_retrieval=not args.no_initial_retrieval, rd_gain_weight=args.rd_gain_weight, qf_gain_weight=args.qf_gain_weight, ki_hit_weight=args.ki_hit_weight, ki_miss_penalty=args.ki_miss_penalty, shaping_scale=args.shaping_scale, advantage_std_norm=args.advantage_std_norm, generate_fn=sample_fn)

    def rollout(batch: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            return list(pool.map(lambda example: make_env(example, step).run(example), batch))

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
    cursor, stop_reason, step = 0, None, 0
    max_attempts = math.ceil(args.questions_per_iteration * (1 + args.max_refill_factor))
    started = time.monotonic()
    progress = tqdm(total=args.iterations, desc="GRPO", unit="step", dynamic_ncols=True)
    for step in range(1, args.iterations + 1):
        actor.eval()  # rollouts and ratios use the same dropout-free policy
        collected: list[dict[str, Any]] = []
        zero_trees: list[dict[str, Any]] = []
        direct, attempts = 0, 0
        rollout_started = time.monotonic()
        while len(collected) < args.questions_per_iteration and attempts < max_attempts:
            need = min(args.questions_per_iteration - len(collected), max_attempts - attempts)
            batch = [examples[(cursor + offset) % len(examples)] for offset in range(need)]
            cursor += need
            attempts += need
            for tree in rollout(batch, step):
                if not any(leaf["actions"] for leaf in tree["leaves"]):
                    direct += 1  # reader answered directly: no proxy tokens to train
                elif tree["zero_signal"]:
                    zero_trees.append(tree)  # all leaves tie: zero advantage, dropped (DAPO dynamic sampling)
                else:
                    collected.append(tree)
        rollout_seconds = time.monotonic() - rollout_started
        signal_trees = len(collected) + len(zero_trees)
        zero_fraction = len(zero_trees) / signal_trees if signal_trees else 1.0
        info: dict[str, Any] = {"step": step, **tree_stats(collected + zero_trees), "zero_signal_groups": len(zero_trees), "zero_signal_fraction": zero_fraction, "direct_answer_trees": direct, "refilled_questions": max(0, attempts - args.questions_per_iteration), "seconds_per_question": rollout_seconds / attempts}
        if collected:
            lr = lr_at(step, args.learning_rate, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            leaves = [leaf for tree in collected for leaf in tree["leaves"]]
            info.update(grpo_backward(actor, reference, tokenizer, leaves, clip_low=args.clip_low, clip_high=args.clip_high, kl_beta=args.kl_beta))
            info["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            optimizer.step()
            info.update(lr=lr, updated=True)
        else:
            info["updated"] = False
        elapsed_hours = (time.monotonic() - started) / 3600
        info.update(elapsed_hours=elapsed_hours, eta_hours=elapsed_hours / step * (args.iterations - step))
        metrics.append(info)
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
    final = output / f"checkpoint-grpo-step-{step:06d}"
    if not final.exists():
        save_adapter(actor, tokenizer, final, {"step": step, "sft_checkpoint": str(Path(args.sft_checkpoint).resolve()), "config": config, "metrics": metrics[-50:]})
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_manifest.json", {"schema_version": "grasp-grpo-tree-v2", "stop_reason": stop_reason, "stop_step": step if stop_reason else None, "completed_steps": step, "sft_f1": stopper.sft_f1, "best_step": stopper.best_step, "best_f1": stopper.best_f1, "best_checkpoint": str(output / "best") if stopper.best_step is not None else None, "final_checkpoint": str(final), "sft_checkpoint": str(Path(args.sft_checkpoint).resolve()), "actor_initialization": str(Path(actor_source).resolve()), "qa": str(Path(args.qa).resolve()), "config": config, "created_at_utc": _now()})
    batcher.close()
    retriever.close()
    print(json.dumps({"stop_reason": stop_reason, "completed_steps": step, "best_step": stopper.best_step, "best_f1": stopper.best_f1, "final_checkpoint": str(final)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all pass.

```bash
git add src/grasp_grpo/train.py tests/test_train_helpers.py
git commit -m "feat(train): refill, warmup, periodic eval, best checkpoint and early stopping"
```

---

### Task 10: Stage gates

**Files:**
- Create: `src/grasp_grpo/stage_gate.py`
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_stage_gate.py`

**Interfaces:**
- Consumes: `stage0_grasp.json` (Task 8 report with `signal_probe`), `metrics.json` / `eval_history.json` / `run_manifest.json` (Task 9).
- Produces: `gate_stage0(report, *, min_f1=0.30, max_zero_signal=0.50, max_seconds=20.0)`, `gate_stage1(metrics, *, adapter_changed, ratio_range=(0.9, 1.1), max_kl=0.05)`, `gate_stage2(metrics, *, window=10, min_gain=0.30)`, `gate_stage3(manifest, eval_history, *, min_improvement=0.02)` — each `-> (passed: bool, reasons: list[str])`; `adapter_changed(before_dir, after_dir) -> bool`; CLI `grasp-stage-gate --stage {0,1,2,3} --run-dir DIR [--sft-checkpoint DIR]` writing `DIR/stage_report.json` and exiting 1 on failure.

- [ ] **Step 1: Write the failing tests** — `tests/test_stage_gate.py`

```python
from grasp_grpo.stage_gate import gate_stage0, gate_stage1, gate_stage2, gate_stage3


def test_stage0():
    ok = {"f1": 0.35, "signal_probe": {"zero_signal_fraction": 0.3, "seconds_per_tree": 12.0}}
    assert gate_stage0(ok) == (True, [])
    passed, reasons = gate_stage0({"f1": 0.2, "signal_probe": {"zero_signal_fraction": 0.7, "seconds_per_tree": 12.0}})
    assert not passed and len(reasons) == 2
    assert not gate_stage0({"f1": 0.35})[0]  # probe missing


def step(**kw):
    return {"updated": True, "loss": 0.01, "ratio": 1.0, "kl": 0.01, "grad_norm": 0.5, "mean_reward": 0.5, **kw}


def test_stage1():
    assert gate_stage1([step()] * 20, adapter_changed=True) == (True, [])
    assert not gate_stage1([step(loss=float("nan"))] + [step()] * 19, adapter_changed=True)[0]
    assert not gate_stage1([step()] * 20, adapter_changed=False)[0]
    assert not gate_stage1([step(grad_norm=0.0)] * 20, adapter_changed=True)[0]
    assert not gate_stage1([step(kl=0.0)] * 20, adapter_changed=True)[0]
    assert not gate_stage1([step(ratio=1.3)] * 20, adapter_changed=True)[0]
    assert not gate_stage1([{"updated": False}] * 20, adapter_changed=True)[0]


def test_stage2():
    rising = [step(mean_reward=0.4)] * 10 + [step(mean_reward=0.45)] * 20 + [step(mean_reward=0.6)] * 10
    assert gate_stage2(rising)[0]
    assert not gate_stage2([step(mean_reward=0.5)] * 40)[0]
    assert not gate_stage2([step()] * 5)[0]


def test_stage3():
    history = [{"step": 0, "f1": 0.40}, {"step": 50, "f1": 0.43}]
    assert gate_stage3({"stop_reason": None, "sft_f1": 0.40}, history)[0]
    assert gate_stage3({"stop_reason": "max_hours", "sft_f1": 0.40}, history)[0]
    assert not gate_stage3({"stop_reason": None, "sft_f1": 0.40}, [{"step": 50, "f1": 0.41}])[0]
    assert not gate_stage3({"stop_reason": "kl_drift", "sft_f1": 0.40}, history)[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stage_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grasp_grpo.stage_gate'`.

- [ ] **Step 3: Create `src/grasp_grpo/stage_gate.py`**

```python
"""Pass/fail gates between the stages of the GRPO protocol (spec §5)."""
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import write_json


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def gate_stage0(report: dict[str, Any], *, min_f1: float = 0.30, max_zero_signal: float = 0.50, max_seconds: float = 20.0) -> tuple[bool, list[str]]:
    reasons = []
    if report["f1"] < min_f1:
        reasons.append(f"SFT F1 {report['f1']:.3f} < {min_f1}: fix the environment before training")
    probe = report.get("signal_probe")
    if not probe:
        reasons.append("signal probe missing: rerun grasp-eval-policy with --signal-probe")
    else:
        if probe["zero_signal_fraction"] > max_zero_signal:
            reasons.append(f"zero-signal trees {probe['zero_signal_fraction']:.0%} > {max_zero_signal:.0%}")
        if probe["seconds_per_tree"] is not None and probe["seconds_per_tree"] > max_seconds:
            reasons.append(f"{probe['seconds_per_tree']:.1f} s per tree > {max_seconds}: recompute the pilot cost")
    return not reasons, reasons


def gate_stage1(metrics: list[dict[str, Any]], *, adapter_changed: bool, ratio_range: tuple[float, float] = (0.9, 1.1), max_kl: float = 0.05) -> tuple[bool, list[str]]:
    updated = [m for m in metrics if m.get("updated")]
    if not updated:
        return False, ["no step produced an update"]
    reasons = []
    if any(not math.isfinite(m["loss"]) for m in updated):
        reasons.append("non-finite loss")
    ratio = _mean([m["ratio"] for m in updated])
    if not ratio_range[0] <= ratio <= ratio_range[1]:
        reasons.append(f"mean ratio {ratio:.3f} outside {ratio_range}")
    kl = _mean([m["kl"] for m in updated])
    if not 0 < kl < max_kl:
        reasons.append(f"mean KL {kl:.4f} not in (0, {max_kl})")
    if any(m["grad_norm"] <= 0 for m in updated):
        reasons.append("zero gradient norm")
    if not adapter_changed:
        reasons.append("LoRA weights did not change")
    return not reasons, reasons


def gate_stage2(metrics: list[dict[str, Any]], *, window: int = 10, min_gain: float = 0.30) -> tuple[bool, list[str]]:
    rewards = [m["mean_reward"] for m in metrics if m.get("mean_reward") is not None]
    if len(rewards) < 2 * window:
        return False, [f"need {2 * window} steps with rewards, got {len(rewards)}"]
    first, last = _mean(rewards[:window]), _mean(rewards[-window:])
    target = first * (1 + min_gain) if first > 0 else first + 0.05
    if last < target:
        return False, [f"reward {first:.3f} -> {last:.3f}, needed >= {target:.3f}: the pipeline cannot fit 8 questions"]
    return True, []


def gate_stage3(manifest: dict[str, Any], eval_history: list[dict[str, Any]], *, min_improvement: float = 0.02) -> tuple[bool, list[str]]:
    reasons = []
    if manifest.get("stop_reason") not in (None, "max_hours"):
        reasons.append(f"stopped early: {manifest['stop_reason']}")
    sft = manifest.get("sft_f1")
    scores = [h["f1"] for h in eval_history if h.get("step", 0) > 0]
    if sft is None or not scores:
        reasons.append("missing SFT baseline or eval history")
    elif max(scores) < sft + min_improvement - 1e-9:
        reasons.append(f"best eval F1 {max(scores):.3f} < SFT {sft:.3f} + {min_improvement}")
    return not reasons, reasons


def adapter_changed(before_dir: str | Path, after_dir: str | Path) -> bool:
    import torch
    from safetensors.torch import load_file

    before = load_file(str(Path(before_dir) / "adapter_model.safetensors"))
    after = load_file(str(Path(after_dir) / "adapter_model.safetensors"))
    shared = set(before) & set(after)
    if not shared:
        raise ValueError("the two adapters share no tensor names")
    return any(not torch.equal(before[name], after[name]) for name in shared)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Decide whether a GRPO protocol stage passed")
    parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sft-checkpoint", help="Required for stage 1 (adapter change check)")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    if args.stage == 0:
        passed, reasons = gate_stage0(_load(run / "stage0_grasp.json"))
    elif args.stage == 1:
        if not args.sft_checkpoint:
            parser.error("--sft-checkpoint is required for stage 1")
        checkpoints = sorted(run.glob("checkpoint-grpo-step-*"))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoint-grpo-step-* in {run}")
        passed, reasons = gate_stage1(_load(run / "metrics.json"), adapter_changed=adapter_changed(args.sft_checkpoint, checkpoints[-1]))
    elif args.stage == 2:
        passed, reasons = gate_stage2(_load(run / "metrics.json"))
    else:
        history = _load(run / "eval_history.json") if (run / "eval_history.json").exists() else []
        passed, reasons = gate_stage3(_load(run / "run_manifest.json"), history)
    report = {"stage": args.stage, "passed": passed, "reasons": reasons, "run_dir": str(run.resolve()), "checked_at_utc": datetime.now(UTC).isoformat()}
    write_json(run / "stage_report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register the CLI** — in `pyproject.toml` under `[project.scripts]` add:

```toml
grasp-stage-gate = "grasp_grpo.stage_gate:main"
```

- [ ] **Step 5: Run the tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stage_gate.py -v`
Expected: 4 passed.

```bash
git add src/grasp_grpo/stage_gate.py tests/test_stage_gate.py pyproject.toml
git commit -m "feat(protocol): stage 0-3 pass/fail gates"
```

---

### Task 11: Stage scripts, settings and docs

**Files:**
- Rewrite: `config/vast.env.example`
- Modify: `scripts/grpo_vast_common.sh` (append helpers)
- Create: `scripts/stage0_baseline.sh`, `scripts/stage1_smoke.sh`, `scripts/stage2_overfit.sh`, `scripts/stage3_pilot.sh`
- Rewrite: `scripts/train_grpo_vast.sh`
- Modify: `scripts/smoke_tree_vast.sh` (drop `--load-in-4bit`)
- Modify: `README.md`

**Interfaces:**
- Consumes: CLIs `grasp-eval-policy`, `grasp-train-grpo`, `grasp-stage-gate` (Tasks 8–10); `config/eval_ids.json` (Task 7).

- [ ] **Step 1: Replace `config/vast.env.example`**

```bash
# Copy this file to config/vast.env and replace every <...> value.
PROJECT_DIR=/workspace/grasp-grpo-trainer
VENV_DIR=/workspace/grasp-grpo-trainer/.venv
READER_VENV=/workspace/grasp-reader-venv

# GPU 0 runs vLLM reader; GPU 1 runs GRPO when using one 2-GPU Vast instance.
READER_GPU=0
TRAIN_GPU=1

SFT_CHECKPOINT=/workspace/checkpoints/grasp-qwen2.5-3b-sft
QA_FILE=/workspace/data/grpo_train_mixed_shuffled.jsonl
CORPUS_CONFIG=/workspace/configs/corpora.json
# Stages 0-3 write to $OUTPUT_ROOT/stage<N>; the full run (train_grpo_vast.sh) writes to $OUTPUT_DIR.
OUTPUT_ROOT=/workspace/outputs/grpo-protocol
OUTPUT_DIR=/workspace/outputs/grpo-final

READER_URL=http://127.0.0.1:8000/v1
READER_API_KEY="<set-a-long-reader-api-key>"
READER_MODEL=Qwen/Qwen2.5-7B-Instruct
# grasp = Project A reader protocol that generated the SFT data; legacy = the first GRPO run.
READER_PROTOCOL=grasp

# Fixed eval set (spec section 4). EVAL_QA is grasp-gold-evidence dev.jsonl.
EVAL_QA=/workspace/data/grasp-gold-evidence/dev.jsonl
EVAL_IDS=/workspace/grasp-grpo-trainer/config/eval_ids.json
EVAL_EVERY=50
# SFT F1 printed by stage 0 (stage0_grasp.json -> f1). Empty = measured again at step 0.
SFT_F1=

# Pilot (stage 3) defaults, spec section 3.4.
ITERATIONS=300
QUESTIONS_PER_ITERATION=16
WORKERS=16
SAVE_EVERY=50
MAX_HOURS=24
BRANCH_FACTOR=2
MAX_STEPS=5
MAX_LEAVES=4
LEARNING_RATE=1e-5
WARMUP_STEPS=20
CLIP_LOW=0.2
CLIP_HIGH=0.28
KL_BETA=0.002

ALPHA=0.3
SHAPING_SCALE=0.1
RETRIEVAL_COST=0.0
RD_GAIN_WEIGHT=0.3
QF_GAIN_WEIGHT=0.7
KI_HIT_WEIGHT=0.3
KI_MISS_PENALTY=0.1

# Full run only: continue from the pilot's best adapter.
ACTOR_CHECKPOINT=
```

- [ ] **Step 2: Append helpers to `scripts/grpo_vast_common.sh`**

```bash

# Tee all output to DIR/logs/console.log and sample nvidia-smi every 30 s until the script exits.
start_logging() {
  local dir="$1"
  mkdir -p "$dir/logs"
  exec > >(tee -a "$dir/logs/console.log") 2>&1
  local gpu_log="$dir/logs/gpu_metrics.csv"
  [[ -f "$gpu_log" ]] || echo "timestamp,index,name,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w" > "$gpu_log"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$gpu_log" || true
      sleep 30
    done
  ) &
  GPU_MONITOR_PID=$!
  trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
}

reader_args() {
  READER_ARGS=(--reader-url "$READER_URL" --reader-api-key "$READER_API_KEY" --reader-model "$READER_MODEL")
}

# Flags shared by every grasp-train-grpo call, built from vast.env.
train_args() {
  reader_args
  TRAIN_ARGS=(--sft-checkpoint "$SFT_CHECKPOINT" --corpus-config "$CORPUS_CONFIG" "${READER_ARGS[@]}"
    --reader-protocol "${READER_PROTOCOL:-grasp}" --workers "$WORKERS"
    --branch-factor "$BRANCH_FACTOR" --max-steps "$MAX_STEPS" --max-leaves "$MAX_LEAVES"
    --alpha "$ALPHA" --shaping-scale "$SHAPING_SCALE" --retrieval-cost "${RETRIEVAL_COST:-0.0}"
    --rd-gain-weight "$RD_GAIN_WEIGHT" --qf-gain-weight "$QF_GAIN_WEIGHT"
    --ki-hit-weight "$KI_HIT_WEIGHT" --ki-miss-penalty "$KI_MISS_PENALTY"
    --learning-rate "$LEARNING_RATE" --warmup-steps "$WARMUP_STEPS"
    --clip-low "$CLIP_LOW" --clip-high "$CLIP_HIGH" --kl-beta "$KL_BETA")
}
```

- [ ] **Step 3: Create `scripts/stage0_baseline.sh`**

```bash
#!/usr/bin/env bash
# Stage 0: SFT baseline on the fixed eval set with both reader protocols, plus speed and signal probe.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL EVAL_QA EVAL_IDS OUTPUT_ROOT; do require_value "$key"; done
RUN="$OUTPUT_ROOT/stage0"
start_logging "$RUN"
reader_args
for protocol in grasp legacy; do
  probe=()
  [[ "$protocol" == grasp ]] && probe=(--signal-probe 50)
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-eval-policy \
    --checkpoint "$SFT_CHECKPOINT" --qa "$EVAL_QA" --eval-ids "$EVAL_IDS" --corpus-config "$CORPUS_CONFIG" \
    "${READER_ARGS[@]}" --reader-protocol "$protocol" \
    --max-steps "$MAX_STEPS" --workers "$WORKERS" --max-leaves "$MAX_LEAVES" \
    --alpha "$ALPHA" --shaping-scale "$SHAPING_SCALE" "${probe[@]}" \
    --output "$RUN/stage0_${protocol}.json"
done
grasp-stage-gate --stage 0 --run-dir "$RUN"
echo "Set SFT_F1 in config/vast.env to the f1 above in $RUN/stage0_grasp.json"
```

- [ ] **Step 4: Create `scripts/stage1_smoke.sh`**

```bash
#!/usr/bin/env bash
# Stage 1: 20 steps x 4 questions; checks finite loss, ratio ~1, small KL, non-zero gradients, LoRA moved.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL OUTPUT_ROOT; do require_value "$key"; done
RUN="$OUTPUT_ROOT/stage1"
start_logging "$RUN"
train_args
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-train-grpo "${TRAIN_ARGS[@]}" \
  --qa "$QA_FILE" --output-dir "$RUN" --iterations 20 --questions-per-iteration 4 --save-every 20
grasp-stage-gate --stage 1 --run-dir "$RUN" --sft-checkpoint "$SFT_CHECKPOINT"
```

- [ ] **Step 5: Create `scripts/stage2_overfit.sh`**

```bash
#!/usr/bin/env bash
# Stage 2: 40 steps on the same 8 questions; reward must rise >= 30% or the pipeline cannot learn.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL OUTPUT_ROOT; do require_value "$key"; done
RUN="$OUTPUT_ROOT/stage2"
start_logging "$RUN"
head -n 8 "$QA_FILE" > "$RUN/overfit8.jsonl"
train_args
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-train-grpo "${TRAIN_ARGS[@]}" \
  --qa "$RUN/overfit8.jsonl" --output-dir "$RUN" --iterations 40 --questions-per-iteration 8 --save-every 40
grasp-stage-gate --stage 2 --run-dir "$RUN"
```

- [ ] **Step 6: Create `scripts/stage3_pilot.sh`**

```bash
#!/usr/bin/env bash
# Stage 3: pilot with periodic eval and early stopping; passes when eval F1 >= SFT + 0.02 with no stop rule fired.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL EVAL_QA EVAL_IDS OUTPUT_ROOT; do require_value "$key"; done
RUN="$OUTPUT_ROOT/stage3"
start_logging "$RUN"
train_args
sft=()
[[ -n "${SFT_F1:-}" ]] && sft=(--sft-f1 "$SFT_F1")
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-train-grpo "${TRAIN_ARGS[@]}" \
  --qa "$QA_FILE" --output-dir "$RUN" --iterations "$ITERATIONS" --questions-per-iteration "$QUESTIONS_PER_ITERATION" \
  --eval-qa "$EVAL_QA" --eval-ids "$EVAL_IDS" --eval-every "$EVAL_EVERY" "${sft[@]}" \
  --max-hours "$MAX_HOURS" --save-every "$SAVE_EVERY"
grasp-stage-gate --stage 3 --run-dir "$RUN"
```

- [ ] **Step 7: Replace `scripts/train_grpo_vast.sh`** (stage 4, full run)

```bash
#!/usr/bin/env bash
# Stage 4: full run, normally continuing from the pilot's best adapter (ACTOR_CHECKPOINT).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG OUTPUT_DIR READER_URL READER_API_KEY READER_MODEL EVAL_QA EVAL_IDS; do require_value "$key"; done
start_logging "$OUTPUT_DIR"
train_args
extra=()
[[ -n "${ACTOR_CHECKPOINT:-}" ]] && extra+=(--actor-checkpoint "$ACTOR_CHECKPOINT")
[[ -n "${SFT_F1:-}" ]] && extra+=(--sft-f1 "$SFT_F1")
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-train-grpo "${TRAIN_ARGS[@]}" "${extra[@]}" \
  --qa "$QA_FILE" --output-dir "$OUTPUT_DIR" --iterations "$ITERATIONS" --questions-per-iteration "$QUESTIONS_PER_ITERATION" \
  --eval-qa "$EVAL_QA" --eval-ids "$EVAL_IDS" --eval-every "$EVAL_EVERY" \
  --max-hours "$MAX_HOURS" --save-every "$SAVE_EVERY"
```

- [ ] **Step 8: Drop 4-bit from `scripts/smoke_tree_vast.sh`**

Delete the final line `  --load-in-4bit` and remove the trailing ` \` from the line above it (`--branch-factor … --max-leaves "$MAX_LEAVES"`).

- [ ] **Step 9: Make scripts executable and lint them**

```bash
git update-index --chmod=+x scripts/stage0_baseline.sh scripts/stage1_smoke.sh scripts/stage2_overfit.sh scripts/stage3_pilot.sh 2>/dev/null || true
for f in scripts/*.sh; do bash -n "$f" && echo "ok $f"; done
```
Expected: `ok` for every script (syntax check only; they run on Vast).

- [ ] **Step 10: Add a protocol section to `README.md`** (append at the end)

```markdown
## Training protocol (stages 0–4)

Every GPU hour is gated; see `docs/superpowers/specs/2026-11-04-grpo-training-protocol-design.md`.

| Stage | Script | Passes when |
|---|---|---|
| 0 | `scripts/stage0_baseline.sh` | SFT F1 ≥ 0.30 with the Project A reader, ≤ 50% zero-signal trees, ≤ 20 s per tree |
| 1 | `scripts/stage1_smoke.sh` | finite loss, ratio ≈ 1, 0 < KL < 0.05, non-zero gradients, LoRA weights moved |
| 2 | `scripts/stage2_overfit.sh` | reward on 8 fixed questions rises ≥ 30% in 40 steps |
| 3 | `scripts/stage3_pilot.sh` | eval F1 ≥ SFT + 0.02 and no early-stop rule fired |
| 4 | `scripts/train_grpo_vast.sh` | early stopping decides; use `best/` |

Each stage writes `$OUTPUT_ROOT/stage<N>/stage_report.json` (`passed`, `reasons`). Do not start a stage before the previous one passed.
Fixed eval ids live in `config/eval_ids.json` (`grasp-make-eval-ids`). Training stops itself on: no eval gain ≥ 0.02 F1 for 3 evals, F1 ≥ 0.03 below SFT twice, format rate < 90%, token KL > 0.3, entropy < 30% of start, > 60% zero-signal trees, or `--max-hours`.
```

- [ ] **Step 11: Commit**

```bash
git add config/vast.env.example scripts/ README.md
git commit -m "feat(protocol): stage scripts, pilot settings and README"
```

---

### Task 12: Final verification

- [ ] **Step 1: Reinstall so new entry points exist, then run everything**

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check src tests --select F,E9
for cli in grasp-train-grpo grasp-eval-policy grasp-stage-gate grasp-make-eval-ids grasp-infer grasp-grpo-rollout; do .venv/Scripts/$cli.exe --help > /dev/null && echo "ok $cli"; done
```
Expected: all tests pass (1 GPU test skipped); ruff reports no `F`/`E9` errors; every CLI prints `ok`.

- [ ] **Step 2: Confirm `chat.py` still imports**

Run: `.venv/Scripts/python.exe -c "import grasp_grpo.chat"`
Expected: no output, exit 0.

- [ ] **Step 3: Review the branch diff**

Run: `git log --oneline main..HEAD && git diff --stat main..HEAD`
Expected: ~12 commits touching only the files in the File Structure table. Do **not** push; report to the user.
