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
