"""Proxy generation on a vLLM server (base model + LoRA), with the trained adapter hot-swapped each step.

HF `generate` decodes one token at a time for a handful of prompts, so a five-step tree took ~8 min
and the GPU sat at ~40% (stage 0, 2026-11-05). vLLM batches every rollout thread's requests
continuously. The trainer keeps the HF copy for the loss; after each optimizer step it writes the
adapter to disk and asks the server to load it under a new name (needs the server started with
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True, see scripts/run_policy_vast.sh).

Prompts are sent as token ids (the exact `_prompt_ids` the loss uses, truncation included), and the
sampled token ids come back through `return_tokens_as_token_ids`, so the loss scores exactly the
tokens that were sampled. The sampler's own log-probs are kept to measure HF/vLLM agreement
(`sampler_logprob_gap` in the step log), not used in the loss.
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

from .model import Action, _prompt_ids


class VLLMPolicy:
    def __init__(self, base_url: str, api_key: str, tokenizer, *, model: str, client=None, admin=None, timeout: float = 900.0):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=3)
        self.tokenizer, self.model, self.client = tokenizer, model, client
        self._admin = admin or _Admin(base_url, api_key)
        self._static = model  # served from --lora-modules; never unloaded
        self._lock = threading.Lock()
        self._files: list[Path] = []  # adapter directories this process wrote and loaded, oldest first
        self._loaded: list[str] = []  # previous (non-static) adapter names still loaded on the server, oldest first
        self.stop_ids = {tokenizer.eos_token_id, *([tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else [])}

    def generate(self, prompt: str, *, samples: int, max_new_tokens: int, temperature: float, top_p: float, model: str | None = None) -> list[Action]:
        """Sample from `model` (an adapter name), or from the current adapter when None."""
        if samples > 1 and temperature <= 0:
            raise ValueError("multiple tree branches require temperature > 0")
        if model is None:
            with self._lock:
                model = self.model
        request: dict[str, Any] = {"model": model, "prompt": _prompt_ids(self.tokenizer, prompt), "max_tokens": max_new_tokens, "n": samples, "temperature": max(temperature, 0.0), "logprobs": 0, "extra_body": {"return_tokens_as_token_ids": True}}
        if temperature > 0:
            request["top_p"] = top_p
        response = self.client.completions.create(**request)
        return [self._action(choice) for choice in sorted(response.choices, key=lambda c: c.index)]

    def _action(self, choice) -> Action:
        tokens = list(choice.logprobs.tokens or [])
        token_ids = [int(str(token).rsplit(":", 1)[-1]) for token in tokens]
        logprobs = [float(x) if x is not None else None for x in (choice.logprobs.token_logprobs or [])]
        if choice.finish_reason == "stop" and (not token_ids or token_ids[-1] not in self.stop_ids):
            # The server stopped on EOS without returning it: the loss must still teach the model to stop.
            token_ids.append(self.tokenizer.eos_token_id)
            logprobs = []  # no sampler log-prob for the added token: skip this action in the gap metric
        if len(logprobs) != len(token_ids) or any(x is None for x in logprobs):
            logprobs = []
        return Action(self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(), token_ids, logprobs)

    def current(self) -> str:
        with self._lock:
            return self.model

    def pinned(self, model: str):
        """A `generate` that always samples from adapter `model`, even after later updates (async rollouts)."""
        return lambda prompt, **kwargs: self.generate(prompt, model=model, **kwargs)

    def update(self, adapter_dir: str | Path, name: str, *, keep_loaded: int = 1, keep_files: int = 2) -> None:
        """Serve `adapter_dir` as `name` from now on. The newest `keep_loaded` adapters stay loaded (2 while a
        rollout pinned to the previous one may still be running); older ones are unloaded unless static. Only
        the newest `keep_files` directories stay on disk (>= keep_loaded: vLLM may reload from CPU cache only)."""
        path = Path(adapter_dir).resolve()
        self._admin.post("load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
        with self._lock:
            previous, self.model = self.model, name
        if previous != self._static:
            self._loaded.append(previous)
        while len(self._loaded) > keep_loaded - 1:
            self._admin.post("unload_lora_adapter", {"lora_name": self._loaded.pop(0)})
        self._files.append(path)
        while len(self._files) > max(keep_files, keep_loaded):
            shutil.rmtree(self._files.pop(0), ignore_errors=True)


class _Admin:
    def __init__(self, base_url: str, api_key: str):
        import httpx

        self.root = base_url.rstrip("/")  # .../v1
        self.http = httpx.Client(timeout=600.0, headers={"Authorization": f"Bearer {api_key}"})

    def post(self, endpoint: str, payload: dict[str, Any]) -> None:
        response = self.http.post(f"{self.root}/{endpoint}", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"vLLM {endpoint} failed ({response.status_code}): {response.text[:500]}")


def add_policy_arguments(parser) -> None:
    parser.add_argument("--policy-url", help="vLLM server for proxy generation (scripts/run_policy_vast.sh); omit for HF generate")
    parser.add_argument("--policy-api-key", default="EMPTY")
    parser.add_argument("--policy-model", default="sft", help="LoRA name the server loaded the starting adapter under")
