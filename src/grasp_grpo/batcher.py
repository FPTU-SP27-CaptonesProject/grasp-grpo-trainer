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
