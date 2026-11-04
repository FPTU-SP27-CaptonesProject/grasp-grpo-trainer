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
