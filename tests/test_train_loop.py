"""The training loop's rollout scheduling (sync and async) with every heavy dependency faked."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from grasp_grpo import train  # noqa: E402

EXAMPLES = [{"id": f"q{i}", "question": f"question {i}", "answers": ["a"], "dataset": "d"} for i in range(6)]


class FakePolicy:
    """Records which adapter each rollout sampled from and when adapters change."""

    def __init__(self, *args, model, **kwargs):
        self.model, self.events, self.lock = model, [], threading.Lock()

    def current(self):
        return self.model

    def generate(self, prompt, *, model=None, **kwargs):
        with self.lock:
            self.events.append(("sample", model or self.model))
        return []

    def pinned(self, model):
        return lambda prompt, **kwargs: self.generate(prompt, model=model, **kwargs)

    def update(self, path, name, **kwargs):
        with self.lock:
            self.events.append(("update", name, kwargs.get("keep_loaded")))
            self.model = name


class FakeEnv:
    calls = 0

    def __init__(self, actor, tokenizer, reader, retriever, *, generate_fn, **kwargs):
        assert actor is None  # a server-side policy must never switch the training model's mode
        self.generate_fn = generate_fn

    def run(self, example):
        FakeEnv.calls += 1
        self.generate_fn("rd", "prompt", 1)
        time.sleep(0.01)
        zero = int(example["id"][1:]) % 3 == 0  # every third question has no learning signal
        action = {"role": "rd", "valid_format": True}
        leaf = {"actions": [action], "reward": {"total": 1.0, "answer_f1": 1.0, "context_recall": 1.0}, "advantage": 0.0 if zero else 1.0}
        return {"leaves": [leaf], "zero_signal": zero}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    model = torch.nn.Linear(2, 2)
    policies = []

    def make_policy(*args, **kwargs):
        policies.append(FakePolicy(*args, **kwargs))
        return policies[-1]

    backward_calls = []

    def fake_backward(actor, reference, tokenizer, leaves, **kwargs):
        backward_calls.append(kwargs["sampler_old"])
        for p in actor.parameters():
            p.grad = torch.ones_like(p)
        return {"loss": 0.0, "ratio": 1.0, "kl": 0.01, "entropy": 0.5}

    monkeypatch.setattr(train, "read_examples", lambda path: list(EXAMPLES))
    monkeypatch.setattr(train, "retriever_from_args", lambda args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(train, "reader_from_args", lambda args: None)
    monkeypatch.setattr(train, "load_policy", lambda *a, **k: (model, SimpleNamespace(), None))
    monkeypatch.setattr(train, "VLLMPolicy", make_policy)
    monkeypatch.setattr(train, "BinaryTreeEnvironment", FakeEnv)
    monkeypatch.setattr(train, "BoundRetriever", lambda *a: None)
    monkeypatch.setattr(train, "grpo_backward", fake_backward)
    monkeypatch.setattr(train, "extraction_stats", lambda retriever: None)
    monkeypatch.setattr(train, "save_adapter", lambda actor, tok, path, meta: path)
    base = ["--sft-checkpoint", "sft", "--qa", "qa.jsonl", "--output-dir", str(tmp_path), "--policy-url", "http://x/v1",
            "--iterations", "3", "--questions-per-iteration", "2", "--workers", "4", "--warmup-steps", "0", "--reader-model", "r", "--corpus-config", "c.json"]
    return SimpleNamespace(base=base, policies=policies, backward_calls=backward_calls, out=tmp_path)


def strip_tag(name):
    """policy-<run tag>-000001 -> policy-000001 (the tag is a per-run timestamp)."""
    parts = name.split("-")
    return name if len(parts) < 3 else f"{parts[0]}-{parts[-1]}"


def metrics(out):
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


def test_sync_loop_samples_each_step_from_the_latest_adapter(harness):
    train.main(harness.base)
    events = harness.policies[0].events
    sampled_before = []  # adapter used by the rollouts preceding each update
    seen: set = set()
    for event in events:
        if event[0] == "sample":
            seen.add(strip_tag(event[1]))
        else:
            sampled_before.append(seen)
            seen = set()
    assert sampled_before == [{"sft"}, {"policy-000001"}, {"policy-000002"}]
    assert harness.backward_calls == [False] * 3
    rows = metrics(harness.out)
    assert [r["step"] for r in rows] == [1, 2, 3] and all(r["updated"] for r in rows)
    assert all(r["leaves"] - r["zero_signal_groups"] == 2 for r in rows)  # exactly questions-per-iteration trained trees


def test_async_loop_samples_one_step_behind_and_trains_on_sampler_logprobs(harness):
    train.main([*harness.base, "--async-rollout"])
    policy = harness.policies[0]
    samples = [strip_tag(e[1]) for e in policy.events if e[0] == "sample"]
    updates = [e for e in policy.events if e[0] == "update"]
    assert len({u[1].split("-")[1] for u in updates}) == 1  # one run tag for the whole run
    assert [(strip_tag(u[1]), u[2]) for u in updates] == [("policy-000001", 2), ("policy-000002", 2), ("policy-000003", 2)]
    # step 1 and step 2 rollouts both start before any update lands; step 3 samples policy-000001.
    assert set(samples) == {"sft", "policy-000001"}
    assert harness.backward_calls == [True] * 3
    rows = metrics(harness.out)
    assert [r["step"] for r in rows] == [1, 2, 3] and all(r["leaves"] - r["zero_signal_groups"] == 2 for r in rows)
    assert all("rollout_wait_seconds" in r for r in rows)


def resumed_run(harness, monkeypatch, *extra):
    """Run 2 steps, then resume from <out>/resume to step 4 (fresh process state: new policy fake, new optimizer)."""
    seen = []
    real_env = FakeEnv.run

    def record(self, example):
        seen.append(example["id"])
        return real_env(self, example)

    monkeypatch.setattr(FakeEnv, "run", record)
    args = [a for a in harness.base]
    args[args.index("--iterations") + 1] = "2"
    args[args.index("--warmup-steps") + 1] = "4"
    train.main([*args, *extra])
    first = list(seen)
    state = torch.load(harness.out / "resume" / "train_state.pt", weights_only=False)
    args[args.index("--iterations") + 1] = "4"
    train.main([*args, *extra, "--resume", str(harness.out / "resume"), "--question-offset", "999"])
    return first, seen[len(first):], state


@pytest.mark.parametrize("extra", [(), ("--async-rollout",)])
def test_resume_continues_step_lr_cursor_and_optimizer(harness, monkeypatch, extra):
    first, second, state = resumed_run(harness, monkeypatch, *extra)
    assert state["step"] == 2 and state["optimizer"]["state"]  # Adam moments saved
    rows = metrics(harness.out)
    assert [r["step"] for r in rows] == [1, 2, 3, 4]  # earlier steps kept in metrics.jsonl
    assert [r["lr"] for r in rows] == pytest.approx([1e-5 * s / 4 for s in (1, 2, 3, 4)])  # warmup not restarted
    assert second[0] == f"q{state['cursor'] % len(EXAMPLES)}"  # picks up at the saved cursor; --question-offset ignored


def test_resume_runs_an_eval_the_crash_skipped(harness, monkeypatch):
    evals = []
    monkeypatch.setattr(train, "load_eval_examples", lambda qa, ids: [{"id": "e"}])
    monkeypatch.setattr(train, "evaluate_policy", lambda *a, **k: evals.append(1) or {"f1": 0.5, "per_question": []})
    real = train.EarlyStopper.after_eval
    crashed = {"done": False}

    def crash_once(self, *, step, f1):
        if step == 2 and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("eval crashed")
        return real(self, step=step, f1=f1)

    monkeypatch.setattr(train.EarlyStopper, "after_eval", crash_once)
    args = [*harness.base, "--eval-qa", "dev.jsonl", "--eval-every", "2", "--sft-f1", "0.5"]
    args[args.index("--iterations") + 1] = "4"
    with pytest.raises(RuntimeError):
        train.main(args)
    assert torch.load(harness.out / "resume" / "train_state.pt", weights_only=False)["step"] == 2  # saved before the eval
    train.main([*args, "--resume", str(harness.out / "resume")])
    history = json.loads((harness.out / "eval_history.json").read_text())
    assert len(evals) == 3  # step 2 (crashed), step 2 again on resume, step 4
    assert [e["step"] for e in history] == [2, 4]  # history comes from the saved state, so no duplicate
    assert [r["step"] for r in metrics(harness.out)] == [1, 2, 3, 4]
