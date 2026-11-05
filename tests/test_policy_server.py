"""vLLM policy client: exact token round trip, EOS handling, adapter hot swap."""
from types import SimpleNamespace

import pytest
import torch

from grasp_grpo.grpo import _backward
from grasp_grpo.policy_server import VLLMPolicy
from grasp_grpo.stage_gate import gate_stage1

EOS, PAD = 99, 98


class FakeTokenizer:
    eos_token_id, pad_token_id = EOS, PAD

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<u>" + messages[0]["content"]

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(c) % 90 for c in text])

    def decode(self, ids, skip_special_tokens):
        return "".join(chr(65 + i % 26) for i in ids if not (skip_special_tokens and i in (EOS, PAD)))


def choice(index, ids, logprobs, finish="stop"):
    return SimpleNamespace(index=index, finish_reason=finish, logprobs=SimpleNamespace(tokens=[f"token_id:{i}" for i in ids], token_logprobs=logprobs))


class FakeCompletions:
    def __init__(self, choices):
        self.choices, self.requests = choices, []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(choices=self.choices)


class FakeAdmin:
    def __init__(self):
        self.calls = []

    def post(self, endpoint, payload):
        self.calls.append((endpoint, payload))


def make(choices):
    completions, admin = FakeCompletions(choices), FakeAdmin()
    policy = VLLMPolicy("http://x/v1", "k", FakeTokenizer(), model="sft", client=SimpleNamespace(completions=completions), admin=admin)
    return policy, completions, admin


def test_generate_sends_token_ids_and_returns_sampled_ids_in_order():
    policy, completions, _ = make([choice(1, [3, 4, EOS], [-0.1, -0.2, -0.3]), choice(0, [5, EOS], [-1.0, -0.5])])
    actions = policy.generate("hi", samples=2, max_new_tokens=16, temperature=0.7, top_p=0.95)
    request = completions.requests[0]
    assert request["prompt"] == [ord(c) % 90 for c in "<u>hi"] and request["n"] == 2 and request["top_p"] == 0.95
    assert request["model"] == "sft" and request["extra_body"] == {"return_tokens_as_token_ids": True}
    assert [a.token_ids for a in actions] == [[5, EOS], [3, 4, EOS]]
    assert actions[0].token_logprobs == [-1.0, -0.5] and actions[0].text == "F"


def test_missing_eos_is_appended_and_length_cut_is_kept():
    policy, _, _ = make([choice(0, [3, 4], [-0.1, -0.2]), choice(1, [7, 8], [-0.3, -0.4], finish="length")])
    stopped, cut = policy.generate("q", samples=2, max_new_tokens=2, temperature=1.0, top_p=1.0)
    assert stopped.token_ids == [3, 4, EOS] and stopped.token_logprobs == []
    assert cut.token_ids == [7, 8] and cut.token_logprobs == [-0.3, -0.4]


def test_greedy_rejects_branching_and_sends_temperature_zero():
    policy, completions, _ = make([choice(0, [1, EOS], [-0.1, -0.1])])
    with pytest.raises(ValueError):
        policy.generate("q", samples=2, max_new_tokens=4, temperature=0.0, top_p=1.0)
    policy.generate("q", samples=1, max_new_tokens=4, temperature=0.0, top_p=1.0)
    assert completions.requests[-1]["temperature"] == 0.0 and "top_p" not in completions.requests[-1]


def test_update_swaps_names_unloads_previous_and_prunes_files(tmp_path):
    policy, completions, admin = make([choice(0, [1, EOS], [-0.1, -0.1])])
    dirs = [tmp_path / f"step-{i}" for i in range(3)]
    for i, path in enumerate(dirs):
        path.mkdir()
        policy.update(path, f"policy-{i}")
    assert admin.calls == [
        ("load_lora_adapter", {"lora_name": "policy-0", "lora_path": str(dirs[0].resolve())}),  # "sft" stays loaded
        ("load_lora_adapter", {"lora_name": "policy-1", "lora_path": str(dirs[1].resolve())}),
        ("unload_lora_adapter", {"lora_name": "policy-0"}),
        ("load_lora_adapter", {"lora_name": "policy-2", "lora_path": str(dirs[2].resolve())}),
        ("unload_lora_adapter", {"lora_name": "policy-1"}),
    ]
    assert [p.exists() for p in dirs] == [False, True, True]
    policy.generate("q", samples=1, max_new_tokens=4, temperature=0.0, top_p=1.0)
    assert completions.requests[-1]["model"] == "policy-2"


class Linear(torch.nn.Module):
    """Stand-in actor: log-probs are a trainable vector, independent of the prompt."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([-0.5, -1.0, -2.0]))


def test_backward_uses_own_logprobs_as_old_and_reports_sampler_gap(monkeypatch):
    from grasp_grpo import grpo

    actor = Linear()
    monkeypatch.setattr(grpo, "response_logprob_entropy", lambda model, tok, prompt, ids: (model.w[: len(ids)], torch.zeros(len(ids))))
    action = {"prompt": "p", "response_token_ids": [1, 2, 3], "old_token_logprobs": [-0.4, -1.0, -2.2]}
    stats = _backward(actor, None, [(action, 1.5)], [[-0.5, -1.0, -2.0]], 3, "cpu", clip_low=0.2, clip_high=0.28, kl_beta=0.0)
    assert stats["ratio"] == pytest.approx(1.0)
    assert stats["sampler_logprob_gap"] == pytest.approx(0.1)
    assert torch.allclose(actor.w.grad, torch.full((3,), -0.5))  # -advantage / tokens


def test_stage1_fails_when_server_samples_from_a_stale_adapter():
    good = {"updated": True, "loss": 0.1, "ratio": 1.0, "kl": 0.01, "grad_norm": 1.0, "sampler_logprob_gap": 0.03}
    stale = {**good, "sampler_logprob_gap": 0.9}
    assert gate_stage1([good] * 4, adapter_changed=True)[0]
    passed, reasons = gate_stage1([good, good, stale, stale], adapter_changed=True)
    assert not passed and "not sampling from the trained adapter" in reasons[0]
