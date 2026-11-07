"""CPU checks of the policy utilities on a tiny random Qwen2 (no download)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from grasp_grpo.model import (
    _token_logprobs_entropy,
    active_adapter,
    batch_token_logprobs,
    set_train_mode,
)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    base = transformers.Qwen2ForCausalLM(config).eval()
    lora = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.1, "init_lora_weights": False}
    model = peft.get_peft_model(base, peft.LoraConfig(**lora))
    model.add_adapter("reference", peft.LoraConfig(**lora, inference_mode=True))
    model.eval()
    return model


ROWS = [([5, 6, 7, 8, 9], [10, 11, 12]), ([3, 4], [20]), ([1, 2, 3, 4, 5, 6, 7, 8], [30, 31, 32, 33, 34]), ([7], [])]


def test_batched_logprobs_equal_one_row_at_a_time(tiny):
    batched = batch_token_logprobs(tiny, ROWS, pad_token_id=0, max_tokens=12)  # several padded batches
    for (prefix, response), got in zip(ROWS, batched, strict=True):
        if not response:
            assert got == []
            continue
        with torch.no_grad():
            single, _ = _token_logprobs_entropy(tiny, prefix, response)
        assert got == pytest.approx(single.tolist(), abs=1e-5)


def test_active_adapter_switches_and_restores_trainable_flags(tiny):
    before = {name: p.requires_grad for name, p in tiny.named_parameters()}
    assert any(flag for name, flag in before.items() if ".default." in name)
    prefix, response = ROWS[0]
    with torch.no_grad():
        actor, _ = _token_logprobs_entropy(tiny, prefix, response)
        with active_adapter(tiny, "reference"):
            reference, _ = _token_logprobs_entropy(tiny, prefix, response)
    assert not torch.allclose(actor, reference)
    assert tiny.active_adapter == "default"
    assert {name: p.requires_grad for name, p in tiny.named_parameters()} == before


def test_train_mode_keeps_dropout_off_with_checkpointing(tiny):
    tiny.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prefix, response = ROWS[2]
    with torch.no_grad():
        eval_logp, _ = _token_logprobs_entropy(tiny, prefix, response)
    set_train_mode(tiny)
    try:
        logp, _ = _token_logprobs_entropy(tiny, prefix, response)
        assert torch.allclose(logp.detach(), eval_logp, atol=1e-6)  # no dropout: ratio starts at 1
        logp.sum().backward()
        grads = [p.grad for name, p in tiny.named_parameters() if ".default." in name and p.requires_grad]
        assert grads and all(g is not None for g in grads)
    finally:
        tiny.eval()
        tiny.zero_grad(set_to_none=True)
        tiny.base_model.model.gradient_checkpointing_disable()


def test_token_batches_respect_budget_and_cover_every_row():
    from grasp_grpo.model import token_batches

    lengths = [5, 50, 7, 30, 30, 200, 6]
    batches = token_batches(lengths, max_tokens=64)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) == 1 or len(batch) * max(lengths[i] for i in batch) <= 64


def test_batched_backward_matches_one_action_at_a_time(tiny):
    from types import SimpleNamespace

    from grasp_grpo import grpo

    rows = [r for r in ROWS if r[1]]
    actions = [{"prompt": str(i), "response_token_ids": response} for i, (_, response) in enumerate(rows)]
    entries = [(action, adv) for action, adv in zip(actions, [1.0, -0.5, 0.7], strict=True)]
    refs = batch_token_logprobs(tiny, rows, pad_token_id=0)
    total = sum(len(r) for _, r in rows)
    tok = SimpleNamespace(pad_token_id=0)
    prompts = {str(i): prefix for i, (prefix, _) in enumerate(rows)}
    original = grpo._prompt_ids
    grpo._prompt_ids = lambda tokenizer, prompt: prompts[prompt]
    try:
        grads = []
        for budget in (1, 10_000):  # one action per pass vs. all actions in one padded pass
            tiny.zero_grad(set_to_none=True)
            stats = grpo._backward(tiny, tok, entries, refs, total, "cpu", clip_low=0.2, clip_high=0.28, kl_beta=0.01, max_batch_tokens=budget)
            grads.append({n: p.grad.clone() for n, p in tiny.named_parameters() if p.grad is not None})
            assert stats["ratio"] == pytest.approx(1.0)
    finally:
        grpo._prompt_ids = original
        tiny.zero_grad(set_to_none=True)
    assert grads[0].keys() == grads[1].keys() and grads[0]
    for name in grads[0]:
        assert torch.allclose(grads[0][name], grads[1][name], atol=1e-5), name
