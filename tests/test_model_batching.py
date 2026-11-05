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
    batched = batch_token_logprobs(tiny, ROWS, pad_token_id=0, chunk=2)
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
