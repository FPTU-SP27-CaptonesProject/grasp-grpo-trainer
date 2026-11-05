"""Trainable PEFT actor, frozen reference, and token-level policy utilities."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Name of the frozen SFT adapter loaded next to the trainable one (one shared base model for both).
REFERENCE_ADAPTER = "reference"


@dataclass
class Action:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]


def _token_logprobs_entropy(model, prefix: list[int], response_ids: list[int]):
    """Float32 log P(token) and entropy of each response position, under raw logits (no sampling warpers).

    Only the response positions get logits (`logits_to_keep`): a 3k-token prompt would otherwise
    materialise a 3k x 152k logit matrix per action.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    ids = torch.tensor([prefix + response_ids], device=device)
    keep = len(response_ids) + 1
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits[0, :-1]
    window = F.log_softmax(logits.float(), dim=-1)
    targets = torch.tensor(response_ids, device=device, dtype=torch.long)
    selected = window.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(window.exp() * window).sum(-1)
    return selected, entropy


def batch_token_logprobs(model, rows: list[tuple[list[int], list[int]]], *, pad_token_id: int, chunk: int = 8) -> list[list[float]]:
    """log P(response token) for many (prefix, response) rows, `chunk` rows per left-padded forward.

    Same quantity as `_token_logprobs_entropy` (no gradient), without one forward pass per row.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    out: list[list[float]] = [[] for _ in rows]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    with torch.no_grad():
        for start in range(0, len(order), chunk):
            idx = [i for i in order[start : start + chunk] if rows[i][1]]
            if not idx:
                continue
            seqs = [rows[i][0] + rows[i][1] for i in idx]
            width = max(len(seq) for seq in seqs)
            input_ids = torch.tensor([[pad_token_id] * (width - len(seq)) + seq for seq in seqs], device=device)
            attention = torch.tensor([[0] * (width - len(seq)) + [1] * len(seq) for seq in seqs], device=device, dtype=torch.long)
            positions = (attention.cumsum(-1) - 1).clamp(min=0)
            keep = max(len(rows[i][1]) for i in idx) + 1
            logits = model(input_ids=input_ids, attention_mask=attention, position_ids=positions, use_cache=False, logits_to_keep=keep).logits
            for j, i in enumerate(idx):
                response = rows[i][1]
                window = F.log_softmax(logits[j, keep - 1 - len(response) : keep - 1].float(), dim=-1)
                targets = torch.tensor(response, device=device, dtype=torch.long)
                out[i] = window.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu().tolist()
    return out


@contextmanager
def active_adapter(model, name: str):
    """Temporarily run `model` with adapter `name`; restores the previous adapter and every requires_grad flag
    (PEFT's set_adapter marks the selected adapter trainable)."""
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    previous = model.active_adapter
    model.set_adapter(name)
    try:
        yield model
    finally:
        model.set_adapter(previous)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def set_train_mode(model) -> None:
    """Training mode for the backward pass (gradient checkpointing only runs when module.training is set)
    with dropout kept off, so the loss scores actions with the same policy that sampled them."""
    import torch

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def load_policy(checkpoint: str, *, trainable: bool, load_in_4bit: bool = False, reference_checkpoint: str | None = None):
    """Base model + PEFT adapter. With `reference_checkpoint`, that adapter is also loaded, frozen, as
    REFERENCE_ADAPTER on the same base, so the KL reference costs no second copy of the base model."""
    import torch
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
        base = prepare_model_for_kbit_training(base)  # also enables (reentrant) gradient checkpointing
    elif trainable:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, root, is_trainable=trainable)
    if reference_checkpoint is not None:
        reference = Path(reference_checkpoint).expanduser().resolve()
        if not (reference / "adapter_config.json").is_file():
            raise ValueError("reference checkpoint must be a local PEFT adapter directory")
        if PeftConfig.from_pretrained(reference).base_model_name_or_path != config.base_model_name_or_path:
            raise ValueError("actor and reference adapters must share the base model")
        model.load_adapter(str(reference), adapter_name=REFERENCE_ADAPTER, is_trainable=False)
    tokenizer = load_tokenizer(checkpoint)
    if torch.cuda.is_available() and not load_in_4bit:
        model.to("cuda")
    model.train(trainable)
    if not trainable:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, tokenizer, str(config.base_model_name_or_path)


def load_tokenizer(checkpoint: str):
    """The adapter's tokenizer (or its base model's), left-padded, as every prompt is rendered with."""
    from peft import PeftConfig
    from transformers import AutoTokenizer

    root = Path(checkpoint).expanduser().resolve()
    source = root if (root / "tokenizer_config.json").exists() else PeftConfig.from_pretrained(root).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


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
    eos = tokenizer.eos_token_id
    results: list[list[Action]] = [[] for _ in prompts]
    for row, ids in enumerate(sequences):
        token_ids = ids.detach().cpu().tolist()
        if eos in token_ids:
            token_ids = token_ids[: token_ids.index(eos) + 1]
        # No behaviour log-probs: one update per rollout batch means the loss's own detached log-probs are
        # exactly the sampling policy's (grpo_backward).
        results[row // samples].append(Action(tokenizer.decode(token_ids, skip_special_tokens=True).strip(), token_ids, []))
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
