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
