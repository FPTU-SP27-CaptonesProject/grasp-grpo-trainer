"""Trainable PEFT actor, frozen reference, and token-level policy utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Action:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]


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


def generate(model, tokenizer, prompt: str, *, samples: int, max_new_tokens: int, temperature: float, top_p: float) -> list[Action]:
    import torch

    if samples > 1 and temperature <= 0:
        raise ValueError("multiple tree branches require temperature > 0")
    prefix = _prompt_ids(tokenizer, prompt)
    device = next(model.parameters()).device
    inputs = {"input_ids": torch.tensor([prefix], device=device), "attention_mask": torch.ones((1, len(prefix)), device=device, dtype=torch.long)}
    generation = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens, "num_return_sequences": samples, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id, "return_dict_in_generate": True, "output_scores": True}
    if temperature > 0:
        generation.update(temperature=temperature, top_p=top_p)
    with torch.no_grad():
        output = model.generate(**inputs, **generation)
        transition = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
    width = inputs["input_ids"].shape[1]
    output_actions = []
    for i, ids in enumerate(output.sequences[:, width:]):
        scores = transition[i, : ids.shape[0]]
        token_ids = ids.detach().cpu().tolist()
        eos = tokenizer.eos_token_id
        if eos in token_ids:
            end = token_ids.index(eos) + 1
            token_ids, scores = token_ids[:end], scores[:end]
        output_actions.append(Action(tokenizer.decode(token_ids, skip_special_tokens=True).strip(), token_ids, scores.detach().float().cpu().tolist()))
    return output_actions


def response_logprob(model, tokenizer, prompt: str, response_ids: list[int]):
    """Differentiable sum log P(response tokens | exact Project A chat prompt)."""
    import torch
    import torch.nn.functional as F

    prefix = _prompt_ids(tokenizer, prompt)
    device = next(model.parameters()).device
    ids = torch.tensor([prefix + response_ids], device=device)
    logits = model(input_ids=ids).logits[:, :-1]
    targets = ids[:, 1:]
    start = len(prefix) - 1
    selected = F.log_softmax(logits[:, start : start + len(response_ids)], dim=-1).gather(-1, targets[:, start : start + len(response_ids)].unsqueeze(-1)).squeeze(-1)
    return selected.sum(), selected
