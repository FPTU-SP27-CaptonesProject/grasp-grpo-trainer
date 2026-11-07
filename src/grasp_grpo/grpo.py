"""Token-level clipped GRPO objective (DAPO aggregation, clip-higher, k3 KL); no critic."""
from __future__ import annotations

import torch

from .model import (
    REFERENCE_ADAPTER,
    _prompt_ids,
    active_adapter,
    batch_token_logprobs,
    padded_logprobs_entropy,
    set_train_mode,
    token_batches,
)


def merge_actions(leaves: list[dict]) -> list[tuple[dict, float]]:
    """One entry per distinct action; an action shared by several leaves gets their summed advantage.

    Every action is scored once per rollout batch and followed by a single optimizer step, so all
    ratios start at 1 and summing the advantages equals scoring the action once per leaf.
    """
    merged: dict[int, list] = {}
    for leaf in leaves:
        for action in leaf["actions"]:
            entry = merged.setdefault(id(action), [action, 0.0])
            entry[1] += float(leaf["advantage"])
    return [(action, advantage) for action, advantage in merged.values()]


def token_loss(logp: torch.Tensor, old: torch.Tensor, ref: torch.Tensor, advantage: torch.Tensor, *, clip_low: float, clip_high: float, kl_beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Summed (not averaged) per-token loss of one action; the caller divides by the batch token count."""
    ratio = torch.exp(torch.clamp(logp - old, -20, 20))
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * advantage
    surrogate = torch.minimum(unclipped, clipped)
    log_ratio_ref = ref - logp
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1  # k3 estimator, >= 0
    loss = (-surrogate + kl_beta * kl).sum()
    with torch.no_grad():
        outside = ((ratio < 1 - clip_low) | (ratio > 1 + clip_high)).sum()
    return loss, {"ratio_sum": float(ratio.detach().sum()), "clipped": float(outside), "kl_sum": float(kl.detach().sum())}


def reference_logprobs(model, tokenizer, actions: list[dict]) -> list[list[float]]:
    """Per-token log P of every action under `model` (no gradient, batched)."""
    rows = [(_prompt_ids(tokenizer, action["prompt"]), list(action["response_token_ids"])) for action in actions]
    return batch_token_logprobs(model, rows, pad_token_id=tokenizer.pad_token_id)


def grpo_backward(actor, reference, tokenizer, leaves: list[dict], *, clip_low: float, clip_high: float, kl_beta: float, max_batch_tokens: int = 12288) -> dict[str, float]:
    """Backpropagate padded batches of actions (<= `max_batch_tokens` tokens each, so activation memory stays
    bounded); loss = sum over tokens / batch tokens.

    `reference=None`: the KL reference is the actor's frozen REFERENCE_ADAPTER, scored in one batched pass first.
    The backward runs in training mode without dropout (gradient checkpointing needs it); the actor is left in eval mode."""
    entries = merge_actions(leaves)
    if not entries:
        raise ValueError("tree batch has no proxy actions")
    total_tokens = sum(len(action["response_token_ids"]) for action, _ in entries)
    if total_tokens == 0:
        raise ValueError("tree batch has no response tokens")
    device = next(actor.parameters()).device
    actions = [action for action, _ in entries]
    if reference is None:
        with active_adapter(actor, REFERENCE_ADAPTER):
            refs = reference_logprobs(actor, tokenizer, actions)
    else:
        refs = reference_logprobs(reference, tokenizer, actions)
    set_train_mode(actor)
    try:
        return _backward(actor, tokenizer, entries, refs, total_tokens, device, clip_low=clip_low, clip_high=clip_high, kl_beta=kl_beta, max_batch_tokens=max_batch_tokens)
    finally:
        actor.eval()


def score_actions(actor, tokenizer, actions: list[dict], max_batch_tokens: int):
    """Yield (action indices, [(logp, entropy), ...]) per padded batch of actions with response tokens."""
    rows = [(_prompt_ids(tokenizer, action["prompt"]), list(action["response_token_ids"])) for action in actions]
    live = [i for i, (_, response) in enumerate(rows) if response]
    for batch in token_batches([len(rows[i][0]) + len(rows[i][1]) for i in live], max_batch_tokens):
        idx = [live[k] for k in batch]
        yield idx, padded_logprobs_entropy(actor, [rows[i] for i in idx], pad_token_id=tokenizer.pad_token_id)


def _backward(actor, tokenizer, entries, refs, total_tokens: int, device, *, clip_low: float, clip_high: float, kl_beta: float, max_batch_tokens: int = 12288) -> dict[str, float]:
    """Old log-probs = the loss's own detached log-probs. Valid because every rollout batch gets exactly one
    optimizer step, so the weights that sampled are the weights being scored (ratio 1, gradient = advantage x
    grad log p, the same as recomputing them at sampling time). The sampler's log-probs, when the action has
    them (vLLM), only feed `sampler_logprob_gap`: mean |HF - sampler| per token, a weight-sync check."""
    sums = {"loss": 0.0, "ratio": 0.0, "clipped": 0.0, "kl": 0.0, "entropy": 0.0}
    gap_sum, gap_tokens = 0.0, 0
    for idx, scored in score_actions(actor, tokenizer, [action for action, _ in entries], max_batch_tokens):
        batch_loss = 0.0
        for i, (logp, entropy) in zip(idx, scored, strict=True):
            action, advantage = entries[i]
            ref = torch.tensor(refs[i], device=device, dtype=logp.dtype)
            old = logp.detach()
            sampler = action.get("old_token_logprobs") or []
            if len(sampler) == len(action["response_token_ids"]):
                gap_sum += float((old - torch.tensor(sampler, device=device, dtype=old.dtype)).abs().sum())
                gap_tokens += len(sampler)
            loss, stats = token_loss(logp, old, ref, torch.tensor(advantage, device=device, dtype=logp.dtype), clip_low=clip_low, clip_high=clip_high, kl_beta=kl_beta)
            batch_loss = batch_loss + loss
            sums["loss"] += float(loss.detach())
            sums["ratio"] += stats["ratio_sum"]
            sums["clipped"] += stats["clipped"]
            sums["kl"] += stats["kl_sum"]
            sums["entropy"] += float(entropy.detach().sum())
        (batch_loss / total_tokens).backward()
    stats = {"loss": sums["loss"] / total_tokens, "ratio": sums["ratio"] / total_tokens, "clip_frac": sums["clipped"] / total_tokens, "kl": sums["kl"] / total_tokens, "entropy": sums["entropy"] / total_tokens, "tokens": float(total_tokens), "action_count": float(len(entries))}
    if gap_tokens:
        stats["sampler_logprob_gap"] = gap_sum / gap_tokens
    return stats
