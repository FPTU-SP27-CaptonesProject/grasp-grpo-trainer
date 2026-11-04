"""Token-level clipped GRPO objective (DAPO aggregation, clip-higher, k3 KL); no critic."""
from __future__ import annotations

import torch

from .model import response_logprob_entropy


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


def grpo_backward(actor, reference, tokenizer, leaves: list[dict], *, clip_low: float, clip_high: float, kl_beta: float) -> dict[str, float]:
    """Backpropagate one action at a time (bounded activation memory); loss = sum over tokens / batch tokens."""
    entries = merge_actions(leaves)
    if not entries:
        raise ValueError("tree batch has no proxy actions")
    total_tokens = sum(len(action["response_token_ids"]) for action, _ in entries)
    if total_tokens == 0:
        raise ValueError("tree batch has no response tokens")
    device = next(actor.parameters()).device
    sums = {"loss": 0.0, "ratio": 0.0, "clipped": 0.0, "kl": 0.0, "entropy": 0.0}
    for action, advantage in entries:
        logp, entropy = response_logprob_entropy(actor, tokenizer, action["prompt"], action["response_token_ids"])
        with torch.no_grad():
            ref, _ = response_logprob_entropy(reference, tokenizer, action["prompt"], action["response_token_ids"])
        old = torch.tensor(action["old_token_logprobs"], device=device, dtype=logp.dtype)
        loss, stats = token_loss(logp, old, ref, torch.tensor(advantage, device=device, dtype=logp.dtype), clip_low=clip_low, clip_high=clip_high, kl_beta=kl_beta)
        (loss / total_tokens).backward()
        sums["loss"] += float(loss.detach())
        sums["ratio"] += stats["ratio_sum"]
        sums["clipped"] += stats["clipped"]
        sums["kl"] += stats["kl_sum"]
        sums["entropy"] += float(entropy.detach().sum())
    return {"loss": sums["loss"] / total_tokens, "ratio": sums["ratio"] / total_tokens, "clip_frac": sums["clipped"] / total_tokens, "kl": sums["kl"] / total_tokens, "entropy": sums["entropy"] / total_tokens, "tokens": float(total_tokens), "action_count": float(len(entries))}
