"""Token-level clipped GRPO objective; no critic/value model is used."""
from __future__ import annotations

import torch

from .model import response_logprob


def grpo_loss(actor, reference, tokenizer, leaves: list[dict], *, clip_epsilon: float, kl_beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    losses, ratios, kls = [], [], []
    for leaf in leaves:
        advantage = torch.tensor(float(leaf["advantage"]), device=next(actor.parameters()).device)
        for action in leaf["actions"]:
            old = torch.tensor(float(action["old_logprob_sum"]), device=advantage.device)
            current, _ = response_logprob(actor, tokenizer, action["prompt"], action["response_token_ids"])
            with torch.no_grad():
                ref, _ = response_logprob(reference, tokenizer, action["prompt"], action["response_token_ids"])
            ratio = torch.exp(torch.clamp(current - old, -20, 20))
            surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage)
            # Sampled-action KL estimator, bounded below for numerical stability.
            kl = torch.clamp(current - ref, min=0.0)
            losses.append(-surrogate + kl_beta * kl)
            ratios.append(ratio.detach())
            kls.append(kl.detach())
    if not losses:
        raise ValueError("tree batch has no proxy actions")
    loss = torch.stack(losses).mean()
    return loss, {"loss": float(loss.detach()), "ratio": float(torch.stack(ratios).mean()), "kl": float(torch.stack(kls).mean()), "action_count": float(len(losses))}
