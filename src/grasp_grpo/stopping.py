"""Early-stopping rules of the GRPO protocol (spec §6). Pure logic: no I/O, no torch."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class StopConfig:
    min_improvement: float = 0.02  # eval F1 gain that counts as real (~2 paired standard errors on 300 questions)
    patience_evals: int = 3
    sft_drop: float = 0.03
    sft_drop_evals: int = 2
    min_format_rate: float = 0.90
    max_kl: float = 0.3
    min_entropy_ratio: float = 0.30
    max_zero_signal: float = 0.60
    window: int = 20
    max_hours: float | None = None


class EarlyStopper:
    def __init__(self, config: StopConfig, sft_f1: float | None):
        self.config = config
        self.sft_f1 = sft_f1
        self.reference_f1 = sft_f1  # the bar a real improvement must clear
        self.history: deque = deque(maxlen=config.window)
        self.initial_entropy: list[float] = []
        self.evals_without_gain = 0
        self.evals_below_sft = 0
        self.best_f1: float | None = None
        self.best_step: int | None = None

    def set_baseline(self, f1: float) -> None:
        self.sft_f1 = self.reference_f1 = f1

    def after_step(self, *, format_rate: float, kl: float, entropy: float | None, zero_signal: float, elapsed_hours: float) -> str | None:
        c = self.config
        if c.max_hours is not None and elapsed_hours > c.max_hours:
            return "max_hours"
        if entropy is not None and len(self.initial_entropy) < c.window:
            self.initial_entropy.append(entropy)
        self.history.append((format_rate, kl, entropy, zero_signal))
        if len(self.history) < c.window:
            return None

        def mean(index: int) -> float | None:
            values = [row[index] for row in self.history if row[index] is not None]
            return sum(values) / len(values) if values else None

        if mean(0) < c.min_format_rate:
            return "format_collapse"
        if mean(1) > c.max_kl:
            return "kl_drift"
        entropy_now = mean(2)
        if entropy_now is not None and len(self.initial_entropy) >= c.window:
            baseline = sum(self.initial_entropy) / len(self.initial_entropy)
            if baseline > 0 and entropy_now < c.min_entropy_ratio * baseline:
                return "entropy_collapse"
        if mean(3) > c.max_zero_signal:
            return "no_signal"
        return None

    def after_eval(self, *, step: int, f1: float) -> tuple[str | None, bool]:
        c = self.config
        is_best = self.best_f1 is None or f1 > self.best_f1
        if is_best:
            self.best_f1, self.best_step = f1, step
        if self.reference_f1 is None:
            self.reference_f1 = f1  # no baseline: the first eval sets the bar
        elif f1 >= self.reference_f1 + c.min_improvement:
            self.reference_f1, self.evals_without_gain = f1, 0
        else:
            self.evals_without_gain += 1
        if self.sft_f1 is not None and f1 <= self.sft_f1 - c.sft_drop:
            self.evals_below_sft += 1
        else:
            self.evals_below_sft = 0
        if self.evals_below_sft >= c.sft_drop_evals:
            return "worse_than_sft", is_best
        if self.evals_without_gain >= c.patience_evals:
            return "no_improvement", is_best
        return None, is_best
