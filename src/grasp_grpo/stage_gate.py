"""Pass/fail gates between the stages of the GRPO protocol (spec §5)."""
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import write_json


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def gate_stage0(report: dict[str, Any], *, min_f1: float = 0.30, max_zero_signal: float = 0.50, max_seconds: float = 20.0) -> tuple[bool, list[str]]:
    reasons = []
    if report["f1"] < min_f1:
        reasons.append(f"SFT F1 {report['f1']:.3f} < {min_f1}: fix the environment before training")
    probe = report.get("signal_probe")
    if not probe:
        reasons.append("signal probe missing: rerun grasp-eval-policy with --signal-probe")
    else:
        if probe["zero_signal_fraction"] > max_zero_signal:
            reasons.append(f"zero-signal trees {probe['zero_signal_fraction']:.0%} > {max_zero_signal:.0%}")
        if probe["seconds_per_tree"] is not None and probe["seconds_per_tree"] > max_seconds:
            reasons.append(f"{probe['seconds_per_tree']:.1f} s per tree > {max_seconds}: recompute the pilot cost")
    return not reasons, reasons


def gate_stage1(metrics: list[dict[str, Any]], *, adapter_changed: bool, ratio_range: tuple[float, float] = (0.9, 1.1), max_kl: float = 0.05) -> tuple[bool, list[str]]:
    updated = [m for m in metrics if m.get("updated")]
    if not updated:
        return False, ["no step produced an update"]
    reasons = []
    if any(not math.isfinite(m["loss"]) for m in updated):
        reasons.append("non-finite loss")
    ratio = _mean([m["ratio"] for m in updated])
    if not ratio_range[0] <= ratio <= ratio_range[1]:
        reasons.append(f"mean ratio {ratio:.3f} outside {ratio_range}")
    kl = _mean([m["kl"] for m in updated])
    if not 0 < kl < max_kl:
        reasons.append(f"mean KL {kl:.4f} not in (0, {max_kl})")
    if any(m["grad_norm"] <= 0 for m in updated):
        reasons.append("zero gradient norm")
    if not adapter_changed:
        reasons.append("LoRA weights did not change")
    return not reasons, reasons


def gate_stage2(metrics: list[dict[str, Any]], *, window: int = 10, min_gain: float = 0.30) -> tuple[bool, list[str]]:
    rewards = [m["mean_reward"] for m in metrics if m.get("mean_reward") is not None]
    if len(rewards) < 2 * window:
        return False, [f"need {2 * window} steps with rewards, got {len(rewards)}"]
    first, last = _mean(rewards[:window]), _mean(rewards[-window:])
    target = first * (1 + min_gain) if first > 0 else first + 0.05
    if last < target:
        return False, [f"reward {first:.3f} -> {last:.3f}, needed >= {target:.3f}: the pipeline cannot fit 8 questions"]
    return True, []


def gate_stage3(manifest: dict[str, Any], eval_history: list[dict[str, Any]], *, min_improvement: float = 0.02) -> tuple[bool, list[str]]:
    reasons = []
    if manifest.get("stop_reason") not in (None, "max_hours"):
        reasons.append(f"stopped early: {manifest['stop_reason']}")
    sft = manifest.get("sft_f1")
    scores = [h["f1"] for h in eval_history if h.get("step", 0) > 0]
    if sft is None or not scores:
        reasons.append("missing SFT baseline or eval history")
    elif max(scores) < sft + min_improvement - 1e-9:
        reasons.append(f"best eval F1 {max(scores):.3f} < SFT {sft:.3f} + {min_improvement}")
    return not reasons, reasons


def adapter_changed(before_dir: str | Path, after_dir: str | Path) -> bool:
    import torch
    from safetensors.torch import load_file

    before = load_file(str(Path(before_dir) / "adapter_model.safetensors"))
    after = load_file(str(Path(after_dir) / "adapter_model.safetensors"))
    shared = set(before) & set(after)
    if not shared:
        raise ValueError("the two adapters share no tensor names")
    return any(not torch.equal(before[name], after[name]) for name in shared)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Decide whether a GRPO protocol stage passed")
    parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sft-checkpoint", help="Required for stage 1 (adapter change check)")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    if args.stage == 0:
        passed, reasons = gate_stage0(_load(run / "stage0_grasp.json"))
    elif args.stage == 1:
        if not args.sft_checkpoint:
            parser.error("--sft-checkpoint is required for stage 1")
        checkpoints = sorted(run.glob("checkpoint-grpo-step-*"))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoint-grpo-step-* in {run}")
        passed, reasons = gate_stage1(_load(run / "metrics.json"), adapter_changed=adapter_changed(args.sft_checkpoint, checkpoints[-1]))
    elif args.stage == 2:
        passed, reasons = gate_stage2(_load(run / "metrics.json"))
    else:
        history = _load(run / "eval_history.json") if (run / "eval_history.json").exists() else []
        passed, reasons = gate_stage3(_load(run / "run_manifest.json"), history)
    report = {"stage": args.stage, "passed": passed, "reasons": reasons, "run_dir": str(run.resolve()), "checked_at_utc": datetime.now(UTC).isoformat()}
    write_json(run / "stage_report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
