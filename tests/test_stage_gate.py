from grasp_grpo.stage_gate import gate_stage0, gate_stage1, gate_stage2, gate_stage3


def test_stage0():
    ok = {"f1": 0.35, "signal_probe": {"zero_signal_fraction": 0.3, "seconds_per_tree": 12.0}}
    assert gate_stage0(ok) == (True, [])
    passed, reasons = gate_stage0({"f1": 0.2, "signal_probe": {"zero_signal_fraction": 0.7, "seconds_per_tree": 12.0}})
    assert not passed and len(reasons) == 2
    assert not gate_stage0({"f1": 0.35})[0]  # probe missing


def step(**kw):
    return {"updated": True, "loss": 0.01, "ratio": 1.0, "kl": 0.01, "grad_norm": 0.5, "mean_reward": 0.5, **kw}


def test_stage1():
    assert gate_stage1([step()] * 20, adapter_changed=True) == (True, [])
    assert not gate_stage1([step(loss=float("nan"))] + [step()] * 19, adapter_changed=True)[0]
    assert not gate_stage1([step()] * 20, adapter_changed=False)[0]
    assert not gate_stage1([step(grad_norm=0.0)] * 20, adapter_changed=True)[0]
    assert not gate_stage1([step(kl=0.0)] * 20, adapter_changed=True)[0]
    assert not gate_stage1([step(ratio=1.3)] * 20, adapter_changed=True)[0]
    assert not gate_stage1([{"updated": False}] * 20, adapter_changed=True)[0]


def test_stage2():
    rising = [step(mean_reward=0.4)] * 10 + [step(mean_reward=0.45)] * 20 + [step(mean_reward=0.6)] * 10
    assert gate_stage2(rising)[0]
    assert not gate_stage2([step(mean_reward=0.5)] * 40)[0]
    assert not gate_stage2([step()] * 5)[0]


def test_stage3():
    history = [{"step": 0, "f1": 0.40}, {"step": 50, "f1": 0.43}]
    assert gate_stage3({"stop_reason": None, "sft_f1": 0.40}, history)[0]
    assert gate_stage3({"stop_reason": "max_hours", "sft_f1": 0.40}, history)[0]
    assert not gate_stage3({"stop_reason": None, "sft_f1": 0.40}, [{"step": 50, "f1": 0.41}])[0]
    assert not gate_stage3({"stop_reason": "kl_drift", "sft_f1": 0.40}, history)[0]
