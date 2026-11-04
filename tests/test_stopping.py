from grasp_grpo.stopping import EarlyStopper, StopConfig


def run_steps(stopper, n, **overrides):
    values = {"format_rate": 1.0, "kl": 0.01, "entropy": 1.0, "zero_signal": 0.1, "elapsed_hours": 0.0, **overrides}
    for _ in range(n):
        reason = stopper.after_step(**values)
        if reason:
            return reason
    return None


def test_healthy_training_never_stops():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 50) is None


def test_rules_wait_for_a_full_window():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 4, format_rate=0.0) is None


def test_format_collapse():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 5, format_rate=0.8) == "format_collapse"


def test_kl_drift():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 5, kl=0.5) == "kl_drift"


def test_entropy_collapse_is_relative_to_first_window():
    stopper = EarlyStopper(StopConfig(window=5), sft_f1=0.4)
    assert run_steps(stopper, 5, entropy=2.0) is None
    assert run_steps(stopper, 5, entropy=0.5) == "entropy_collapse"


def test_steps_without_update_do_not_feed_entropy():
    assert run_steps(EarlyStopper(StopConfig(window=3), sft_f1=0.4), 6, entropy=None) is None


def test_no_signal():
    assert run_steps(EarlyStopper(StopConfig(window=5), sft_f1=0.4), 5, zero_signal=0.9) == "no_signal"


def test_budget():
    stopper = EarlyStopper(StopConfig(window=5, max_hours=2.0), sft_f1=0.4)
    assert stopper.after_step(format_rate=1.0, kl=0.0, entropy=1.0, zero_signal=0.0, elapsed_hours=2.5) == "max_hours"


def test_no_improvement_after_patience():
    stopper = EarlyStopper(StopConfig(), sft_f1=0.40)
    assert stopper.after_eval(step=50, f1=0.41) == (None, True)
    assert stopper.after_eval(step=100, f1=0.415) == (None, True)
    assert stopper.after_eval(step=150, f1=0.40) == ("no_improvement", False)


def test_real_gain_resets_patience():
    stopper = EarlyStopper(StopConfig(), sft_f1=0.40)
    stopper.after_eval(step=50, f1=0.41)
    stopper.after_eval(step=100, f1=0.41)
    assert stopper.after_eval(step=150, f1=0.43) == (None, True)
    assert stopper.after_eval(step=200, f1=0.43) == (None, False)
    assert stopper.after_eval(step=250, f1=0.44) == (None, True)
    assert stopper.after_eval(step=300, f1=0.44)[0] == "no_improvement"
    assert (stopper.best_f1, stopper.best_step) == (0.44, 250)


def test_worse_than_sft_needs_consecutive_evals():
    stopper = EarlyStopper(StopConfig(patience_evals=10), sft_f1=0.40)
    assert stopper.after_eval(step=50, f1=0.36)[0] is None
    assert stopper.after_eval(step=100, f1=0.39)[0] is None
    assert stopper.after_eval(step=150, f1=0.365)[0] is None
    assert stopper.after_eval(step=200, f1=0.36)[0] == "worse_than_sft"


def test_baseline_can_be_set_after_construction():
    stopper = EarlyStopper(StopConfig(), sft_f1=None)
    stopper.set_baseline(0.30)
    assert stopper.after_eval(step=50, f1=0.26)[0] is None
    assert stopper.after_eval(step=100, f1=0.25)[0] == "worse_than_sft"
