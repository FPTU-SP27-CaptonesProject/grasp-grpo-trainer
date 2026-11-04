import argparse

import pytest

from grasp_grpo.train import lr_at, parse_args, public_config, tree_stats


def test_warmup_is_linear_then_constant():
    assert lr_at(1, 1e-5, 20) == pytest.approx(5e-7)
    assert lr_at(20, 1e-5, 20) == pytest.approx(1e-5)
    assert lr_at(50, 1e-5, 20) == pytest.approx(1e-5)
    assert lr_at(3, 1e-5, 0) == pytest.approx(1e-5)


def test_tree_stats_counts_each_shared_action_once():
    shared = {"role": "rd", "valid_format": True}
    leaves = [
        {"actions": [shared, {"role": "qf", "valid_format": False}], "reward": {"total": 1.0, "answer_f1": 1.0, "context_recall": None}},
        {"actions": [shared], "reward": {"total": 0.0, "answer_f1": 0.0, "context_recall": 1.0}},
    ]
    stats = tree_stats([{"leaves": leaves}])
    assert stats["format_rate"] == 0.5
    assert (stats["format_rate_rd"], stats["format_rate_qf"], stats["format_rate_ki"]) == (1.0, 0.0, None)
    assert stats["mean_reward"] == 0.5 and stats["mean_context_recall"] == 0.5 and stats["leaves"] == 2


def test_tree_stats_without_trees():
    stats = tree_stats([])
    assert stats["format_rate"] == 1.0 and stats["mean_reward"] is None


def test_public_config_drops_secrets():
    assert public_config(argparse.Namespace(reader_api_key="secret", learning_rate=1e-5)) == {"learning_rate": 1e-5}


def test_pilot_defaults_match_the_spec():
    args = parse_args(["--sft-checkpoint", "s", "--qa", "q", "--corpus-config", "c", "--reader-model", "m", "--output-dir", "o"])
    assert (args.learning_rate, args.questions_per_iteration, args.max_leaves, args.max_steps) == (1e-5, 16, 4, 5)
    assert (args.alpha, args.shaping_scale, args.kl_beta, args.warmup_steps) == (0.3, 0.1, 0.002, 20)
    assert (args.clip_low, args.clip_high, args.eval_every, args.reader_protocol) == (0.2, 0.28, 50, "grasp")
