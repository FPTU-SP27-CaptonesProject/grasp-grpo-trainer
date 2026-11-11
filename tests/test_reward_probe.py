"""The advantage split adds up, and a better answer with a lower advantage is counted."""
from grasp_grpo.reward_probe import analyse


def leaf(f1, rec, sh):
    return {"f1_part": 0.7 * f1, "recall_part": 0.3 * rec, "shaping_part": sh, "answer_f1": f1, "total": 0.7 * f1 + 0.3 * rec + sh,
            "terminal": "reader_answer"}


def tree(*leaves):
    leaves = list(leaves)
    mean = sum(x["total"] for x in leaves) / len(leaves)
    for x in leaves:
        x["advantage"] = x["total"] - mean
    return {"leaves": leaves, "zero_signal": all(abs(x["advantage"]) < 1e-6 for x in leaves)}


def test_shares_add_up_and_flips_are_counted():
    trees = [
        tree(leaf(1.0, 0.5, 0.0), leaf(0.0, 0.5, 0.0)),  # only F1 differs
        tree(leaf(0.5, 1.0, 0.0), leaf(1.0, 0.0, 0.0)),  # F1 and recall pull opposite ways
        tree(leaf(0.0, 0.0, 0.0), leaf(0.0, 0.0, 0.0)),  # no signal
    ]
    s = analyse(trees)
    assert s["signal_trees"] == 2 and s["zero_signal_trees"] == 1
    assert abs(sum(s["advantage_variance_share"].values()) - 1.0) < 1e-6
    # tree 2: totals 0.65 vs 0.70 -> the F1=1.0 leaf has the higher total; no flip there
    assert s["leaf_pairs_with_different_f1"] == 2 and s["pairs_where_better_answer_gets_lower_advantage"] == 0
    flipped = analyse([tree(leaf(0.5, 1.0, 0.1), leaf(0.6, 0.0, 0.0))])
    assert flipped["pairs_where_better_answer_gets_lower_advantage"] == 1
