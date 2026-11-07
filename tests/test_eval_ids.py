import pytest

from grasp_grpo.eval_ids import filter_examples, select_eval_ids


def rows():
    return [{"id": f"{ds}{i}", "dataset": ds, "gold_doc_ids": ["g"] if i < 8 else []} for ds in ("a", "b") for i in range(10)]


def test_selection_is_per_dataset_deterministic_and_gold_only():
    first = select_eval_ids(rows(), per_dataset=3, seed=42)
    assert first == select_eval_ids(rows(), per_dataset=3, seed=42)
    assert len(first) == 6 and sum(x.startswith("a") for x in first) == 3
    assert not set(first) & {"a8", "a9", "b8", "b9"}


def test_excluded_ids_are_never_selected():
    chosen = select_eval_ids(rows(), per_dataset=7, seed=1, exclude={"a0"})
    assert "a0" not in chosen and len(chosen) == 14


def test_too_few_questions_raises():
    with pytest.raises(ValueError, match="only 8"):
        select_eval_ids(rows(), per_dataset=9, seed=0)


def test_filter_examples_keeps_order_and_reports_missing():
    examples = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    assert [e["id"] for e in filter_examples(examples, ["z", "x"])] == ["x", "z"]
    with pytest.raises(ValueError, match="missing"):
        filter_examples(examples, ["x", "nope"])


def test_interleave_datasets_round_robins_and_keeps_every_example():
    from grasp_grpo.policy_eval import interleave_datasets

    rows = [{"id": f"a::{i}"} for i in range(3)] + [{"id": "b::0", "dataset": "b"}] + [{"id": f"c::{i}"} for i in range(2)]
    assert [r["id"] for r in interleave_datasets(rows)] == ["a::0", "b::0", "c::0", "a::1", "c::1", "a::2"]
    assert interleave_datasets([]) == []
