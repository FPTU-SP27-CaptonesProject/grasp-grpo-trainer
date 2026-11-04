import pytest

from grasp_grpo.environment import BinaryTreeEnvironment
from grasp_grpo.model import Action
from grasp_grpo.reader import ReaderStep

EXAMPLE = {"id": "q", "question": "Question", "gold_answers": ["answer"], "gold_doc_ids": ["gold"]}
QF_TEXT = "Reasoning Chain: None\nKnowledge Gap: need evidence\nQuery: gold query"


class FakeActor:
    def eval(self):
        return self


class FakeReader:
    def __init__(self):
        self.calls = 0

    def step(self, question, context, thoughts, transcript):
        self.calls += 1
        if self.calls == 1:
            return ReaderStep("<think> Need evidence. </think>\n", "Need evidence.", None)
        return ReaderStep("<answer>answer</answer>", None, "answer")

    def final(self, question, context, thoughts, transcript):
        return "answer"


class FakeRetriever:
    def search(self, query):
        if query == "gold query":
            return [{"doc_id": "gold", "title": "Gold", "text": "evidence", "triples": [{"doc_id": "gold", "head": "Gold", "relation": "answers", "tail": "answer"}]}]
        return [{"doc_id": "other", "title": "Other", "text": "noise", "triples": []}]

    def candidate_triples(self, _query, passages):
        return [triple for passage in passages for triple in passage["triples"]]


def make_environment(monkeypatch, rd_texts=("Yes (retrieval is required)",) * 2, **overrides):
    def fake_generate(_actor, _tokenizer, prompt, **_kwargs):
        if "retrieval decision assistant" in prompt:
            return [Action(text, [1], [-0.1]) for text in rd_texts]
        text = QF_TEXT if "query formulation assistant" in prompt else "<Gold; answers; answer>"
        return [Action(text, [1], [-0.1]), Action(text, [1], [-0.1])]

    monkeypatch.setattr("grasp_grpo.environment.generate", fake_generate)
    settings = {"branch_factor": 2, "max_steps": 1, "max_leaves": 2, "temperature": 0.7, "top_p": 0.95, "alpha": 0.7, "retrieval_cost": 0.0, "behavior_checkpoint": "test", **overrides}
    return BinaryTreeEnvironment(FakeActor(), object(), FakeReader(), FakeRetriever(), **settings)


def test_gold_evidence_adds_auditable_rd_qf_and_ki_rewards(monkeypatch):
    tree = make_environment(monkeypatch).run(EXAMPLE)
    leaf = tree["leaves"][0]
    assert leaf["reward"]["answer_f1"] == 1.0
    assert leaf["reward"]["context_recall"] == 1.0
    assert leaf["reward"]["action_shaping"] == pytest.approx(1.3)
    assert leaf["reward"]["total"] == pytest.approx(2.3)
    shaped = [action["reward_shaping"]["total"] for action in leaf["actions"]]
    assert shaped == [0.3, 0.7, 0.3]


def test_shaping_scale_and_identical_branches_give_no_signal(monkeypatch):
    tree = make_environment(monkeypatch, shaping_scale=0.1).run(EXAMPLE)
    assert tree["leaves"][0]["reward"]["total"] == pytest.approx(1.0 + 0.13)
    assert tree["zero_signal"] is True
    assert all(leaf["advantage"] == 0.0 for leaf in tree["leaves"])


def test_advantage_is_reward_minus_tree_mean(monkeypatch):
    rd = ("Yes (retrieval is required)", "No (retrieval is not required)")
    tree = make_environment(monkeypatch, rd_texts=rd).run(EXAMPLE)
    by_choice = {leaf["actions"][0]["response"]: leaf["advantage"] for leaf in tree["leaves"]}
    assert by_choice == {rd[0]: pytest.approx(1.0), rd[1]: pytest.approx(-1.0)}
    assert tree["zero_signal"] is False


def test_std_normalised_advantage_is_optional(monkeypatch):
    rd = ("Yes (retrieval is required)", "No (retrieval is not required)")
    tree = make_environment(monkeypatch, rd_texts=rd, advantage_std_norm=True).run(EXAMPLE)
    assert sorted(round(leaf["advantage"], 4) for leaf in tree["leaves"]) == [-1.0, 1.0]


def test_qf_and_ki_draw_one_sample_through_generate_fn(monkeypatch):
    calls = []

    def generate_fn(role, prompt, samples):
        calls.append((role, samples))
        text = {"rd": "Yes (retrieval is required)", "qf": QF_TEXT, "ki": "<Gold; answers; answer>"}[role]
        return [Action(text, [1], [-0.1]) for _ in range(samples)]

    make_environment(monkeypatch, generate_fn=generate_fn).run(EXAMPLE)
    assert calls == [("rd", 2), ("qf", 1), ("ki", 1), ("qf", 1), ("ki", 1)]
