import pytest

from grasp_grpo.environment import BinaryTreeEnvironment
from grasp_grpo.model import Action


class FakeActor:
    def eval(self):
        return self


class FakeReader:
    def __init__(self):
        self.calls = 0

    def generate(self, _prompt):
        self.calls += 1
        return "<think>Need evidence.</think>" if self.calls == 1 else "<answer>answer</answer>"


class FakeRetriever:
    def search(self, query):
        if query == "gold query":
            return [{"doc_id": "gold", "title": "Gold", "text": "evidence", "triples": [{"doc_id": "gold", "head": "Gold", "relation": "answers", "tail": "answer"}]}]
        return [{"doc_id": "other", "title": "Other", "text": "noise", "triples": []}]

    def candidate_triples(self, _query, passages):
        return [triple for passage in passages for triple in passage["triples"]]


def test_gold_evidence_adds_auditable_rd_qf_and_ki_rewards(monkeypatch):
    def fake_generate(_actor, _tokenizer, prompt, **_kwargs):
        if "retrieval decision assistant" in prompt:
            text = "Yes (retrieval is required)"
        elif "query formulation assistant" in prompt:
            text = "Reasoning Chain: None\nKnowledge Gap: need evidence\nQuery: gold query"
        else:
            text = "<Gold; answers; answer>"
        return [Action(text, [1], [-0.1]), Action(text, [1], [-0.1])]

    monkeypatch.setattr("grasp_grpo.environment.generate", fake_generate)
    environment = BinaryTreeEnvironment(
        FakeActor(), object(), FakeReader(), FakeRetriever(), branch_factor=2,
        max_steps=1, max_leaves=2, temperature=0.7, top_p=0.95, alpha=0.7,
        retrieval_cost=0.0, behavior_checkpoint="test",
    )
    tree = environment.run({"id": "q", "question": "Question", "gold_answers": ["answer"], "gold_doc_ids": ["gold"]})

    leaf = tree["leaves"][0]
    assert leaf["reward"]["answer_f1"] == 1.0
    assert leaf["reward"]["context_recall"] == 1.0
    assert leaf["reward"]["action_shaping"] == pytest.approx(1.3)
    assert leaf["reward"]["total"] == pytest.approx(2.3)
    shaped = [action["reward_shaping"]["total"] for action in leaf["actions"]]
    assert shaped == [0.3, 0.7, 0.3]
