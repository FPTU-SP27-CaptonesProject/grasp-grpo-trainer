from grasp_grpo.infer import run_example
from grasp_grpo.model import Action
from grasp_grpo.policy_eval import signal_probe, summarize
from grasp_grpo.reader import ReaderStep


class Reader:
    def __init__(self):
        self.calls = 0

    def step(self, question, context, thoughts, transcript):
        self.calls += 1
        if self.calls == 1:
            return ReaderStep("<think> need </think>\n", "need", None)
        return ReaderStep("<answer> answer </answer>", None, "answer")

    def final(self, *args):
        raise AssertionError("the reader answered before max_steps")


class Retriever:
    def search(self, query):
        return [{"doc_id": "gold" if query == "gold query" else "q", "title": "", "text": ""}]

    def candidate_triples(self, query, passages, question=None):
        return [{"doc_id": p["doc_id"], "head": "H", "relation": "r", "tail": "T"} for p in passages]


def generate_fn(role, prompt, samples):
    text = {"rd": "Yes (retrieval is required)", "qf": "Reasoning Chain: None\nKnowledge Gap: need\nQuery: gold query", "ki": "<H; r; T>"}[role]
    return [Action(text, [1], [0.0])]


def test_run_example_follows_reader_and_proxy():
    example = {"id": "q1", "question": "Q?", "gold_answers": ["answer"], "gold_doc_ids": ["gold"], "metadata": {}}
    result = run_example(generate_fn, Reader(), Retriever(), example, max_steps=5)
    assert result["prediction"] == "answer"
    assert result["context_doc_ids"] == ["q", "gold"]
    assert [a["role"] for a in result["actions"]] == ["rd", "qf", "ki"]


def test_summarize_reports_quality_format_and_speed():
    examples = [{"id": "a", "gold_answers": ["x"], "gold_doc_ids": ["d1", "d2"]}, {"id": "b", "gold_answers": ["y"], "gold_doc_ids": []}]
    predictions = [
        {"id": "a", "prediction": "x", "context_doc_ids": ["d1"], "actions": [{"role": "rd", "valid_format": True}, {"role": "qf", "valid_format": False}]},
        {"id": "b", "prediction": "z", "context_doc_ids": [], "actions": [{"role": "rd", "valid_format": True}]},
    ]
    report = summarize(examples, predictions, elapsed_seconds=10.0)
    assert report["f1"] == 0.5 and report["em"] == 0.5
    assert report["context_recall"] == 0.5
    assert report["format_rate"] == {"rd": 1.0, "qf": 0.0, "ki": None}
    assert report["mean_retrievals"] == 0.5
    assert report["seconds_per_question"] == 5.0


def test_signal_probe_counts_zero_signal_trees():
    trees = iter([
        {"zero_signal": True, "leaves": [{"actions": [1]}]},
        {"zero_signal": False, "leaves": [{"actions": [1]}]},
        {"zero_signal": True, "leaves": [{"actions": []}]},  # reader answered directly: ignored
    ])

    class Env:
        def __init__(self, tree):
            self.tree = tree

        def run(self, example):
            return self.tree

    probe = signal_probe(lambda example: Env(next(trees)), [{}, {}, {}], workers=1)
    assert probe["trees"] == 3 and probe["action_trees"] == 2
    assert probe["zero_signal_fraction"] == 0.5
