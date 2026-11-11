"""Report baselines: LLM-only, single-step RAG, IRCoT, and the chat reader for a hosted model."""
import json

from grasp_grpo import baselines
from grasp_grpo.policy_eval import summarize
from grasp_grpo.reader import ReaderStep

EXAMPLE = {"id": "q1", "question": "Q?", "gold_answers": ["answer"], "gold_doc_ids": ["g1", "g2"], "metadata": {"dataset": "d"}}


class Retriever:
    """search(query) returns docs named after the query: q-0.. for the question, <thought>-0.. otherwise."""

    def __init__(self):
        self.queries, self.selected = [], None

    def select(self, example):
        self.selected = example["id"]

    def search(self, query):
        self.queries.append(query)
        if query == "Q?":
            return [{"doc_id": f"q-{i}", "title": "", "text": ""} for i in range(4)]
        # the first two results repeat docs already in context; IRCoT must skip them
        return [{"doc_id": "q-0", "title": "", "text": ""}, {"doc_id": "q-1", "title": "", "text": ""}] + [
            {"doc_id": f"{query}-{i}", "title": "", "text": ""} for i in range(5)]


class Reader:
    def __init__(self, thoughts=("t1", "t2"), answer_after=None):
        self.thoughts, self.answer_after = list(thoughts), answer_after
        self.contexts, self.final_calls = [], []

    def step(self, question, context, thoughts, transcript):
        self.contexts.append([d["doc_id"] for d in context])
        if self.answer_after is not None and len(thoughts) >= self.answer_after:
            return ReaderStep("<answer> answer </answer>", None, "answer")
        thought = self.thoughts[len(thoughts) % len(self.thoughts)]
        return ReaderStep(f"<think> {thought} </think>\n", thought, None)

    def final(self, question, context, thoughts, transcript):
        self.final_calls.append(([d["doc_id"] for d in context], list(thoughts)))
        return "final"


def test_llm_only_never_retrieves():
    retriever, reader = Retriever(), Reader()
    out = baselines.run_llm_only(reader, retriever, EXAMPLE)
    assert retriever.queries == [] and reader.final_calls == [([], [])]
    assert out == {"id": "q1", "prediction": "final", "context_doc_ids": [], "actions": [], "retrievals": 0}


def test_single_step_retrieves_once_with_the_question():
    retriever, reader = Retriever(), Reader()
    out = baselines.run_single_step(reader, retriever, EXAMPLE)
    assert retriever.selected == "q1" and retriever.queries == ["Q?"]
    assert reader.final_calls == [(["q-0", "q-1", "q-2", "q-3"], [])]
    assert out["prediction"] == "final" and out["retrievals"] == 1 and out["context_doc_ids"] == ["q-0", "q-1", "q-2", "q-3"]


def test_ircot_queries_with_each_thought_and_adds_new_docs():
    retriever, reader = Retriever(), Reader(answer_after=2)
    events = []
    out = baselines.run_ircot(reader, retriever, EXAMPLE, max_steps=5, docs_per_step=3, emit=lambda kind, data: events.append(kind))
    assert retriever.queries == ["Q?", "t1", "t2"]  # question first, then one search per thought
    assert out["context_doc_ids"] == ["q-0", "q-1", "q-2", "q-3", "t1-0", "t1-1", "t1-2", "t2-0", "t2-1", "t2-2"]
    assert reader.contexts[1][-3:] == ["t1-0", "t1-1", "t1-2"]  # the next thought sees the new docs
    assert out["prediction"] == "answer" and out["retrievals"] == 3 and reader.final_calls == []
    assert events == ["retrieve", "thought", "retrieve", "thought", "retrieve", "answer"]


def test_ircot_falls_back_to_final_after_max_steps():
    retriever, reader = Retriever(), Reader()
    out = baselines.run_ircot(reader, retriever, EXAMPLE, max_steps=2, docs_per_step=1)
    assert retriever.queries == ["Q?", "t1", "t2"] and out["prediction"] == "final"
    assert reader.final_calls == [(["q-0", "q-1", "q-2", "q-3", "t1-0", "t2-0"], ["t1", "t2"])]


def test_summarize_counts_baseline_retrievals():
    out = baselines.run_single_step(Reader(), Retriever(), EXAMPLE)
    report = summarize([EXAMPLE], [out], 1.0)
    assert report["mean_retrievals"] == 1.0 and report["context_recall"] == 0.0


def test_chat_reader_answers_with_the_reader_prompt():
    sent = []

    class Client:
        class chat:  # noqa: N801 - mimics openai.OpenAI().chat.completions.create
            class completions:
                @staticmethod
                def create(**kwargs):
                    sent.append(kwargs)
                    message = type("M", (), {"content": "<think> x </think>\n<answer> Paris </answer>"})
                    return type("R", (), {"choices": [type("C", (), {"message": message})]})

    reader = baselines.ChatReader(Client(), "big-model")
    answer = reader.final("Where?", [{"title": "T", "text": "Paris is"}], [], "")
    assert answer == "Paris"
    prompt = sent[0]["messages"][0]["content"]
    assert "Title: T\nText: Paris is" in prompt and "Question: Where?" in prompt
    assert "<think>" not in prompt and "<reason>" in prompt  # Qwen3 tokenizes <think> as a special token
    assert sent[0]["model"] == "big-model" and sent[0]["temperature"] == 0.0


def test_chat_reader_without_answer_tag_keeps_the_last_line():
    assert baselines.answer_or_last_line("I think\nParis") == "Paris"
    assert baselines.answer_or_last_line("<answer>Rome</answer>") == "Rome"


def test_cli_writes_a_report(tmp_path, monkeypatch):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({**EXAMPLE, "question": "Q?", "answers": ["answer"]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(baselines, "retriever_from_args", lambda args: type("R", (Retriever,), {"close": lambda self: None})())
    monkeypatch.setattr(baselines, "reader_from_args", lambda args: Reader(answer_after=0))
    monkeypatch.setattr(baselines, "read_examples", lambda path: [EXAMPLE])
    out = tmp_path / "report.json"
    baselines.main(["--method", "ircot", "--qa", str(qa), "--corpus-config", "c.json", "--reader-model", "r", "--output", str(out), "--workers", "2"])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["method"] == "ircot" and report["questions"] == 1 and report["f1"] == 1.0
    assert report["predictions"][0]["prediction"] == "answer"


def test_report_table_orders_methods(tmp_path):
    from grasp_grpo.report_table import table

    def rep(name, f1, method=True):
        r = {"questions": 999, "f1": f1, "em": 0.4, "context_recall": 0.7, "mean_retrievals": 1.0, "by_dataset": {"wiki18": {"f1": f1}}}
        return (tmp_path / f"{name}.json", {**r, "method": name} if method else r)

    text = table([rep("grasp-grpo", 0.6, method=False), rep("ircot", 0.5), rep("llm", 0.2)])
    rows = [line.split("|")[1].strip() for line in text.splitlines()[2:]]
    assert rows == ["llm", "ircot", "grasp-grpo"] and "0.600" in text and "| – |" in text


def test_ircot_caps_the_context():
    retriever, reader = Retriever(), Reader()
    out = baselines.run_ircot(reader, retriever, EXAMPLE, max_steps=5, docs_per_step=3, max_context_docs=6)
    assert len(out["context_doc_ids"]) == 6  # 4 from the question + 2 (room), then no more searches
    assert retriever.queries == ["Q?", "t1"] and out["retrievals"] == 2


def test_cli_scores_a_failing_question_zero(tmp_path, monkeypatch):
    class Broken(Reader):
        def final(self, *args):
            raise RuntimeError("context too long")

    monkeypatch.setattr(baselines, "retriever_from_args", lambda args: type("R", (Retriever,), {"close": lambda self: None})())
    monkeypatch.setattr(baselines, "reader_from_args", lambda args: Broken())
    monkeypatch.setattr(baselines, "read_examples", lambda path: [EXAMPLE])
    out = tmp_path / "r.json"
    baselines.main(["--method", "single", "--qa", "x", "--corpus-config", "c.json", "--reader-model", "r", "--output", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["f1"] == 0.0 and report["error_count"] == 1 and "context too long" in report["errors"][0]["error"]
