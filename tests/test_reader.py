import hashlib

from grasp_grpo.contracts import READER, reader_context
from grasp_grpo.reader import GraspReader, LegacyReader, ReaderStep, extract_answer

GRASP_READER_SHA256 = "63d28c7d8045eb3ce2760504d4341893f0b3bc888b3752cfc2b51a29e62e54aa"


def test_reader_template_matches_grasp_byte_for_byte():
    assert hashlib.sha256(READER.encode("utf-8")).hexdigest() == GRASP_READER_SHA256


def test_reader_context_uses_title_text_blocks():
    items = [{"title": "A", "text": "x"}, {"title": None, "text": "y"}]
    assert reader_context(items) == "Title: A\nText: x\n\nTitle: \nText: y"
    assert reader_context([]) == ""


class ScriptedCompletion:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def __call__(self, prompt, *, stop, max_tokens):
        self.calls.append({"prompt": prompt, "stop": stop, "max_tokens": max_tokens})
        return self.outputs.pop(0)


def render(user):
    return f"<user>{user}</user><assistant>"


def test_grasp_reader_continues_transcript_and_stops_at_think():
    complete = ScriptedCompletion(["<think> Who directed it?", " <answer> Ann </answer>"])
    reader = GraspReader(complete, render)
    context = [{"title": "Film", "text": "Directed by Ann."}]
    first = reader.step("Q?", context, [], "")
    assert first == ReaderStep("<think> Who directed it? </think>\n", "Who directed it?", None)
    second = reader.step("Q?", context, ["Who directed it?"], first.piece)
    assert second.answer == "Ann" and second.thought is None
    assert complete.calls[0]["stop"] == ["</think>"]
    assert complete.calls[1]["prompt"].endswith("<assistant>" + first.piece)
    assert "Title: Film\nText: Directed by Ann." in complete.calls[0]["prompt"]


def test_grasp_reader_wraps_bare_thought_and_closes_open_answer():
    reader = GraspReader(ScriptedCompletion(["Bare thought  ", "<answer> Bob"]), render)
    step = reader.step("Q?", [], [], "")
    assert step.piece == "<think> Bare thought </think>\n" and step.thought == "Bare thought"
    answer = reader.step("Q?", [], ["Bare thought"], step.piece)
    assert answer.piece == "<answer> Bob </answer>" and answer.answer == "Bob"


def test_grasp_reader_empty_output_and_final_answer():
    complete = ScriptedCompletion(["   ", "<think> a </think>\n<answer> Carol </answer>"])
    reader = GraspReader(complete, render)
    assert reader.step("Q?", [], [], "") == ReaderStep("", None, None)
    assert reader.final("Q?", [], [], "ignored") == "Carol"
    assert complete.calls[1]["stop"] is None and complete.calls[1]["max_tokens"] == 1024
    assert complete.calls[1]["prompt"] == render(READER.format(context="None", question="Q?"))


class FakeChat:
    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def test_legacy_reader_keeps_first_run_behaviour():
    chat = FakeChat(["<think>Need X</think>", "no tags"])
    reader = LegacyReader(chat)
    step = reader.step("Q?", [], [], "")
    assert step.thought == "Need X" and step.answer is None
    assert reader.final("Q?", [], ["Need X"], step.piece) == "no tags"
    assert "Return the final short answer now" in chat.prompts[1]


def test_extract_answer_normalises_whitespace():
    assert extract_answer("x <answer>  a   b </answer>") == "a b"
    assert extract_answer("<answer> open") == "open"
    assert extract_answer("none") is None
