"""Lazy Bedrock triple extraction: Project A prompt/parse/cache format, once per document, budget cap."""
import json
import sqlite3
import threading
import time

import pytest
from test_retrieval_parity import FakeEncoder, FakeIndex, passages, project_a_cache  # noqa: F401

from grasp_grpo.extractor import EXTRACT_PROMPT, TripleExtractor, _BadRequest, parse_triples
from grasp_grpo.retrieval import BoundRetriever, DenseRetriever, attach_extractor, extraction_stats

MODEL = "qwen.qwen3-235b-a22b-2507"


class FakeChat:
    def __init__(self, reply="<A; is; B>\n<C; has; D>", delay=0.0, error=None):
        self.prompts, self.reply, self.delay, self.error = [], reply, delay, error
        self.lock = threading.Lock()

    def __call__(self, prompt, max_tokens):
        with self.lock:
            self.prompts.append(prompt)
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.reply, 1_000_000, 0  # 1M prompt tokens -> $0.2266


def rows(path):
    db = sqlite3.connect(path)
    try:
        return {doc: (model, json.loads(payload)) for doc, model, payload in db.execute("SELECT doc_id, model, triples FROM triples")}
    finally:
        db.close()


def test_parse_matches_project_a():
    assert parse_triples("x <A ; born in; Paris > y\n<bad; line>") == [["A", "born in", "Paris"]]


def test_fill_extracts_once_and_writes_project_a_rows(tmp_path):
    cache = tmp_path / "triples.sqlite"
    chat = FakeChat(delay=0.05)
    extractor = TripleExtractor(str(cache), chat=chat, model=MODEL)
    first = [{"doc_id": "d9", "title": "T", "text": "body", "triples": [], "needs_triples": True}]
    second = [{"doc_id": "d9", "title": "T", "text": "body", "triples": [], "needs_triples": True}]
    threads = [threading.Thread(target=extractor.fill, args=(x,)) for x in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(chat.prompts) == 1  # the same document requested twice concurrently
    assert chat.prompts[0] == EXTRACT_PROMPT.format(title="T", text="body")
    assert first[0]["triples"] == second[0]["triples"] == [{"doc_id": "d9", "head": "A", "relation": "is", "tail": "B"}, {"doc_id": "d9", "head": "C", "relation": "has", "tail": "D"}]
    assert "needs_triples" not in first[0]
    extractor.close()
    assert rows(cache) == {"d9": (MODEL, [["A", "is", "B"], ["C", "has", "D"]])}


def test_budget_stops_extraction_without_caching(tmp_path):
    cache = tmp_path / "triples.sqlite"
    extractor = TripleExtractor(str(cache), chat=FakeChat(), model=MODEL, budget_usd=0.1, concurrency=1)
    docs = [{"doc_id": f"d{i}", "title": "", "text": "t", "triples": [], "needs_triples": True} for i in range(3)]
    extractor.fill(docs)
    assert [bool(d["triples"]) for d in docs] == [True, False, False]
    assert extractor.stats["usd"] == pytest.approx(0.2266)
    assert extractor.stats["skipped_budget"] == 2
    extractor.close()
    assert set(rows(cache)) == {"d0"}  # skipped documents stay missing for a later run


def test_bad_request_is_cached_empty_other_errors_are_not(tmp_path):
    cache = tmp_path / "triples.sqlite"
    bad = TripleExtractor(str(cache), chat=FakeChat(error=_BadRequest("too long")), model=MODEL)
    doc = [{"doc_id": "long", "title": "", "text": "t", "triples": [], "needs_triples": True}]
    bad.fill(doc)
    bad.close()
    failing = TripleExtractor(str(cache), chat=FakeChat(error=TimeoutError()), model=MODEL)
    other = [{"doc_id": "flaky", "title": "", "text": "t", "triples": [], "needs_triples": True}]
    failing.fill(other)
    assert other[0]["triples"] == [] and failing.stats["failed"] == 1
    failing.close()
    assert rows(cache) == {"long": (MODEL, [])}


def test_retriever_flags_only_documents_missing_from_cache(tmp_path, passages):  # noqa: F811
    cache = project_a_cache(tmp_path)  # d0, d1 cached; d2 missing
    retriever = DenseRetriever.from_parts(FakeIndex(), FakeEncoder(), passages, top_k=3, triple_cache=str(cache))
    chat = FakeChat(reply="<gamma; is; third>")
    attach_extractor(retriever, chat=chat, model=MODEL)
    lock = threading.Lock()
    result = BoundRetriever(retriever, {"id": "q"}, lock).search("gamma question")
    by_doc = {x["doc_id"]: x for x in result}
    assert len(chat.prompts) == 1 and "gamma fact" in chat.prompts[0]
    assert by_doc["d2"]["triples"] == [{"doc_id": "d2", "head": "gamma", "relation": "is", "tail": "third"}]
    assert retriever.stats == {"docs": 3, "docs_without_triples": 0, "cache_misses": 1}
    assert extraction_stats(retriever)["extracted"] == 1
    BoundRetriever(retriever, {"id": "q"}, lock).search("gamma question")
    assert len(chat.prompts) == 1  # now read from the cache
    assert retriever.stats["cache_misses"] == 1
    retriever.close()
