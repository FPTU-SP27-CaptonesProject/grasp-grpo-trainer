"""Prefetch: each uncached document is extracted once, cached ones never, and a dry run extracts nothing."""
import sqlite3
import threading
from types import SimpleNamespace

from grasp_grpo.prefetch_triples import prefetch


class FakeExtractor:
    def __init__(self):
        self.filled, self.lock = [], threading.Lock()

    def fill(self, passages):
        with self.lock:
            self.filled.extend(p["doc_id"] for p in passages if p.pop("needs_triples", False))


class FakeChild:
    """A corpus retriever: question -> doc ids; doc ids starting with 'c' are already cached."""

    def __init__(self, docs, extractor):
        self.docs, self.extractor, self.top_k = docs, extractor, 10
        self.triple_db = sqlite3.connect(":memory:", check_same_thread=False)
        self.triple_db.execute("CREATE TABLE triples (doc_id TEXT, model TEXT, triples TEXT)")
        cached = {d for ids in docs.values() for d in ids if d.startswith("c")}
        self.triple_db.executemany("INSERT INTO triples VALUES (?, 'm', '[]')", [(d,) for d in cached])

    def search(self, query, *, fill=True):
        assert not fill
        return [{"doc_id": d, "title": "", "text": d, "triples": []} for d in self.docs[query][: self.top_k]]


def router(extractor):
    retrievers = {
        "a": FakeChild({"qa1": ["c1", "n1", "n2"], "qa2": ["n2", "n3", "c2"]}, extractor),
        "b": FakeChild({"qb1": ["n4", "c3", "n5", "n6"]}, extractor),
    }
    return SimpleNamespace(retrievers=retrievers)


EXAMPLES = [
    {"id": "1", "question": "qa1", "metadata": {"dataset": "a"}},
    {"id": "2", "question": "qa2", "metadata": {"dataset": "a"}},
    {"id": "3", "question": "qb1", "metadata": {"dataset": "b"}},
]


def test_prefetch_extracts_each_missing_document_once(monkeypatch):
    monkeypatch.setattr("grasp_grpo.prefetch_triples.extraction_stats", lambda r: None)
    extractor = FakeExtractor()
    report = prefetch(router(extractor), EXAMPLES, depth=3, dry_run=False)
    assert sorted(extractor.filled) == ["n1", "n2", "n3", "n4", "n5"]  # depth 3 drops n6; n2 only once
    assert report["questions"] == 3 and report["documents"] == 8 and report["missing"] == 5


def test_dry_run_counts_without_extracting(monkeypatch):
    monkeypatch.setattr("grasp_grpo.prefetch_triples.extraction_stats", lambda r: None)
    extractor = FakeExtractor()
    report = prefetch(router(extractor), EXAMPLES, depth=10, dry_run=True)
    assert extractor.filled == []
    assert report["missing"] == 6 and report["estimated_usd"] >= 0
