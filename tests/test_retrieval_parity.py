"""DenseRetriever must reproduce Project A retrieval (the setup SFT data and the SFT eval used)."""
import json
import sqlite3

import numpy as np
import pytest

from grasp_grpo.retrieval import DenseRetriever

VEC = {"alpha": [1.0, 0.0], "beta": [0.8, 0.6], "gamma": [0.0, 1.0]}


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **_):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            word = next((w for w in VEC if w in text), "gamma")
            out.append(VEC[word])
        return np.asarray(out, dtype=np.float32)


class FakeIndex:
    """IVF-PQ stand-in: approximate scores rank rows in the wrong order."""

    def __init__(self):
        self.nprobe = 1
        self.asked = None

    def search(self, vector, n):
        self.asked = n
        rows = np.array([[2, 1, 0, -1]], dtype=np.int64)
        scores = np.array([[0.9, 0.8, 0.7, 0.0]], dtype=np.float32)
        return scores[:, :n], rows[:, :n]


@pytest.fixture
def passages(tmp_path):
    path = tmp_path / "passages.sqlite"
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE passages ("row" INTEGER PRIMARY KEY, doc_id TEXT, title TEXT, text TEXT)')
    db.executemany("INSERT INTO passages VALUES (?, ?, ?, ?)", [(0, "d0", "A", "alpha fact"), (1, "d1", "B", "beta fact"), (2, "d2", "C", "gamma fact")])
    db.commit()
    db.close()
    return path


def project_a_cache(tmp_path):
    path = tmp_path / "triples.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE triples (doc_id TEXT, model TEXT, triples TEXT, PRIMARY KEY (doc_id, model))")
    db.execute("INSERT INTO triples VALUES (?, ?, ?)", ("d0", "qwen.qwen3-235b-a22b-2507", json.dumps([["X", "is", "alpha"], ["x", "is", "Alpha."]])))
    db.execute("INSERT INTO triples VALUES (?, ?, ?)", ("d1", "qwen.qwen3-235b-a22b-2507", json.dumps([["Y", "has", "beta"]])))
    db.commit()
    db.close()
    return path


def test_search_sets_nprobe_and_rescores_candidates(passages):
    index, encoder = FakeIndex(), FakeEncoder()
    retriever = DenseRetriever.from_parts(index, encoder, passages, top_k=2, candidates=100, nprobe=64)
    assert index.nprobe == 64
    result = retriever.search("alpha question")
    assert index.asked == 100
    assert [x["doc_id"] for x in result] == ["d0", "d1"]  # exact scores, not the IVF-PQ order
    assert result[0]["score"] == pytest.approx(1.0)
    assert encoder.calls[0] == ["query: alpha question"]
    assert sorted(encoder.calls[1]) == ["passage: A. alpha fact", "passage: B. beta fact", "passage: C. gamma fact"]


def test_search_without_rescore_keeps_index_order(passages):
    retriever = DenseRetriever.from_parts(FakeIndex(), FakeEncoder(), passages, top_k=2, rescore=False)
    assert [x["doc_id"] for x in retriever.search("alpha")] == ["d2", "d1"]


def test_reads_project_a_triple_cache_and_counts_misses(passages, tmp_path):
    retriever = DenseRetriever.from_parts(FakeIndex(), FakeEncoder(), passages, top_k=3, triple_cache=project_a_cache(tmp_path))
    by_id = {x["doc_id"]: x for x in retriever.search("alpha")}
    assert by_id["d1"]["triples"] == [{"doc_id": "d1", "head": "Y", "relation": "has", "tail": "beta"}]
    assert len(by_id["d0"]["triples"]) == 2
    assert by_id["d2"]["triples"] == []
    assert retriever.stats == {"docs": 3, "docs_without_triples": 1, "cache_misses": 1}


def test_reads_legacy_four_column_cache(passages, tmp_path):
    path = tmp_path / "legacy.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE triples (doc_id TEXT, head TEXT, relation TEXT, tail TEXT)")
    db.execute("INSERT INTO triples VALUES ('d2', 'Z', 'in', 'gamma')")
    db.commit()
    db.close()
    retriever = DenseRetriever.from_parts(FakeIndex(), FakeEncoder(), passages, top_k=3, triple_cache=path)
    by_id = {x["doc_id"]: x for x in retriever.search("alpha")}
    assert by_id["d2"]["triples"] == [{"doc_id": "d2", "head": "Z", "relation": "in", "tail": "gamma"}]


def test_candidate_triples_rank_by_question_and_query_and_dedupe(passages, tmp_path):
    encoder = FakeEncoder()
    retriever = DenseRetriever.from_parts(FakeIndex(), encoder, passages, top_k=3, max_candidate_triples=20, triple_cache=project_a_cache(tmp_path))
    found = retriever.search("alpha")
    ranked = retriever.candidate_triples("beta query", found, question="who")
    assert ["query: who beta query"] in encoder.calls
    assert [x["tail"] for x in ranked] == ["beta", "alpha"]  # case/punctuation duplicate dropped
