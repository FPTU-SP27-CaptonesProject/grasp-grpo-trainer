import argparse
import threading

import pytest

from grasp_grpo.retrieval import BoundRetriever, add_retrieval_arguments, retriever_from_args


class FakeRouter:
    def __init__(self):
        self.current = None

    def select(self, example):
        self.current = example["metadata"]["dataset"]

    def search(self, query):
        return [{"doc_id": f"{self.current}:{query}"}]

    def candidate_triples(self, query, passages, question=None):
        return [{"corpus": self.current}]


def test_each_bound_view_searches_its_own_corpus():
    router, lock = FakeRouter(), threading.Lock()
    a = BoundRetriever(router, {"metadata": {"dataset": "wiki18"}}, lock)
    b = BoundRetriever(router, {"metadata": {"dataset": "2wiki"}}, lock)
    assert a.search("q") == [{"doc_id": "wiki18:q"}]
    assert b.search("q") == [{"doc_id": "2wiki:q"}]
    assert a.candidate_triples("q", []) == [{"corpus": "wiki18"}]
    a.select({"metadata": {"dataset": "2wiki"}})  # binding is fixed per tree
    assert a.search("q") == [{"doc_id": "wiki18:q"}]


def test_retriever_from_args_needs_exactly_one_source():
    parser = argparse.ArgumentParser()
    add_retrieval_arguments(parser)
    with pytest.raises(ValueError, match="exactly one"):
        retriever_from_args(parser.parse_args([]))
