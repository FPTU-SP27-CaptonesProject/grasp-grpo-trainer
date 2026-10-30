"""E5/FAISS retrieval compatible with Project A passage and triple stores."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class DenseRetriever:
    def __init__(self, index_dir: str, encoder: str, *, top_k: int, max_candidate_triples: int, triple_cache: str | None = None, query_prefix: str = "query: ", passage_prefix: str = "passage: "):
        import faiss
        from sentence_transformers import SentenceTransformer

        root = Path(index_dir).resolve()
        candidates = [root / "index.faiss", root / encoder.rsplit("/", 1)[-1] / "index.faiss"]
        self.index_path = next((x for x in candidates if x.is_file()), None)
        if self.index_path is None:
            raise FileNotFoundError(f"No index.faiss below {root}")
        stores = [root / "passages.sqlite", self.index_path.parent / "passages.sqlite", self.index_path.parent.parent / "passages.sqlite"]
        self.passage_path = next((x for x in stores if x.is_file()), None)
        if self.passage_path is None:
            raise FileNotFoundError("passages.sqlite is required for Project A-compatible retrieval")
        self.index = faiss.read_index(str(self.index_path))
        self.encoder = SentenceTransformer(encoder)
        self.top_k, self.max_candidate_triples = top_k, max_candidate_triples
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.db = sqlite3.connect(f"file:{self.passage_path.as_posix()}?mode=ro", uri=True)
        self.triple_db = sqlite3.connect(f"file:{Path(triple_cache).resolve().as_posix()}?mode=ro", uri=True) if triple_cache else None

    def search(self, query: str) -> list[dict[str, Any]]:
        import numpy as np

        vector = self.encoder.encode([self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, rows = self.index.search(vector, self.top_k)
        valid = [(int(row), float(score)) for row, score in zip(rows[0], scores[0], strict=True) if row >= 0]
        if not valid:
            return []
        marks = ",".join("?" for _ in valid)
        fetched = self.db.execute(f'SELECT "row", doc_id, title, text FROM passages WHERE "row" IN ({marks})', [x[0] for x in valid]).fetchall()
        by_row = {int(row): {"doc_id": str(doc_id), "title": title, "text": str(text), "triples": []} for row, doc_id, title, text in fetched}
        result = []
        for row, score in valid:
            item = by_row.get(row)
            if item:
                item["score"] = score
                result.append(item)
        if self.triple_db and result:
            ids = [x["doc_id"] for x in result]
            marks = ",".join("?" for _ in ids)
            triples: dict[str, list[dict[str, str]]] = {x: [] for x in ids}
            for doc_id, head, relation, tail in self.triple_db.execute(f"SELECT doc_id, head, relation, tail FROM triples WHERE doc_id IN ({marks})", ids):
                triples[str(doc_id)].append({"doc_id": str(doc_id), "head": str(head), "relation": str(relation), "tail": str(tail)})
            for item in result:
                item["triples"] = triples[item["doc_id"]]
        return result

    def candidate_triples(self, query: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import numpy as np

        candidates = [{**triple, "doc_id": str(triple.get("doc_id") or passage["doc_id"])} for passage in passages for triple in passage.get("triples", [])]
        if not candidates:
            return []
        texts = [self.passage_prefix + f"{x['head']} {x['relation']} {x['tail']}" for x in candidates]
        vectors = self.encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        query_vector = self.encoder.encode([self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
        order = np.argsort(-(vectors @ query_vector))[: self.max_candidate_triples]
        return [candidates[int(i)] for i in order]

    def close(self) -> None:
        self.db.close()
        if self.triple_db:
            self.triple_db.close()


class CorpusRouter:
    """Routes each mixed-dataset example to its own frozen corpus artifacts."""

    def __init__(self, config_path: str, encoder: str, *, top_k: int, max_candidate_triples: int):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        entries = config.get("corpora", config)
        if not isinstance(entries, dict) or not entries:
            raise ValueError("corpus config must be a non-empty object or contain a corpora object")
        self.retrievers = {
            str(name): DenseRetriever(str(item["index_dir"]), encoder, top_k=top_k, max_candidate_triples=max_candidate_triples, triple_cache=item.get("triple_cache"), query_prefix=item.get("query_prefix", "query: "), passage_prefix=item.get("passage_prefix", "passage: "))
            for name, item in entries.items()
        }
        self.current: DenseRetriever | None = None

    def select(self, example: dict[str, Any]) -> None:
        dataset = str(example.get("metadata", {}).get("dataset", ""))
        if dataset not in self.retrievers:
            known = ", ".join(self.retrievers)
            raise KeyError(f"example {example.get('id')} has dataset={dataset!r}; corpus config has {known}")
        self.current = self.retrievers[dataset]

    def search(self, query: str) -> list[dict[str, Any]]:
        if self.current is None: raise RuntimeError("select(example) must be called before retrieval")
        return self.current.search(query)

    def candidate_triples(self, query: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.current is None: raise RuntimeError("select(example) must be called before retrieval")
        return self.current.candidate_triples(query, passages)

    def close(self) -> None:
        for retriever in self.retrievers.values(): retriever.close()
