"""E5/FAISS retrieval compatible with Project A passage and triple stores."""
from __future__ import annotations

import json
import sqlite3
import threading
from functools import cache
from pathlib import Path
from typing import Any


def _triple_key(head: str, relation: str, tail: str) -> str:
    """Project A grasp.kg.triples.triple_key: case, spacing and trailing punctuation do not matter."""
    return " | ".join(" ".join(x.lower().strip(" .\"'").split()) for x in (head, relation, tail))


@cache
def _load_encoder(name: str):
    """One E5 model shared by every corpus (Project A E5Encoder: max_seq_length 512, fp16 on GPU)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    model.max_seq_length = 512
    if model.device.type == "cuda":
        model.half()
    return model


class DenseRetriever:
    """Project A retrieval (configs/retriever/e5_large.yaml): FAISS returns `candidates` rows with
    `nprobe` lists probed, those rows are re-scored exactly by re-encoding "title. text", and the
    top_k are kept.

    Without the re-score and nprobe, the IVF-PQ corpora (hotpotqa, wiki18) return other documents
    than the ones the SFT data and the SFT eval were produced with.
    """

    def __init__(self, index_dir: str, encoder: str, *, top_k: int, max_candidate_triples: int, triple_cache: str | None = None, query_prefix: str = "query: ", passage_prefix: str = "passage: ", candidates: int = 100, nprobe: int = 64, rescore: bool = True):
        import faiss

        root = Path(index_dir).resolve()
        found = [root / "index.faiss", root / encoder.rsplit("/", 1)[-1] / "index.faiss"]
        index_path = next((x for x in found if x.is_file()), None)
        if index_path is None:
            raise FileNotFoundError(f"No index.faiss below {root}")
        stores = [root / "passages.sqlite", index_path.parent / "passages.sqlite", index_path.parent.parent / "passages.sqlite"]
        passage_path = next((x for x in stores if x.is_file()), None)
        if passage_path is None:
            raise FileNotFoundError("passages.sqlite is required for Project A-compatible retrieval")
        self._setup(faiss.read_index(str(index_path)), _load_encoder(encoder), passage_path, top_k=top_k, max_candidate_triples=max_candidate_triples, triple_cache=triple_cache, query_prefix=query_prefix, passage_prefix=passage_prefix, candidates=candidates, nprobe=nprobe, rescore=rescore)
        self.index_path = index_path

    @classmethod
    def from_parts(cls, index, encoder, passages_path, *, top_k: int = 10, max_candidate_triples: int = 20, triple_cache=None, query_prefix: str = "query: ", passage_prefix: str = "passage: ", candidates: int = 100, nprobe: int = 64, rescore: bool = True) -> DenseRetriever:
        """Build from an already loaded index and encoder (tests, notebooks)."""
        self = cls.__new__(cls)
        self.index_path = None
        self._setup(index, encoder, Path(passages_path), top_k=top_k, max_candidate_triples=max_candidate_triples, triple_cache=triple_cache, query_prefix=query_prefix, passage_prefix=passage_prefix, candidates=candidates, nprobe=nprobe, rescore=rescore)
        return self

    def _setup(self, index, encoder, passage_path: Path, *, top_k, max_candidate_triples, triple_cache, query_prefix, passage_prefix, candidates, nprobe, rescore) -> None:
        self.index, self.encoder, self.passage_path = index, encoder, passage_path
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
        self.top_k, self.max_candidate_triples = top_k, max_candidate_triples
        self.candidates, self.rescore = candidates, rescore
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.db = sqlite3.connect(f"file:{Path(passage_path).resolve().as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self.triple_db = sqlite3.connect(f"file:{Path(triple_cache).resolve().as_posix()}?mode=ro", uri=True, check_same_thread=False) if triple_cache else None
        self.triple_schema = None
        if self.triple_db:
            columns = {row[1] for row in self.triple_db.execute("PRAGMA table_info(triples)")}
            self.triple_schema = "project_a" if "triples" in columns else "rows"
        self.stats = {"docs": 0, "docs_without_triples": 0}

    def _encode(self, texts: list[str]):
        import numpy as np

        return np.asarray(self.encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)

    @staticmethod
    def _passage_text(item: dict[str, Any]) -> str:
        """Project A grasp.retrieval.encoder.passage_text."""
        return f"{item['title']}. {item['text']}" if item["title"] else item["text"]

    def search(self, query: str) -> list[dict[str, Any]]:
        import numpy as np

        vector = self._encode([self.query_prefix + query])
        wanted = max(self.top_k, self.candidates) if self.rescore else self.top_k
        scores, rows = self.index.search(vector, wanted)
        valid = [(int(row), float(score)) for row, score in zip(rows[0], scores[0], strict=True) if row >= 0]
        if not valid:
            return []
        marks = ",".join("?" for _ in valid)
        fetched = self.db.execute(f'SELECT "row", doc_id, title, text FROM passages WHERE "row" IN ({marks})', [x[0] for x in valid]).fetchall()
        by_row = {int(row): {"doc_id": str(doc_id), "title": title, "text": str(text), "triples": []} for row, doc_id, title, text in fetched}
        valid = [(row, score) for row, score in valid if row in by_row]
        if self.rescore and valid:
            texts = [self.passage_prefix + self._passage_text(by_row[row]) for row, _ in valid]
            # Project A stores passage embeddings in fp16; round the same way so ties break identically.
            exact = self._encode(texts).astype(np.float16).astype(np.float32) @ vector[0]
            valid = sorted(((row, float(score)) for (row, _), score in zip(valid, exact, strict=True)), key=lambda x: -x[1])
        result = []
        for row, score in valid[: self.top_k]:
            item = by_row[row]
            item["score"] = score
            result.append(item)
        if self.triple_db and result:
            self._attach_triples(result)
        return result

    def _attach_triples(self, result: list[dict[str, Any]]) -> None:
        ids = [x["doc_id"] for x in result]
        marks = ",".join("?" for _ in ids)
        triples: dict[str, list[dict[str, str]]] = {x: [] for x in ids}
        if self.triple_schema == "project_a":
            # Project A cache: one JSON list of [head, relation, tail] per (doc_id, extractor model).
            seen: set[str] = set()
            for doc_id, payload in self.triple_db.execute(f"SELECT doc_id, triples FROM triples WHERE doc_id IN ({marks}) ORDER BY doc_id, model", ids):
                if doc_id in seen:  # one extractor model per document
                    continue
                seen.add(doc_id)
                for head, relation, tail in json.loads(payload):
                    triples[str(doc_id)].append({"doc_id": str(doc_id), "head": str(head), "relation": str(relation), "tail": str(tail)})
        else:
            for doc_id, head, relation, tail in self.triple_db.execute(f"SELECT doc_id, head, relation, tail FROM triples WHERE doc_id IN ({marks})", ids):
                triples[str(doc_id)].append({"doc_id": str(doc_id), "head": str(head), "relation": str(relation), "tail": str(tail)})
        for item in result:
            item["triples"] = triples[item["doc_id"]]
        self.stats["docs"] += len(result)
        self.stats["docs_without_triples"] += sum(1 for item in result if not item["triples"])

    def candidate_triples(self, query: str, passages: list[dict[str, Any]], question: str | None = None) -> list[dict[str, Any]]:
        """Project A grasp.kg.rank.rank_triples: dedupe, rank by E5 against "question query", keep the top ones."""
        import numpy as np

        candidates, seen = [], set()
        for passage in passages:
            for triple in passage.get("triples", []):
                key = _triple_key(str(triple["head"]), str(triple["relation"]), str(triple["tail"]))
                if key not in seen:
                    seen.add(key)
                    candidates.append({**triple, "doc_id": str(triple.get("doc_id") or passage["doc_id"])})
        if not candidates:
            return []
        vectors = self._encode([self.passage_prefix + f"{x['head']} {x['relation']} {x['tail']}" for x in candidates])
        query_vector = self._encode([self.query_prefix + f"{question or ''} {query}".strip()])[0]
        order = np.argsort(-(vectors @ query_vector), kind="stable")[: self.max_candidate_triples]
        return [candidates[int(i)] for i in order]

    def close(self) -> None:
        self.db.close()
        if self.triple_db:
            self.triple_db.close()


class CorpusRouter:
    """Routes each mixed-dataset example to its own frozen corpus artifacts."""

    def __init__(self, config_path: str, encoder: str, *, top_k: int, max_candidate_triples: int, candidates: int = 100, nprobe: int = 64, rescore: bool = True):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        entries = config.get("corpora", config)
        if not isinstance(entries, dict) or not entries:
            raise ValueError("corpus config must be a non-empty object or contain a corpora object")
        self.retrievers = {
            str(name): DenseRetriever(str(item["index_dir"]), encoder, top_k=top_k, max_candidate_triples=max_candidate_triples, triple_cache=item.get("triple_cache"), query_prefix=item.get("query_prefix", "query: "), passage_prefix=item.get("passage_prefix", "passage: "), candidates=candidates, nprobe=nprobe, rescore=rescore)
            for name, item in entries.items()
        }
        self.current: DenseRetriever | None = None

    @property
    def stats(self) -> dict[str, dict[str, int]]:
        return {name: dict(r.stats) for name, r in self.retrievers.items()}

    def select(self, example: dict[str, Any]) -> None:
        dataset = str(example.get("metadata", {}).get("dataset", ""))
        if dataset not in self.retrievers:
            known = ", ".join(self.retrievers)
            raise KeyError(f"example {example.get('id')} has dataset={dataset!r}; corpus config has {known}")
        self.current = self.retrievers[dataset]

    def search(self, query: str) -> list[dict[str, Any]]:
        if self.current is None: raise RuntimeError("select(example) must be called before retrieval")
        return self.current.search(query)

    def candidate_triples(self, query: str, passages: list[dict[str, Any]], question: str | None = None) -> list[dict[str, Any]]:
        if self.current is None: raise RuntimeError("select(example) must be called before retrieval")
        return self.current.candidate_triples(query, passages, question=question)

    def close(self) -> None:
        for retriever in self.retrievers.values(): retriever.close()


class BoundRetriever:
    """Pins a shared retriever to one example's corpus and serialises calls across rollout threads."""

    def __init__(self, retriever, example: dict[str, Any], lock: threading.Lock):
        self.retriever, self.example, self.lock = retriever, example, lock

    def select(self, example: dict[str, Any]) -> None:
        return None  # bound at construction

    def _bind(self) -> None:
        select = getattr(self.retriever, "select", None)
        if select is not None:
            select(self.example)

    def search(self, query: str) -> list[dict[str, Any]]:
        with self.lock:
            self._bind()
            return self.retriever.search(query)

    def candidate_triples(self, query: str, passages: list[dict[str, Any]], question: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            self._bind()
            return self.retriever.candidate_triples(query, passages, question=question)


def add_retrieval_arguments(parser) -> None:
    parser.add_argument("--index-dir")
    parser.add_argument("--corpus-config", help="JSON mapping metadata.dataset to index/triple artifacts")
    parser.add_argument("--triple-cache")
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-candidate-triples", type=int, default=20)
    # Project A configs/retriever/e5_large.yaml
    parser.add_argument("--retrieval-candidates", type=int, default=100, help="FAISS rows re-scored exactly before keeping top-k")
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--no-rescore", action="store_true")


def retriever_from_args(args) -> DenseRetriever | CorpusRouter:
    if bool(args.index_dir) == bool(args.corpus_config):
        raise ValueError("provide exactly one of --index-dir or --corpus-config")
    extra = {"candidates": getattr(args, "retrieval_candidates", 100), "nprobe": getattr(args, "nprobe", 64), "rescore": not getattr(args, "no_rescore", False)}
    if args.corpus_config:
        return CorpusRouter(args.corpus_config, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples, **extra)
    return DenseRetriever(args.index_dir, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples, triple_cache=args.triple_cache, **extra)
