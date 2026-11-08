"""E5/FAISS retrieval compatible with Project A passage and triple stores."""
from __future__ import annotations

import functools

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
@functools.lru_cache(maxsize=None)
def _load_encoder(name: str, device: str | None = None):
    """One E5 model shared by every corpus (Project A E5Encoder: max_seq_length 512, fp16 on GPU).

    Cached so the three corpora share a copy (each loaded its own before: ~2 GB). `device` lets the trainer
    put it on the vLLM GPU: on the training GPU, the 100-passage re-score of every search queued behind the
    backward pass and async rollouts ran ~240 s instead of ~140 s (stage 2, 2026-11-09)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=device)
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

    def __init__(self, index_dir: str, encoder: str, *, top_k: int, max_candidate_triples: int, triple_cache: str | None = None, query_prefix: str = "query: ", passage_prefix: str = "passage: ", candidates: int = 100, nprobe: int = 64, rescore: bool = True, encoder_device: str | None = None):
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
        self._setup(faiss.read_index(str(index_path)), _load_encoder(encoder, encoder_device), passage_path, top_k=top_k, max_candidate_triples=max_candidate_triples, triple_cache=triple_cache, query_prefix=query_prefix, passage_prefix=passage_prefix, candidates=candidates, nprobe=nprobe, rescore=rescore)
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
        self.stats = {"docs": 0, "docs_without_triples": 0, "cache_misses": 0}
        self.extractor = None  # grasp_grpo.extractor.TripleExtractor: fills documents missing from the cache

    def _encode(self, texts: list[str]):
        import numpy as np

        return np.asarray(self.encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)

    @staticmethod
    def _passage_text(item: dict[str, Any]) -> str:
        """Project A grasp.retrieval.encoder.passage_text."""
        return f"{item['title']}. {item['text']}" if item["title"] else item["text"]

    def search(self, query: str, *, fill: bool = True) -> list[dict[str, Any]]:
        """Top-k passages with their cached triples. Documents missing from the cache are extracted
        when an extractor is set; `fill=False` leaves that (and `record`) to the caller, which
        BoundRetriever does outside its lock."""
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
            if fill:
                if self.extractor is not None:
                    self.extractor.fill(result)
                self.record(result)
        return result

    def record(self, result: list[dict[str, Any]]) -> None:
        """Coverage counters, after any extraction."""
        if not self.triple_db:
            return
        self.stats["docs"] += len(result)
        self.stats["docs_without_triples"] += sum(1 for item in result if not item["triples"])

    def _attach_triples(self, result: list[dict[str, Any]]) -> None:
        ids = [x["doc_id"] for x in result]
        marks = ",".join("?" for _ in ids)
        triples: dict[str, list[dict[str, str]]] = {x: [] for x in ids}
        cached: set[str] = set()
        if self.triple_schema == "project_a":
            # Project A cache: one JSON list of [head, relation, tail] per (doc_id, extractor model).
            seen: set[str] = set()
            for doc_id, payload in self.triple_db.execute(f"SELECT doc_id, triples FROM triples WHERE doc_id IN ({marks}) ORDER BY doc_id, model", ids):
                if doc_id in seen:  # one extractor model per document
                    continue
                seen.add(doc_id)
                cached.add(str(doc_id))
                for head, relation, tail in json.loads(payload):
                    triples[str(doc_id)].append({"doc_id": str(doc_id), "head": str(head), "relation": str(relation), "tail": str(tail)})
        else:
            for doc_id, head, relation, tail in self.triple_db.execute(f"SELECT doc_id, head, relation, tail FROM triples WHERE doc_id IN ({marks})", ids):
                cached.add(str(doc_id))
                triples[str(doc_id)].append({"doc_id": str(doc_id), "head": str(head), "relation": str(relation), "tail": str(tail)})
        for item in result:
            item["triples"] = triples[item["doc_id"]]
            if item["doc_id"] not in cached:
                self.stats["cache_misses"] += 1
                if self.extractor is not None:
                    item["needs_triples"] = True

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
        if self.extractor is not None:
            self.extractor.close()
            self.extractor = None


class CorpusRouter:
    """Routes each mixed-dataset example to its own frozen corpus artifacts."""

    def __init__(self, config_path: str, encoder: str, *, top_k: int, max_candidate_triples: int, candidates: int = 100, nprobe: int = 64, rescore: bool = True, encoder_device: str | None = None):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        entries = config.get("corpora", config)
        if not isinstance(entries, dict) or not entries:
            raise ValueError("corpus config must be a non-empty object or contain a corpora object")
        self.retrievers = {
            str(name): DenseRetriever(str(item["index_dir"]), encoder, top_k=top_k, max_candidate_triples=max_candidate_triples, triple_cache=item.get("triple_cache"), query_prefix=item.get("query_prefix", "query: "), passage_prefix=item.get("passage_prefix", "passage: "), candidates=candidates, nprobe=nprobe, rescore=rescore, encoder_device=encoder_device)
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

    def search(self, query: str, *, fill: bool = True) -> list[dict[str, Any]]:
        if self.current is None: raise RuntimeError("select(example) must be called before retrieval")
        return self.current.search(query, fill=fill)

    def candidate_triples(self, query: str, passages: list[dict[str, Any]], question: str | None = None) -> list[dict[str, Any]]:
        if self.current is None: raise RuntimeError("select(example) must be called before retrieval")
        return self.current.candidate_triples(query, passages, question=question)

    def close(self) -> None:
        extractors = {id(r.extractor): r.extractor for r in self.retrievers.values() if r.extractor is not None}
        for retriever in self.retrievers.values():
            retriever.extractor = None
            retriever.close()
        for extractor in extractors.values():
            extractor.close()


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
            target = getattr(self.retriever, "current", None) or self.retriever
            if getattr(target, "extractor", None) is None:
                return self.retriever.search(query)
            result = target.search(query, fill=False)
        target.extractor.fill(result)  # Bedrock calls run outside the lock; other rollouts keep going
        with self.lock:
            target.record(result)
        return result

    def candidate_triples(self, query: str, passages: list[dict[str, Any]], question: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            self._bind()
            return self.retriever.candidate_triples(query, passages, question=question)


def add_retrieval_arguments(parser) -> None:
    parser.add_argument("--index-dir")
    parser.add_argument("--corpus-config", help="JSON mapping metadata.dataset to index/triple artifacts")
    parser.add_argument("--triple-cache")
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--encoder-device", help="torch device for the E5 encoder (default: cuda:0 / cpu); scripts set RETRIEVER_GPU")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-candidate-triples", type=int, default=20)
    # Project A configs/retriever/e5_large.yaml
    parser.add_argument("--retrieval-candidates", type=int, default=100, help="FAISS rows re-scored exactly before keeping top-k")
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--no-rescore", action="store_true")
    from .extractor import add_extraction_arguments

    add_extraction_arguments(parser)


def retriever_from_args(args) -> DenseRetriever | CorpusRouter:
    if bool(args.index_dir) == bool(args.corpus_config):
        raise ValueError("provide exactly one of --index-dir or --corpus-config")
    extra = {"candidates": getattr(args, "retrieval_candidates", 100), "nprobe": getattr(args, "nprobe", 64), "rescore": not getattr(args, "no_rescore", False), "encoder_device": getattr(args, "encoder_device", None)}
    if args.corpus_config:
        retriever = CorpusRouter(args.corpus_config, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples, **extra)
    else:
        retriever = DenseRetriever(args.index_dir, args.encoder, top_k=args.top_k, max_candidate_triples=args.max_candidate_triples, triple_cache=args.triple_cache, **extra)
    if getattr(args, "extract_triples", False):
        attach_extractor(retriever, model=args.extract_model, budget_usd=args.extract_budget_usd, concurrency=args.extract_concurrency)
    return retriever


def _children(retriever) -> list[DenseRetriever]:
    return list(retriever.retrievers.values()) if isinstance(retriever, CorpusRouter) else [retriever]


def attach_extractor(retriever, **kwargs) -> None:
    """One extractor per triple cache file (the Vast layout shares one cache across corpora)."""
    from .extractor import TripleExtractor

    by_cache: dict[Path, TripleExtractor] = {}
    for child in _children(retriever):
        if child.triple_db is None:
            continue
        path = Path(child.triple_db.execute("PRAGMA database_list").fetchone()[2]).resolve()
        if child.triple_schema != "project_a":
            raise ValueError(f"{path}: triple extraction needs a Project A cache (triples table with a JSON column)")
        if path not in by_cache:
            by_cache[path] = TripleExtractor(str(path), **kwargs)
        child.extractor = by_cache[path]


def extraction_stats(retriever) -> dict[str, Any] | None:
    """Summed extractor counters (requests, extracted/empty/failed documents, tokens, USD), or None."""
    extractors = {id(c.extractor): c.extractor for c in _children(retriever) if getattr(c, "extractor", None) is not None}
    if not extractors:
        return None
    total: dict[str, Any] = {}
    for extractor in extractors.values():
        for key, value in extractor.stats.items():
            total[key] = total.get(key, 0) + value
    return total
