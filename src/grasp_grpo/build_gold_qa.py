"""Build auditable GRPO QA files with gold passage IDs.

The reward uses ``gold_doc_ids`` only after a rollout.  This tool never writes
gold evidence into model prompts; it only joins benchmark annotations to the
document IDs in the retrieval artifact that will be used for GRPO.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import read_examples, write_json


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    value = [json.loads(line) for line in text.splitlines() if line.strip()] if source.suffix == ".jsonl" else json.loads(text)
    if isinstance(value, dict):
        for key in ("data", "examples", "records"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{source}: expected a JSON/JSONL list of objects")
    return value


def normal(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", value).split())


def answers(record: dict[str, Any]) -> list[str]:
    value = record.get("gold_answers", record.get("answers", record.get("answer", [])))
    value = [value] if isinstance(value, str) else value
    return [str(item) for item in (value or []) if str(item).strip()]


def source_id(record: dict[str, Any]) -> str | None:
    for key in ("id", "_id", "qid", "question_id"):
        if record.get(key) is not None:
            return str(record[key])
    return None


def evidence(record: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return explicit source IDs and titles from standard QA annotations."""
    ids: set[str] = set()
    titles: set[str] = set()
    for context in record.get("gold_ctxs", record.get("gold_contexts", [])) or []:
        if not isinstance(context, dict):
            continue
        for key in ("id", "doc_id", "document_id"):
            if context.get(key) is not None:
                ids.add(str(context[key]))
                break
        if context.get("title"):
            titles.add(str(context["title"]))
    for fact in record.get("supporting_facts", []) or []:
        if isinstance(fact, (list, tuple)) and fact:
            titles.add(str(fact[0]))
        elif isinstance(fact, dict) and fact.get("title"):
            titles.add(str(fact["title"]))
    for item in record.get("supporting_titles", record.get("supporting_documents", [])) or []:
        titles.add(str(item.get("title")) if isinstance(item, dict) else str(item))
    return ids, {title for title in titles if title.strip()}


def passage_db(index_dir: str) -> Path:
    root = Path(index_dir)
    candidates = [root / "passages.sqlite", *root.glob("**/passages.sqlite")]
    found = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(found) != 1:
        raise ValueError(f"{root}: expected exactly one passages.sqlite, found {found}")
    return found[0]


def corpus_lookup(index_dir: str) -> tuple[set[str], dict[str, set[str]]]:
    database = sqlite3.connect(f"file:{passage_db(index_dir).as_posix()}?mode=ro", uri=True)
    try:
        rows = database.execute("SELECT doc_id, title FROM passages").fetchall()
    finally:
        database.close()
    ids = {str(doc_id) for doc_id, _ in rows}
    titles: dict[str, set[str]] = defaultdict(set)
    for doc_id, title in rows:
        if title:
            titles[normal(title)].add(str(doc_id))
    return ids, titles


def parse_sources(values: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--gold-source must be DATASET=PATH")
        dataset, path = value.split("=", 1)
        if not dataset or not path:
            raise ValueError("--gold-source must be DATASET=PATH")
        if dataset in result:
            raise ValueError(f"duplicate --gold-source for dataset {dataset}")
        result[dataset] = read_records(path)
    return result


def build_source_indexes(records: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        question = str(record.get("question", "")).strip()
        if question:
            by_question[normal(question)].append(record)
        identifier = source_id(record)
        if identifier:
            by_id[identifier] = record
    return by_question, by_id


def choose_source(row: dict[str, Any], by_question: dict[str, list[dict[str, Any]]], by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    row_id = str(row["id"])
    candidates = [row_id]
    if "::" in row_id:
        candidates.append(row_id.rsplit("::", 1)[1])
    for candidate in candidates:
        if candidate in by_id:
            return by_id[candidate], "id"
    matches = by_question.get(normal(row["question"]), [])
    if not matches:
        return None, "question_not_found"
    wanted_answers = {normal(answer) for answer in row["gold_answers"]}
    compatible = [item for item in matches if wanted_answers & {normal(answer) for answer in answers(item)}]
    if len(compatible) == 1:
        return compatible[0], "question_and_answer"
    if len(compatible) > 1:
        return None, "ambiguous_question"
    return None, "answer_mismatch"


def resolve_ids(record: dict[str, Any], valid_ids: set[str], title_ids: dict[str, set[str]]) -> tuple[list[str] | None, str]:
    explicit_ids, titles = evidence(record)
    resolved = {identifier for identifier in explicit_ids if identifier in valid_ids}
    invalid_explicit = explicit_ids - valid_ids
    ambiguous = False
    for title in titles:
        candidates = title_ids.get(normal(title), set())
        if len(candidates) == 1:
            resolved.update(candidates)
        elif len(candidates) > 1:
            ambiguous = True
        else:
            return None, "title_not_in_passages"
    if invalid_explicit and not titles:
        return None, "source_ids_not_in_passages"
    if ambiguous:
        return None, "ambiguous_title"
    if not resolved:
        return None, "missing_gold_evidence"
    return sorted(resolved), "mapped"


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich GRPO QA rows with verified gold_doc_ids")
    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--qa", help="Existing GRPO JSONL/JSON to enrich")
    input_mode.add_argument("--from-gold", action="store_true", help="Create canonical QA rows directly from --gold-source annotations")
    parser.add_argument("--corpus-config", required=True, help="JSON mapping dataset to index_dir")
    parser.add_argument("--gold-source", action="append", required=True, metavar="DATASET=PATH", help="Processed benchmark annotations; repeat once per corpus")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", help="Defaults to OUTPUT.report.json")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    config = json.loads(Path(args.corpus_config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("corpus config must be a JSON object")
    source_rows = parse_sources(args.gold_source)
    source_indexes = {name: build_source_indexes(rows) for name, rows in source_rows.items()}
    lookups: dict[str, tuple[set[str], dict[str, set[str]]]] = {}
    output: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)

    if args.qa:
        input_rows = read_examples(args.qa)
    else:
        input_rows = []
        for dataset, records in source_rows.items():
            for index, record in enumerate(records):
                question = str(record.get("question", "")).strip()
                gold_answers = answers(record)
                identifier = source_id(record)
                if question and gold_answers and identifier:
                    input_rows.append({"id": f"{dataset}::{identifier}", "question": question, "gold_answers": gold_answers, "metadata": {"dataset": dataset}, "_source": record})
                else:
                    counters[dataset]["invalid_source_row"] += 1

    for row in input_rows:
        dataset = str(row["metadata"].get("dataset", ""))
        counters[dataset]["total"] += 1
        if dataset not in source_indexes:
            status = "missing_gold_source"
        elif dataset not in config or not isinstance(config[dataset], dict) or not config[dataset].get("index_dir"):
            status = "missing_corpus_config"
        else:
            if dataset not in lookups:
                lookups[dataset] = corpus_lookup(str(config[dataset]["index_dir"]))
            source = row.get("_source") if args.from_gold else None
            if source is not None:
                status = "source_row"
            else:
                source, status = choose_source(row, *source_indexes[dataset])
            if source is not None:
                ids, status = resolve_ids(source, *lookups[dataset])
                if ids is not None:
                    output.append({"id": row["id"], "dataset": dataset, "split": args.split, "question": row["question"], "gold_answers": row["gold_answers"], "gold_doc_ids": ids})
                    counters[dataset]["mapped"] += 1
                    counters[dataset]["gold_doc_count"] += len(ids)
                    continue
        counters[dataset][status] += 1
        if len(samples[dataset]) < 20:
            samples[dataset].append({"id": row["id"], "question": row["question"], "status": status})

    report = {
        "input": "--from-gold" if args.from_gold else str(Path(args.qa).resolve()),
        "output": str(Path(args.output).resolve()),
        "rows_written": len(output),
        "datasets": {
            dataset: {
                **dict(counter),
                "mean_gold_doc_count": counter["gold_doc_count"] / counter["mapped"] if counter["mapped"] else 0.0,
                "examples_not_written": samples[dataset],
            }
            for dataset, counter in sorted(counters.items())
        },
    }
    write_jsonl(args.output, output)
    report_path = args.report or f"{args.output}.report.json"
    write_json(report_path, report)
    print(json.dumps({"rows_written": len(output), "report": str(Path(report_path).resolve())}))


if __name__ == "__main__":
    main()
