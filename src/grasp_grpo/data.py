from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_examples(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    raw = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()] if source.suffix == ".jsonl" else json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("QA data must be a JSON list or JSONL")
    result = []
    for i, item in enumerate(raw):
        if item.get("status") not in (None, "ok"):
            continue
        answers = item.get("gold_answers", item.get("answers", []))
        answers = [answers] if isinstance(answers, str) else answers
        if not str(item.get("question", "")).strip() or not answers:
            raise ValueError(f"row {i}: question and gold_answers are required")
        metadata = dict(item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {})
        for key in ("dataset", "split"):
            if item.get(key) is not None:
                metadata.setdefault(key, item[key])
        result.append({"id": str(item.get("id", item.get("qid", i))), "question": str(item["question"]), "gold_answers": [str(x) for x in answers], "gold_doc_ids": [str(x) for x in item.get("gold_doc_ids", [])], "metadata": metadata})
    return result


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
