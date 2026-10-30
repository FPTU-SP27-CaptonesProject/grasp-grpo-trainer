"""Shared answer-file evaluator for GRASP and baseline RAG systems."""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from .data import read_examples, write_json


def normal(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", str(text).lower()).split())


def f1(prediction: str, answers: list[str]) -> float:
    predicted = normal(prediction).split()
    values = []
    for answer in answers:
        gold = normal(answer).split()
        common = sum((Counter(predicted) & Counter(gold)).values())
        values.append(0.0 if not common else 2 * common / (len(predicted) + len(gold)))
    return max(values, default=0.0)


def exact(prediction: str, answers: list[str]) -> float:
    return float(any(normal(prediction) == normal(answer) for answer in answers))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GRASP and baselines with identical QA data")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--predictions", required=True, help="JSON/JSONL objects: id and prediction/final_answer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import json
    raw = Path(args.predictions).read_text(encoding="utf-8")
    predictions = [json.loads(line) for line in raw.splitlines() if line.strip()] if args.predictions.endswith(".jsonl") else json.loads(raw)
    by_id = {str(x.get("id", x.get("question_id"))): str(x.get("prediction", x.get("final_answer", ""))) for x in predictions}
    rows = []
    for item in read_examples(args.qa):
        prediction = by_id.get(item["id"], "")
        rows.append({"id": item["id"], "f1": f1(prediction, item["gold_answers"]), "em": exact(prediction, item["gold_answers"])})
    if not rows:
        raise ValueError("no evaluation rows")
    report = {"questions": len(rows), "missing_predictions": sum(x["id"] not in by_id for x in read_examples(args.qa)), "f1": sum(x["f1"] for x in rows) / len(rows), "em": sum(x["em"] for x in rows) / len(rows), "per_question": rows}
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_question"}))


if __name__ == "__main__":
    main()
