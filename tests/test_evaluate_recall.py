import json
import sys

from grasp_grpo.evaluate import main


def test_evaluator_reports_context_recall_when_gold_ids_exist(tmp_path, monkeypatch):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({"id": "q1", "dataset": "demo", "question": "Question?", "gold_answers": ["answer"], "gold_doc_ids": ["a", "b"]}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps([{"id": "q1", "prediction": "answer", "context_doc_ids": ["a", "other"]}]), encoding="utf-8")
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", ["grasp-evaluate", "--qa", str(qa), "--predictions", str(predictions), "--output", str(output)])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["f1"] == 1.0
    assert report["em"] == 1.0
    assert report["context_recall"] == 0.5
