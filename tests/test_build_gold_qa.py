import json
import sqlite3
import sys

from grasp_grpo.build_gold_qa import main


def test_build_gold_qa_maps_ids_and_supporting_titles(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    database = sqlite3.connect(corpus / "passages.sqlite")
    database.execute("CREATE TABLE passages (row INTEGER, doc_id TEXT, title TEXT, text TEXT)")
    database.executemany(
        "INSERT INTO passages VALUES (?, ?, ?, ?)",
        [(0, "doc-a", "Alpha Page", "alpha"), (1, "doc-b", "Beta Page", "beta")],
    )
    database.commit()
    database.close()

    config = tmp_path / "corpora.json"
    config.write_text(json.dumps({"demo": {"index_dir": str(corpus)}}), encoding="utf-8")
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        "\n".join(
            [
                json.dumps({"id": "demo::one", "dataset": "demo", "question": "One?", "gold_answers": ["one"]}),
                json.dumps({"id": "demo::two", "dataset": "demo", "question": "Two?", "gold_answers": ["two"]}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {"id": "one", "question": "One?", "answers": ["one"], "gold_ctxs": [{"id": "doc-a"}]},
                {"id": "two", "question": "Two?", "answers": ["two"], "supporting_facts": [["Beta Page", 0]]},
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grasp-build-gold-qa", "--qa", str(qa), "--corpus-config", str(config),
            "--gold-source", f"demo={source}", "--output", str(output),
        ],
    )
    main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["gold_doc_ids"] for row in rows] == [["doc-a"], ["doc-b"]]
    report = json.loads((tmp_path / "output.jsonl.report.json").read_text(encoding="utf-8"))
    assert report["datasets"]["demo"]["mapped"] == 2
