"""Build the Project A-compatible triples.sqlite cache from teacher traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def build(source: Path, destination: Path, overwrite: bool = False) -> dict[str, int]:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} exists; pass --overwrite to rebuild")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(destination)
    database.execute("CREATE TABLE triples (doc_id TEXT NOT NULL, head TEXT NOT NULL, relation TEXT NOT NULL, tail TEXT NOT NULL, PRIMARY KEY (doc_id, head, relation, tail)) WITHOUT ROWID")
    database.execute("CREATE INDEX triples_doc_id ON triples(doc_id)")
    total = inserted = 0
    try:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("status") not in (None, "ok"):
                    continue
                rows = []
                for step in item.get("steps", []):
                    for candidate in step.get("candidates", []):
                        if not isinstance(candidate, dict):
                            continue
                        row = tuple(str(candidate.get(key) or "").strip() for key in ("doc_id", "head", "relation", "tail"))
                        if all(row): rows.append(row)
                total += len(rows)
                if rows:
                    before = database.total_changes
                    database.executemany("INSERT OR IGNORE INTO triples VALUES (?, ?, ?, ?)", rows)
                    inserted += database.total_changes - before
        database.commit()
        return {"candidate_rows": total, "unique_triples": inserted}
    finally:
        database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build triples.sqlite from one Project A teacher trajectory JSONL")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build(Path(args.trajectories), Path(args.output), args.overwrite)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
