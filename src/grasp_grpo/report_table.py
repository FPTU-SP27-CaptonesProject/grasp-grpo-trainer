"""Markdown table of eval reports (baselines.py and policy_eval.py JSON) for the thesis.

    python -m grasp_grpo.report_table outputs/report/test/*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ORDER = ("llm", "single", "single-bedrock", "ircot", "grasp-sft", "grasp-grpo")
DATASETS = ("2wiki_ircot", "hotpotqa_abstracts_2017", "wiki18")


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def row_name(path: Path, report: dict[str, Any]) -> str:
    return str(report.get("method") or path.stem)


def table(reports: list[tuple[Path, dict[str, Any]]]) -> str:
    rank = {name: i for i, name in enumerate(ORDER)}
    reports = sorted(reports, key=lambda item: (rank.get(row_name(*item), len(ORDER)), row_name(*item)))
    head = ["Method", "n", "F1", "EM", "Recall", "Retrievals", *(f"F1 {d.split('_')[0]}" for d in DATASETS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for path, r in reports:
        by = r.get("by_dataset", {})
        cells = [row_name(path, r), str(r.get("questions")), _fmt(r.get("f1")), _fmt(r.get("em")), _fmt(r.get("context_recall")),
                 _fmt(r.get("mean_retrievals")), *(_fmt(by.get(d, {}).get("f1")) for d in DATASETS)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+")
    p.add_argument("--output", help="also write the table here")
    args = p.parse_args(argv)
    loaded = [(Path(x), json.loads(Path(x).read_text(encoding="utf-8"))) for x in args.reports]
    text = table(loaded)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
