"""Push a run's small log files to a private Hugging Face dataset so a local dashboard can follow it.

Only top-level *.json / *.jsonl, logs/gpu_metrics.csv and the last lines of logs/console.log are sent;
checkpoints stay on the machine. Logs survive the instance being stopped (e.g. out of credit).
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path


def stage_files(run_dir: Path, staging: Path, tail_lines: int = 300) -> None:
    run_dir, staging = Path(run_dir), Path(staging)
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "logs").mkdir(parents=True)
    for path in sorted(run_dir.glob("*.json")) + sorted(run_dir.glob("*.jsonl")):
        if not path.name.endswith(".tmp"):
            shutil.copy2(path, staging / path.name)
    gpu = run_dir / "logs" / "gpu_metrics.csv"
    if gpu.is_file():
        shutil.copy2(gpu, staging / "logs" / gpu.name)
    console = run_dir / "logs" / "console.log"
    if console.is_file():
        with console.open(encoding="utf-8", errors="replace") as handle:
            tail = deque(handle, maxlen=tail_lines)
        (staging / "logs" / "console_tail.log").write_text("".join(tail), encoding="utf-8")


def sync_once(api, run_dir: Path, repo: str, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / name
        stage_files(run_dir, staging)
        # Unchanged files make an empty commit, which the Hub skips.
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(staging), path_in_repo=f"runs/{name}", commit_message=f"logs {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo", required=True, help="private HF dataset, created if missing")
    parser.add_argument("--name", help="folder under runs/ (default: run dir name)")
    parser.add_argument("--every", type=int, default=180, help="seconds between uploads")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    name = args.name or run_dir.name
    while True:
        try:
            sync_once(api, run_dir, args.repo, name)
        except Exception as error:  # noqa: BLE001 - a failed upload must never stop training
            print(f"[log-sync] upload failed, retrying later: {error}", flush=True)
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
