"""Serve the dashboard to the team as a private static HF Space in the org (free; Docker Spaces need a paid plan).

The page cannot read the private log dataset itself, so this script pulls the logs the same way
scripts/dashboard.py does, writes them to data.json and uploads that file to the Space every --every seconds.
Org members who are signed in to HF can open the Space; nobody enters a token in the page.

    python scripts/deploy_dashboard_space.py             # deploy the page, then keep data.json fresh (tmux)
    python scripts/deploy_dashboard_space.py --once      # deploy and upload data.json once
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dashboard import Store  # noqa: E402

README = """---
title: GRASP GRPO dashboard
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
---

Live GRASP GRPO charts. data.json is refreshed by scripts/deploy_dashboard_space.py (grasp-grpo-trainer).
"""
MAX_GPU_ROWS = 3000  # 30 s samples: ~12 h per GPU pair, keeps data.json small


def page() -> str:
    html = (HERE / "dashboard.html").read_text(encoding="utf-8")
    assert html.count('fetch("/api/data"') == 1
    return html.replace('fetch("/api/data"', 'fetch("data.json?t=" + Date.now()')


def snapshot(store: Store) -> bytes:
    data = store.data()
    for run in data["runs"]:
        run["gpu"] = run["gpu"][-MAX_GPU_ROWS:]
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default="doancuavahk/grasp-grpo-dashboard")
    parser.add_argument("--repo", default="doancuavahk/grasp-grpo-logs", help="log dataset")
    parser.add_argument("--chain", action="append", help="as in dashboard.py; default grpo-final=stage3:100")
    parser.add_argument("--every", type=int, default=300, help="seconds between data.json uploads")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "grasp-grpo-dashboard-space"))
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=README.encode("utf-8"), path_in_repo="README.md", repo_id=args.space, repo_type="space", commit_message="Space card")
    api.upload_file(path_or_fileobj=page().encode("utf-8"), path_in_repo="index.html", repo_id=args.space, repo_type="space", commit_message="Dashboard page")
    print(f"https://huggingface.co/spaces/{args.space}", flush=True)
    store = Store(args.repo, Path(args.cache), 0, None, args.chain if args.chain is not None else ["grpo-final=stage3:100"])
    last = b""
    while True:
        try:
            body = snapshot(store)
            if body != last:
                api.upload_file(path_or_fileobj=body, path_in_repo="data.json", repo_id=args.space, repo_type="space", commit_message="data")
                last = body
                print(time.strftime("%H:%M:%S"), f"data.json {len(body) / 1e6:.2f} MB", flush=True)
        except Exception as error:  # noqa: BLE001 - keep serving the last upload
            print(time.strftime("%H:%M:%S"), "upload failed:", error, flush=True)
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
