"""Local live dashboard for GRPO runs on Vast.

The Vast scripts push small logs to a private HF dataset (grasp_grpo.log_sync). This server pulls them
every --refresh seconds and serves a page that redraws the charts. Needs a local `hf auth login`.

    .venv\\Scripts\\python scripts\\dashboard.py            # then open http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "dashboard.html"


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _strip(report):
    if isinstance(report, dict):
        report = {k: v for k, v in report.items() if k != "per_question"}
        if isinstance(report.get("signal_probe"), dict):
            report["signal_probe"] = {k: v for k, v in report["signal_probe"].items() if k != "trees"}
    return report


def read_run(run: Path) -> dict:
    out: dict = {"name": run.name, "metrics": [], "eval_history": [], "reports": {}, "gpu": [], "console": ""}
    live = run / "metrics.jsonl"
    if live.is_file():
        for line in live.read_text(encoding="utf-8").splitlines():
            try:
                out["metrics"].append(json.loads(line))
            except ValueError:
                pass  # a line cut by an in-flight upload
    if not out["metrics"]:
        out["metrics"] = _json(run / "metrics.json") or []
    out["eval_history"] = _json(run / "eval_history.json") or []
    for path in sorted(run.glob("*.json")):
        if path.name not in ("metrics.json", "eval_history.json"):
            out["reports"][path.stem] = _strip(_json(path))
    gpu = run / "logs" / "gpu_metrics.csv"
    if gpu.is_file():
        with gpu.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle):
                if len(row) < 6 or row[0] == "timestamp":
                    continue
                try:
                    out["gpu"].append({"t": row[0].strip().replace("/", "-").split(".")[0], "gpu": int(row[1]), "util": float(row[3]), "mem_gb": float(row[4]) / 1024, "total_gb": float(row[5]) / 1024})
                except ValueError:
                    pass
    tail = run / "logs" / "console_tail.log"
    if tail.is_file():
        out["console"] = tail.read_text(encoding="utf-8", errors="replace")[-20000:]
    return out


def chain_run(parent: dict, child: dict, at: int) -> dict:
    """One continuous run: the parent's steps 1..at, then the child (which started from the parent's
    step-`at` adapter) shifted by `at` steps and by the parent's hours up to that step."""
    head = [m for m in parent["metrics"] if m.get("step", 0) <= at]
    hours = next((m.get("elapsed_hours") for m in reversed(head) if m.get("elapsed_hours") is not None), 0.0) or 0.0

    def shift(row: dict) -> dict:
        row = {**row, "step": row.get("step", 0) + at}
        if row.get("elapsed_hours") is not None:
            row["elapsed_hours"] += hours
        return row

    return {
        **child,
        "name": f"{child['name']} + {parent['name']} 1-{at}",
        "chain": {"parent": parent["name"], "child": child["name"], "at": at},
        "metrics": head + [shift(m) for m in child["metrics"]],
        "eval_history": [e for e in parent["eval_history"] if e.get("step", 0) <= at] + [shift(e) for e in child["eval_history"] if e.get("step", 0) > 0],
    }


def add_chains(runs: list[dict], chains: list[str]) -> list[dict]:
    by_name = {r["name"]: r for r in runs}
    out = []
    for spec in chains:  # CHILD=PARENT:STEP
        child, _, rest = spec.partition("=")
        parent, _, at = rest.partition(":")
        if child in by_name and parent in by_name and at.isdigit():
            out.append(chain_run(by_name[parent], by_name[child], int(at)))
    return out + runs


class Store:
    def __init__(self, repo: str, cache: Path, refresh: int, local: Path | None, chains: list[str] | None = None):
        self.repo, self.cache, self.refresh, self.local, self.chains = repo, cache, refresh, local, chains or []
        self.lock, self.fetched, self.error = threading.Lock(), 0.0, None
        self.remote_runs: set[str] | None = None  # runs that still exist on HF (local cache never deletes)

    def data(self) -> dict:
        with self.lock:
            if self.local is None and time.time() - self.fetched >= self.refresh:
                try:
                    from huggingface_hub import HfApi, snapshot_download

                    snapshot_download(self.repo, repo_type="dataset", local_dir=str(self.cache), allow_patterns=["runs/**"])
                    files = HfApi().list_repo_files(self.repo, repo_type="dataset")
                    self.remote_runs = {f.split("/")[1] for f in files if f.startswith("runs/") and f.count("/") >= 2}
                    self.error = None
                except Exception as error:  # noqa: BLE001 - show it on the page, keep the last copy
                    self.error = f"{type(error).__name__}: {error}"
                self.fetched = time.time()
            root = self.local or (self.cache / "runs")
            keep = self.remote_runs if self.local is None else None
            runs = [read_run(p) for p in sorted(root.iterdir()) if p.is_dir() and (keep is None or p.name in keep)] if root.is_dir() else []
            return {"repo": self.repo, "fetched": self.fetched, "error": self.error, "runs": add_chains(runs, self.chains)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="doancuavahk/grasp-grpo-logs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh", type=int, default=60, help="seconds between pulls from HF")
    parser.add_argument("--cache", default=str(Path.home() / ".cache" / "grasp-grpo-dashboard"))
    parser.add_argument("--local-runs", help="read run folders from disk instead of HF (debugging)")
    parser.add_argument("--no-browser", action="store_true")
    # The full run continues from the pilot's step-100 adapter (pilot-best-step100).
    parser.add_argument("--chain", action="append", metavar="CHILD=PARENT:STEP",
                        help="also show CHILD drawn after PARENT's steps 1..STEP as one run (repeatable); default grpo-final=stage3:100")
    args = parser.parse_args()
    store = Store(args.repo, Path(args.cache), args.refresh, Path(args.local_runs) if args.local_runs else None,
                  args.chain if args.chain is not None else ["grpo-final=stage3:100"])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/data"):
                body, kind = json.dumps(store.data()).encode("utf-8"), "application/json"
            elif self.path in ("/", "/index.html"):
                body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"dashboard: {url}  (logs from {args.local_runs or args.repo}, refresh {args.refresh}s; Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
