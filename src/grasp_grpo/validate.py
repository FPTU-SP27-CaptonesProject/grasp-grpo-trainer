from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Project A SFT or Project B GRPO PEFT adapter")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-dir")
    parser.add_argument("--triple-cache")
    args = parser.parse_args()
    root = Path(args.checkpoint).expanduser().resolve()
    needed = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [name for name in needed if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"Invalid PEFT adapter {root}: missing {', '.join(missing)}")
    if args.index_dir:
        index = Path(args.index_dir).resolve()
        if not any(index.glob("**/index.faiss")):
            raise SystemExit(f"No FAISS index below {index}")
        if not any(index.glob("**/passages.sqlite")):
            raise SystemExit(f"No passages.sqlite below {index}")
    if args.triple_cache and not Path(args.triple_cache).is_file():
        raise SystemExit("triple cache does not exist")
    print(f"OK: {root}")


if __name__ == "__main__":
    main()
