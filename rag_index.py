from __future__ import annotations

import argparse
import pathlib

from rag_runner.config import load_config
from rag_runner.runner import rebuild_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or rebuild the Codex RAG vector index.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the configured vector index")
    args = parser.parse_args()
    config = load_config(pathlib.Path(args.config))
    count = rebuild_index(config)
    print("Indexed {} chunks".format(count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
