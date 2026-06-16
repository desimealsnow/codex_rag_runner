from __future__ import annotations

import argparse
import pathlib

from .runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone Codex RAG GitHub runner.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    once = args.once or not args.loop
    return run(pathlib.Path(args.config), once=once, loop=args.loop, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
