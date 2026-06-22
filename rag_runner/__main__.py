from __future__ import annotations

import argparse
import pathlib

from .config import load_config
from .runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone Codex RAG GitHub runner.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start an interactive study session in the terminal")
    parser.add_argument("--web", action="store_true", help="Start the Gradio chat UI in your browser")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (for Colab or mobile access)")
    parser.add_argument("--host", default="127.0.0.1", help="Host for --web (use 0.0.0.0 for LAN/mobile on same network)")
    parser.add_argument("--port", type=int, default=7860, help="Port for --web")
    args = parser.parse_args()

    if args.interactive:
        from .interactive import run_interactive
        config = load_config(pathlib.Path(args.config))
        return run_interactive(config, dry_run=args.dry_run)

    if args.web:
        from .chat_ui import launch_chat_ui
        launch_chat_ui(
            config_path=pathlib.Path(args.config),
            dry_run=args.dry_run,
            share=args.share,
            host=args.host,
            port=args.port,
        )
        return 0

    once = args.once or not args.loop
    return run(pathlib.Path(args.config), once=once, loop=args.loop, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
