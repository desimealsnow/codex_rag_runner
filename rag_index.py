import argparse
import pathlib

from rag_runner.config import load_config
from rag_runner.runner import rebuild_index

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Codex RAG vector index")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index")
    args = parser.parse_args()

    config = load_config(pathlib.Path(args.config))
    if args.rebuild:
        count = rebuild_index(config)
        print(f"Successfully indexed {count} chunks.")
    else:
        print("Pass --rebuild to build the index.")

if __name__ == "__main__":
    main()
