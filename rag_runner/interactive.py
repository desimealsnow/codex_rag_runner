"""Local interactive CLI (REPL) for the NISM exam study assistant."""
from __future__ import annotations

import traceback
from typing import Any, Dict

from .prompting import MODE_INSTRUCTIONS
from .session import RagSession


HELP_TEXT = """
Available commands:
  /mode [name]     Show or change study mode (ask, teach, quiz, flashcards, revise)
  /files           List indexed source files
  /stats           Show index statistics
  /rebuild         Rebuild the vector index from source files
  /help            Show this help message
  /quit, /exit     Exit the session

Study modes:
  ask        Direct answer with explanation and key terms
  teach      Step-by-step tutor with exam tips and self-checks
  quiz       Practice questions with cited answer key
  flashcards Concise Q/A cards with citations
  revise     Revision sheet with must-know points and checklist
""".strip()


def _print_banner(file_count: int, chunk_count: int, study_mode: str) -> None:
    """Print the session startup banner."""
    print()
    print("=" * 60)
    print("  NISM Exam Study Assistant")
    print("=" * 60)
    print("  Files: {}  |  Chunks: {}  |  Mode: {}".format(file_count, chunk_count, study_mode))
    print("  Type /help for commands, /quit to exit")
    print("=" * 60)
    print()


def run_interactive(config: Dict[str, Any], dry_run: bool = False) -> int:
    """Run the interactive REPL session."""
    session = RagSession(config, dry_run=dry_run)

    print("\nBuilding vector index from source files...")
    try:
        session.build_index()
    except Exception as exc:
        print("Error building index: {}".format(exc))
        traceback.print_exc()
        return 1

    if session.chunk_count == 0:
        rag = config.get("rag") or {}
        print("\nWarning: No chunks were indexed from '{}'.".format(rag.get("source_dir")))
        print("Make sure your study files are in that directory.")
        print("Supported extensions: {}".format(", ".join(rag.get("allowed_extensions") or [])))

    _print_banner(session.file_count, session.chunk_count, session.study_mode)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Good luck with your exam!")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            command = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if command in ("/quit", "/exit", "/q"):
                print("\nGoodbye! Good luck with your exam!")
                return 0

            elif command == "/help":
                print()
                print(HELP_TEXT)
                print()

            elif command == "/mode":
                if arg:
                    session.set_study_mode(arg)
                    label = MODE_INSTRUCTIONS.get(session.study_mode, {}).get("label", session.study_mode)
                    print("\n  Switched to {} mode.\n".format(label))
                else:
                    label = MODE_INSTRUCTIONS.get(session.study_mode, {}).get("label", session.study_mode)
                    print("\n  Current mode: {} ({})\n".format(session.study_mode, label))
                    print("  Available modes: {}".format(", ".join(sorted(MODE_INSTRUCTIONS.keys()))))
                    print()

            elif command == "/files":
                files = session.get_file_list()
                if files:
                    print("\n  Indexed files:")
                    for path in files:
                        print("    - {}".format(path))
                    print()
                else:
                    print("\n  No files indexed.\n")

            elif command == "/stats":
                stats = session.get_stats()
                print()
                print("  Index Statistics:")
                print("    Source directory: {}".format(stats["source_dir"]))
                print("    Files indexed:   {}".format(stats["file_count"]))
                print("    Total chunks:    {}".format(stats["chunk_count"]))
                print("    Chunk size:      {} chars".format(stats["chunk_chars"]))
                print("    Chunk overlap:   {} chars".format(stats["chunk_overlap"]))
                print("    Top-K retrieval: {}".format(stats["top_k"]))
                print("    Min support:     {}".format(stats["min_support_score"]))
                print("    Model:           {}".format(stats["model_name"]))
                print()

            elif command == "/rebuild":
                print("\n  Rebuilding vector index...")
                try:
                    session.build_index()
                    print("  Done! {} chunks from {} files.\n".format(session.chunk_count, session.file_count))
                except Exception as exc:
                    print("  Error rebuilding index: {}\n".format(exc))

            else:
                print("\n  Unknown command: {}. Type /help for available commands.\n".format(command))

            continue

        print()
        try:
            result = session.query(user_input)
            if result.chunks:
                print(
                    "  [Retrieved {} chunks, best support: {:.3f}]".format(
                        len(result.chunks),
                        result.best_support,
                    )
                )
                print()
            print(result.answer)
            print()
        except KeyboardInterrupt:
            print("\n  (Interrupted)")
            print()
        except Exception as exc:
            print("  Error: {}".format(exc))
            traceback.print_exc()
            print()
