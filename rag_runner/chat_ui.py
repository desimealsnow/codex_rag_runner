"""Gradio web chat UI for the RAG study assistant (local + Colab/mobile)."""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Tuple

from .config import DEFAULT_CONFIG, deep_merge, resolve_paths, validate_config
from .prompting import MODE_INSTRUCTIONS
from .session import QueryResult, RagSession, chunk_to_dict


STUDY_MODES = list(MODE_INSTRUCTIONS.keys())


def build_config(
    config: Dict[str, Any] | None = None,
    config_path: pathlib.Path | str | None = None,
) -> Dict[str, Any]:
    """Load config from a file or dict, merging with defaults."""
    if config is not None:
        merged = deep_merge(DEFAULT_CONFIG, config)
        base_dir = pathlib.Path(".").resolve()
    elif config_path is not None:
        from .config import load_config

        return load_config(pathlib.Path(config_path))
    else:
        path = pathlib.Path("config.json")
        if path.exists():
            from .config import load_config

            return load_config(path)
        merged = deep_merge(DEFAULT_CONFIG, {})
        base_dir = pathlib.Path(".").resolve()

    validate_config(merged)
    resolve_paths(merged, base_dir)
    return merged


def format_sources(result: QueryResult) -> str:
    """Format retrieved chunks as markdown for display under the answer."""
    if not result.chunks:
        return ""

    lines = [
        "",
        "---",
        "**Retrieved sources** (best support: {:.3f})".format(result.best_support),
        "",
    ]
    for index, chunk in enumerate(result.chunks, start=1):
        info = chunk_to_dict(chunk)
        lines.append(
            "**[{}]** `{}` chunk {} · score {:.3f}".format(
                index,
                info["file"],
                info["chunk_index"],
                info["support_score"],
            )
        )
        lines.append("> {}".format(info["preview"].replace("\n", "\n> ")))
        lines.append("")
    return "\n".join(lines)


def format_stats(session: RagSession) -> str:
    """Format index stats for the sidebar."""
    stats = session.get_stats()
    files = stats.get("files") or []
    file_lines = "\n".join("- `{}`".format(name) for name in files[:12])
    if len(files) > 12:
        file_lines += "\n- … and {} more".format(len(files) - 12)

    return """**Index**
- Files: **{file_count}**
- Chunks: **{chunk_count}**
- Mode: **{study_mode_label}** (`{study_mode}`)
- Model: `{model_name}`

**Indexed files**
{file_lines}
""".format(
        file_count=stats["file_count"],
        chunk_count=stats["chunk_count"],
        study_mode_label=stats["study_mode_label"],
        study_mode=stats["study_mode"],
        model_name=stats["model_name"],
        file_lines=file_lines or "_No files indexed yet._",
    )


def create_chat_blocks(session: RagSession):
    """Build the Gradio Blocks UI."""
    import gradio as gr

    def chat(message: str, history: List[Tuple[str, str]], study_mode: str):
        if not message or not message.strip():
            return history, ""
        session.set_study_mode(study_mode)
        result = session.query(message.strip())
        reply = result.answer + format_sources(result)
        history = history + [(message.strip(), reply)]
        return history, ""

    def on_mode_change(study_mode: str):
        session.set_study_mode(study_mode)
        return format_stats(session)

    def on_rebuild():
        session.build_index(force_rebuild=True)
        return format_stats(session)

    with gr.Blocks(title="RAG Study Assistant") as demo:
        gr.Markdown(
            "# RAG Study Assistant\n"
            "Ask questions grounded in your indexed study files. "
            "Answers cite retrieved chunks."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=480,
                    show_copy_button=True,
                )
                with gr.Row():
                    message = gr.Textbox(
                        label="Your question",
                        placeholder="Ask about your study material…",
                        scale=4,
                        lines=1,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                clear = gr.Button("Clear chat")

            with gr.Column(scale=1):
                study_mode = gr.Dropdown(
                    label="Study mode",
                    choices=STUDY_MODES,
                    value=session.study_mode,
                )
                stats = gr.Markdown(format_stats(session))
                rebuild = gr.Button("Rebuild index")

        send.click(chat, [message, chatbot, study_mode], [chatbot, message])
        message.submit(chat, [message, chatbot, study_mode], [chatbot, message])
        clear.click(lambda: ([], ""), outputs=[chatbot, message])
        study_mode.change(on_mode_change, study_mode, stats)
        rebuild.click(on_rebuild, outputs=stats)

    return demo


def launch_chat_ui(
    config: Dict[str, Any] | None = None,
    config_path: pathlib.Path | str | None = None,
    *,
    dry_run: bool = False,
    share: bool = False,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    """Build the index and launch the Gradio chat UI."""
    resolved = build_config(config=config, config_path=config_path)
    session = RagSession(resolved, dry_run=dry_run)
    print("Preparing vector index…")
    status = session.build_index()
    if status == "loaded":
        print("Reused saved index (no re-indexing).")
    elif status == "built":
        print("Built new index from source files.")
    else:
        print("Warning: no chunks indexed — add study files and click Rebuild index.")
    print(
        "Ready: {} chunks from {} files. Opening chat UI on http://{}:{}".format(
            session.chunk_count,
            session.file_count,
            host,
            port,
        )
    )
    if share:
        print("Public share link will appear below (valid ~72h on Colab).")

    demo = create_chat_blocks(session)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        share=share,
        server_name=host,
        server_port=port,
        show_error=True,
    )
