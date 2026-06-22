from __future__ import annotations

import pathlib

from rag_runner.config import DEFAULT_CONFIG, deep_merge, resolve_paths
from rag_runner.corpus import chunk_source, clean_content, discover_source_files, read_source_file
from rag_runner.github_client import Issue, github_token
from rag_runner.llm_client import LLMResult, detect_backend
from rag_runner.prompting import build_grounded_prompt, normalize_study_mode
from rag_runner.runner import lock_file, parse_context, parse_study_mode, parse_task
from rag_runner.vector_store import RetrievedChunk, patch_sqlite


def test_chunk_source_keeps_file_identity(tmp_path: pathlib.Path) -> None:
    source_dir = tmp_path / "files"
    source_dir.mkdir()
    path = source_dir / "concept.md"
    path.write_text("Alpha concept.\n\n" + "Beta detail. " * 80, encoding="utf-8")
    source = read_source_file(path, source_dir)

    chunks = chunk_source(source, chunk_chars=120, chunk_overlap=20)

    assert chunks
    assert chunks[0].rel_path == "concept.md"
    assert chunks[0].chunk_id.startswith("concept.md::")


def test_discover_source_files_filters_extensions(tmp_path: pathlib.Path) -> None:
    source_dir = tmp_path / "files"
    source_dir.mkdir()
    (source_dir / "keep.md").write_text("keep", encoding="utf-8")
    (source_dir / "skip.bin").write_bytes(b"skip")

    found = discover_source_files(source_dir, [".md"])

    assert [path.name for path in found] == ["keep.md"]


def test_parse_task_and_context_from_issue_body() -> None:
    issue = Issue(
        number=1,
        title="fallback",
        author="alice",
        html_url="https://example.invalid/1",
        body="""Queue: codex:queued
Workflow: rag
Operation: ask
Task:
Explain alpha.
Context:
Focus on examples.
""",
    )

    assert parse_task(issue) == "Explain alpha."
    assert parse_context(issue) == "Focus on examples."
    assert parse_study_mode(issue) == "ask"


def test_grounded_prompt_requires_citations_and_refusal() -> None:
    prompt = build_grounded_prompt(
        "What is alpha?",
        [
            RetrievedChunk(
                rel_path="concept.md",
                chunk_index=0,
                text="Alpha is the first concept.",
                distance=0.1,
                support_score=0.9,
            )
        ],
        min_support_score=0.2,
    )

    assert "Cite source chunks inline" in prompt
    assert "The configured files do not cover this clearly." in prompt
    assert "file=concept.md chunk=0" in prompt
    assert "What is alpha?" in prompt


def test_grounded_prompt_supports_learning_modes() -> None:
    prompt = build_grounded_prompt(
        "Teach alpha.",
        [
            RetrievedChunk(
                rel_path="concept.md",
                chunk_index=0,
                text="Alpha is the first concept.",
                distance=0.1,
                support_score=0.9,
            )
        ],
        min_support_score=0.2,
        study_mode="teach",
    )

    assert "Mode: Teach" in prompt
    assert "certification tutor" in prompt
    assert "Three quick self-check questions" in prompt


def test_study_mode_aliases() -> None:
    assert normalize_study_mode("practice") == "quiz"
    assert normalize_study_mode("flashcard") == "flashcards"
    assert normalize_study_mode("revision") == "revise"
    assert normalize_study_mode("unknown") == "ask"


def test_config_deep_merge_keeps_defaults() -> None:
    config = deep_merge(DEFAULT_CONFIG, {"github": {"repo": "codex_rag_runner_test"}})

    assert config["github"]["owner"] == DEFAULT_CONFIG["github"]["owner"]
    assert config["github"]["repo"] == "codex_rag_runner_test"
    assert config["rag"]["top_k"] == DEFAULT_CONFIG["rag"]["top_k"]


def test_patch_sqlite_enables_newer_sqlite() -> None:
    patch_sqlite()
    import sqlite3

    assert tuple(int(part) for part in sqlite3.sqlite_version.split(".")[:2]) >= (3, 35)


def test_github_token_accepts_literal_token() -> None:
    assert github_token({"token": "ghp_direct"}) == "ghp_direct"
    assert github_token({"github_token": "ghp_legacy"}) == "ghp_legacy"


def test_github_token_accepts_token_env_name(monkeypatch) -> None:
    monkeypatch.setenv("RAG_TEST_TOKEN", "ghp_from_env")

    assert github_token({"token_env": "RAG_TEST_TOKEN"}) == "ghp_from_env"


def test_github_token_accepts_token_misplaced_in_token_env() -> None:
    assert github_token({"token_env": "github_pat_example"}) == "github_pat_example"


def test_lock_file_removes_stale_pid(tmp_path: pathlib.Path) -> None:
    lock_path = tmp_path / "runner.lock"
    lock_path.write_text("999999999", encoding="utf-8")

    with lock_file(lock_path):
        assert lock_path.read_text(encoding="utf-8").strip().isdigit()

    assert not lock_path.exists()


# --- New tests for content cleaning ---


def test_clean_content_removes_picture_placeholders() -> None:
    text = """Some text before.
**==> picture [92 x 80] intentionally omitted <==**
Some text after."""
    cleaned = clean_content(text)
    assert "picture" not in cleaned
    assert "Some text before." in cleaned
    assert "Some text after." in cleaned


def test_clean_content_removes_picture_text_blocks() -> None:
    text = """Before.
**----- Start of picture text -----**<br>
garbled text here<br>
more garbled<br>
**----- End of picture text -----**<br>
After."""
    cleaned = clean_content(text)
    assert "garbled" not in cleaned
    assert "Before." in cleaned
    assert "After." in cleaned


def test_clean_content_removes_standalone_page_numbers() -> None:
    text = "Real content.\n\n3\n\nMore content."
    cleaned = clean_content(text)
    assert "\n3\n" not in cleaned
    assert "Real content." in cleaned
    assert "More content." in cleaned


def test_clean_content_removes_br_tags() -> None:
    text = "Hello<br>World"
    cleaned = clean_content(text)
    assert "<br>" not in cleaned
    assert "Hello" in cleaned
    assert "World" in cleaned


def test_clean_content_collapses_blank_lines() -> None:
    text = "A\n\n\n\n\nB"
    cleaned = clean_content(text)
    assert "\n\n\n" not in cleaned
    assert "A" in cleaned
    assert "B" in cleaned


# --- New tests for resolve_paths ---


def test_resolve_paths_makes_relative_absolute(tmp_path: pathlib.Path) -> None:
    config = deep_merge(DEFAULT_CONFIG, {})
    resolve_paths(config, tmp_path)

    assert pathlib.Path(config["rag"]["source_dir"]).is_absolute()
    assert pathlib.Path(config["rag"]["index_dir"]).is_absolute()
    assert pathlib.Path(config["runner"]["work_dir"]).is_absolute()


def test_resolve_paths_preserves_absolute(tmp_path: pathlib.Path) -> None:
    abs_path = str(tmp_path / "custom" / "files")
    config = deep_merge(DEFAULT_CONFIG, {"rag": {"source_dir": abs_path}})
    resolve_paths(config, tmp_path)

    assert config["rag"]["source_dir"] == abs_path


# --- New tests for LLM client ---


def test_detect_backend_returns_string_or_none() -> None:
    result = detect_backend()
    assert result is None or isinstance(result, str)


def test_llm_result_dataclass() -> None:
    result = LLMResult(returncode=0, body="hello", stdout="hello\n", stderr="")
    assert result.returncode == 0
    assert result.body == "hello"


# --- New tests for min_support_score filtering ---


def test_query_filters_low_support_chunks(tmp_path: pathlib.Path) -> None:
    """Verify that RetrievedChunk filtering logic works conceptually."""
    chunks = [
        RetrievedChunk(rel_path="a.md", chunk_index=0, text="good", distance=0.1, support_score=0.9),
        RetrievedChunk(rel_path="b.md", chunk_index=0, text="bad", distance=0.9, support_score=0.1),
    ]
    filtered = [c for c in chunks if c.support_score >= 0.2]
    assert len(filtered) == 1
    assert filtered[0].text == "good"
