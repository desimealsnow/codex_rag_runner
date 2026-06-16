from __future__ import annotations

import pathlib

from rag_runner.config import DEFAULT_CONFIG, deep_merge
from rag_runner.corpus import chunk_source, discover_source_files, read_source_file
from rag_runner.github_client import Issue, github_token
from rag_runner.prompting import build_grounded_prompt
from rag_runner.runner import lock_file, parse_context, parse_task
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
