from __future__ import annotations

import contextlib
import datetime as dt
import os
import pathlib
import re
import time
from typing import Any, Dict, Iterator, List

from .codex_client import run_codex
from .config import load_config
from .corpus import load_corpus
from .github_client import GitHubClient, Issue
from .prompting import build_grounded_prompt
from .vector_store import ChromaIndex, RetrievedChunk


def log(message: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print("[{}] {}".format(now, message), flush=True)


@contextlib.contextmanager
def lock_file(path: pathlib.Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def field_value(body: str, field_name: str) -> str:
    pattern = re.compile(r"^\s*{}\s*:\s*(.*)$".format(re.escape(field_name)), re.IGNORECASE | re.MULTILINE)
    match = pattern.search(body or "")
    return match.group(1).strip() if match else ""


def parse_task(issue: Issue) -> str:
    body = issue.body or ""
    lines = body.splitlines()
    task_lines: List[str] = []
    active = False
    field_pattern = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_-]*\s*:")
    for line in lines:
        if re.match(r"^\s*Task\s*:", line, re.IGNORECASE):
            active = True
            first = line.split(":", 1)[1].strip()
            if first:
                task_lines.append(first)
            continue
        if active and field_pattern.match(line):
            break
        if active:
            task_lines.append(line)
    task = "\n".join(task_lines).strip()
    return task or issue.title.strip()


def parse_context(issue: Issue) -> str:
    body = issue.body or ""
    lines = body.splitlines()
    context_lines: List[str] = []
    active = False
    field_pattern = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_-]*\s*:")
    for line in lines:
        if re.match(r"^\s*Context\s*:", line, re.IGNORECASE):
            active = True
            first = line.split(":", 1)[1].strip()
            if first:
                context_lines.append(first)
            continue
        if active and field_pattern.match(line):
            break
        if active:
            context_lines.append(line)
    return "\n".join(context_lines).strip()


def rebuild_index(config: Dict[str, Any]) -> int:
    chunks = load_corpus(config)
    ChromaIndex(config).rebuild(chunks)
    return len(chunks)


def retrieve(config: Dict[str, Any], question: str) -> List[RetrievedChunk]:
    rebuild_index(config)
    rag = config.get("rag") or {}
    return ChromaIndex(config).query(question, int(rag.get("top_k") or 6))


def answer_issue(issue: Issue, config: Dict[str, Any], dry_run: bool = False) -> str:
    question = parse_task(issue)
    context = parse_context(issue)
    if context:
        question = question + "\n\nAdditional user context:\n" + context
    chunks = retrieve(config, question)
    min_support = float((config.get("rag") or {}).get("min_support_score") or 0.2)
    prompt = build_grounded_prompt(question, chunks, min_support)
    work_dir = pathlib.Path(str((config.get("runner") or {}).get("work_dir") or "/tmp/codex-rag-runs")).expanduser() / "issue-{}".format(issue.number)
    result = run_codex(prompt, config, work_dir, dry_run=dry_run)
    prefix = "RAG answer for issue #{}\n\n".format(issue.number)
    body = result.body.strip() or "Codex returned no answer."
    max_chars = int((config.get("runner") or {}).get("comment_max_chars") or 60000)
    return (prefix + body)[:max_chars]


def process_once(config: Dict[str, Any], dry_run: bool = False) -> int:
    client = GitHubClient(config)
    github = config.get("github") or {}
    count = 0
    for issue in client.list_queued_issues():
        count += 1
        log("Processing issue #{}: {}".format(issue.number, issue.title))
        if not dry_run:
            client.add_labels(issue.number, [str(github.get("running_label") or "codex:running")])
            client.remove_label(issue.number, str(github.get("queued_label") or "codex:queued"))
        try:
            answer = answer_issue(issue, config, dry_run=dry_run)
            if dry_run:
                print(answer)
            else:
                client.add_comment(issue.number, answer)
                client.add_labels(issue.number, [str(github.get("done_label") or "codex:done")])
                client.remove_label(issue.number, str(github.get("running_label") or "codex:running"))
        except Exception as exc:
            message = "RAG runner failed for issue #{}: {}".format(issue.number, exc)
            log(message)
            if not dry_run:
                client.add_comment(issue.number, message)
                client.add_labels(issue.number, [str(github.get("failed_label") or "codex:failed")])
                client.remove_label(issue.number, str(github.get("running_label") or "codex:running"))
    return count


def run(config_path: pathlib.Path, once: bool, loop: bool, dry_run: bool) -> int:
    config = load_config(config_path)
    lock_path = pathlib.Path(str((config.get("runner") or {}).get("lock_file") or "/tmp/codex-rag.lock")).expanduser()
    with lock_file(lock_path):
        while True:
            process_once(config, dry_run=dry_run)
            if once or not loop:
                return 0
            time.sleep(int((config.get("github") or {}).get("poll_interval_seconds") or 30))
