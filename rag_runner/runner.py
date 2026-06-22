from __future__ import annotations

import contextlib
import datetime as dt
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, Iterator, List

from .llm_client import run_llm
from .config import load_config
from .corpus import load_corpus
from .github_client import GitHubClient, Issue
from .prompting import build_grounded_prompt, normalize_study_mode
from .vector_store import ChromaIndex, RetrievedChunk


def log(message: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print("[{}] {}".format(now, message), flush=True)


@contextlib.contextmanager
def lock_file(path: pathlib.Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            holder = lock_holder(path)
            if holder and pid_exists(holder):
                raise RuntimeError("Runner lock already exists at {} for live PID {}".format(path, holder))
            log("Removing stale runner lock at {}".format(path))
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
    if fd is None:
        raise RuntimeError("Unable to acquire runner lock at {}".format(path))
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def lock_holder(path: pathlib.Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return int(text)
    except (FileNotFoundError, TypeError, ValueError):
        return 0


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # On Windows, os.kill() behaves differently; use ctypes kernel32 check
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


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


def parse_study_mode(issue: Issue) -> str:
    body = issue.body or ""
    return normalize_study_mode(field_value(body, "Study Mode") or field_value(body, "Operation"))


def rebuild_index(config: Dict[str, Any]) -> int:
    chunks = load_corpus(config)
    log("Indexing {} chunks from {}".format(len(chunks), (config.get("rag") or {}).get("source_dir")))
    ChromaIndex(config).rebuild(chunks)
    return len(chunks)


def retrieve(config: Dict[str, Any], question: str) -> List[RetrievedChunk]:
    """Query the existing vector index. Index must be built beforehand."""
    rag = config.get("rag") or {}
    index = ChromaIndex(config)
    min_support = float(rag.get("min_support_score") or 0.2)
    return index.query(question, int(rag.get("top_k") or 10), min_support_score=min_support)


def answer_issue(issue: Issue, config: Dict[str, Any], dry_run: bool = False) -> str:
    question = parse_task(issue)
    context = parse_context(issue)
    study_mode = parse_study_mode(issue)
    if context:
        question = question + "\n\nAdditional user context:\n" + context
    chunks = retrieve(config, question)
    min_support = float((config.get("rag") or {}).get("min_support_score") or 0.2)
    prompt = build_grounded_prompt(question, chunks, min_support, study_mode)
    work_dir = pathlib.Path(str((config.get("runner") or {}).get("work_dir") or "./runs")).expanduser() / "issue-{}".format(issue.number)
    result = run_llm(prompt, config, work_dir, dry_run=dry_run)
    prefix = "RAG {} response for issue #{}\n\n".format(study_mode, issue.number)
    body = result.body.strip() or "LLM returned no answer."
    max_chars = int((config.get("runner") or {}).get("comment_max_chars") or 60000)
    return (prefix + body)[:max_chars]


def process_once(config: Dict[str, Any], dry_run: bool = False) -> int:
    client = GitHubClient(config)
    github = config.get("github") or {}
    queued_label = str(github.get("queued_label") or "codex:queued")
    log("Polling {}/{} for label {}".format(github.get("owner"), github.get("repo"), queued_label))
    count = 0
    issues = client.list_queued_issues()
    if not issues:
        log("No queued issues found")
    for issue in issues:
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
                log("Completed issue #{}".format(issue.number))
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
    lock_path = pathlib.Path(str((config.get("runner") or {}).get("lock_file") or "./runner.lock")).expanduser()
    github = config.get("github") or {}
    rag = config.get("rag") or {}
    log("Starting RAG runner for {}/{} using {}".format(github.get("owner"), github.get("repo"), config_path))
    log("Watching {} with queued label {}".format(rag.get("source_dir"), github.get("queued_label") or "codex:queued"))
    with lock_file(lock_path):
        while True:
            process_once(config, dry_run=dry_run)
            if once or not loop:
                return 0
            time.sleep(int((config.get("github") or {}).get("poll_interval_seconds") or 30))
