from __future__ import annotations

import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class CodexResult:
    returncode: int
    body: str
    stdout: str
    stderr: str


def build_codex_command(config: Dict[str, Any], output_path: pathlib.Path, cwd: pathlib.Path) -> List[str]:
    codex = config.get("codex") or {}
    args = [str(arg) for arg in (codex.get("extra_args") or [])]
    return [
        str(codex.get("binary") or "codex"),
        "exec",
        *(["--json"] if "--json" not in args else []),
        "--cd",
        str(cwd),
        "--sandbox",
        str(codex.get("sandbox") or "workspace-write"),
        "--output-last-message",
        str(output_path),
        *args,
        "-",
    ]


def run_codex(prompt: str, config: Dict[str, Any], work_dir: pathlib.Path, dry_run: bool = False) -> CodexResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "codex-answer.md"
    stdout_path = work_dir / "codex-stdout.log"
    stderr_path = work_dir / "codex-stderr.log"
    cmd = build_codex_command(config, output_path, pathlib.Path.cwd())
    if dry_run:
        return CodexResult(0, "DRY RUN: would execute {}\n\n{}".format(" ".join(cmd), prompt), "", "")
    timeout_seconds = int((config.get("codex") or {}).get("timeout_seconds") or 900)
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stdout_handle, stderr=stderr_handle)
        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8"))
        process.stdin.close()
        returncode = process.wait(timeout=timeout_seconds)
    body = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    if not body.strip():
        body = stdout.strip() or stderr.strip()
    return CodexResult(returncode, body.strip(), stdout, stderr)
