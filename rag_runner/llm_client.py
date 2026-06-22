"""LLM backend client for the RAG runner.

Supports CLI backends (claude, claude-npx) and API backends (gemini) for cloud/Colab.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMResult:
    """Result from an LLM backend call."""
    returncode: int
    body: str
    stdout: str
    stderr: str


def detect_backend() -> Optional[str]:
    """Auto-detect which LLM CLI backend is available on PATH."""
    # Check for claude (Claude Code CLI)
    if shutil.which("claude"):
        return "claude"
    # Check for npx (can run claude via npx)
    if shutil.which("npx"):
        return "claude-npx"
    return None


def _build_claude_command(prompt: str, config: Dict[str, Any]) -> List[str]:
    """Build command to invoke Claude Code CLI."""
    llm = config.get("llm") or {}
    extra_args = [str(a) for a in (llm.get("claude_args") or [])]
    return [
        "claude",
        "-p", prompt,
        "--no-input",
        "--max-turns", "1",
        *extra_args,
    ]


def _build_claude_npx_command(prompt: str, config: Dict[str, Any]) -> List[str]:
    """Build command to invoke Claude Code CLI via npx."""
    llm = config.get("llm") or {}
    extra_args = [str(a) for a in (llm.get("claude_args") or [])]
    return [
        "npx", "-y", "@anthropic-ai/claude-code",
        "-p", prompt,
        "--no-input",
        "--max-turns", "1",
        *extra_args,
    ]


def run_gemini(prompt: str, config: Dict[str, Any]) -> LLMResult:
    """Run prompt through the Google Gemini API (for Colab / browser hosting)."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        return LLMResult(
            1,
            "Gemini backend requires google-generativeai. Install: pip install google-generativeai",
            "",
            str(exc),
        )

    llm = config.get("llm") or {}
    api_key = str(llm.get("api_key") or "").strip()
    if not api_key:
        api_key_env = str(llm.get("api_key_env") or "GEMINI_API_KEY")
        api_key = str(os.environ.get(api_key_env) or "").strip()
    if not api_key:
        return LLMResult(
            1,
            "Set llm.api_key or {} environment variable for Gemini backend".format(
                llm.get("api_key_env") or "GEMINI_API_KEY"
            ),
            "",
            "",
        )

    model_name = str(llm.get("model") or "gemini-2.0-flash")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        body = (getattr(response, "text", None) or "").strip()
        if not body:
            return LLMResult(1, "Gemini returned an empty response", "", "")
        return LLMResult(0, body, body, "")
    except Exception as exc:
        return LLMResult(1, "Gemini API error: {}".format(exc), "", str(exc))


def run_llm(prompt: str, config: Dict[str, Any], work_dir: pathlib.Path, dry_run: bool = False) -> LLMResult:
    """Run prompt through the configured LLM backend and return the result."""
    work_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = work_dir / "llm-stdout.log"
    stderr_path = work_dir / "llm-stderr.log"

    llm = config.get("llm") or {}
    backend = str(llm.get("backend") or "auto").strip().lower()

    if backend == "auto":
        detected = detect_backend()
        if not detected:
            return LLMResult(
                1,
                "No LLM CLI backend found. Install Claude Code CLI or set llm.backend to gemini for Colab.",
                "",
                "",
            )
        backend = detected

    if backend == "gemini":
        if dry_run:
            return LLMResult(
                0,
                "DRY RUN: would call Gemini API (prompt length: {} chars)".format(len(prompt)),
                "",
                "",
            )
        return run_gemini(prompt, config)

    if backend == "claude":
        cmd = _build_claude_command(prompt, config)
    elif backend == "claude-npx":
        cmd = _build_claude_npx_command(prompt, config)
    else:
        return LLMResult(
            1,
            "Unknown LLM backend: {}. Supported: claude, claude-npx, gemini, auto".format(backend),
            "",
            "",
        )

    if dry_run:
        return LLMResult(0, "DRY RUN: would execute {}\n\nPrompt length: {} chars".format(" ".join(cmd[:5]) + " ...", len(prompt)), "", "")

    timeout_seconds = int(llm.get("timeout_seconds") or 300)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(work_dir),
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        body = result.stdout.strip()
        if not body:
            body = result.stderr.strip()
        return LLMResult(result.returncode, body, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return LLMResult(1, "LLM backend timed out after {} seconds".format(timeout_seconds), "", "")
    except FileNotFoundError as exc:
        return LLMResult(1, "LLM backend not found: {}".format(exc), "", "")
