from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "github": {
        "api_url": "https://api.github.com",
        "owner": "rameshchandranerolu",
        "repo": "codex_rag_runner",
        "token_env": "GITHUB_TOKEN",
        "queued_label": "codex:queued",
        "running_label": "codex:running",
        "done_label": "codex:done",
        "failed_label": "codex:failed",
        "allowed_authors": [],
        "poll_interval_seconds": 30,
        "request_timeout_seconds": 90,
    },
    "rag": {
        "source_dir": "/scratch/rnerolu/codex-rag/files",
        "index_dir": "/scratch/rnerolu/codex-rag/index",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_cache_dir": "/scratch/rnerolu/codex-rag/model-cache",
        "collection_name": "study_files",
        "chunk_chars": 1800,
        "chunk_overlap": 250,
        "top_k": 8,
        "min_support_score": 0.2,
        "allowed_extensions": [".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".sql", ".py", ".java", ".js", ".html", ".xml"],
    },
    "codex": {
        "binary": "codex",
        "sandbox": "workspace-write",
        "extra_args": [
            "--skip-git-repo-check",
            "--config",
            'approval_policy=on-request',
            "--config",
            'model_service_tier="instant"',
            "--config",
            'model_reasoning_effort="medium"',
            "--model",
            "gpt-5.5",
        ],
        "timeout_seconds": 900,
    },
    "runner": {
        "work_dir": "/scratch/rnerolu/codex-rag/runs",
        "lock_file": "/scratch/rnerolu/codex-rag/runner.lock",
        "max_issues_per_poll": 1,
        "comment_max_chars": 60000,
    },
}


class ConfigError(Exception):
    pass


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError("Config file not found: {}".format(path)) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError("Invalid JSON in {}: {}".format(path, exc)) from exc
    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a JSON object: {}".format(path))
    config = deep_merge(DEFAULT_CONFIG, data)
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: Dict[str, Any]) -> None:
    github = config.get("github") or {}
    for key in ("owner", "repo"):
        if not str(github.get(key) or "").strip():
            raise ConfigError("Missing github.{}".format(key))
    rag = config.get("rag") or {}
    for key in ("source_dir", "index_dir", "collection_name"):
        if not str(rag.get(key) or "").strip():
            raise ConfigError("Missing rag.{}".format(key))
    if int(rag.get("chunk_chars") or 0) <= int(rag.get("chunk_overlap") or 0):
        raise ConfigError("rag.chunk_chars must be greater than rag.chunk_overlap")
    if int(rag.get("top_k") or 0) <= 0:
        raise ConfigError("rag.top_k must be positive")
