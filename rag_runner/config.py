from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "github": {
        "api_url": "https://api.github.com",
        "owner": "desimealsnow",
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
        "source_dir": "./files",
        "index_dir": "./index",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_cache_dir": "./model-cache",
        "collection_name": "study_files",
        "chunk_chars": 2500,
        "chunk_overlap": 400,
        "top_k": 10,
        "min_support_score": 0.2,
        "allowed_extensions": [".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".pdf"],
    },
    "llm": {
        "backend": "auto",
        "timeout_seconds": 300,
        "claude_args": ["--max-turns", "1", "--no-input"],
    },
    "runner": {
        "work_dir": "./runs",
        "lock_file": "./runner.lock",
        "max_issues_per_poll": 1,
        "comment_max_chars": 60000,
    },
}


class ConfigError(Exception):
    pass


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a deep copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_paths(config: Dict[str, Any], config_dir: pathlib.Path) -> None:
    """Resolve relative paths in *config* against *config_dir* in-place.

    The following keys are resolved:
      - rag.source_dir
      - rag.index_dir
      - rag.model_cache_dir
      - runner.work_dir
      - runner.lock_file
    """
    rag = config.get("rag") or {}
    for key in ("source_dir", "index_dir", "model_cache_dir"):
        raw = rag.get(key)
        if raw:
            p = pathlib.Path(raw)
            if not p.is_absolute():
                rag[key] = str((config_dir / p).resolve())

    runner = config.get("runner") or {}
    for key in ("work_dir", "lock_file"):
        raw = runner.get(key)
        if raw:
            p = pathlib.Path(raw)
            if not p.is_absolute():
                runner[key] = str((config_dir / p).resolve())


def validate_config(config: Dict[str, Any]) -> None:
    """Validate required configuration fields.

    Raises :class:`ConfigError` if a required field is missing or invalid.
    The ``llm`` section is optional — when present its values are used, but
    its absence does not cause a validation failure.
    """
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


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    """Load configuration from *path*, merging with defaults.

    Relative paths in the loaded config are resolved against the directory
    that contains *path*.
    """
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
    resolve_paths(config, path.parent.resolve())
    config["_config_path"] = str(path.resolve())
    return config
