from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class SourceFile:
    path: pathlib.Path
    rel_path: str
    content: str
    content_hash: str
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    rel_path: str
    chunk_index: int
    text: str
    content_hash: str
    start_char: int
    end_char: int


def normalize_extensions(values: Iterable[str]) -> List[str]:
    result = []
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = "." + text
        if text not in result:
            result.append(text)
    return result


def discover_source_files(source_dir: pathlib.Path, allowed_extensions: Iterable[str]) -> List[pathlib.Path]:
    extensions = set(normalize_extensions(allowed_extensions))
    if not source_dir.is_dir():
        return []
    files = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return sorted(files)


def read_source_file(path: pathlib.Path, source_dir: pathlib.Path) -> SourceFile:
    content = path.read_text(encoding="utf-8", errors="replace")
    stat = path.stat()
    rel_path = path.relative_to(source_dir).as_posix()
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    return SourceFile(path=path, rel_path=rel_path, content=content, content_hash=digest, mtime_ns=stat.st_mtime_ns, size=stat.st_size)


def chunk_source(source: SourceFile, chunk_chars: int, chunk_overlap: int) -> List[Chunk]:
    text = source.content
    if not text.strip():
        return []
    chunks: List[Chunk] = []
    step = max(1, chunk_chars - chunk_overlap)
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            split = text.rfind("\n\n", start, end)
            if split > start + chunk_chars // 2:
                end = split
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = "{}::{}::{}".format(source.rel_path, source.content_hash[:12], index)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    rel_path=source.rel_path,
                    chunk_index=index,
                    text=chunk_text,
                    content_hash=source.content_hash,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + step)
    return chunks


def load_corpus(config: Dict[str, object]) -> List[Chunk]:
    rag = config.get("rag") if isinstance(config.get("rag"), dict) else {}
    source_dir = pathlib.Path(str(rag.get("source_dir") or "")).expanduser()
    extensions = normalize_extensions(rag.get("allowed_extensions") or [])
    chunk_chars = int(rag.get("chunk_chars") or 1800)
    chunk_overlap = int(rag.get("chunk_overlap") or 250)
    chunks: List[Chunk] = []
    for path in discover_source_files(source_dir, extensions):
        chunks.extend(chunk_source(read_source_file(path, source_dir), chunk_chars, chunk_overlap))
    return chunks
