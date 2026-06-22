"""Shared RAG session state and query logic for CLI and web UIs."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .corpus import load_corpus
from .llm_client import run_llm
from .prompting import MODE_INSTRUCTIONS, build_grounded_prompt, normalize_study_mode
from .vector_store import ChromaIndex, RetrievedChunk


@dataclass
class QueryResult:
    answer: str
    chunks: List[RetrievedChunk]
    best_support: float


@dataclass
class RagSession:
    config: Dict[str, Any]
    dry_run: bool = False
    study_mode: str = "ask"
    index: ChromaIndex | None = None
    chunk_count: int = 0
    file_count: int = 0

    def build_index(self, force_rebuild: bool = False) -> str:
        """Build or load the vector index.

        When *force_rebuild* is False and a non-empty index already exists at
        ``rag.index_dir``, the existing Chroma collection is reused.

        Returns ``"loaded"``, ``"built"``, or ``"empty"``.
        """
        from .runner import log

        self.index = ChromaIndex(self.config)

        if not force_rebuild:
            existing = self.index.collection.get(include=[])
            ids = existing.get("ids") or []
            if ids:
                result = self.index.collection.get(include=["metadatas"])
                metadatas = result.get("metadatas") or []
                paths = set()
                for meta in metadatas:
                    if isinstance(meta, dict) and meta.get("rel_path"):
                        paths.add(str(meta["rel_path"]))
                self.chunk_count = len(ids)
                self.file_count = len(paths)
                log("Loaded existing index: {} chunks from {} files".format(self.chunk_count, self.file_count))
                return "loaded"

        chunks = load_corpus(self.config)
        log("Indexing {} chunks".format(len(chunks)))
        if chunks:
            self.index.rebuild(chunks)
        self.chunk_count = len(chunks)
        self.file_count = len({chunk.rel_path for chunk in chunks})
        if self.chunk_count:
            return "built"
        return "empty"

    def get_file_list(self) -> List[str]:
        if not self.index:
            return []
        try:
            result = self.index.collection.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            paths = set()
            for meta in metadatas:
                if isinstance(meta, dict):
                    rel_path = meta.get("rel_path")
                    if rel_path:
                        paths.add(str(rel_path))
            return sorted(paths)
        except Exception:
            return []

    def get_stats(self) -> Dict[str, Any]:
        rag = self.config.get("rag") or {}
        return {
            "source_dir": rag.get("source_dir"),
            "file_count": self.file_count,
            "chunk_count": self.chunk_count,
            "chunk_chars": rag.get("chunk_chars"),
            "chunk_overlap": rag.get("chunk_overlap"),
            "top_k": rag.get("top_k"),
            "min_support_score": rag.get("min_support_score"),
            "model_name": rag.get("model_name"),
            "study_mode": self.study_mode,
            "study_mode_label": MODE_INSTRUCTIONS.get(self.study_mode, {}).get("label", self.study_mode),
            "files": self.get_file_list(),
        }

    def set_study_mode(self, mode: str) -> str:
        self.study_mode = normalize_study_mode(mode)
        return self.study_mode

    def query(self, question: str, study_mode: str | None = None) -> QueryResult:
        if not self.index:
            raise RuntimeError("Vector index is not built")

        mode = normalize_study_mode(study_mode or self.study_mode)
        rag = self.config.get("rag") or {}
        top_k = int(rag.get("top_k") or 10)
        min_support = float(rag.get("min_support_score") or 0.2)

        chunks = self.index.query(question, top_k, min_support_score=min_support)
        if not chunks:
            return QueryResult(
                answer="No relevant content found in the indexed files for this question.",
                chunks=[],
                best_support=0.0,
            )

        best_support = max(chunk.support_score for chunk in chunks)
        prompt = build_grounded_prompt(question, chunks, min_support, mode)
        work_dir = (
            pathlib.Path(str((self.config.get("runner") or {}).get("work_dir") or "./runs")).expanduser()
            / "interactive"
        )
        result = run_llm(prompt, self.config, work_dir, dry_run=self.dry_run)

        if result.returncode != 0 and not result.body:
            answer = "LLM backend error (code {}): {}".format(
                result.returncode,
                result.stderr[:500] if result.stderr else "no output",
            )
        else:
            answer = result.body

        return QueryResult(answer=answer, chunks=chunks, best_support=best_support)


def chunk_to_dict(chunk: RetrievedChunk) -> Dict[str, Any]:
    preview = chunk.text.strip()
    if len(preview) > 280:
        preview = preview[:277] + "..."
    return {
        "file": chunk.rel_path,
        "chunk_index": chunk.chunk_index,
        "support_score": round(chunk.support_score, 3),
        "preview": preview,
    }
