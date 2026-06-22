from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from .corpus import Chunk


def patch_sqlite() -> None:
    try:
        import pysqlite3  # type: ignore
    except ImportError:
        return
    sys.modules["sqlite3"] = pysqlite3


@dataclass(frozen=True)
class RetrievedChunk:
    rel_path: str
    chunk_index: int
    text: str
    distance: float
    support_score: float


class ChromaIndex:
    def __init__(self, config: Dict[str, Any]):
        patch_sqlite()
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        rag = config.get("rag") or {}
        self.client = chromadb.PersistentClient(
            path=str(rag.get("index_dir")),
            settings=Settings(anonymized_telemetry=False),
        )
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=str(rag.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"),
            device="cpu",
            normalize_embeddings=True,
            cache_folder=str(rag.get("model_cache_dir") or ""),
        )
        self.collection = self.client.get_or_create_collection(
            name=str(rag.get("collection_name") or "study_files"),
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )

    def rebuild(self, chunks: List[Chunk]) -> None:
        existing = self.collection.get(include=[])
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        if not chunks:
            return
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "rel_path": chunk.rel_path,
                    "chunk_index": chunk.chunk_index,
                    "content_hash": chunk.content_hash,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                for chunk in chunks
            ],
        )

    def query(self, question: str, top_k: int, min_support_score: float = 0.0) -> List[RetrievedChunk]:
        result = self.collection.query(query_texts=[question], n_results=top_k, include=["documents", "metadatas", "distances"])
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        retrieved: List[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            numeric_distance = float(distance)
            retrieved.append(
                RetrievedChunk(
                    rel_path=str((metadata or {}).get("rel_path") or ""),
                    chunk_index=int((metadata or {}).get("chunk_index") or 0),
                    text=str(document or ""),
                    distance=numeric_distance,
                    support_score=max(0.0, 1.0 - numeric_distance),
                )
            )
        return [r for r in retrieved if r.support_score >= min_support_score]
