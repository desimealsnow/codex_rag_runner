from __future__ import annotations

from typing import Iterable

from .vector_store import RetrievedChunk


def build_grounded_prompt(question: str, chunks: Iterable[RetrievedChunk], min_support_score: float) -> str:
    chunk_list = list(chunks)
    context_parts = []
    for index, chunk in enumerate(chunk_list, start=1):
        context_parts.append(
            "[{}] file={} chunk={} support={:.3f}\n{}".format(
                index,
                chunk.rel_path,
                chunk.chunk_index,
                chunk.support_score,
                chunk.text.strip(),
            )
        )
    context = "\n\n".join(context_parts) if context_parts else "(no retrieved context)"
    best_support = max([chunk.support_score for chunk in chunk_list], default=0.0)
    return """You are a source-grounded study assistant.

Answer the user's question only through the lens of the retrieved file chunks.
Do not use outside knowledge as source truth. You may use general language only
to explain what the cited chunks already support.

Rules:
- Cite source chunks inline using [1], [2], etc.
- If the retrieved chunks do not support an answer, say: "The configured files do not cover this clearly."
- If the chunks conflict, describe the conflict and cite both sides.
- Do not mention implementation details of this RAG runner.
- Keep the answer concise but educational.

Retrieval support:
- Best support score: {best_support:.3f}
- Minimum support score for confident answers: {min_support_score:.3f}
- If best support is below the minimum, refuse as unsupported unless the answer is plainly present in the chunks.

Retrieved chunks:
{context}

Question:
{question}
""".format(
        best_support=best_support,
        min_support_score=min_support_score,
        context=context,
        question=question.strip(),
    )
