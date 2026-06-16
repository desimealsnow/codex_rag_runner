from __future__ import annotations

from typing import Iterable

from .vector_store import RetrievedChunk


MODE_INSTRUCTIONS = {
    "ask": {
        "label": "Ask",
        "instruction": """Answer the user's question directly. Structure the answer with:
- A short direct answer.
- The explanation needed to understand it.
- Key terms or distinctions that matter for certification prep.
- A brief source coverage note if the retrieved chunks are narrow.""",
    },
    "teach": {
        "label": "Teach",
        "instruction": """Teach the concept as a certification tutor. Structure the answer with:
- Plain-English intuition first.
- Step-by-step explanation.
- Why this matters for the exam.
- Common traps or confusions, only when supported by the chunks.
- Three quick self-check questions with answers.""",
    },
    "quiz": {
        "label": "Quiz",
        "instruction": """Create exam practice from the retrieved chunks. Structure the answer with:
- 5 to 8 practice questions.
- A mix of multiple-choice and short-answer questions when the chunks support it.
- An answer key with one-sentence cited rationale for each answer.
- Do not invent plausible distractors that contradict or go beyond the chunks.""",
    },
    "flashcards": {
        "label": "Flashcards",
        "instruction": """Create flashcards from the retrieved chunks. Structure the answer with:
- 8 to 12 concise Q/A cards.
- One concept per card.
- Citations on each answer.
- A final "Review first" list of the most important cards.""",
    },
    "revise": {
        "label": "Revise",
        "instruction": """Create a certification revision sheet from the retrieved chunks. Structure the answer with:
- Must-know points.
- Definitions and formulas, if present in the chunks.
- Comparisons or distinctions.
- Exam cues and memory hooks grounded in the chunks.
- A final checklist for revision.""",
    },
}


MODE_ALIASES = {
    "": "ask",
    "answer": "ask",
    "explain": "teach",
    "learn": "teach",
    "study": "teach",
    "practice": "quiz",
    "questions": "quiz",
    "test": "quiz",
    "flashcard": "flashcards",
    "cards": "flashcards",
    "revision": "revise",
    "review": "revise",
    "notes": "revise",
    "cheatsheet": "revise",
}


def normalize_study_mode(value: str) -> str:
    key = str(value or "").strip().lower().replace(" ", "-")
    key = key.replace("_", "-")
    if key in MODE_INSTRUCTIONS:
        return key
    return MODE_ALIASES.get(key, "ask")


def build_grounded_prompt(question: str, chunks: Iterable[RetrievedChunk], min_support_score: float, study_mode: str = "ask") -> str:
    chunk_list = list(chunks)
    mode = normalize_study_mode(study_mode)
    mode_config = MODE_INSTRUCTIONS[mode]
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
- Keep the answer useful for certification preparation.
- Do not create exam facts, examples, traps, or answer options unless they are grounded in the chunks.
- When creating quiz questions or flashcards, every answer must cite the chunks that support it.

Retrieval support:
- Best support score: {best_support:.3f}
- Minimum support score for confident answers: {min_support_score:.3f}
- If best support is below the minimum, refuse as unsupported unless the answer is plainly present in the chunks.

Study mode:
- Mode: {mode_label}
- Instructions:
{mode_instruction}

Retrieved chunks:
{context}

Question:
{question}
""".format(
        best_support=best_support,
        min_support_score=min_support_score,
        mode_label=mode_config["label"],
        mode_instruction=mode_config["instruction"],
        context=context,
        question=question.strip(),
    )
