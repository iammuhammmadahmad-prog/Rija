"""
Retrieval-augmented prompt builder.

Small models answer much better when the fact is placed in context.
"""

from __future__ import annotations

from typing import Optional

from myai_datasets.qa_dataset import format_prompt
from memory.compress import compress_context


def build_rag_prompt(
    question: str,
    search_index,
    *,
    top_k: int = 3,
    max_chars: int = 1200,
    qa_style: bool = True,
) -> str:
    """
    Build a prompt with retrieved context.
    qa_style=True uses the fine-tune template so v5_qa checkpoints respond well.
    """
    context = ""
    if search_index is not None:
        raw = search_index.build_context(question, top_k=top_k, max_chars=max_chars, compress=True)
        if raw:
            context = compress_context(raw, max_chars=max_chars).text

    if qa_style:
        if context:
            return (
                f"### Context:\n{context}\n\n"
                f"{format_prompt(question)}"
            )
        return format_prompt(question)

    # Legacy wiki-continuation style
    if context:
        return (
            f"Context:\n{context}\n\n"
            f"Based on the context, answer clearly.\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    return f"Question: {question}\nAnswer:"


def extract_answer(text: str) -> str:
    """Strip prompt scaffolding from model output."""
    for marker in ("### Answer:", "\nAnswer:", "Answer:"):
        if marker in text:
            text = text.split(marker, 1)[1]
    # Stop at next section if model rambles into wiki templates
    for stop in ("\n### ", "\nReferences", "\nExternal links", "\nCategory:"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()
