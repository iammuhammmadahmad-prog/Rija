"""
Seed the knowledge base with factual snippets for RAG.
Also indexes them into SearchIndex.

Usage:
    python -m memory.seed_facts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from memory.knowledge_base import KnowledgeBase
from memory.search import SearchIndex
from myai_datasets.qa_dataset import load_qa_jsonl


def seed_from_qa(
    qa_path: str = "myai_datasets/qa_seed.jsonl",
    base_dir: str = "memory/store",
) -> int:
    kb = KnowledgeBase(base_dir=base_dir)
    rows = load_qa_jsonl(qa_path)
    for i, row in enumerate(rows):
        doc_id = f"fact_{i:03d}"
        content = f"Q: {row['question']}\nA: {row['answer']}"
        kb.add_document(doc_id, content, metadata={"source": "qa_seed", "question": row["question"]})
    return len(rows)


def build_index(base_dir: str = "memory/store") -> SearchIndex:
    kb = KnowledgeBase(base_dir=base_dir)
    index = SearchIndex()
    for doc in kb.load_all_documents():
        index.add(doc["id"], doc["content"])
    return index


def main():
    n = seed_from_qa()
    index = build_index()
    print(f"Seeded {n} fact documents into memory/store/documents/")
    print(f"Search index size: {len(index.documents)} docs")
    hits = index.search("capital of France", top_k=2)
    for doc_id, score, snip in hits:
        print(f"  hit {doc_id} ({score:.2f}): {snip[:80]}")


if __name__ == "__main__":
    main()
