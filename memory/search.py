"""
Phase 10: Simple keyword search index for retrieval-augmented generation.
No external vector DB — uses TF-style term scoring for transparency.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Tuple


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class SearchIndex:
    """In-memory inverted index with TF-IDF-like scoring."""

    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.doc_freq: Counter = Counter()

    def add(self, doc_id: str, text: str) -> None:
        self.documents[doc_id] = text
        unique_terms = set(_tokenize(text))
        for term in unique_terms:
            self.doc_freq[term] += 1

    def remove(self, doc_id: str) -> None:
        if doc_id not in self.documents:
            return
        unique_terms = set(_tokenize(self.documents[doc_id]))
        del self.documents[doc_id]
        for term in unique_terms:
            self.doc_freq[term] -= 1
            if self.doc_freq[term] <= 0:
                del self.doc_freq[term]

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        n_docs = len(self.documents)
        scores: List[Tuple[str, float]] = []

        for doc_id, text in self.documents.items():
            terms = _tokenize(text)
            tf = Counter(terms)
            score = 0.0
            for qt in query_terms:
                if qt in tf:
                    idf = math.log((n_docs + 1) / (self.doc_freq.get(qt, 0) + 1)) + 1
                    score += (1 + math.log(1 + tf[qt])) * idf
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in scores[:top_k]:
            snippet = self.documents[doc_id][:300]
            results.append((doc_id, score, snippet))
        return results

    def build_context(self, query: str, top_k: int = 3, max_chars: int = 1500) -> str:
        """Return concatenated relevant snippets to prepend to a prompt."""
        hits = self.search(query, top_k=top_k)
        parts = []
        total = 0
        for doc_id, score, snippet in hits:
            block = f"[{doc_id}] {snippet}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)
