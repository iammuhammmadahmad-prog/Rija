"""Phase 10: Long-term memory — profiles, documents, chat history, search."""

from memory.knowledge_base import KnowledgeBase
from memory.chat_history import ChatHistory
from memory.search import SearchIndex
from memory.compress import compress_context, compress_snippets

__all__ = [
    "KnowledgeBase",
    "ChatHistory",
    "SearchIndex",
    "compress_context",
    "compress_snippets",
]
