"""
Two-tier response cache for deterministic generations.

Adapted from OmniRoute's semantic cache idea (LRU + optional disk),
simplified for local MyAI: in-memory LRU keyed by SHA-256 of
(model_id + prompt + generation settings). Only caches greedy /
temperature=0 style requests by default.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def make_cache_key(
    model_id: str,
    prompt: str,
    *,
    temperature: float,
    strategy: str,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
) -> str:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "temperature": temperature,
        "strategy": strategy,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


class ResponseCache:
    """Thread-safe LRU cache with optional JSONL persistence."""

    def __init__(
        self,
        max_size: int = 128,
        ttl_seconds: Optional[float] = 1800.0,
        disk_path: Optional[str] = None,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.disk_path = Path(disk_path) if disk_path else None
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()
        if self.disk_path:
            self._load_disk()

    def _expired(self, entry: Dict[str, Any]) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - entry["ts"]) > self.ttl_seconds

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            if self._expired(entry):
                del self._store[key]
                self.stats.misses += 1
                return None
            self._store.move_to_end(key)
            self.stats.hits += 1
            return entry["text"]

    def put(self, key: str, text: str) -> None:
        with self._lock:
            self._store[key] = {"text": text, "ts": time.time()}
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
            if self.disk_path:
                self._append_disk(key, text)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.stats = CacheStats()

    def _append_disk(self, key: str, text: str) -> None:
        try:
            self.disk_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.disk_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "text": text, "ts": time.time()}) + "\n")
        except OSError:
            pass

    def _load_disk(self) -> None:
        if not self.disk_path or not self.disk_path.exists():
            return
        try:
            for line in self.disk_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self._store[row["key"]] = {"text": row["text"], "ts": row.get("ts", time.time())}
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
        except (OSError, json.JSONDecodeError):
            pass
