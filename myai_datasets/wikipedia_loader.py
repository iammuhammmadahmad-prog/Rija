"""
Fast Wikipedia token stream for pretraining.

Faster than the original per-article / per-window path:
- encode_ordinary_batch on article groups
- numpy packing instead of one torch.tensor per sequence
- <|endoftext|> document separators (cleaner next-token signal)
- a producer thread so tokenization overlaps with the training step
"""

from __future__ import annotations

import os
import threading
from queue import Empty, Full, Queue
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


_SENTINEL = object()


def append_document(buffer: list[int], token_ids: list[int], eos_id: int) -> None:
    """Append one document and an EOS boundary (unless already present)."""
    if not token_ids:
        return
    buffer.extend(token_ids)
    if token_ids[-1] != eos_id:
        buffer.append(eos_id)


def windows_from_tokens(tokens: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack tokens into (N, seq_len+1) windows with a 1-token overlap
    (same stride as the original loader: buffer = buffer[seq_len:]).
    """
    arr = np.asarray(tokens, dtype=np.int64)
    need = seq_len + 1
    if arr.size < need:
        return np.empty((0, need), dtype=np.int64), arr
    n = (arr.size - need) // seq_len + 1
    starts = np.arange(n, dtype=np.int64) * seq_len
    index = starts[:, None] + np.arange(need, dtype=np.int64)
    leftover_start = n * seq_len
    return arr[index], arr[leftover_start:]


def _wiki_text_batches(dataset, batch_articles: int, min_chars: int) -> Iterator[list[str]]:
    try:
        for batch in dataset.iter(batch_size=batch_articles):
            texts = batch["text"] if isinstance(batch, dict) else [batch["text"]]
            kept = [t for t in texts if t and len(t) >= min_chars]
            if kept:
                yield kept
        return
    except AttributeError:
        pass
    except Exception:
        pass

    bucket: list[str] = []
    for item in dataset:
        text = item.get("text") or ""
        if len(text) < min_chars:
            continue
        bucket.append(text)
        if len(bucket) >= batch_articles:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def iter_stream_windows(
    seq_len: int,
    lang: str = "en",
    split: str = "train",
    timeout: float = 60.0,
    min_chars: int = 200,
    batch_articles: int = 32,
) -> Iterator[np.ndarray]:
    from datasets import load_dataset
    import tiktoken

    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split=split,
        streaming=True,
        storage_options={"client_kwargs": {"timeout": timeout}},
    )
    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc.eot_token
    pending: list[int] = []

    for texts in _wiki_text_batches(dataset, batch_articles, min_chars):
        encoded = enc.encode_ordinary_batch(texts)
        for ids in encoded:
            append_document(pending, ids, eos_id)
        windows, leftover = windows_from_tokens(
            np.asarray(pending, dtype=np.int64), seq_len
        )
        pending = leftover.tolist()
        for row in windows:
            yield row


def iter_wiki_batches(
    batch_size: int,
    seq_len: int,
    **window_kwargs,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    rows: list[np.ndarray] = []
    for row in iter_stream_windows(seq_len=seq_len, **window_kwargs):
        rows.append(row)
        if len(rows) >= batch_size:
            packed = np.stack(rows, axis=0)
            rows.clear()
            x = torch.from_numpy(np.ascontiguousarray(packed[:, :-1]))
            y = torch.from_numpy(np.ascontiguousarray(packed[:, 1:]))
            yield x, y


class WikipediaStreamDataset(IterableDataset):
    """Iterable of single (x, y) sequences — kept for compatibility."""

    def __init__(
        self,
        seq_len=256,
        lang="en",
        split="train",
        timeout=60.0,
        min_chars: int = 200,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.lang = lang
        self.split = split
        self.timeout = timeout
        self.min_chars = min_chars

    def __iter__(self):
        for row in iter_stream_windows(
            seq_len=self.seq_len,
            lang=self.lang,
            split=self.split,
            timeout=self.timeout,
            min_chars=self.min_chars,
        ):
            yield torch.from_numpy(row[:-1].copy()), torch.from_numpy(row[1:].copy())


class WikiPrefetchLoader:
    """Yields pinned (x, y) batches from a background tokenize/pack thread."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        prefetch: int = 8,
        pin_memory: Optional[bool] = None,
        worker_affinity: Optional[Iterable[int]] = None,
        **window_kwargs,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.prefetch = max(2, prefetch)
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.worker_affinity = set(worker_affinity) if worker_affinity is not None else None
        self.window_kwargs = window_kwargs

    def __iter__(self):
        q: Queue = Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def _produce():
            if self.worker_affinity:
                try:
                    os.sched_setaffinity(0, self.worker_affinity)
                except (OSError, AttributeError):
                    pass
            try:
                for x, y in iter_wiki_batches(
                    self.batch_size, self.seq_len, **self.window_kwargs
                ):
                    if stop.is_set():
                        break
                    if self.pin_memory:
                        x = x.pin_memory()
                        y = y.pin_memory()
                    while not stop.is_set():
                        try:
                            q.put((x, y), timeout=0.2)
                            break
                        except Full:
                            continue
                q.put(_SENTINEL)
            except Exception as exc:
                q.put(exc)

        worker = threading.Thread(target=_produce, name="wiki-prefetch", daemon=True)
        worker.start()
        try:
            while True:
                try:
                    item = q.get(timeout=600.0)
                except Empty as exc:
                    raise RuntimeError(
                        "Wikipedia prefetch timed out after 600s — "
                        "check network access to Hugging Face."
                    ) from exc
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()


def get_wiki_dataloader(
    batch_size=32,
    seq_len=256,
    prefetch: int = 8,
    pin_memory: Optional[bool] = None,
    threaded: bool = True,
    worker_affinity: Optional[Iterable[int]] = None,
):
    """
    Return an iterable of (x, y) LongTensor batches.

    threaded=True (default) overlaps tokenization with training.
    Set threaded=False to get a plain DataLoader (tests / debugging).
    """
    if threaded:
        return WikiPrefetchLoader(
            batch_size=batch_size,
            seq_len=seq_len,
            prefetch=prefetch,
            pin_memory=pin_memory,
            worker_affinity=worker_affinity,
        )
    ds = WikipediaStreamDataset(seq_len=seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
    )


if __name__ == "__main__":
    print("Testing Wikipedia streaming pipeline...")
    loader = get_wiki_dataloader(batch_size=4, seq_len=128)
    for x, y in loader:
        print(f"Batch input shape: {x.shape}, Batch target shape: {y.shape}")
        break
    print("Streaming successful!")
