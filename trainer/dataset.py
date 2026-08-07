"""
Phase 5: Dataset loading, chunking, and batching for language-model training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional

from tokenizer.tokenizer import BPETokenizer


class TextChunkDataset(Dataset):
    """
    Reads a text file, encodes it once with the tokenizer, and yields
    fixed-length chunks of token IDs for next-token prediction.
    """

    def __init__(
        self,
        text_path: str,
        tokenizer: BPETokenizer,
        seq_len: int = 128,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len

        text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
        self.tokens: List[int] = tokenizer.encode(text)

        # Each sample is seq_len+1 tokens so we can form input/target pairs
        self.chunks: List[List[int]] = []
        for start in range(0, len(self.tokens) - seq_len, self.stride):
            chunk = self.tokens[start : start + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.chunks.append(chunk)

        if not self.chunks and len(self.tokens) > 1:
            padded = self.tokens + [self.tokens[-1]] * (seq_len + 1 - len(self.tokens))
            self.chunks.append(padded[: seq_len + 1])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def create_dataloader(
    text_path: str,
    tokenizer: BPETokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TextChunkDataset(text_path, tokenizer, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
