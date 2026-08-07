"""
Phase 4: Embedding layer + positional encoding, implemented from scratch.
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Maps token IDs -> dense vectors. Just a learned lookup table,
    scaled by sqrt(d_model) as in the original Transformer paper so the
    embedding magnitudes are on a similar scale to the positional encodings."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) sinusoidal positional encoding, from
    'Attention Is All You Need'. Added to token embeddings so the model
    knows where each token sits in the sequence."""

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        # register as buffer: moves with .to(device), not trained, saved in state_dict
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """Alternative to sinusoidal: a learned embedding table for positions
    (like GPT-2 uses). Swap this in for SinusoidalPositionalEncoding if
    you'd rather the model learn position representations from data."""

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        return x + self.position_embedding(positions)
