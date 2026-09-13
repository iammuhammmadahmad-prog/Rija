"""
Transformer decoder block: classic (GELU+LayerNorm) or modern (SwiGLU+RMSNorm).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention, KVCache
from model.norms import RMSNorm


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    """LLaMA-style SwiGLU MLP used in ChatGPT-era open models."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        architecture: str = "classic",
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.architecture = architecture
        modern = architecture == "modern"

        Norm = RMSNorm if modern else nn.LayerNorm
        self.ln1 = Norm(d_model)
        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            n_kv_heads=n_kv_heads if modern else None,
            use_rope=modern,
            use_sdpa=True,
            max_seq_len=max_seq_len,
        )
        self.ln2 = Norm(d_model)
        self.ff = (
            SwiGLUFeedForward(d_model, d_ff, dropout=dropout)
            if modern
            else FeedForward(d_model, d_ff, dropout=dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        past_kv: KVCache = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed = self.ln1(x)
        attn_out, present_kv = self.attn(
            normed, normed, normed, mask=mask, past_kv=past_kv, start_pos=start_pos
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x, present_kv
