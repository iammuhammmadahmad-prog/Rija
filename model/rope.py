"""Rotary Positional Embeddings (RoPE) — used by LLaMA, GPT-NeoX, ChatGPT-era models."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim/2)
        # Duplicate for pair-wise rotate
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.max_seq_len = seq_len

    def forward(self, seq_len: int, start_pos: int = 0):
        end = start_pos + seq_len
        if end > self.max_seq_len:
            self._build_cache(end)
        cos = self.cos_cached[:, :, start_pos:end, :]
        sin = self.sin_cached[:, :, start_pos:end, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (batch, heads, seq, head_dim)
    cos/sin: broadcastable to x
    """
    return (x * cos) + (rotate_half(x) * sin)
