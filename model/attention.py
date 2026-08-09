"""
Self-attention with optional RoPE, GQA, SDPA, and KV cache.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding, apply_rope

KVCache = Optional[Tuple[torch.Tensor, torch.Tensor]]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple:
    head_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        n_kv_heads: Optional[int] = None,
        use_rope: bool = False,
        use_sdpa: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads or num_heads
        assert num_heads % self.n_kv_heads == 0, "num_heads must be divisible by n_kv_heads"
        self.head_dim = d_model // num_heads
        self.n_rep = num_heads // self.n_kv_heads
        self.use_rope = use_rope
        self.use_sdpa = use_sdpa

        # Classic GPT-2-style layers keep bias for checkpoint compatibility.
        # Modern LLaMA-style layers are bias-free.
        bias = not use_rope
        self.w_q = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.w_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.w_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.w_out = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len) if use_rope else None

    def _shape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, heads, self.head_dim).transpose(1, 2)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        b, kv_h, t, d = x.shape
        x = x[:, :, None, :, :].expand(b, kv_h, self.n_rep, t, d)
        return x.reshape(b, kv_h * self.n_rep, t, d)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        past_kv: KVCache = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        q = self._shape(self.w_q(query), self.num_heads)
        k = self._shape(self.w_k(key), self.n_kv_heads)
        v = self._shape(self.w_v(value), self.n_kv_heads)

        if self.use_rope and self.rope is not None:
            cos, sin = self.rope(q.size(2), start_pos=start_pos)
            q = apply_rope(q, cos, sin)
            # keys use same absolute positions as the new tokens
            k = apply_rope(k, cos[..., : k.size(2), :], sin[..., : k.size(2), :])

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v)

        k_rep = self._repeat_kv(k)
        v_rep = self._repeat_kv(v)

        if self.use_sdpa and mask is None and past_kv is None:
            # Pure causal prefills — use fused SDPA / flash when available
            attn_output = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=None, dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        elif self.use_sdpa:
            # Build additive mask for SDPA from boolean mask
            attn_mask = None
            if mask is not None:
                # mask True/1 = keep; SDPA expects True = keep for bool masks in recent torch
                attn_mask = mask
            attn_output = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_output, _ = scaled_dot_product_attention(q, k_rep, v_rep, mask=mask)

        b, h, t, d = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, h * d)
        return self.dropout(self.w_out(attn_output)), present_kv


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def causal_mask_with_past(q_len: int, k_len: int, device=None) -> torch.Tensor:
    mask = torch.ones(q_len, k_len, device=device, dtype=torch.bool)
    if q_len > 1:
        past_len = k_len - q_len
        triangle = torch.tril(torch.ones(q_len, q_len, device=device, dtype=torch.bool))
        mask[:, past_len:] = triangle
    return mask.unsqueeze(0).unsqueeze(0)
