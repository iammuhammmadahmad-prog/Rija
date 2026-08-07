"""
Phase 4: Self-attention and multi-head attention, implemented from scratch
(i.e. explicit Q/K/V projections and the scaled dot-product formula, rather
than nn.MultiheadAttention).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query: torch.Tensor,
                                  key: torch.Tensor,
                                  value: torch.Tensor,
                                  mask: torch.Tensor = None) -> tuple:
    """
    query, key, value: (batch, num_heads, seq_len, head_dim)
    mask: broadcastable to (batch, num_heads, seq_len, seq_len); positions
          with mask == 0 are disallowed (get -inf before softmax).

    Returns: (output, attention_weights)
    """
    head_dim = query.size(-1)

    # (batch, heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)  # (batch, heads, seq_len, head_dim)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Splits d_model into num_heads parallel attention heads so the model can
    attend to information from different representation subspaces at once,
    then concatenates the results and projects back to d_model.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_output = self._combine_heads(attn_output)

        return self.dropout(self.w_out(attn_output))


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Lower-triangular mask so position i can only attend to positions <= i.
    This is what makes the model a proper autoregressive (next-token
    predicting) language model instead of seeing the future.
    Shape: (1, 1, seq_len, seq_len), broadcastable over batch and heads.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)
