"""
Phase 4: Feed-forward network, layer normalization, residual connections,
and the full transformer decoder block, built from the pieces in
attention.py.
"""

import torch
import torch.nn as nn

from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network: two linear layers with a
    nonlinearity between them, applied independently to every position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One decoder block:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    (pre-norm architecture, used by GPT-2 and most modern LLMs, because it
    trains more stably than the original post-norm design).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention sub-layer with residual connection
        normed = self.ln1(x)
        x = x + self.dropout(self.attn(normed, normed, normed, mask=mask))

        # Feed-forward sub-layer with residual connection
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
