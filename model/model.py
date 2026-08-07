"""
Phase 4: The full transformer-based language model. Ties together:
  - token embedding
  - positional encoding
  - a stack of transformer blocks
  - final layer norm
  - output projection (tied to the token embedding weights)

This is a decoder-only, GPT-style architecture: good for next-token
prediction / text generation.
"""

import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding, LearnedPositionalEncoding
from model.transformer import TransformerBlock
from model.attention import causal_mask


class GPTConfig:
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 d_ff: int = 1024,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.positional_encoding = LearnedPositionalEncoding(config.d_model, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

        # Output projection: tied to the input embedding weights (weight
        # tying) — a well-known trick that reduces parameter count and
        # tends to improve small-model quality.
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        token_ids: (batch, seq_len) integer token IDs
        targets:   (batch, seq_len) integer token IDs shifted by one, or None

        Returns: (logits, loss) where loss is None if targets is None.
        """
        batch, seq_len = token_ids.shape
        assert seq_len <= self.config.max_seq_len, "sequence longer than max_seq_len"

        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        mask = causal_mask(seq_len, device=token_ids.device)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.final_norm(x)
        logits = self.output_projection(x)  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test with random data (no real training)
    config = GPTConfig(vocab_size=1000, d_model=64, num_heads=4, num_layers=2,
                        d_ff=256, max_seq_len=32)
    model = GPTLanguageModel(config)
    print(f"Parameters: {model.num_parameters():,}")

    dummy_input = torch.randint(0, config.vocab_size, (2, 16))
    dummy_targets = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(dummy_input, dummy_targets)
    print(f"Logits shape: {logits.shape}")  # (2, 16, 1000)
    print(f"Loss: {loss.item():.4f}")
