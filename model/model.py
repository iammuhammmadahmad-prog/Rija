"""
Decoder-only GPT language model.

architecture='classic' — GPT-2 style (learned PE, LayerNorm, GELU) — loads v1–v4
architecture='modern'  — ChatGPT-era style (RoPE, RMSNorm, SwiGLU, SDPA, optional GQA)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.embeddings import TokenEmbedding, LearnedPositionalEncoding
from model.transformer import TransformerBlock
from model.attention import causal_mask, causal_mask_with_past, KVCache
from model.norms import RMSNorm


class GPTConfig:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        architecture: str = "classic",
        n_kv_heads: Optional[int] = None,
        tie_weights: bool = True,
        gradient_checkpointing: bool = False,
        scale_embeddings: Optional[bool] = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.architecture = architecture
        self.n_kv_heads = n_kv_heads
        self.tie_weights = tie_weights
        self.gradient_checkpointing = gradient_checkpointing
        # Classic scales embeddings by sqrt(d); modern (LLaMA-like) usually does not
        if scale_embeddings is None:
            scale_embeddings = architecture != "modern"
        self.scale_embeddings = scale_embeddings


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        modern = config.architecture == "modern"

        self.token_embedding = TokenEmbedding(
            config.vocab_size, config.d_model, scale=config.scale_embeddings
        )
        # RoPE replaces absolute PE in modern mode
        self.positional_encoding = (
            None if modern else LearnedPositionalEncoding(config.d_model, config.max_seq_len)
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    config.dropout,
                    architecture=config.architecture,
                    n_kv_heads=config.n_kv_heads,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(config.d_model) if modern else nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.output_projection.weight = self.token_embedding.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor = None,
        past_kvs: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ):
        batch, seq_len = token_ids.shape
        past_len = 0
        if past_kvs is not None and past_kvs[0] is not None:
            past_len = past_kvs[0][0].size(2)

        total_len = past_len + seq_len
        if total_len > self.config.max_seq_len:
            raise AssertionError(
                f"sequence length {total_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        x = self.token_embedding(token_ids)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, start_pos=past_len)
        x = self.dropout(x)

        # For modern+SDPA prefills we can omit explicit mask; keep mask for cache/classic
        modern = self.config.architecture == "modern"
        if past_len == 0 and modern and not use_cache:
            mask = None
        elif past_len == 0:
            mask = causal_mask(seq_len, device=token_ids.device)
        else:
            mask = causal_mask_with_past(seq_len, total_len, device=token_ids.device)

        present_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and past_kvs is None
            and not use_cache
        )

        for i, block in enumerate(self.blocks):
            layer_past = past_kvs[i] if past_kvs is not None else None

            if use_ckpt:
                # checkpoint needs tensor outputs; wrap to also return kv (discarded in train)
                def _run(hidden, m=mask, blk=block, sp=past_len):
                    out, _kv = blk(hidden, mask=m, past_kv=None, start_pos=sp)
                    return out

                x = checkpoint(_run, x, use_reentrant=False)
                if use_cache:
                    present_kvs.append(None)  # unreachable in train path
            else:
                x, present = block(x, mask=mask, past_kv=layer_past, start_pos=past_len)
                if use_cache:
                    present_kvs.append(present)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        if use_cache:
            return logits, loss, present_kvs
        return logits, loss

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    for arch in ("classic", "modern"):
        config = GPTConfig(
            vocab_size=1000, d_model=64, num_heads=4, num_layers=2,
            d_ff=256, max_seq_len=32, architecture=arch,
        )
        model = GPTLanguageModel(config)
        x = torch.randint(0, 1000, (2, 16))
        logits, loss = model(x, x)
        print(arch, model.num_parameters(), logits.shape, float(loss))
