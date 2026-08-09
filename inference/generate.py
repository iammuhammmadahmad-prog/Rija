"""
Phase 8: Inference engine — load weights, encode input, generate tokens,
decode back to text. Supports greedy, temperature, top-k, and top-p decoding.

Optimizations (borrowed patterns from OmniRoute gateway ideas):
  - KV-cache for O(n) decode instead of recomputing the full context
  - Streaming token yields (SSE-style incremental delivery)
  - Response LRU cache for deterministic generations
  - Unified checkpoint loading (BPE + tiktoken wiki paths)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Literal

import torch
import torch.nn.functional as F

from model.model import GPTLanguageModel
from inference.cache import ResponseCache, make_cache_key
from inference.checkpoint import load_checkpoint, find_latest_checkpoint


DecodeStrategy = Literal["greedy", "temperature", "top_k", "top_p"]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    strategy: DecodeStrategy = "temperature"
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    use_cache: bool = True
    use_kv_cache: bool = True


class InferenceEngine:
    def __init__(
        self,
        model: GPTLanguageModel,
        tokenizer,
        device: Optional[torch.device] = None,
        model_id: str = "local",
        response_cache: Optional[ResponseCache] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id
        self.response_cache = response_cache

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: str = "tokenizer/vocab.json",
        tokenizer_kind: Optional[str] = None,
        enable_cache: bool = True,
    ):
        loaded = load_checkpoint(
            checkpoint_path,
            tokenizer_path=tokenizer_path,
            tokenizer_kind=tokenizer_kind,
        )
        cache = ResponseCache(max_size=64, disk_path="memory/store/response_cache.jsonl") if enable_cache else None
        return cls(
            loaded.model,
            loaded.tokenizer,
            model_id=checkpoint_path,
            response_cache=cache,
        )

    @classmethod
    def from_latest(cls, version: str = "v4_wiki", **kwargs):
        return cls.from_checkpoint(find_latest_checkpoint(version), **kwargs)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def _stop_ids(self, config: GenerationConfig) -> set:
        stop_ids = set(config.stop_token_ids or [])
        eos = None
        if hasattr(self.tokenizer, "token_to_id"):
            eos = self.tokenizer.token_to_id.get("<eos>")
        if eos is not None:
            stop_ids.add(eos)
        return stop_ids

    def _cacheable(self, config: GenerationConfig) -> bool:
        if self.response_cache is None:
            return False
        if config.strategy == "greedy":
            return True
        return config.strategy == "temperature" and config.temperature <= 0.0

    @torch.no_grad()
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Return the completion text only (not including the prompt)."""
        config = config or GenerationConfig()
        key = None

        if self._cacheable(config):
            key = make_cache_key(
                self.model_id,
                prompt,
                temperature=config.temperature,
                strategy=config.strategy,
                top_k=config.top_k,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
            )
            hit = self.response_cache.get(key)
            if hit is not None:
                return hit

        pieces: List[str] = []
        for piece in self.generate_stream(prompt, config):
            pieces.append(piece)
        text = "".join(pieces)

        if key is not None and self.response_cache is not None:
            self.response_cache.put(key, text)
        return text

    @torch.no_grad()
    def generate_stream(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Yield decoded text increments as tokens are produced
        (OmniRoute-style streaming delivery for the local model).
        """
        config = config or GenerationConfig()
        ids = self.encode(prompt)
        if not ids:
            ids = [0]

        stop_ids = self._stop_ids(config)
        max_seq = self.model.config.max_seq_len
        generated = list(ids)
        past_kvs = None
        prev_decoded = self.decode(generated)

        # Prefill: run full prompt once (possibly truncated to max_seq)
        context = generated[-max_seq:]
        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)

        if config.use_kv_cache:
            logits, _, past_kvs = self.model(input_ids, use_cache=True)
        else:
            logits, _ = self.model(input_ids)
            past_kvs = None

        next_logits = logits[0, -1, :]
        next_logits = self._apply_repetition_penalty(next_logits, generated, config.repetition_penalty)
        next_id = self._sample(next_logits, config)

        for _ in range(config.max_new_tokens):
            if next_id in stop_ids:
                break

            generated.append(next_id)
            decoded = self.decode(generated)
            # Yield only the new suffix (handles multi-byte BPE merges cleanly)
            if decoded.startswith(prev_decoded):
                delta = decoded[len(prev_decoded) :]
            else:
                delta = decoded
            if delta:
                yield delta
            prev_decoded = decoded

            # Decode step
            if config.use_kv_cache and past_kvs is not None:
                # Drop cache if we would exceed max_seq_len; fall back to window recompute
                cache_len = past_kvs[0][0].size(2)
                if cache_len >= max_seq:
                    past_kvs = None
                    context = generated[-max_seq:]
                    step_ids = torch.tensor([context], dtype=torch.long, device=self.device)
                    logits, _, past_kvs = self.model(step_ids, use_cache=True)
                else:
                    step_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    logits, _, past_kvs = self.model(step_ids, past_kvs=past_kvs, use_cache=True)
            else:
                context = generated[-max_seq:]
                step_ids = torch.tensor([context], dtype=torch.long, device=self.device)
                logits, _ = self.model(step_ids)

            next_logits = logits[0, -1, :]
            next_logits = self._apply_repetition_penalty(
                next_logits, generated, config.repetition_penalty
            )
            next_id = self._sample(next_logits, config)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, token_ids: List[int], penalty: float
    ) -> torch.Tensor:
        if penalty is None or abs(penalty - 1.0) < 1e-6:
            return logits
        out = logits.clone()
        for token_id in set(token_ids):
            if out[token_id] < 0:
                out[token_id] *= penalty
            else:
                out[token_id] /= penalty
        return out

    def _sample(self, logits: torch.Tensor, config: GenerationConfig) -> int:
        if config.strategy == "greedy" or config.temperature <= 0:
            return int(torch.argmax(logits).item())

        scaled = logits / max(config.temperature, 1e-6)

        if config.strategy == "top_k":
            return self._top_k_sample(scaled, config.top_k)
        if config.strategy == "top_p":
            return self._top_p_sample(scaled, config.top_p)

        # temperature with optional top-k / top-p soft filters
        if config.top_k and config.top_k > 0:
            k = min(config.top_k, scaled.size(-1))
            top_vals, _ = torch.topk(scaled, k)
            scaled = scaled.masked_fill(scaled < top_vals[-1], float("-inf"))
        if config.top_p is not None and config.top_p < 1.0:
            return self._top_p_sample(scaled, config.top_p)

        probs = F.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def _top_k_sample(logits: torch.Tensor, k: int) -> int:
        k = min(k, logits.size(-1))
        top_vals, top_idx = torch.topk(logits, k)
        probs = F.softmax(top_vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(top_idx[choice].item())

    @staticmethod
    def _top_p_sample(logits: torch.Tensor, p: float) -> int:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)

        # Keep tokens with cumulative prob <= p (always keep at least one)
        mask = cumulative - probs > p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

        filtered_probs = F.softmax(sorted_logits, dim=-1)
        choice = torch.multinomial(filtered_probs, num_samples=1)
        return int(sorted_idx[choice].item())
