"""
Phase 8: Inference engine — load weights, encode input, generate tokens,
decode back to text. Supports greedy, temperature, top-k, and top-p decoding.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Literal

import torch
import torch.nn.functional as F

from tokenizer.tokenizer import BPETokenizer
from model.model import GPTConfig, GPTLanguageModel


DecodeStrategy = Literal["greedy", "temperature", "top_k", "top_p"]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    strategy: DecodeStrategy = "temperature"
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    stop_token_ids: Optional[List[int]] = None


class InferenceEngine:
    def __init__(
        self,
        model: GPTLanguageModel,
        tokenizer: BPETokenizer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer_path: str = "tokenizer/vocab.json"):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Unpack tuple if saved as a single-element tuple
        if isinstance(state, tuple):
            state = state[0]
            
        model_cfg = state["model_config"]
        
        # Convert config object to dict if it wasn't serialized as a dict
        if hasattr(model_cfg, "__dict__"):
            model_cfg = vars(model_cfg)
            
        config = GPTConfig(**model_cfg)
        model = GPTLanguageModel(config)
        model.load_state_dict(state["model_state_dict"])

        tokenizer = BPETokenizer.load(tokenizer_path)
        return cls(model, tokenizer)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        ids = self.encode(prompt)
        generated = list(ids)

        eos_id = self.tokenizer.token_to_id.get("<eos>")
        stop_ids = set(config.stop_token_ids or [])
        if eos_id is not None:
            stop_ids.add(eos_id)

        for _ in range(config.max_new_tokens):
            context = generated[-self.model.config.max_seq_len :]
            input_ids = torch.tensor([context], dtype=torch.long, device=self.device)

            logits, _ = self.model(input_ids)
            next_logits = logits[0, -1, :]

            next_id = self._sample(next_logits, config)

            if next_id in stop_ids:
                break

            generated.append(next_id)

        return self.decode(generated)

    def _sample(self, logits: torch.Tensor, config: GenerationConfig) -> int:
        if config.strategy == "greedy":
            return int(torch.argmax(logits).item())

        scaled = logits / max(config.temperature, 1e-6)

        if config.strategy == "top_k":
            return self._top_k_sample(scaled, config.top_k)
        if config.strategy == "top_p":
            return self._top_p_sample(scaled, config.top_p)

        # temperature (full-vocab sampling)
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
