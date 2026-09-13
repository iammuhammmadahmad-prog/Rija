"""
Unified checkpoint / tokenizer loading.

Inspired by OmniRoute's format-translation layer: normalize incompatible
checkpoint schemas (BPE Trainer vs Wikipedia/tiktoken) into one interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import torch

from model.model import GPTConfig, GPTLanguageModel


@runtime_checkable
class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list: ...
    def decode(self, ids: list) -> str: ...


@dataclass
class LoadedCheckpoint:
    model: GPTLanguageModel
    tokenizer: Any
    config: GPTConfig
    tokenizer_kind: str  # "bpe" | "tiktoken"
    step: Optional[int]
    meta: Dict[str, Any]


def _as_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    raise TypeError(f"Unsupported config type: {type(cfg)}")


_GPT_CONFIG_KEYS = (
    "vocab_size",
    "d_model",
    "num_heads",
    "num_layers",
    "d_ff",
    "max_seq_len",
    "dropout",
    "architecture",
    "n_kv_heads",
    "tie_weights",
    "gradient_checkpointing",
    "scale_embeddings",
)


def _extract_model_config(state: Dict[str, Any]) -> GPTConfig:
    # Prefer explicit model_config (Trainer format)
    for key in ("model_config", "config"):
        if key not in state:
            continue
        raw = _as_dict(state[key])
        filtered = {k: raw[k] for k in _GPT_CONFIG_KEYS if k in raw}
        if "vocab_size" in filtered:
            # Older checkpoints default to classic architecture
            filtered.setdefault("architecture", "classic")
            return GPTConfig(**filtered)

    raise KeyError(
        "Checkpoint missing model config. Expected 'model_config' or GPT fields in 'config'."
    )


def _extract_state_dict(state: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        return state

    for key in ("model_state_dict", "model_state", "model"):
        if key in state and isinstance(state[key], dict):
            sd = state[key]
            break
    else:
        # Raw state_dict (tensor values only)
        if all(isinstance(v, torch.Tensor) for v in state.values()):
            sd = state
        else:
            raise KeyError("Could not find model weights in checkpoint")

    # Strip torch.compile prefix
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def _infer_tokenizer_kind(config: GPTConfig, explicit: Optional[str] = None) -> str:
    if explicit in ("bpe", "tiktoken"):
        return explicit
    # GPT-2 BPE vocab used by the Wikipedia training path
    if config.vocab_size == 50257:
        return "tiktoken"
    return "bpe"


def _load_tokenizer(kind: str, tokenizer_path: str = "tokenizer/vocab.json"):
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

        class TikTokenWrapper:
            """Thin adapter so InferenceEngine can treat tiktoken like BPETokenizer."""

            def __init__(self, encoding):
                self._enc = encoding
                self.vocab_size = encoding.n_vocab
                # GPT-2 EOT; used as stop token when present
                self.token_to_id = {"<eos>": encoding.eot_token}

            def encode(self, text: str):
                return self._enc.encode(text, allowed_special={"<|endoftext|>"})

            def decode(self, ids):
                return self._enc.decode(ids)

        return TikTokenWrapper(enc)

    from tokenizer.tokenizer import BPETokenizer

    return BPETokenizer.load(tokenizer_path)


def find_latest_checkpoint(version: str = "v4_wiki", checkpoint_dir: str = "checkpoints") -> str:
    """Prefer latest.pt, else highest step_*.pt under checkpoints/{version}/."""
    root = Path(checkpoint_dir) / version
    latest = root / "latest.pt"
    if latest.exists():
        return str(latest)

    steps = sorted(root.glob("step_*.pt"))
    if steps:
        return str(steps[-1])

    raise FileNotFoundError(
        f"No checkpoint found under {root}. Train a model first "
        f"(e.g. python -m trainer.train_wiki)."
    )


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    tokenizer_path: str = "tokenizer/vocab.json",
    tokenizer_kind: Optional[str] = None,
) -> LoadedCheckpoint:
    """
    Load any Rija checkpoint format and pair it with the correct tokenizer.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, tuple):
        state = state[0]

    config = _extract_model_config(state)
    state_dict = _extract_state_dict(state)

    model = GPTLanguageModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    kind = _infer_tokenizer_kind(config, tokenizer_kind)
    tokenizer = _load_tokenizer(kind, tokenizer_path)

    return LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        config=config,
        tokenizer_kind=kind,
        step=state.get("step") if isinstance(state, dict) else None,
        meta={
            "path": str(path),
            "keys": list(state.keys()) if isinstance(state, dict) else [],
        },
    )
