"""
ChatGPT-scale ladder for Rija v5.

True ChatGPT / GPT-3.5 class models are ~175B params and need datacenter GPUs.
This module defines a practical scale ladder from laptop → single GPU → cluster,
using modern architecture defaults (RoPE, RMSNorm, SwiGLU).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

from model.model import GPTConfig


@dataclass
class ScalePreset:
    name: str
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    max_seq_len: int
    n_kv_heads: Optional[int]
    batch_size: int
    grad_accum: int
    learning_rate: float
    total_steps: int
    dropout: float
    architecture: str  # classic | modern
    hardware: str
    approx_params: str
    notes: str

    def to_gpt_config(self, vocab_size: int = 50257) -> GPTConfig:
        return GPTConfig(
            vocab_size=vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            architecture=self.architecture,
            n_kv_heads=self.n_kv_heads,
            tie_weights=True,
            gradient_checkpointing=self.name in {"large", "xl", "chatgpt"},
        )


# Rough GPT-3 / LLaMA-style width-depth points
SCALE_PRESETS: Dict[str, ScalePreset] = {
    "tiny": ScalePreset(
        name="tiny",
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        max_seq_len=256,
        n_kv_heads=None,
        batch_size=8,
        grad_accum=1,
        learning_rate=3e-4,
        total_steps=100_000,
        dropout=0.1,
        architecture="modern",
        hardware="laptop CPU/GPU",
        approx_params="~20M",
        notes="Fast iteration; similar to current v4 size with modern arch",
    ),
    "small": ScalePreset(
        name="small",
        d_model=512,
        num_layers=8,
        num_heads=8,
        d_ff=2048,
        max_seq_len=512,
        n_kv_heads=None,
        batch_size=4,
        grad_accum=2,
        learning_rate=2e-4,
        total_steps=100_000,
        dropout=0.1,
        architecture="modern",
        hardware="laptop GPU / 8–12GB VRAM",
        approx_params="~50–80M",
        notes="Default v5_smart modernized",
    ),
    "base": ScalePreset(
        name="base",
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=1024,
        n_kv_heads=None,
        batch_size=2,
        grad_accum=8,
        learning_rate=1.5e-4,
        total_steps=200_000,
        dropout=0.1,
        architecture="modern",
        hardware="single 16–24GB GPU",
        approx_params="~125M (GPT-2 small class)",
        notes="GPT-2 Small scale",
    ),
    "medium": ScalePreset(
        name="medium",
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
        max_seq_len=2048,
        n_kv_heads=8,  # GQA for memory
        batch_size=1,
        grad_accum=16,
        learning_rate=1e-4,
        total_steps=300_000,
        dropout=0.05,
        architecture="modern",
        hardware="24–48GB GPU",
        approx_params="~350M",
        notes="Serious single-GPU pretrain",
    ),
    "large": ScalePreset(
        name="large",
        d_model=2048,
        num_layers=24,
        num_heads=16,
        d_ff=8192,
        max_seq_len=2048,
        n_kv_heads=8,
        batch_size=1,
        grad_accum=32,
        learning_rate=6e-5,
        total_steps=500_000,
        dropout=0.05,
        architecture="modern",
        hardware="multi-GPU (A100/H100 class)",
        approx_params="~1.5B",
        notes="Needs gradient checkpointing + multi-GPU",
    ),
    "xl": ScalePreset(
        name="xl",
        d_model=4096,
        num_layers=32,
        num_heads=32,
        d_ff=11008,
        max_seq_len=4096,
        n_kv_heads=8,
        batch_size=1,
        grad_accum=64,
        learning_rate=3e-5,
        total_steps=1_000_000,
        dropout=0.0,
        architecture="modern",
        hardware="multi-node GPU cluster",
        approx_params="~7B (LLaMA-7B class)",
        notes="Open-weight ChatGPT-competitor size class",
    ),
    "chatgpt": ScalePreset(
        name="chatgpt",
        d_model=12288,
        num_layers=96,
        num_heads=96,
        d_ff=49152,
        max_seq_len=8192,
        n_kv_heads=8,
        batch_size=1,
        grad_accum=512,
        learning_rate=1e-5,
        total_steps=1_000_000,
        dropout=0.0,
        architecture="modern",
        hardware="datacenter (thousands of GPUs)",
        approx_params="~175B (GPT-3 / ChatGPT-era class)",
        notes="Config reference only — not trainable on a laptop",
    ),
}


def get_preset(name: str) -> ScalePreset:
    key = name.lower().strip()
    if key not in SCALE_PRESETS:
        known = ", ".join(SCALE_PRESETS)
        raise ValueError(f"Unknown size '{name}'. Choose from: {known}")
    return SCALE_PRESETS[key]


def estimate_parameters(cfg: GPTConfig) -> int:
    """Rough non-compiled parameter count for a GPTConfig."""
    v, d, L, h = cfg.vocab_size, cfg.d_model, cfg.num_layers, cfg.num_heads
    d_ff = cfg.d_ff
    n_kv = cfg.n_kv_heads or h
    head_dim = d // h

    emb = v * d
    # attention: q,k,v,o (GQA reduces k/v)
    attn = L * (d * d + 2 * (n_kv * head_dim) * d + d * d)
    if cfg.architecture == "modern":
        # SwiGLU has 3 matrices
        ffn = L * (3 * d * d_ff)
        norms = L * 2 * d + d  # rms weights
        pe = 0
    else:
        ffn = L * (2 * d * d_ff + d_ff + d)  # with biases approx
        norms = L * 4 * d + 2 * d  # layernorm weight+bias
        pe = cfg.max_seq_len * d
    return emb + attn + ffn + norms + pe


def print_scale_table() -> None:
    print(f"{'size':<10} {'params':<18} {'arch':<8} {'hardware'}")
    print("-" * 72)
    for name, p in SCALE_PRESETS.items():
        cfg = p.to_gpt_config()
        est = estimate_parameters(cfg)
        print(f"{name:<10} {p.approx_params:<18} {est/1e6:>8.1f}M  {p.hardware}")
    print()
    print("ChatGPT-scale ≈ 'chatgpt' (~175B). Train 'small'/'base' locally;")
    print("use cloud multi-GPU for 'large'/'xl'. 'chatgpt' is a reference config.")


def preset_summary(name: str) -> dict:
    p = get_preset(name)
    cfg = p.to_gpt_config()
    return {
        **asdict(p),
        "estimated_parameters": estimate_parameters(cfg),
        "gpt_config": cfg.__dict__,
    }
