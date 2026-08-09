"""
Training configuration presets for different model sizes.
Phase 7 / Phase 15: start tiny, scale up as hardware allows.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class TrainConfig:
    # Data
    train_path: str = "myai_datasets/processed/train.txt"
    val_path: str = "myai_datasets/processed/val.txt"
    tokenizer_path: str = "tokenizer/vocab.json"

    # Model (tiny v1 defaults — ~1-5M params depending on vocab)
    vocab_size: int = 2000
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 2000
    eval_every: int = 200
    save_every: int = 500
    grad_clip: float = 1.0
    seed: int = 42

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    model_version: str = "v1"

    # Resume
    resume_from: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def tiny_v1() -> TrainConfig:
    """Smallest useful model for CPU / laptop proof-of-concept."""
    return TrainConfig(
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_len=64,
        batch_size=8,
        max_steps=1000,
    )


def small_v2() -> TrainConfig:
    """Step up after v1 works — still single-GPU friendly."""
    return TrainConfig(
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=256,
        batch_size=16,
        max_steps=10000,
        model_version="v2",
    )

def v4_wiki() -> TrainConfig:
    """Preset matching trainer/train_wiki.py (tiktoken GPT-2 vocab)."""
    return TrainConfig(
        train_path="myai_datasets/processed/train.txt",
        val_path="myai_datasets/processed/val.txt",
        tokenizer_path="tiktoken:gpt2",
        vocab_size=50257,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=128,
        batch_size=16,
        learning_rate=3e-4,
        max_steps=200000,
        eval_every=500,
        save_every=1000,
        model_version="v4_wiki",
    )


def v5_smart() -> TrainConfig:
    """
    Default v5 = scale preset 'small' (~50-80M, modern arch).
    For ChatGPT-scale ladder see: python -m trainer.train_v5 --list-sizes
    """
    return TrainConfig(
        train_path="myai_datasets/processed/train.txt",
        val_path="myai_datasets/processed/val.txt",
        tokenizer_path="tiktoken:gpt2",
        vocab_size=50257,
        d_model=512,
        num_heads=8,
        num_layers=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        batch_size=4,
        learning_rate=2e-4,
        max_steps=100000,
        eval_every=500,
        save_every=1000,
        model_version="v5_smart",
    )
