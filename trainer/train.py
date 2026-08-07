"""
Phase 5: Full training system — forward pass, loss, backprop, checkpoints,
validation, resume. Phase 7 trains v1 from here.
"""

import json
import time
import random
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Any

import torch
from torch.optim import AdamW

from tokenizer.tokenizer import BPETokenizer
from model.model import GPTConfig, GPTLanguageModel
from trainer.config import TrainConfig
from trainer.dataset import create_dataloader


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Tokenizer
        tok_path = Path(config.tokenizer_path)
        if tok_path.exists():
            self.tokenizer = BPETokenizer.load(str(tok_path))
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {tok_path}. "
                "Run dataset preparation first or train a tokenizer."
            )

        # Model
        model_config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.model = GPTLanguageModel(model_config).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.paused = False
        self.log_file = Path(config.log_dir) / f"train_{config.model_version}.jsonl"

        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: Optional[str] = None, tag: str = "latest") -> str:
        ckpt_dir = Path(self.config.checkpoint_dir) / self.config.model_version
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            path = str(ckpt_dir / f"step_{self.step:06d}_{tag}.pt")

        state = {
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
           "model_config": vars(self.model.config) if hasattr(self.model.config, "__dict__") else self.model.config,
                "vocab_size": self.model.config.vocab_size,
                "d_model": self.model.config.d_model,
                "num_heads": self.model.config.num_heads,
                "num_layers": self.model.config.num_layers,
                "d_ff": self.model.config.d_ff,
                "max_seq_len": self.model.config.max_seq_len,
                "dropout": self.model.config.dropout,
            },
        
        torch.save(state, path)

        # Also write a convenient "latest" symlink-like copy
        latest = ckpt_dir / "latest.pt"
        torch.save(state, latest)
        return path

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.step = state.get("step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"[trainer] resumed from {path} at step {self.step}")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log(self, record: Dict[str, Any]) -> None:
        record["timestamp"] = time.time()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self) -> float:
        val_path = Path(self.config.val_path)
        if not val_path.exists():
            return float("nan")

        loader = create_dataloader(
            str(val_path),
            self.tokenizer,
            seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            _, loss = self.model(inputs, targets)
            total_loss += loss.item()
            total_batches += 1

        self.model.train()
        return total_loss / max(total_batches, 1)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self, progress_callback=None) -> Dict[str, Any]:
        train_loader = create_dataloader(
            self.config.train_path,
            self.tokenizer,
            seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        while self.step < self.config.max_steps:
            if self.paused:
                time.sleep(0.1)
                continue

            for inputs, targets in train_loader:
                if self.step >= self.config.max_steps:
                    break
                if self.paused:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                _, loss = self.model(inputs, targets)
                loss.backward()

                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                self.optimizer.step()
                self.step += 1
                running_loss += loss.item()

                if self.step % 50 == 0:
                    avg = running_loss / 50
                    elapsed = time.time() - start_time
                    record = {
                        "step": self.step,
                        "train_loss": avg,
                        "elapsed_s": elapsed,
                    }
                    self._log(record)
                    if progress_callback:
                        progress_callback(record)
                    else:
                        print(f"[step {self.step}] train_loss={avg:.4f}")
                    running_loss = 0.0

                if self.step % self.config.eval_every == 0:
                    val_loss = self.validate()
                    record = {"step": self.step, "val_loss": val_loss}
                    self._log(record)
                    if progress_callback:
                        progress_callback(record)
                    else:
                        print(f"[step {self.step}] val_loss={val_loss:.4f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(tag="best")

                if self.step % self.config.save_every == 0:
                    path = self.save_checkpoint()
                    print(f"[step {self.step}] checkpoint saved -> {path}")

        final_path = self.save_checkpoint(tag="final")
        summary = {
            "steps": self.step,
            "best_val_loss": self.best_val_loss,
            "checkpoint": final_path,
            "parameters": self.model.num_parameters(),
        }
        self._register_version(summary)
        return summary

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def _register_version(self, summary: Dict[str, Any]) -> None:
        """Phase 9: record what data this version learned from."""
        registry_path = Path("checkpoints") / "registry.json"
        registry: Dict[str, Any] = {}
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))

        registry[self.config.model_version] = {
            "train_path": self.config.train_path,
            "val_path": self.config.val_path,
            "tokenizer_path": self.config.tokenizer_path,
            "steps": summary["steps"],
            "best_val_loss": summary["best_val_loss"],
            "checkpoint": summary["checkpoint"],
            "parameters": summary["parameters"],
            "config": self.config.to_dict(),
        }
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def train_tokenizer_from_files(
    paths: list,
    output_path: str = "tokenizer/vocab.json",
    vocab_size: int = 2000,
) -> BPETokenizer:
    """Convenience: train and save a BPE tokenizer from raw text files."""
    tok = BPETokenizer()
    tok.train_from_files(paths, vocab_size=vocab_size)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(output_path)
    print(f"[tokenizer] vocab_size={tok.vocab_size}, saved to {output_path}")
    return tok
