"""
Phase 14: Evaluation suite — perplexity, accuracy, speed, memory usage.
"""

import time
import math
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from model.model import GPTLanguageModel
from trainer.dataset import TextChunkDataset
from tokenizer.tokenizer import BPETokenizer


@torch.no_grad()
def compute_perplexity(model: GPTLanguageModel, dataloader: DataLoader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))  # cap to avoid overflow


@torch.no_grad()
def compute_accuracy(model: GPTLanguageModel, dataloader: DataLoader, device) -> float:
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, _ = model(inputs)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()

    return correct / max(total, 1)


def benchmark_speed(
    model: GPTLanguageModel,
    seq_len: int,
    device,
    num_runs: int = 20,
) -> Dict[str, float]:
    model.eval()
    dummy = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    # Warmup
    for _ in range(3):
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (num_runs * seq_len) / elapsed
    return {
        "forward_passes": num_runs,
        "total_seconds": elapsed,
        "tokens_per_second": tokens_per_sec,
    }


def evaluate_model(
    model: GPTLanguageModel,
    tokenizer: BPETokenizer,
    val_path: str,
    seq_len: int,
    batch_size: int = 8,
    device=None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TextChunkDataset(val_path, tokenizer, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    perplexity = compute_perplexity(model, loader, device)
    accuracy = compute_accuracy(model, loader, device)
    speed = benchmark_speed(model, min(seq_len, 64), device)

    param_count = model.num_parameters()
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    return {
        "perplexity": perplexity,
        "token_accuracy": accuracy,
        "parameters": param_count,
        "model_size_mb": mem_mb,
        **speed,
    }
