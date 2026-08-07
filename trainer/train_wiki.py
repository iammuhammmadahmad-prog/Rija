import sys
import time
import math
import torch
import torch.nn as nn
from pathlib import Path

# Fix path resolution
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from model.model import GPTConfig, GPTLanguageModel
from myai_datasets.wikipedia_loader import get_wiki_dataloader

def get_lr(step, total_steps, max_lr=3e-4, min_lr=3e-5, warmup_steps=1000):
    """Cosine learning rate decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train_wikipedia(
    total_steps: int = 100000,
    batch_size: int = 32,      # Increased batch size for T4 GPU efficiency
    seq_len: int = 128,
    lr: float = 3e-4,
    save_every: int = 5000,
    resume_checkpoint: str = "checkpoints/v4_wiki/step_001000.pt" # Set to None to start fresh
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = GPTConfig(
        vocab_size=50257,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=seq_len,
        dropout=0.1,
    )

    model = GPTLanguageModel(config).to(device)

    start_step = 1
    ckpt_dir = Path("checkpoints/v4_wiki")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume logic
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"[Checkpoint] Resuming training from {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        start_step = checkpoint.get("step", 0) + 1
        print(f"[Checkpoint] Successfully loaded! Resuming from step {start_step}.")

    print(f"[Model Init] Parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dataloader = get_wiki_dataloader(batch_size=batch_size, seq_len=seq_len)
    data_iter = iter(dataloader)

    model.train()
    start_time = time.time()

    print(f"\n--- Starting Wikipedia Training Run ({start_step} -> {total_steps}) ---")
    for step in range(start_step, total_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Dynamic Learning Rate Schedule
        current_lr = get_lr(step, total_steps, max_lr=lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log progress every 100 steps
        if step % 100 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            print(f"[step {step}/{total_steps}] loss={loss.item():.4f} | lr={current_lr:.2e} | time={elapsed:.1f}s")

        # Save checkpoint periodically
        if step % save_every == 0 or step == total_steps:
            save_path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "step": step,
            }, save_path)
            print(f" -> Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_wikipedia(total_steps=100000, batch_size=32, seq_len=128, save_every=5000)