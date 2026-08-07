import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
import torch
import torch.nn as nn

from model.model import GPTConfig, GPTLanguageModel
from myai_datasets.wikipedia_loader import get_wiki_dataloader

def train_wikipedia(
    steps: int = 5000,
    batch_size: int = 16,
    seq_len: int = 128,
    lr: float = 3e-4,
    save_every: int = 1000,
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
    print(f"[Model Init] Parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dataloader = get_wiki_dataloader(batch_size=batch_size, seq_len=seq_len)
    data_iter = iter(dataloader)

    model.train()
    start_time = time.time()
    
    ckpt_dir = Path("checkpoints/v4_wiki")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting Wikipedia Training Run ---")
    for step in range(1, steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            elapsed = time.time() - start_time
            print(f"[step {step}/{steps}] loss={loss.item():.4f} | time={elapsed:.1f}s")

        if step % save_every == 0 or step == steps:
            save_path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "step": step,
            }, save_path)
            print(f" -> Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_wikipedia(steps=1000, batch_size=8, seq_len=128)