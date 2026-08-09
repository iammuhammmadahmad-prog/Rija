import argparse
import math
import sys
import time
from pathlib import Path

import torch

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


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Prefer latest.pt, else highest step_XXXXXX.pt."""
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    steps = sorted(ckpt_dir.glob("step_*.pt"))
    return steps[-1] if steps else None


def _unwrap_state_dict(model: torch.nn.Module):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def _checkpoint_payload(model, config, step, optimizer=None):
    state_dict = _unwrap_state_dict(model)
    payload = {
        "model_state": state_dict,
        "model_state_dict": state_dict,
        "config": config.__dict__,
        "model_config": config.__dict__,
        "tokenizer_kind": "tiktoken",
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def train_wikipedia(
    total_steps: int = 200000,
    batch_size: int = 16,
    seq_len: int = 128,
    lr: float = 3e-4,
    save_every: int = 1000,
    resume_checkpoint: str | None = None,
    no_resume: bool = False,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    start_step = 1
    ckpt_dir = Path("checkpoints/v4_wiki")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if not no_resume:
        if resume_checkpoint:
            resume_path = Path(resume_checkpoint)
        else:
            resume_path = find_latest_checkpoint(ckpt_dir)

    if resume_path is not None and resume_path.exists():
        print(f"[Checkpoint] Resuming training from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state = (
            checkpoint.get("model_state")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
        # Strip torch.compile prefix if present
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("[Checkpoint] Optimizer state restored.")
            except Exception as e:
                print(f"[Checkpoint] Could not restore optimizer ({e}); continuing with fresh AdamW.")
        start_step = int(checkpoint.get("step", 0)) + 1
        print(f"[Checkpoint] Loaded! Resuming from step {start_step}.")
    else:
        print("[Checkpoint] No checkpoint found — starting from scratch.")

    if start_step > total_steps:
        print(f"[Done] Already at step {start_step - 1} >= total_steps={total_steps}. Nothing to do.")
        return

    # Optional torch.compile block with fallback guard
    try:
        model = torch.compile(model)
        print("[PyTorch] Model compiled successfully with torch.compile()")
    except Exception as e:
        print(f"[PyTorch] Skipping torch.compile (running in standard eager mode): {e}")

    dataloader = get_wiki_dataloader(batch_size=batch_size, seq_len=seq_len)
    data_iter = iter(dataloader)

    model.train()
    start_time = time.time()
    running_loss = 0.0
    running_count = 0

    print(f"\n--- Starting Wikipedia Training Run ({start_step} -> {total_steps}) ---")
    for step in range(start_step, total_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        current_lr = get_lr(step, total_steps, max_lr=lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        if device == "cpu":
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        running_count += 1

        if step % 100 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            avg = running_loss / max(running_count, 1)
            print(
                f"[step {step}/{total_steps}] loss={loss.item():.4f} | "
                f"avg100={avg:.4f} | lr={current_lr:.2e} | time={elapsed:.1f}s"
            )
            running_loss = 0.0
            running_count = 0

        if step % save_every == 0 or step == total_steps:
            payload = _checkpoint_payload(model, config, step, optimizer=optimizer)
            save_path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save(payload, save_path)
            torch.save(payload, ckpt_dir / "latest.pt")
            print(f" -> Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Resume / start Wikipedia pretraining")
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path (default: auto-detect latest under checkpoints/v4_wiki/)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and train from scratch",
    )
    args = parser.parse_args()

    train_wikipedia(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        save_every=args.save_every,
        resume_checkpoint=args.resume,
        no_resume=args.no_resume,
    )


if __name__ == "__main__":
    main()
