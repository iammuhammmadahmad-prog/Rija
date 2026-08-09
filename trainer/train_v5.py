"""
v5 scale trainer — modern GPT architecture with a ChatGPT-scale size ladder.

Examples:
  python -m trainer.train_v5 --list-sizes
  python -m trainer.train_v5 --size small
  python -m trainer.train_v5 --size base --batch-size 1 --grad-accum 8
  python -m trainer.train_v5 --size chatgpt   # blocked unless --force-huge
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from model.model import GPTLanguageModel
from model.scaling import SCALE_PRESETS, estimate_parameters, get_preset, print_scale_table
from myai_datasets.wikipedia_loader import get_wiki_dataloader


HUGE_SIZES = {"large", "xl", "chatgpt"}


def get_lr(step, total_steps, max_lr, min_lr=None, warmup_steps=2000):
    if min_lr is None:
        min_lr = max_lr * 0.1
    if step < warmup_steps:
        return max_lr * (step / max(warmup_steps, 1))
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    steps = sorted(ckpt_dir.glob("step_*.pt"))
    return steps[-1] if steps else None


def _unwrap_state_dict(model: torch.nn.Module):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def _payload(model, config, step, optimizer=None, size_name: str = "small"):
    state_dict = _unwrap_state_dict(model)
    out = {
        "model_state": state_dict,
        "model_state_dict": state_dict,
        "config": config.__dict__,
        "model_config": config.__dict__,
        "tokenizer_kind": "tiktoken",
        "step": step,
        "model_version": f"v5_{size_name}",
        "size": size_name,
    }
    if optimizer is not None:
        out["optimizer_state_dict"] = optimizer.state_dict()
    return out


def train_v5(
    size: str = "small",
    total_steps: int | None = None,
    batch_size: int | None = None,
    seq_len: int | None = None,
    lr: float | None = None,
    grad_accum: int | None = None,
    save_every: int = 1000,
    resume_checkpoint: str | None = None,
    no_resume: bool = False,
    force_huge: bool = False,
    compile_model: bool = True,
):
    preset = get_preset(size)
    if size in HUGE_SIZES and not force_huge:
        print(
            f"\nRefusing to train size='{size}' ({preset.approx_params}) on this machine.\n"
            f"Hardware target: {preset.hardware}\n"
            f"Notes: {preset.notes}\n\n"
            f"Use --size small|base|medium for local training.\n"
            f"Pass --force-huge only if you really have the cluster for it.\n"
        )
        print_scale_table()
        return

    total_steps = total_steps if total_steps is not None else preset.total_steps
    batch_size = batch_size if batch_size is not None else preset.batch_size
    seq_len = seq_len if seq_len is not None else preset.max_seq_len
    lr = lr if lr is not None else preset.learning_rate
    grad_accum = grad_accum if grad_accum is not None else preset.grad_accum

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = preset.to_gpt_config()
    # Allow seq_len override without rebuilding unrelated dims
    config.max_seq_len = seq_len

    print(f"Using device: {device}")
    print(
        f"v5 size={size} arch={config.architecture} "
        f"d_model={config.d_model} layers={config.num_layers} "
        f"heads={config.num_heads} kv_heads={config.n_kv_heads} "
        f"d_ff={config.d_ff} seq={seq_len} "
        f"batch={batch_size} accum={grad_accum} "
        f"eff_batch={batch_size * grad_accum}"
    )
    print(f"Target hardware: {preset.hardware} | approx {preset.approx_params}")

    model = GPTLanguageModel(config).to(device)
    n_params = model.num_parameters()
    print(f"Parameters: {n_params:,} (estimate helper={estimate_parameters(config):,})")

    if device == "cpu" and n_params > 200_000_000:
        print(
            "[warn] >200M params on CPU will be extremely slow. "
            "Prefer --size tiny/small or a CUDA GPU."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1 if config.architecture == "modern" else 0.01,
        betas=(0.9, 0.95),
    )

    start_step = 1
    ckpt_dir = Path(f"checkpoints/v5_{size}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if not no_resume:
        resume_path = Path(resume_checkpoint) if resume_checkpoint else find_latest_checkpoint(ckpt_dir)

    if resume_path is not None and resume_path.exists():
        print(f"[Checkpoint] Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state") or checkpoint.get("model_state_dict")
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"[Checkpoint] Optimizer not restored ({e})")
        start_step = int(checkpoint.get("step", 0)) + 1
        print(f"[Checkpoint] Resuming from step {start_step}")
    else:
        print(f"[Checkpoint] Starting v5_{size} from scratch")

    if start_step > total_steps:
        print(f"[Done] Already past total_steps={total_steps}")
        return

    if compile_model:
        try:
            model = torch.compile(model)
            print("[PyTorch] torch.compile enabled")
        except Exception as e:
            print(f"[PyTorch] compile skipped: {e}")

    use_cuda_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    dataloader = get_wiki_dataloader(batch_size=batch_size, seq_len=seq_len)
    data_iter = iter(dataloader)
    model.train()
    start_time = time.time()
    running_loss = 0.0
    running_count = 0
    optimizer.zero_grad(set_to_none=True)

    warmup = max(100, min(2000, total_steps // 50))
    print(f"\n--- v5_{size} pretrain ({start_step} -> {total_steps}) ---")
    for step in range(start_step, total_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        current_lr = get_lr(step, total_steps, max_lr=lr, warmup_steps=warmup)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        if device == "cpu":
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum
            loss.backward()
        elif use_cuda_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                _, loss = model(x, y)
            loss = loss / grad_accum
            scaler.scale(loss).backward()
        else:
            _, loss = model(x, y)
            loss = loss / grad_accum
            loss.backward()

        running_loss += loss.item() * grad_accum
        running_count += 1

        if step % grad_accum == 0:
            if use_cuda_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0 or step == total_steps:
            avg = running_loss / max(running_count, 1)
            elapsed = time.time() - start_time
            print(
                f"[step {step}/{total_steps}] loss={loss.item() * grad_accum:.4f} | "
                f"avg100={avg:.4f} | lr={current_lr:.2e} | time={elapsed:.1f}s"
            )
            running_loss = 0.0
            running_count = 0

        if step % save_every == 0 or step == total_steps:
            payload = _payload(model, config, step, optimizer=optimizer, size_name=size)
            path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save(payload, path)
            torch.save(payload, ckpt_dir / "latest.pt")
            # Convenience alias used by finetune_qa defaults
            if size == "small":
                alias = Path("checkpoints/v5_smart")
                alias.mkdir(parents=True, exist_ok=True)
                torch.save(payload, alias / "latest.pt")
            print(f" -> Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MyAI v5 (modern GPT, ChatGPT-scale ladder)"
    )
    parser.add_argument(
        "--size",
        default="small",
        choices=list(SCALE_PRESETS.keys()),
        help="Model size preset (chatgpt = 175B reference, blocked by default)",
    )
    parser.add_argument("--list-sizes", action="store_true")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--force-huge",
        action="store_true",
        help="Allow large/xl/chatgpt sizes (needs serious multi-GPU hardware)",
    )
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    if args.list_sizes:
        print_scale_table()
        return

    train_v5(
        size=args.size,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        grad_accum=args.grad_accum,
        save_every=args.save_every,
        resume_checkpoint=args.resume,
        no_resume=args.no_resume,
        force_huge=args.force_huge,
        compile_model=not args.no_compile,
    )


if __name__ == "__main__":
    main()
