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

from model.embeddings import resize_learned_pos_weight
from model.model import GPTConfig, GPTLanguageModel
from myai_datasets.wikipedia_loader import get_wiki_dataloader
from trainer.cpu_runtime import configure_cpu_runtime, has_cpu_bf16_accel


DEFAULT_MAX_LR = 6e-4


def get_lr(step, total_steps, max_lr=6e-4, min_lr=None, warmup_steps=2000):
    """Cosine learning rate decay with linear warmup."""
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


def _checkpoint_payload(model, config, step, optimizer=None, schedule=None):
    state_dict = {k: v.detach().cpu() for k, v in _unwrap_state_dict(model).items()}
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
    if schedule is not None:
        # The cosine schedule is a function of (max_lr, total_steps). Storing it
        # means a resume replays the same curve instead of rebuilding it from
        # whatever --lr happens to be on the command line.
        payload["schedule"] = dict(schedule)
    return payload


def _configure_backends(device: str) -> None:
    torch.manual_seed(42)
    if device != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, device: str):
    decay, no_decay, seen = [], [], set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        # Embedding tables must stay out of the decay group. Each row only gets a
        # gradient on the steps where its token/position actually appears, so at
        # this batch size the decay pull outweighs the gradient and the table
        # shrinks toward zero -- which is what destroyed the position embeddings.
        is_embedding = "embedding" in name
        if (
            param.ndim < 2
            or is_embedding
            or name.endswith("bias")
            or "norm" in name.lower()
            or ".ln" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
    if device == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, **kwargs)


def _restore_optimizer_state(optimizer, model: torch.nn.Module, saved: dict) -> str:
    """Load AdamW state, tolerating a change in how params are grouped.

    Checkpoints written before the decay/no-decay split hold a single param
    group, so a plain load_state_dict raises and the moments get discarded --
    which is enough to knock a converged model out of its basin. The parameters
    themselves are unchanged, so remap the moments by each tensor's position in
    model.parameters() and verify shapes before trusting the result.
    """
    try:
        optimizer.load_state_dict(saved)
        return "restored"
    except ValueError as exc:
        if "parameter groups" not in str(exc):
            raise

    saved_state = saved.get("state") or {}
    saved_ids = [pid for group in saved.get("param_groups", []) for pid in group["params"]]
    ordered = list(model.parameters())
    if len(saved_ids) != len(ordered):
        return f"discarded (checkpoint holds {len(saved_ids)} params, model has {len(ordered)})"

    position = {id(p): i for i, p in enumerate(ordered)}
    flat = [p for group in optimizer.param_groups for p in group["params"]]
    new_state = {}
    for new_idx, param in enumerate(flat):
        i = position.get(id(param))
        if i is None:
            return "discarded (parameter not found in model order)"
        entry = saved_state.get(saved_ids[i])
        if entry is None:
            continue
        moment = entry.get("exp_avg")
        if moment is not None and tuple(moment.shape) != tuple(param.shape):
            # Ordering assumption is wrong; a silent mismatch is worse than none.
            return "discarded (moment/param shape mismatch)"
        new_state[new_idx] = entry

    if not new_state:
        return "discarded (no reusable moments)"

    groups, cursor = [], 0
    for group in optimizer.param_groups:
        meta = {k: v for k, v in group.items() if k != "params"}
        meta["params"] = list(range(cursor, cursor + len(group["params"])))
        cursor += len(group["params"])
        groups.append(meta)

    optimizer.load_state_dict({"state": new_state, "param_groups": groups})
    return f"remapped across regrouped params ({len(new_state)}/{len(flat)} moments recovered)"


def _load_compatible_state(model: GPTLanguageModel, state: dict) -> None:
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    pe_key = "positional_encoding.position_embedding.weight"
    if (
        pe_key in state
        and model.positional_encoding is not None
        and state[pe_key].shape != model.positional_encoding.position_embedding.weight.shape
    ):
        new_len = model.positional_encoding.position_embedding.weight.size(0)
        state[pe_key] = resize_learned_pos_weight(state[pe_key], new_len)
        print(f"[Checkpoint] Resized positional embeddings to max_seq_len={new_len}")
    model.load_state_dict(state)


class _CudaPrefetcher:
    """Overlap H2D copies with compute on a side CUDA stream."""

    def __init__(self, iterator, device: torch.device):
        self.iterator = iterator
        self.device = device
        self.stream = torch.cuda.Stream()
        self._batch = None
        self._preload()

    def _preload(self):
        try:
            x, y = next(self.iterator)
        except StopIteration:
            self._batch = None
            return
        with torch.cuda.stream(self.stream):
            self._batch = (
                x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True),
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self._batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._batch
        batch[0].record_stream(torch.cuda.current_stream())
        batch[1].record_stream(torch.cuda.current_stream())
        self._preload()
        return batch


def _next_batch(data_iter, dataloader):
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


def train_wikipedia(
    total_steps: int = 200000,
    batch_size: int | None = None,
    seq_len: int = 128,
    lr: float | None = None,
    save_every: int = 1000,
    resume_checkpoint: str | None = None,
    no_resume: bool = False,
    grad_accum: int = 1,
    dropout: float = 0.0,
    compile_model: bool = True,
    use_amp: bool = True,
    weight_decay: float = 0.01,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_info = configure_cpu_runtime() if device == "cpu" else None
    if batch_size is None:
        batch_size = 64 if device == "cuda" else 8
    grad_accum = max(1, grad_accum)
    _configure_backends(device)
    print(f"Using device: {device}")
    if cpu_info is not None:
        print(
            f"[laptop] P-cores={cpu_info['p_cores']} ({cpu_info['threads']} threads) | "
            f"E-cores={cpu_info['e_cores']} for data prefetch | "
            f"skipped LPE={cpu_info['lpe_cores']}"
        )
        if cpu_info["power_profile"]:
            print(
                f"[laptop] power profile -> {cpu_info['power_profile']} "
                "(restore later with: powerprofilesctl set balanced)"
            )
        else:
            print(
                "[laptop] Could not switch power profile. For speed, plug in the charger "
                "and run: powerprofilesctl set performance"
            )

    config = GPTConfig(
        vocab_size=50257,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=seq_len,
        dropout=dropout,
    )

    model = GPTLanguageModel(config).to(device)
    n_params = model.num_parameters()

    start_step = 1
    ckpt_dir = Path("checkpoints/v4_wiki")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if not no_resume:
        if resume_checkpoint:
            resume_path = Path(resume_checkpoint)
        else:
            resume_path = find_latest_checkpoint(ckpt_dir)

    checkpoint = None
    if resume_path is not None and resume_path.exists():
        print(f"[Checkpoint] Resuming training from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state = (
            checkpoint.get("model_state")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
        _load_compatible_state(model, state)
        start_step = int(checkpoint.get("step", 0)) + 1

    # Resolve the schedule before building the optimizer: an explicit --lr wins,
    # then whatever the checkpoint recorded, then the default.
    saved_schedule = (checkpoint or {}).get("schedule") or {}
    saved_lr = saved_schedule.get("max_lr")
    if lr is not None:
        max_lr, lr_source = float(lr), "--lr"
        if saved_lr is not None and abs(saved_lr - max_lr) > 1e-12:
            print(
                f"[Schedule] Overriding the checkpoint's max_lr={saved_lr:.2e} "
                f"with --lr {max_lr:.2e}. The cosine curve will shift."
            )
    elif saved_lr is not None:
        max_lr, lr_source = float(saved_lr), f"checkpoint ({resume_path.name})"
    else:
        max_lr, lr_source = DEFAULT_MAX_LR, "default"
        if checkpoint is not None:
            print(
                f"[Schedule] This checkpoint predates schedule persistence and no "
                f"--lr was given, so max_lr falls back to {max_lr:.2e}. If the run "
                "was using a lower value, pass it explicitly."
            )

    saved_total = saved_schedule.get("total_steps")
    if saved_total is not None and int(saved_total) != int(total_steps):
        print(
            f"[Schedule] total_steps changed ({saved_total} -> {total_steps}); "
            "the cosine curve is stretched relative to the earlier run."
        )

    optimizer = _build_optimizer(model, lr=max_lr, weight_decay=weight_decay, device=device)

    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            try:
                outcome = _restore_optimizer_state(
                    optimizer, model, checkpoint["optimizer_state_dict"]
                )
                print(f"[Checkpoint] Optimizer state {outcome}.")
            except Exception as e:
                print(f"[Checkpoint] Could not restore optimizer ({e}); continuing with fresh AdamW.")
        else:
            print("[Checkpoint] No optimizer state in checkpoint; continuing with fresh AdamW.")
        print(f"[Checkpoint] Loaded! Resuming from step {start_step}.")
    else:
        print("[Checkpoint] No checkpoint found — starting from scratch.")

    if start_step > total_steps:
        print(f"[Done] Already at step {start_step - 1} >= total_steps={total_steps}. Nothing to do.")
        return

    if compile_model and device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[PyTorch] Model compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            try:
                model = torch.compile(model)
                print("[PyTorch] Model compiled with torch.compile()")
            except Exception as e2:
                print(f"[PyTorch] Skipping torch.compile: {e2 or e}")

    amp_device = "cuda" if device == "cuda" else "cpu"
    amp_dtype = torch.bfloat16
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    # AVX2-only Intel CPUs (this ThinkPad) are slower in bf16 than fp32.
    amp_on = bool(use_amp) and (device == "cuda" or has_cpu_bf16_accel())
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on and device == "cuda" and amp_dtype == torch.float16)

    e_cores = (cpu_info or {}).get("e_cores") or None
    dataloader = get_wiki_dataloader(
        batch_size=batch_size,
        seq_len=seq_len,
        prefetch=4 if device == "cpu" else 8,
        pin_memory=device == "cuda",
        worker_affinity=e_cores,
    )
    data_iter = iter(dataloader)
    if device == "cuda":
        data_iter = _CudaPrefetcher(data_iter, torch.device(device))

    model.train()
    start_time = time.time()
    log_time = start_time
    running_loss = 0.0
    running_count = 0
    tokens_per_step = batch_size * seq_len * grad_accum
    warmup = max(100, min(2000, total_steps // 50))
    schedule = {"max_lr": max_lr, "total_steps": int(total_steps), "warmup_steps": int(warmup)}
    optimizer.zero_grad(set_to_none=True)

    print(
        f"\n--- v4_wiki ({start_step} -> {total_steps}) | "
        f"params={n_params:,} | batch={batch_size} accum={grad_accum} "
        f"seq={seq_len} tokens/step={tokens_per_step} | "
        f"amp={amp_on} {amp_dtype if amp_on else 'fp32'} | dropout={dropout} "
        f"max_lr={max_lr:.2e} (from {lr_source}) ---"
    )
    if device == "cpu":
        print(
            "[laptop] No NVIDIA GPU — training on the 6 P-cores. "
            "About 0.3–0.4s/step at batch 8 (~20h for 200k steps). "
            "Keep the charger plugged in and the lid open."
        )
    for step in range(start_step, total_steps + 1):
        x, y = None, None
        try:
            if isinstance(data_iter, _CudaPrefetcher):
                x, y = next(data_iter)
            else:
                (x, y), data_iter = _next_batch(data_iter, dataloader)
        except StopIteration:
            data_iter = iter(dataloader)
            if device == "cuda":
                data_iter = _CudaPrefetcher(data_iter, torch.device(device))
                x, y = next(data_iter)
            else:
                (x, y), data_iter = _next_batch(data_iter, dataloader)

        if device != "cuda":
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        current_lr = get_lr(step, total_steps, max_lr=max_lr, warmup_steps=warmup)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        if amp_on:
            with torch.amp.autocast(amp_device, dtype=amp_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss = loss / grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * grad_accum
        running_count += 1

        if step % grad_accum == 0 or step == total_steps:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0 or step == total_steps:
            now = time.time()
            avg = running_loss / max(running_count, 1)
            dt = max(now - log_time, 1e-6)
            toks = running_count * batch_size * seq_len / dt
            print(
                f"[step {step}/{total_steps}] loss={loss.item() * grad_accum:.4f} | "
                f"avg100={avg:.4f} | lr={current_lr:.2e} | "
                f"{toks:.0f} tok/s | time={now - start_time:.1f}s"
            )
            running_loss = 0.0
            running_count = 0
            log_time = now

        if step % save_every == 0 or step == total_steps:
            payload = _checkpoint_payload(
                model, config, step, optimizer=optimizer, schedule=schedule
            )
            save_path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save(payload, save_path)
            torch.save(payload, ckpt_dir / "latest.pt")
            print(f" -> Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Resume / start Wikipedia pretraining")
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Micro-batch size (default: 64 on CUDA, 8 on CPU)",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Peak LR for the cosine schedule. Omit to reuse the value stored in the "
            f"checkpoint, falling back to {DEFAULT_MAX_LR:.0e} when it has none."
        ),
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
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
        grad_accum=args.grad_accum,
        dropout=args.dropout,
        compile_model=not args.no_compile,
        use_amp=not args.no_amp,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
