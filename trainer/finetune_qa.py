"""
Fine-tune a Rija checkpoint on Q&A pairs for smarter answers.

Recommended flow:
  1) Pretrain:  python -m trainer.train_v5
  2) Fine-tune: python -m trainer.finetune_qa --base checkpoints/v5_smart/latest.pt
  Or bootstrap from v4: python -m trainer.finetune_qa --base checkpoints/v4_wiki/latest.pt

Usage:
  python -m trainer.finetune_qa
  python -m trainer.finetune_qa --base checkpoints/v4_wiki/step_070000.pt --steps 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from inference.checkpoint import load_checkpoint
from myai_datasets.qa_dataset import (
    augment_rows,
    create_qa_dataloader,
    format_prompt,
    load_qa_jsonl,
)
from model.model import GPTConfig, GPTLanguageModel


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _save(model, config, step, out_dir: Path, optimizer=None, tag: str = "latest"):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _unwrap(model).state_dict()
    payload = {
        "model_state": state,
        "model_state_dict": state,
        "config": config.__dict__ if hasattr(config, "__dict__") else dict(config),
        "model_config": config.__dict__ if hasattr(config, "__dict__") else dict(config),
        "tokenizer_kind": "tiktoken",
        "step": step,
        "model_version": "v5_qa",
        "finetune": "qa",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path = out_dir / f"step_{step:06d}.pt"
    torch.save(payload, path)
    torch.save(payload, out_dir / f"{tag}.pt")
    print(f" -> Saved {path}")
    return path


@torch.no_grad()
def _sample_answer(model, tokenizer, device, question: str, max_new: int = 40):
    prompt = format_prompt(question)
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new):
        cond = x[:, -model.config.max_seq_len :]
        logits, _ = model(cond)
        next_id = int(torch.argmax(logits[0, -1]).item())
        # GPT-2 EOT
        if hasattr(tokenizer, "token_to_id") and next_id == tokenizer.token_to_id.get("<eos>"):
            break
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    text = tokenizer.decode(x[0].tolist())
    if "### Answer:" in text:
        return text.split("### Answer:", 1)[1].strip()
    return text[len(prompt) :].strip()


def finetune_qa(
    base_checkpoint: str,
    qa_path: str = "myai_datasets/qa_seed.jsonl",
    out_dir: str = "checkpoints/v5_qa",
    steps: int = 2000,
    batch_size: int = 4,
    lr: float = 1e-4,
    seq_len: int | None = None,
    save_every: int = 500,
    augment: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading base checkpoint: {base_checkpoint}")

    loaded = load_checkpoint(base_checkpoint, device=device)
    model = loaded.model
    tokenizer = loaded.tokenizer
    config = loaded.config
    if seq_len is None:
        seq_len = min(config.max_seq_len, 256)

    rows = load_qa_jsonl(qa_path)
    if not rows:
        raise RuntimeError(f"No Q&A rows found in {qa_path}")
    rows = augment_rows(rows, n_extra=augment)
    print(f"Q&A examples: {len(rows)} (after light augmentation)")

    eot = None
    if hasattr(tokenizer, "token_to_id"):
        eot = tokenizer.token_to_id.get("<eos>")

    loader = create_qa_dataloader(
        rows,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        eot_token=eot,
    )
    data_iter = iter(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    out_path = Path(out_dir)
    start = time.time()
    running = 0.0
    n = 0

    # Quick before sample
    demo_q = "What is the capital of France?"
    print(f"\n[before] {demo_q}")
    print(" ", _sample_answer(model, tokenizer, device, demo_q))

    print(f"\n--- Q&A fine-tune ({steps} steps) ---")
    for step in range(1, steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        # model already uses ignore_index=-100
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running += loss.item()
        n += 1
        if step % 50 == 0 or step == steps:
            avg = running / max(n, 1)
            print(
                f"[step {step}/{steps}] loss={loss.item():.4f} | "
                f"avg={avg:.4f} | time={time.time() - start:.1f}s"
            )
            running = 0.0
            n = 0

        if step % save_every == 0 or step == steps:
            _save(model, config, step, out_path, optimizer=optimizer)

    print(f"\n[after] {demo_q}")
    print(" ", _sample_answer(model, tokenizer, device, demo_q))
    print(f"\nDone. Use: python generate.py --checkpoint {out_path / 'latest.pt'} --qa")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Rija on Q&A data")
    parser.add_argument(
        "--base",
        default=None,
        help="Base checkpoint (default: v5_smart/latest, else v4_wiki/latest)",
    )
    parser.add_argument("--qa", default="myai_datasets/qa_seed.jsonl")
    parser.add_argument("--out", default="checkpoints/v5_qa")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--augment", type=int, default=2)
    args = parser.parse_args()

    base = args.base
    if base is None:
        for candidate in (
            "checkpoints/v5_smart/latest.pt",
            "checkpoints/v4_wiki/latest.pt",
            "checkpoints/v4_wiki/step_070000.pt",
        ):
            if Path(candidate).exists():
                base = candidate
                break
    if base is None:
        raise FileNotFoundError(
            "No base checkpoint found. Train v5 first or pass --base path/to.pt"
        )

    finetune_qa(
        base_checkpoint=base,
        qa_path=args.qa,
        out_dir=args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        save_every=args.save_every,
        augment=args.augment,
    )


if __name__ == "__main__":
    main()
