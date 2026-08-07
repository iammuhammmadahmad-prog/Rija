"""
MyAI — CLI entry point for all phases.

Usage:
    python main.py prepare          # Phase 6: build dataset + tokenizer
    python main.py train            # Phase 5/7: train v1
    python main.py generate "..."   # Phase 8: run inference
    python main.py evaluate         # Phase 14: metrics
    python main.py gui              # Phase 12/13: launch desktop app
    python main.py smoke            # Quick end-to-end smoke test
"""

import argparse
import json
from pathlib import Path

from tokenizer.tokenizer import BPETokenizer
import myai_datasets.prepare


def cmd_smoke(_args=None):
    """Phases 3+4 smoke test."""
    import torch
    from model.model import GPTConfig, GPTLanguageModel

    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning models learn patterns from data.
    A transformer uses self-attention to relate tokens to each other.
    """
    tok = BPETokenizer()
    tok.train_from_text(sample_text, vocab_size=300, min_frequency=2)
    print(f"[tokenizer] vocab={tok.vocab_size}")

    config = GPTConfig(
        vocab_size=tok.vocab_size, d_model=64, num_heads=4,
        num_layers=2, d_ff=256, max_seq_len=64,
    )
    model = GPTLanguageModel(config)
    print(f"[model] params={model.num_parameters():,}")

    ids = tok.encode("The transformer learns from data.")
    inputs = torch.tensor([ids[:-1]])
    targets = torch.tensor([ids[1:]])
    _, loss = model(inputs, targets)
    print(f"[model] loss={loss.item():.4f}")


def cmd_prepare(args):
    result = myai_datasets.prepare.prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        vocab_size=args.vocab_size,
    )
    print(json.dumps(result, indent=2))


def cmd_train(args):
    from trainer.config import TrainConfig, tiny_v1
    from trainer.train import Trainer

    if args.prepare:
        myai_datasets.prepare.prepare_dataset(vocab_size=args.vocab_size)

    config = tiny_v1() if args.preset == "tiny" else TrainConfig()
    config.max_steps = args.steps
    config.model_version = args.version
    if args.resume:
        config.resume_from = args.resume

    trainer = Trainer(config)
    summary = trainer.train()
    print(json.dumps(summary, indent=2))


def cmd_generate(args):
    from inference.generate import InferenceEngine, GenerationConfig

    ckpt = args.checkpoint or _find_latest_checkpoint(args.version)
    engine = InferenceEngine.from_checkpoint(ckpt, args.tokenizer)

    config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    output = engine.generate(args.prompt, config)
    print(output)


def cmd_evaluate(args):
    from inference.generate import InferenceEngine
    from evaluation.metrics import evaluate_model

    ckpt = args.checkpoint or _find_latest_checkpoint(args.version)
    engine = InferenceEngine.from_checkpoint(ckpt, args.tokenizer)

    results = evaluate_model(
        engine.model,
        engine.tokenizer,
        val_path=args.val_path,
        seq_len=engine.model.config.max_seq_len,
        batch_size=args.batch_size,
        device=engine.device,
    )
    print(json.dumps(results, indent=2))


def _find_latest_checkpoint(version: str = "v1") -> str:
    latest = Path("checkpoints") / version / "latest.pt"
    if latest.exists():
        return str(latest)
    raise FileNotFoundError(f"No checkpoint at {latest}. Train a model first.")


def cmd_gui(_args):
    from ui.app import launch_app
    launch_app()


def main():
    parser = argparse.ArgumentParser(description="MyAI — build your own LLM")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("smoke", help="Quick tokenizer + model smoke test")

    p_prep = sub.add_parser("prepare", help="Prepare dataset (Phase 6)")
    p_prep.add_argument("--raw-dir", default="datasets/raw")
    p_prep.add_argument("--output-dir", default="datasets/processed")
    p_prep.add_argument("--val-ratio", type=float, default=0.1)
    p_prep.add_argument("--vocab-size", type=int, default=2000)

    p_train = sub.add_parser("train", help="Train model (Phase 5/7)")
    p_train.add_argument("--preset", choices=["tiny", "default"], default="tiny")
    p_train.add_argument("--steps", type=int, default=1000)
    p_train.add_argument("--version", default="v1")
    p_train.add_argument("--resume", default=None)
    p_train.add_argument("--prepare", action="store_true", help="Prepare dataset first")
    p_train.add_argument("--vocab-size", type=int, default=2000)

    p_gen = sub.add_parser("generate", help="Generate text (Phase 8)")
    p_gen.add_argument("prompt", nargs="?", default="The transformer")
    p_gen.add_argument("--checkpoint", default=None)
    p_gen.add_argument("--tokenizer", default="tokenizer/vocab.json")
    p_gen.add_argument("--version", default="v1")
    p_gen.add_argument("--max-tokens", type=int, default=80)
    p_gen.add_argument("--strategy", default="temperature",
                       choices=["greedy", "temperature", "top_k", "top_p"])
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top-k", type=int, default=40)
    p_gen.add_argument("--top-p", type=float, default=0.9)

    p_eval = sub.add_parser("evaluate", help="Evaluate model (Phase 14)")
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--tokenizer", default="tokenizer/vocab.json")
    p_eval.add_argument("--version", default="v1")
    p_eval.add_argument("--val-path", default="datasets/processed/val.txt")
    p_eval.add_argument("--batch-size", type=int, default=8)

    sub.add_parser("gui", help="Launch desktop GUI (Phase 12/13)")

    args = parser.parse_args()

    commands = {
        "smoke": cmd_smoke,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "generate": cmd_generate,
        "evaluate": cmd_evaluate,
        "gui": cmd_gui,
    }

    if args.command is None:
        cmd_smoke()
    else:
        commands[args.command](args)


if __name__ == "__main__":
    main()
