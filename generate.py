"""
Interactive generation for Rija checkpoints (BPE, wiki, or Q&A fine-tuned).

Usage:
  python generate.py --checkpoint checkpoints/v5_qa/latest.pt --qa
  python generate.py --checkpoint checkpoints/v5_qa/latest.pt --qa --rag
  python generate.py --prompt "What is the Sun?" --qa --rag
"""

from __future__ import annotations

import argparse

from inference.checkpoint import find_latest_checkpoint, load_checkpoint
from inference.generate import InferenceEngine, GenerationConfig
from inference.rag import build_rag_prompt, extract_answer


def _maybe_load_rag_index():
    from memory.seed_facts import build_index, seed_from_qa
    from pathlib import Path

    docs = Path("memory/store/documents")
    if not docs.exists() or not any(docs.glob("*.json")):
        print("[RAG] Seeding fact documents from qa_seed.jsonl ...")
        seed_from_qa()
    return build_index()


def main():
    parser = argparse.ArgumentParser(description="Rija interactive text generation")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--version", default="v5_qa", help="Checkpoint version folder fallback")
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument(
        "--strategy",
        default="temperature",
        choices=["greedy", "temperature", "top_k", "top_p"],
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Use ### Question / ### Answer prompt template",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Retrieve facts from memory and put them in the prompt",
    )
    args = parser.parse_args()

    # Prefer explicit checkpoint; else try version, then v4_wiki
    ckpt = args.checkpoint
    if ckpt is None:
        for version in (args.version, "v5_smart", "v4_wiki"):
            try:
                ckpt = find_latest_checkpoint(version)
                break
            except FileNotFoundError:
                continue
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found. Train or fine-tune a model first.")

    print(f"Loading checkpoint from {ckpt}...")
    loaded = load_checkpoint(ckpt)
    engine = InferenceEngine(loaded.model, loaded.tokenizer, model_id=ckpt)
    print(
        f"Ready | tokenizer={loaded.tokenizer_kind} | "
        f"vocab={loaded.config.vocab_size} | "
        f"step={loaded.step} | params={loaded.model.num_parameters():,}"
    )
    print("=" * 50)

    search_index = _maybe_load_rag_index() if args.rag else None

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    def build_prompt(user_text: str) -> str:
        if args.rag or args.qa:
            return build_rag_prompt(
                user_text,
                search_index,
                qa_style=True,
            )
        return user_text

    def run_once(user_text: str) -> None:
        prompt = build_prompt(user_text)
        print(prompt, end="", flush=True)
        pieces = []
        for piece in engine.generate_stream(prompt, gen_cfg):
            pieces.append(piece)
            print(piece, end="", flush=True)
        print()
        if args.qa or args.rag:
            clean = extract_answer(prompt + "".join(pieces))
            if clean and clean != "".join(pieces).strip():
                print("-" * 40)
                print("Answer:", clean)

    if args.prompt is not None:
        run_once(args.prompt)
        return

    while True:
        try:
            prompt = input("\nEnter prompt (or 'q' to quit): ")
            if prompt.strip().lower() == "q":
                break
            if not prompt.strip():
                continue
            print("-" * 40)
            run_once(prompt)
        except (KeyboardInterrupt, EOFError):
            print()
            break


if __name__ == "__main__":
    main()
