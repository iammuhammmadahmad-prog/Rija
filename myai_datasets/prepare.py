"""
Phase 6: Build your dataset.

Reads raw text files, removes duplicates and corrupted lines, normalizes
formatting, and splits into training and validation sets.
"""

import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Tuple


def normalize_text(text: str) -> str:
    """Clean up whitespace, line endings, and common formatting issues."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_corrupted(line: str, min_chars: int = 3, max_non_printable_ratio: float = 0.3) -> bool:
    """Heuristic filter for garbled or empty lines."""
    stripped = line.strip()
    if len(stripped) < min_chars:
        return True
    non_printable = sum(1 for c in stripped if ord(c) < 32 and c not in "\n\t")
    if non_printable / max(len(stripped), 1) > max_non_printable_ratio:
        return True
    return False


def deduplicate_lines(lines: List[str]) -> List[str]:
    seen = set()
    unique = []
    for line in lines:
        key = hashlib.md5(line.strip().encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(line)
    return unique


def read_raw_files(raw_dir: str) -> str:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return ""

    chunks = []
    for p in sorted(raw_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".py", ".json", ".csv"}:
            chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(chunks)


def split_train_val(text: str, val_ratio: float = 0.1) -> Tuple[str, str]:
    """Split by paragraphs so train/val don't share exact chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        split_at = int(len(text) * (1 - val_ratio))
        return text[:split_at], text[split_at:]

    val_count = max(1, int(len(paragraphs) * val_ratio))
    train_paras = paragraphs[:-val_count]
    val_paras = paragraphs[-val_count:]
    return "\n\n".join(train_paras), "\n\n".join(val_paras)


def prepare_dataset(
    raw_dir: str = "datasets/raw",
    output_dir: str = "datasets/processed",
    val_ratio: float = 0.1,
    train_tokenizer: bool = True,
    vocab_size: int = 2000,
) -> dict:
    """
    Full Phase 6 pipeline:
      1. Read raw files
      2. Normalize + deduplicate + filter corrupted lines
      3. Split train/val
      4. Optionally train tokenizer on combined text
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_text = read_raw_files(raw_dir)
    if not raw_text.strip():
        raise FileNotFoundError(
            f"No text found in {raw_dir}. Add .txt/.md/.py files there first."
        )

    normalized = normalize_text(raw_text)
    lines = normalized.split("\n")
    clean_lines = [ln for ln in lines if not is_corrupted(ln)]
    clean_lines = deduplicate_lines(clean_lines)
    clean_text = "\n".join(clean_lines)

    train_text, val_text = split_train_val(clean_text, val_ratio=val_ratio)

    train_path = out / "train.txt"
    val_path = out / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    result = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "lines_kept": len(clean_lines),
    }

    if train_tokenizer:
        tok = _train_and_save_tokenizer(
            [str(train_path)],
            output_path="tokenizer/vocab.json",
            vocab_size=vocab_size,
        )
        result["vocab_size"] = tok.vocab_size

    print(f"[dataset] train={result['train_chars']:,} chars, val={result['val_chars']:,} chars")
    return result


def _train_and_save_tokenizer(paths: list, output_path: str, vocab_size: int):
    """Train BPE tokenizer without importing torch-dependent trainer modules."""
    from tokenizer.tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.train_from_files(paths, vocab_size=vocab_size)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(output_path)
    print(f"[tokenizer] vocab_size={tok.vocab_size}, saved to {output_path}")
    return tok
