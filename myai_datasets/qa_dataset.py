"""
Q&A dataset helpers for instruction-style fine-tuning.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


QA_TEMPLATE = "### Question:\n{question}\n### Answer:\n{answer}"
PROMPT_TEMPLATE = "### Question:\n{question}\n### Answer:\n"


def format_qa(question: str, answer: str) -> str:
    return QA_TEMPLATE.format(question=question.strip(), answer=answer.strip())


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def load_qa_jsonl(path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        q = str(obj["question"]).strip()
        a = str(obj["answer"]).strip()
        if q and a:
            rows.append({"question": q, "answer": a})
    return rows


def save_qa_jsonl(rows: Iterable[Dict[str, str]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


class QADataset(Dataset):
    """
    Tokenizes Q&A pairs for causal LM fine-tuning.
    Labels mask the question prefix with -100 so loss focuses on the answer.
    """

    def __init__(
        self,
        rows: List[Dict[str, str]],
        tokenizer,
        seq_len: int = 256,
        eot_token: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.samples: List[Tuple[List[int], List[int]]] = []

        for row in rows:
            full = format_qa(row["question"], row["answer"])
            prompt = format_prompt(row["question"])
            ids = list(tokenizer.encode(full))
            prompt_ids = list(tokenizer.encode(prompt))
            if eot_token is not None:
                ids.append(eot_token)

            if len(ids) < 2:
                continue

            # Keep the end of the sequence (answer side) if too long
            if len(ids) > seq_len + 1:
                overflow = len(ids) - (seq_len + 1)
                ids = ids[overflow:]
                prompt_len = max(1, len(prompt_ids) - overflow)
            else:
                prompt_len = len(prompt_ids)

            x = ids[:-1][:seq_len]
            y = ids[1:][:seq_len]
            labels = list(y)
            mask_until = max(0, min(prompt_len - 1, len(labels)))
            for i in range(mask_until):
                labels[i] = -100

            pad_id = 0
            if len(x) < seq_len:
                pad_n = seq_len - len(x)
                x = x + [pad_id] * pad_n
                labels = labels + [-100] * pad_n

            self.samples.append((x, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_qa_dataloader(
    rows: List[Dict[str, str]],
    tokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    eot_token: Optional[int] = None,
) -> DataLoader:
    ds = QADataset(rows, tokenizer, seq_len=seq_len, eot_token=eot_token)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def augment_rows(rows: List[Dict[str, str]], n_extra: int = 0, seed: int = 42) -> List[Dict[str, str]]:
    """Light prompt variants to reduce overfitting on tiny seeds."""
    rng = random.Random(seed)
    out = list(rows)
    for row in rows:
        variants = [
            row["question"],
            f"Answer this: {row['question']}",
            f"Q: {row['question']}",
        ]
        for q in variants[1 : 1 + n_extra]:
            out.append({"question": q, "answer": row["answer"]})
    rng.shuffle(out)
    return out
