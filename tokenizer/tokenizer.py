"""
Phase 3: A tokenizer built from scratch (Byte-Pair Encoding).

No external tokenizer libraries are used. This implements:
  - reading text files
  - pre-tokenizing (splitting text into word-ish chunks)
  - iteratively merging the most frequent adjacent symbol pairs (BPE)
  - building a vocabulary with IDs
  - encoding text -> IDs
  - decoding IDs -> text

Usage:
    tok = BPETokenizer()
    tok.train_from_files(["datasets/raw/some.txt"], vocab_size=2000)
    ids = tok.encode("Hello, world!")
    text = tok.decode(ids)
    tok.save("tokenizer/vocab.json")
    tok2 = BPETokenizer.load("tokenizer/vocab.json")
"""

import json
import re
import collections
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

# A regex-based pre-tokenizer: splits text into words, numbers, punctuation
# and whitespace runs, similar in spirit to GPT-2's pre-tokenizer (but simpler).
_PRETOKEN_PATTERN = re.compile(
    r"""[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]|\s+"""
)

END_OF_WORD = "</w>"  # marks the end of a word so the model can learn word boundaries
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def read_text_files(paths: Iterable[str]) -> str:
    """Read and concatenate a list of text files."""
    chunks = []
    for p in paths:
        chunks.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def pre_tokenize(text: str) -> List[str]:
    """Split raw text into pre-tokens (words / numbers / punctuation / whitespace)."""
    return _PRETOKEN_PATTERN.findall(text)


class BPETokenizer:
    def __init__(self):
        # token string -> id
        self.token_to_id: Dict[str, int] = {}
        # id -> token string
        self.id_to_token: Dict[int, str] = {}
        # ordered list of merges learned during training: (a, b) -> merged
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train_from_files(self, paths: Iterable[str], vocab_size: int = 2000,
                          min_frequency: int = 2) -> None:
        text = read_text_files(paths)
        self.train_from_text(text, vocab_size=vocab_size, min_frequency=min_frequency)

    def train_from_text(self, text: str, vocab_size: int = 2000,
                         min_frequency: int = 2) -> None:
        # 1. Pre-tokenize into words (skip pure-whitespace pre-tokens for counting)
        words = [w for w in pre_tokenize(text) if w.strip() != ""]
        word_freq = collections.Counter(words)

        # 2. Represent each word as a tuple of characters + end-of-word marker
        splits: Dict[str, List[str]] = {
            word: list(word) + [END_OF_WORD] for word in word_freq
        }

        # 3. Seed vocabulary with every character we saw + specials
        vocab = set(SPECIAL_TOKENS)
        for word in splits:
            vocab.update(splits[word])

        # 4. Iteratively merge the most frequent adjacent pair
        num_merges = max(0, vocab_size - len(vocab))
        merges: List[Tuple[str, str]] = []

        for _ in range(num_merges):
            pair_freq = collections.Counter()
            for word, freq in word_freq.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pair_freq[(symbols[i], symbols[i + 1])] += freq

            if not pair_freq:
                break

            best_pair, best_count = pair_freq.most_common(1)[0]
            if best_count < min_frequency:
                break

            merged_token = best_pair[0] + best_pair[1]
            vocab.add(merged_token)
            merges.append(best_pair)

            # Apply merge to every word containing this pair
            for word in list(splits.keys()):
                symbols = splits[word]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if (i < len(symbols) - 1
                            and symbols[i] == best_pair[0]
                            and symbols[i + 1] == best_pair[1]):
                        new_symbols.append(merged_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[word] = new_symbols

        # 5. Finalize vocabulary: specials first, then everything else sorted
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        ordered_vocab = list(SPECIAL_TOKENS) + sorted(vocab - set(SPECIAL_TOKENS))
        self.token_to_id = {tok: i for i, tok in enumerate(ordered_vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def _bpe_word(self, word: str) -> List[str]:
        """Apply learned merges to a single word, returning subword tokens."""
        symbols = list(word) + [END_OF_WORD]
        if len(symbols) == 1:
            return symbols

        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # pick the pair with the lowest merge rank (i.e. learned earliest)
            candidate = None
            candidate_rank = None
            for pair in pairs:
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if candidate_rank is None or rank < candidate_rank:
                        candidate = pair
                        candidate_rank = rank
            if candidate is None:
                break

            merged_token = candidate[0] + candidate[1]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1
                        and symbols[i] == candidate[0]
                        and symbols[i + 1] == candidate[1]):
                    new_symbols.append(merged_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str, add_bos_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos_eos:
            ids.append(self.token_to_id[BOS_TOKEN])

        for pre_tok in pre_tokenize(text):
            if pre_tok.strip() == "":
                # whitespace: encode as unk-safe pass-through if present in vocab,
                # otherwise skip individual chars won't exist so map via UNK
                if pre_tok in self.token_to_id:
                    ids.append(self.token_to_id[pre_tok])
                continue
            for sub in self._bpe_word(pre_tok):
                ids.append(self.token_to_id.get(sub, self.token_to_id[UNK_TOKEN]))

        if add_bos_eos:
            ids.append(self.token_to_id[EOS_TOKEN])
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(i, UNK_TOKEN) for i in ids]
        text = "".join(tokens)
        text = text.replace(END_OF_WORD, " ")
        for special in SPECIAL_TOKENS:
            text = text.replace(special, "")
        return text.strip()

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        data = {
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        return tok


if __name__ == "__main__":
    # Tiny smoke test / demo
    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "The dog barks. The fox runs away quickly. "
        "Quick foxes and lazy dogs are common in stories."
    )
    tok = BPETokenizer()
    tok.train_from_text(sample_text, vocab_size=120, min_frequency=2)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Learned merges: {len(tok.merges)}")

    test_str = "The quick fox barks."
    ids = tok.encode(test_str)
    decoded = tok.decode(ids)
    print(f"Original : {test_str}")
    print(f"Token IDs: {ids}")
    print(f"Decoded  : {decoded}")
