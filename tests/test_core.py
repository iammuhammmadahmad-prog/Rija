"""Unit tests for MyAI components."""

import tempfile
from pathlib import Path

import torch

from tokenizer.tokenizer import BPETokenizer
from model.model import GPTConfig, GPTLanguageModel
from trainer.dataset import TextChunkDataset
from inference.generate import InferenceEngine, GenerationConfig
from memory.search import SearchIndex
from tools.registry import ToolRegistry
from datasets.prepare import normalize_text, deduplicate_lines


def test_tokenizer_roundtrip():
    text = "Hello world! The fox runs fast."
    tok = BPETokenizer()
    tok.train_from_text(text * 5, vocab_size=100)
    ids = tok.encode("Hello world")
    decoded = tok.decode(ids)
    assert "Hello" in decoded or "world" in decoded


def test_model_forward():
    config = GPTConfig(vocab_size=100, d_model=32, num_heads=4, num_layers=2, d_ff=64, max_seq_len=16)
    model = GPTLanguageModel(config)
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 100)
    assert loss.item() > 0


def test_dataset_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.txt"
        path.write_text("word " * 500, encoding="utf-8")
        tok = BPETokenizer()
        tok.train_from_text(path.read_text(), vocab_size=80)
        ds = TextChunkDataset(str(path), tok, seq_len=32)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.shape == (32,)


def test_search_index():
    idx = SearchIndex()
    idx.add("doc1", "Python is a programming language")
    idx.add("doc2", "Machine learning uses neural networks")
    results = idx.search("Python programming")
    assert len(results) >= 1
    assert results[0][0] == "doc1"


def test_tools():
    reg = ToolRegistry()
    assert reg.call("calculate", expression="2+2") == "4"
    assert reg.call("calculate", expression="sqrt(16)") == "4.0"


def test_normalize_and_dedupe():
    text = "Hello\n\n\nWorld"
    assert "\n\n\n" not in normalize_text(text)
    lines = deduplicate_lines(["same", "same", "different"])
    assert len(lines) == 2


def test_generation_strategies():
    config = GPTConfig(vocab_size=50, d_model=32, num_heads=4, num_layers=1, d_ff=64, max_seq_len=16)
    model = GPTLanguageModel(config)
    tok = BPETokenizer()
    tok.train_from_text("test test test", vocab_size=50)
    engine = InferenceEngine(model, tok, device=torch.device("cpu"))

    for strategy in ("greedy", "temperature", "top_k", "top_p"):
        out = engine.generate("test", GenerationConfig(max_new_tokens=5, strategy=strategy))
        assert isinstance(out, str)
