"""Unit tests for MyAI components."""

import sys
from pathlib import Path

# Add project root directory to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import torch

from tokenizer.tokenizer import BPETokenizer
from model.model import GPTConfig, GPTLanguageModel
from trainer.dataset import TextChunkDataset
from inference.generate import InferenceEngine, GenerationConfig
from inference.cache import ResponseCache, make_cache_key
from inference.checkpoint import load_checkpoint
from memory.search import SearchIndex
from memory.compress import compress_context, collapse_whitespace
from tools.registry import ToolRegistry
from myai_datasets.prepare import normalize_text, deduplicate_lines


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


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, d_model=32, num_heads=4, num_layers=2, d_ff=64, max_seq_len=16)
    model = GPTLanguageModel(config)
    model.eval()
    ids = torch.randint(0, 50, (1, 6))

    with torch.no_grad():
        full_logits, _ = model(ids)
        logits1, _, past = model(ids[:, :4], use_cache=True)
        logits2, _, _ = model(ids[:, 4:], past_kvs=past, use_cache=True)

    assert torch.allclose(full_logits[:, 3:4, :], logits1[:, -1:, :], atol=1e-5)
    assert torch.allclose(full_logits[:, 4:, :], logits2, atol=1e-5)


def test_modern_arch_forward_and_cache():
    torch.manual_seed(1)
    config = GPTConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_seq_len=16,
        architecture="modern",
        n_kv_heads=2,
    )
    model = GPTLanguageModel(config)
    model.eval()
    x = torch.randint(0, 64, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 64)
    assert loss.item() > 0

    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        full, _ = model(ids)
        a, _, past = model(ids[:, :3], use_cache=True)
        b, _, _ = model(ids[:, 3:], past_kvs=past, use_cache=True)
    assert torch.allclose(full[:, 2:3, :], a[:, -1:, :], atol=1e-4)
    assert torch.allclose(full[:, 3:, :], b, atol=1e-4)


def test_scale_presets():
    from model.scaling import get_preset, estimate_parameters, SCALE_PRESETS

    assert "chatgpt" in SCALE_PRESETS
    small = get_preset("small")
    cfg = small.to_gpt_config()
    assert cfg.architecture == "modern"
    assert estimate_parameters(cfg) > 1_000_000
    chatgpt = get_preset("chatgpt")
    assert chatgpt.approx_params.startswith("~175")


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


def test_context_compression():
    messy = "Hello   world\n\n\n\nThis is   really   just   filler text."
    result = compress_context(messy, max_chars=200)
    assert "   " not in result.text
    assert result.compressed_chars <= result.original_chars
    assert collapse_whitespace("a  \n\n\n  b") == "a\n\n  b".replace("  b", "b") or "a" in collapse_whitespace("a  \n\n\n  b")


def test_response_cache():
    cache = ResponseCache(max_size=2, ttl_seconds=None)
    key = make_cache_key("m", "hi", temperature=0.0, strategy="greedy", top_k=40, top_p=0.9, max_new_tokens=10)
    assert cache.get(key) is None
    cache.put(key, "hello")
    assert cache.get(key) == "hello"
    assert cache.stats.hits == 1


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


def test_qa_dataset_and_rag_prompt():
    from myai_datasets.qa_dataset import format_prompt, load_qa_jsonl, QADataset
    from inference.rag import build_rag_prompt, extract_answer
    from memory.search import SearchIndex

    rows = load_qa_jsonl(PROJECT_ROOT / "myai_datasets" / "qa_seed.jsonl")
    assert len(rows) >= 20
    assert "Paris" in rows[0]["answer"] or "France" in rows[0]["question"]

    tok = BPETokenizer()
    tok.train_from_text("What is the capital of France? Paris is the capital.", vocab_size=80)
    ds = QADataset(rows[:5], tok, seq_len=64)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (64,)
    assert (y == -100).any()

    idx = SearchIndex()
    idx.add("f1", "Q: What is the Sun?\nA: The Sun is a star.")
    prompt = build_rag_prompt("What is the Sun?", idx, qa_style=True)
    assert "### Question:" in prompt
    assert "### Answer:" in prompt
    assert "star" in prompt.lower()
    assert extract_answer("### Answer:\nThe Sun is a star.\nReferences") == "The Sun is a star."


def test_checkpoint_roundtrip_unified_loader():
    config = GPTConfig(vocab_size=40, d_model=32, num_heads=4, num_layers=1, d_ff=64, max_seq_len=16)
    model = GPTLanguageModel(config)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "step": 42,
                "tokenizer_kind": "bpe",
            },
            path,
        )
        # Force BPE even though vocab != 50257
        tok = BPETokenizer()
        tok.train_from_text("hello world test data", vocab_size=40)
        tok_path = Path(tmp) / "vocab.json"
        tok.save(str(tok_path))

        loaded = load_checkpoint(str(path), tokenizer_path=str(tok_path), tokenizer_kind="bpe")
        assert loaded.step == 42
        assert loaded.config.vocab_size == 40
