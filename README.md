# Rija — Build Your Own LLM From Scratch

A phased roadmap for building a transformer language model without relying on
external tokenizer or model libraries (no `tiktoken`, no `transformers`).
Everything uses plain Python + PyTorch.

## Quick start

```bash
pip install -r requirements.txt

# Smoke test (Phases 3+4)
python main.py smoke

# Full pipeline
python main.py prepare          # Phase 6: clean data + train tokenizer
python main.py train --prepare  # Phase 5/7: train v1 (~1-5M params)
python main.py generate "The transformer" --strategy temperature
python main.py evaluate
python main.py gui              # Desktop app (Phases 12/13)
```

## What's implemented

| Phase | Module | Status |
|-------|--------|--------|
| 3 | `tokenizer/` | BPE tokenizer — encode/decode/save/load |
| 4 | `model/` | GPT-style transformer (embeddings, attention, blocks) |
| 5 | `trainer/` | Dataset, batching, training loop, checkpoints |
| 6 | `datasets/` | Clean, dedupe, split, sample data in `raw/` |
| 7 | `trainer/config.py` | Tiny v1 preset for first training run |
| 8 | `inference/` | Greedy, temperature, top-k, top-p generation |
| 9 | `checkpoints/registry.json` | Model versioning + dataset records |
| 10 | `memory/` | Knowledge base, chat history, search index |
| 11 | `tools/` | File I/O, calculator, code runner, registry |
| 12 | `ui/app.py` | Desktop GUI — chat window |
| 13 | `ui/app.py` | Trainer tab — start/pause/resume training |
| 14 | `evaluation/` | Perplexity, accuracy, speed benchmarks |
| 15 | `trainer/config.py` | `tiny_v1()` → `small_v2()` scaling presets |

## Project structure

```
Rija/
├── datasets/
│   ├── raw/              # Put your text files here
│   ├── processed/        # train.txt + val.txt (generated)
│   └── prepare.py        # Phase 6 pipeline
├── tokenizer/
│   └── tokenizer.py      # Phase 3: BPE tokenizer
├── model/
│   ├── embeddings.py     # Token + positional embeddings
│   ├── attention.py      # Multi-head self-attention
│   ├── transformer.py    # Transformer block
│   └── model.py          # Full GPT language model
├── trainer/
│   ├── config.py         # Training presets (tiny v1, small v2)
│   ├── dataset.py        # Text chunk dataset + DataLoader
│   └── train.py          # Training loop + checkpoints
├── inference/
│   └── generate.py       # Phase 8: decoding strategies
├── memory/
│   ├── knowledge_base.py # Documents + user profiles
│   ├── chat_history.py   # Conversation persistence
│   └── search.py         # TF-IDF retrieval for RAG
├── tools/
│   ├── file_tools.py     # Read/write/list files
│   ├── calculator.py     # Safe math evaluation
│   ├── code_runner.py    # Approved Python execution
│   └── registry.py       # Tool dispatch
├── evaluation/
│   └── metrics.py        # Perplexity, accuracy, speed
├── ui/
│   └── app.py            # tkinter GUI (chat + trainer)
├── checkpoints/          # Saved model weights
├── logs/                 # Training JSONL logs
├── tests/
│   └── test_core.py
└── main.py               # CLI entry point
```

## Training your first model (Phase 7)

1. Add text files to `datasets/raw/` (sample files included).
2. Prepare the dataset:
   ```bash
   python main.py prepare
   ```
3. Train v1 (tiny, CPU-friendly):
   ```bash
   python main.py train --preset tiny --steps 1000 --version v1
   ```
4. Generate text:
   ```bash
   python main.py generate "Machine learning" --strategy top_p
   ```
5. Evaluate:
   ```bash
   python main.py evaluate --version v1
   ```

## Scaling up (Phase 15)

After v1 works, edit `trainer/config.py` or use the GUI to increase:
- `d_model`, `num_layers`, `num_heads` (model size)
- `max_steps`, `batch_size` (training duration)
- `vocab_size` (tokenizer capacity)

Train v2 with `--version v2` — the registry tracks which data each version saw.

## GUI

```bash
python main.py gui
```

Tabs:
- **Chat** — load a checkpoint, chat with memory retrieval
- **Trainer** — prepare data, start/pause/resume training
- **Models** — view version registry
- **Tools** — test file/calc/code tools
- **Logs** — view training metrics

## Tests

```bash
python -m pytest tests/ -v
```

## Design principles

- **From scratch** — you own every line; no black-box libraries for core ML
- **Start tiny** — v1 fits on CPU with ~1-5M parameters
- **Version everything** — checkpoints, datasets, and configs are tracked
- **Modular** — memory, tools, and UI are separate from the model core

---

**Rija — Build your own LLM from scratch.**
