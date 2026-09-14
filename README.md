# Rija! Build Your Own LLM From Scratch

Rija is a from-scratch language model stack: tokenizer, transformer, trainer, RAG, CLI, and a local Studio UI. Weights start random. There is no Hugging Face model download for the network itself.

The current laptop run is **v5 tiny 19.2M parameters**, pretrained on Wikipedia with a 100,000-step budget.

## Stack

| Layer | What it is |
|---|---|
| Tokenizer | Custom BPE for v1, GPT-2 `tiktoken` (50,257) for Wikipedia / v5 |
| Model | Decoder-only GPT. v1–v4 are classic (learned PE, LayerNorm, GELU). v5 is modern (RoPE, RMSNorm, SwiGLU) |
| Train | Local text, Wikipedia stream, QA fine-tune, cosine LR, checkpoint resume |
| Inference | Greedy, temperature, top-k, top-p |
| Memory | TF-IDF RAG over a local knowledge base |
| Apps | React Studio at `http://127.0.0.1:8765`, plus a Tk desktop app |
| CLI | `prepare` → `train` / `train-v5` → `finetune-qa` → `evaluate` → `generate` → `studio` |

## Quick start

Python 3.10+ and a virtualenv:

```bash
git clone https://github.com/iammuhammmadahmad-prog/Rija.git
cd Rija
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py smoke
python main.py studio
```

Studio opens at [http://127.0.0.1:8765](http://127.0.0.1:8765). Load a `.pt` checkpoint from the sidebar and type a prompt.

## Train

**Tiny v1** (custom BPE, CPU, minutes):

```bash
python main.py prepare
python main.py train --preset tiny --steps 1000 --version v1
python main.py generate "Machine learning" --strategy top_p --version v1
```

**v5 Wikipedia pretrain** (tiktoken, laptop-safe tiny preset):

```bash
python -m trainer.train_v5 --size tiny --seq-len 128 --batch-size 8 --no-compile --no-resume
# or
python main.py train-v5 --size tiny --seq-len 128 --batch-size 8
```

Resume is the default. Checkpoints land in `checkpoints/v5_tiny/`. Watch the log with `tail -f logs/train_v5_tiny.log`.

**QA fine-tune** on a finished base:

```bash
python main.py finetune-qa --base checkpoints/v5_tiny/latest.pt --out checkpoints/v5_qa
python main.py generate "What is a transformer?" --qa --rag --version v5_qa
```

## v5 size ladder

| Size | Params (approx) | Hardware |
|---|---|---|
| `tiny` | ~19M | Laptop CPU / GPU |
| `small` | ~50–80M | Laptop GPU, 8–12 GB |
| `base` | ~125M | Single 16–24 GB GPU |
| `medium` | ~350M | 24–48 GB GPU |
| `large` / `xl` / `chatgpt` | 1.5B–175B | Blocked unless `--force-huge` |

`tiny` on CPU is the supported path. Larger sizes are configs, not a promise this machine can train them.

## Studio and desktop

```bash
python main.py studio --host 127.0.0.1 --port 8765
python main.py gui
```

Studio streams tokens from a local checkpoint. The desktop app has Chat, Trainer, Models, Tools, and Logs.

## CLI

```text
python main.py smoke         # tokenizer + tiny forward pass
python main.py prepare       # clean raw text, train BPE
python main.py train         # v1 loop
python main.py train-v5      # modern GPT scale ladder
python main.py finetune-qa   # instruction / Q&A pass
python main.py seed-facts    # load qa_seed.jsonl into RAG
python main.py generate      # sample a checkpoint
python main.py evaluate      # perplexity / accuracy
python main.py studio        # web playground
python main.py gui           # Tk desktop
```

## Layout

```text
Rija/
├── main.py                 # CLI
├── generate.py             # interactive generation
├── tokenizer/              # custom BPE
├── model/                  # GPT (classic + modern)
│   └── scaling.py          # v5 size presets
├── trainer/
│   ├── train.py            # v1
│   ├── train_wiki.py       # Wikipedia (classic)
│   ├── train_v5.py         # Wikipedia (modern)
│   ├── finetune_qa.py
│   └── cpu_runtime.py      # laptop P/E-core pinning
├── myai_datasets/          # raw / processed / Wikipedia loader / QA seed
├── inference/              # decode, RAG, checkpoint load
├── memory/                 # knowledge base, chat history, search
├── tools/                  # files, calculator, code runner
├── evaluation/
├── ui/
│   ├── app.py              # desktop
│   ├── studio.py           # FastAPI + SSE
│   └── static/             # React Studio
├── checkpoints/
└── tests/
```

## Tests

```bash
python -m pytest tests/ -v
```

## Honest status

v5 tiny has learned Wikipedia-shaped English. It is not a chat assistant. Samples can look fluent and still be empty or circular. That is expected at ~19M parameters with a 50k GPT-2 vocab on a laptop.

Next useful upgrades are a smaller custom vocab (4k–8k) trained on the same data, or more tokens on this architecture not a larger size preset on CPU.
