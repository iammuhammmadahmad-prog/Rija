"""
Rija Studio — local web playground for testing checkpoints.

  python main.py studio
  python -m ui.studio
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

STATIC_DIR = Path(__file__).resolve().parent / "static"
CHECKPOINTS = root_dir / "checkpoints"
PREFERRED_VERSIONS = ("v4_wiki", "v5_qa", "v5_smart", "v2", "v1")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    strategy: str = "temperature"
    temperature: float = 0.8
    max_tokens: int = 80
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    qa: bool = True
    rag: bool = True


class LoadRequest(BaseModel):
    path: str


class StudioState:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        self.path: Optional[str] = None
        self.meta: dict[str, Any] = {}
        self.error: Optional[str] = None
        self.loading = False
        self.search_index = None
        self.cancel = threading.Event()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ready": self.engine is not None,
                "loading": self.loading,
                "path": self.path,
                "error": self.error,
                "meta": dict(self.meta),
            }


state = StudioState()


def list_checkpoints(base: Path = CHECKPOINTS) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not base.exists():
        return items
    seen: set[str] = set()
    for version_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        candidates = []
        latest = version_dir / "latest.pt"
        if latest.exists():
            candidates.append(latest)
        steps = sorted(version_dir.glob("step_*.pt"))
        if steps:
            newest = steps[-1]
            if newest.resolve() != latest.resolve() if latest.exists() else True:
                candidates.append(newest)
        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "version": version_dir.name,
                    "name": path.name,
                    "path": str(path.relative_to(root_dir))
                    if path.is_relative_to(root_dir)
                    else str(path),
                    "abs_path": str(path),
                    "mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "preferred": version_dir.name in PREFERRED_VERSIONS,
                }
            )
    items.sort(key=lambda x: (0 if x["preferred"] else 1, x["version"], x["name"]))
    return items


def _pick_default_checkpoint() -> Optional[str]:
    available = {item["version"]: item["abs_path"] for item in list_checkpoints()}
    for version in PREFERRED_VERSIONS:
        if version in available:
            return available[version]
    items = list_checkpoints()
    return items[0]["abs_path"] if items else None


def _ensure_rag_index():
    if state.search_index is not None:
        return state.search_index
    from memory.seed_facts import build_index, seed_from_qa

    docs = root_dir / "memory" / "store" / "documents"
    if not docs.exists() or not any(docs.glob("*.json")):
        seed_from_qa()
    state.search_index = build_index()
    return state.search_index


def load_checkpoint_into_state(path: str) -> dict[str, Any]:
    from inference.checkpoint import load_checkpoint
    from inference.generate import InferenceEngine

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root_dir / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")

    with state.lock:
        state.loading = True
        state.error = None
    try:
        loaded = load_checkpoint(str(resolved))
        engine = InferenceEngine(loaded.model, loaded.tokenizer, model_id=str(resolved))
        meta = {
            "version": resolved.parent.name,
            "file": resolved.name,
            "tokenizer": loaded.tokenizer_kind,
            "vocab_size": loaded.config.vocab_size,
            "d_model": loaded.config.d_model,
            "num_layers": loaded.config.num_layers,
            "max_seq_len": loaded.config.max_seq_len,
            "architecture": getattr(loaded.config, "architecture", "classic"),
            "step": loaded.step,
            "parameters": loaded.model.num_parameters(),
        }
        with state.lock:
            state.engine = engine
            state.path = str(resolved)
            state.meta = meta
            state.loading = False
            state.error = None
        return meta
    except Exception as exc:
        with state.lock:
            state.loading = False
            state.error = str(exc)
        raise


def _autoload():
    path = _pick_default_checkpoint()
    if path is None:
        with state.lock:
            state.error = "No checkpoint found. Train v4_wiki first."
        return
    try:
        load_checkpoint_into_state(path)
    except Exception as exc:
        with state.lock:
            state.error = str(exc)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        threading.Thread(target=_autoload, name="myai-autoload", daemon=True).start()
        yield

    app = FastAPI(title="Rija Studio", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return FileResponse(STATIC_DIR / "favicon.ico", media_type="image/x-icon")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/status")
    def status():
        snap = state.snapshot()
        snap["models"] = list_checkpoints()
        return snap

    @app.get("/api/models")
    def models():
        return {"models": list_checkpoints()}

    @app.post("/api/models/load")
    def load_model(req: LoadRequest):
        try:
            meta = load_checkpoint_into_state(req.path)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "meta": meta}

    @app.get("/api/examples")
    def examples():
        return {
            "examples": [
                "Rija, I was thinking about",
                "Tell me about a quiet Sunday",
                "A letter that starts with your name",
                "What would you say about the sea",
                "Begin with the word tomorrow",
            ]
        }

    @app.post("/api/chat/cancel")
    def cancel_chat():
        state.cancel.set()
        return {"ok": True}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        snap = state.snapshot()
        if snap["loading"]:
            raise HTTPException(409, "Model is still loading.")
        if state.engine is None:
            raise HTTPException(409, snap["error"] or "No model loaded.")

        from inference.generate import GenerationConfig
        from inference.rag import build_rag_prompt, extract_answer

        prompt = req.message
        if req.qa or req.rag:
            index = _ensure_rag_index() if req.rag else None
            prompt = build_rag_prompt(req.message, index, qa_style=True)

        config = GenerationConfig(
            max_new_tokens=max(1, min(req.max_tokens, 256)),
            strategy=req.strategy,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )

        state.cancel.clear()
        token_q: queue.Queue = queue.Queue()

        def _run():
            pieces: list[str] = []
            try:
                for piece in state.engine.generate_stream(prompt, config):
                    if state.cancel.is_set():
                        break
                    pieces.append(piece)
                    token_q.put(("token", piece))
                raw = "".join(pieces)
                answer = extract_answer(prompt + raw) if (req.qa or req.rag) else raw
                token_q.put(("done", {"text": raw, "answer": answer, "stopped": state.cancel.is_set()}))
            except Exception as exc:
                token_q.put(("error", str(exc)))

        threading.Thread(target=_run, name="myai-generate", daemon=True).start()

        def _sse():
            while True:
                kind, payload = token_q.get()
                if kind == "token":
                    yield f"data: {json.dumps({'token': payload})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'done': True, **payload})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'error': payload})}\n\n"
                    break

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return app


app = create_app()


def launch_studio(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    print(f"Rija Studio → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="Launch Rija Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    launch_studio(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
