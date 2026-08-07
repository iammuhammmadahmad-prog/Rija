"""
Phase 12 + 13: Desktop GUI with chat, training controls, and model manager.

Uses tkinter (stdlib) so no extra GUI dependencies are required.
Run: python -m ui.app
"""

import json
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
from typing import Optional

from memory.chat_history import ChatHistory
from memory.knowledge_base import KnowledgeBase
from memory.search import SearchIndex
from tools.registry import ToolRegistry
from inference.generate import InferenceEngine, GenerationConfig
from trainer.config import TrainConfig, tiny_v1
from trainer.train import Trainer
from myai_datasets.prepare import prepare_dataset


class MyAIApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MyAI — Build Your Own LLM")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        self.chat_history = ChatHistory()
        self.knowledge_base = KnowledgeBase()
        self.search_index = SearchIndex()
        self.tools = ToolRegistry()
        self.engine: Optional[InferenceEngine] = None
        self.current_conv_id: Optional[str] = None
        self.trainer: Optional[Trainer] = None
        self.train_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._load_knowledge_into_index()
        self._new_conversation()

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.chat_tab = ttk.Frame(notebook)
        self.train_tab = ttk.Frame(notebook)
        self.models_tab = ttk.Frame(notebook)
        self.tools_tab = ttk.Frame(notebook)
        self.logs_tab = ttk.Frame(notebook)

        notebook.add(self.chat_tab, text="Chat")
        notebook.add(self.train_tab, text="Trainer")
        notebook.add(self.models_tab, text="Models")
        notebook.add(self.tools_tab, text="Tools")
        notebook.add(self.logs_tab, text="Logs")

        self._build_chat_tab()
        self._build_train_tab()
        self._build_models_tab()
        self._build_tools_tab()
        self._build_logs_tab()

    # ------------------------------------------------------------------ #
    # Chat tab
    # ------------------------------------------------------------------ #
    def _build_chat_tab(self):
        top = ttk.Frame(self.chat_tab)
        top.pack(fill=tk.X, pady=4)

        ttk.Label(top, text="Model:").pack(side=tk.LEFT, padx=4)
        self.model_var = tk.StringVar(value="(none loaded)")
        ttk.Label(top, textvariable=self.model_var).pack(side=tk.LEFT)

        ttk.Button(top, text="Load Checkpoint", command=self._load_model_dialog).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="New Chat", command=self._new_conversation).pack(side=tk.RIGHT, padx=4)

        settings = ttk.LabelFrame(self.chat_tab, text="Generation Settings")
        settings.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(settings, text="Strategy:").grid(row=0, column=0, padx=4, pady=2)
        self.strategy_var = tk.StringVar(value="temperature")
        ttk.Combobox(settings, textvariable=self.strategy_var,
                     values=["greedy", "temperature", "top_k", "top_p"], width=12).grid(row=0, column=1, padx=4)

        ttk.Label(settings, text="Temperature:").grid(row=0, column=2, padx=4)
        self.temp_var = tk.DoubleVar(value=0.8)
        ttk.Entry(settings, textvariable=self.temp_var, width=6).grid(row=0, column=3, padx=4)

        ttk.Label(settings, text="Max tokens:").grid(row=0, column=4, padx=4)
        self.max_tokens_var = tk.IntVar(value=80)
        ttk.Entry(settings, textvariable=self.max_tokens_var, width=6).grid(row=0, column=5, padx=4)

        self.use_memory_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings, text="Use memory retrieval", variable=self.use_memory_var).grid(row=0, column=6, padx=8)

        self.chat_display = scrolledtext.ScrolledText(self.chat_tab, wrap=tk.WORD, state=tk.DISABLED, height=25)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        input_frame = ttk.Frame(self.chat_tab)
        input_frame.pack(fill=tk.X, padx=4, pady=4)

        self.user_input = tk.Text(input_frame, height=3, wrap=tk.WORD)
        self.user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.user_input.bind("<Control-Return>", lambda e: self._send_message())

        ttk.Button(input_frame, text="Send (Ctrl+Enter)", command=self._send_message).pack(side=tk.RIGHT, padx=4)

    def _append_chat(self, role: str, text: str):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n[{role.upper()}]\n{text}\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _new_conversation(self):
        self.current_conv_id = self.chat_history.create_conversation()
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self._append_chat("system", "New conversation started. Load a checkpoint to chat with your model.")

    def _load_model_dialog(self):
        path = filedialog.askopenfilename(
            title="Load checkpoint",
            initialdir="checkpoints",
            filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self._load_model(path)

    def _load_model(self, checkpoint_path: str):
        try:
            self.engine = InferenceEngine.from_checkpoint(checkpoint_path)
            self.model_var.set(Path(checkpoint_path).name)
            self._append_chat("system", f"Loaded model from {checkpoint_path}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _send_message(self):
        text = self.user_input.get("1.0", tk.END).strip()
        if not text:
            return
        self.user_input.delete("1.0", tk.END)
        self._append_chat("user", text)
        self.chat_history.add_message(self.current_conv_id, "user", text)

        if self.engine is None:
            self._append_chat("assistant", "No model loaded. Go to Models tab or click Load Checkpoint.")
            return

        prompt = text
        if self.use_memory_var.get():
            context = self.search_index.build_context(text)
            if context:
                prompt = f"Context:\n{context}\n\nUser: {text}\nAssistant:"

        config = GenerationConfig(
            max_new_tokens=self.max_tokens_var.get(),
            strategy=self.strategy_var.get(),
            temperature=self.temp_var.get(),
        )

        try:
            response = self.engine.generate(prompt, config)
            self._append_chat("assistant", response)
            self.chat_history.add_message(self.current_conv_id, "assistant", response)
        except Exception as e:
            self._append_chat("system", f"Generation error: {e}")

    # ------------------------------------------------------------------ #
    # Trainer tab (Phase 13)
    # ------------------------------------------------------------------ #
    def _build_train_tab(self):
        cfg_frame = ttk.LabelFrame(self.train_tab, text="Training Configuration")
        cfg_frame.pack(fill=tk.X, padx=4, pady=4)

        self.train_status = tk.StringVar(value="Idle")
        ttk.Label(cfg_frame, textvariable=self.train_status, font=("", 10, "bold")).grid(row=0, column=0, columnspan=4, pady=4)

        fields = [
            ("Model version", "model_version", "v1"),
            ("Max steps", "max_steps", "1000"),
            ("Batch size", "batch_size", "8"),
            ("Learning rate", "learning_rate", "3e-4"),
            ("d_model", "d_model", "64"),
            ("num_layers", "num_layers", "2"),
        ]
        self.train_fields = {}
        for i, (label, key, default) in enumerate(fields):
            ttk.Label(cfg_frame, text=label + ":").grid(row=1 + i // 3, column=(i % 3) * 2, padx=4, pady=2, sticky=tk.E)
            var = tk.StringVar(value=default)
            ttk.Entry(cfg_frame, textvariable=var, width=12).grid(row=1 + i // 3, column=(i % 3) * 2 + 1, padx=4, pady=2)
            self.train_fields[key] = var

        btn_frame = ttk.Frame(self.train_tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(btn_frame, text="Prepare Dataset", command=self._prepare_dataset).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Start Training", command=self._start_training).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Pause", command=self._pause_training).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Resume", command=self._resume_training).pack(side=tk.LEFT, padx=4)

        self.train_log = scrolledtext.ScrolledText(self.train_tab, wrap=tk.WORD, height=20, state=tk.DISABLED)
        self.train_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _train_log_append(self, msg: str):
        self.train_log.config(state=tk.NORMAL)
        self.train_log.insert(tk.END, msg + "\n")
        self.train_log.config(state=tk.DISABLED)
        self.train_log.see(tk.END)

    def _prepare_dataset(self):
        try:
            result = prepare_dataset()
            self._train_log_append(
                f"Dataset ready: train={result['train_chars']:,} chars, "
                f"val={result['val_chars']:,} chars, vocab={result.get('vocab_size', '?')}"
            )
            self.train_status.set("Dataset prepared")
        except Exception as e:
            messagebox.showerror("Dataset Error", str(e))

    def _get_train_config(self) -> TrainConfig:
        cfg = tiny_v1()
        cfg.model_version = self.train_fields["model_version"].get()
        cfg.max_steps = int(self.train_fields["max_steps"].get())
        cfg.batch_size = int(self.train_fields["batch_size"].get())
        cfg.learning_rate = float(self.train_fields["learning_rate"].get())
        cfg.d_model = int(self.train_fields["d_model"].get())
        cfg.num_layers = int(self.train_fields["num_layers"].get())
        return cfg

    def _start_training(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("Training", "Training already running.")
            return

        def run():
            try:
                config = self._get_train_config()
                self.trainer = Trainer(config)
                self.train_status.set(f"Training {config.model_version}...")
                self._train_log_append(f"Started training: {config.model_version}, {self.trainer.model.num_parameters():,} params")

                def on_progress(record):
                    msg = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
                    self.root.after(0, lambda m=msg: self._train_log_append(m))

                summary = self.trainer.train(progress_callback=on_progress)
                self.root.after(0, lambda: self.train_status.set("Training complete"))
                self.root.after(0, lambda: self._train_log_append(f"Done: {summary}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
                self.root.after(0, lambda: self.train_status.set("Error"))

        self.train_thread = threading.Thread(target=run, daemon=True)
        self.train_thread.start()

    def _pause_training(self):
        if self.trainer:
            self.trainer.pause()
            self.train_status.set("Paused")

    def _resume_training(self):
        if self.trainer:
            self.trainer.resume()
            self.train_status.set("Training...")

    # ------------------------------------------------------------------ #
    # Models tab
    # ------------------------------------------------------------------ #
    def _build_models_tab(self):
        ttk.Label(self.models_tab, text="Model Registry (Phase 9)", font=("", 12, "bold")).pack(pady=8)
        self.registry_text = scrolledtext.ScrolledText(self.models_tab, wrap=tk.WORD, height=20)
        self.registry_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        ttk.Button(self.models_tab, text="Refresh", command=self._refresh_registry).pack(pady=4)
        self._refresh_registry()

    def _refresh_registry(self):
        registry_path = Path("checkpoints/registry.json")
        self.registry_text.delete("1.0", tk.END)
        if registry_path.exists():
            self.registry_text.insert(tk.END, registry_path.read_text(encoding="utf-8"))
        else:
            self.registry_text.insert(tk.END, "No models trained yet. Use the Trainer tab to train v1.")

    # ------------------------------------------------------------------ #
    # Tools tab
    # ------------------------------------------------------------------ #
    def _build_tools_tab(self):
        ttk.Label(self.tools_tab, text="Available Tools (Phase 11)", font=("", 12, "bold")).pack(pady=8)

        tools_list = scrolledtext.ScrolledText(self.tools_tab, wrap=tk.WORD, height=8)
        tools_list.pack(fill=tk.X, padx=8, pady=4)
        for name, desc in self.tools.list_tools().items():
            tools_list.insert(tk.END, f"  {name}: {desc}\n")

        ttk.Label(self.tools_tab, text="Try a tool:").pack(anchor=tk.W, padx=8)
        tool_frame = ttk.Frame(self.tools_tab)
        tool_frame.pack(fill=tk.X, padx=8, pady=4)

        self.tool_name_var = tk.StringVar(value="calculate")
        ttk.Combobox(tool_frame, textvariable=self.tool_name_var,
                     values=list(self.tools.list_tools()), width=15).pack(side=tk.LEFT, padx=4)
        self.tool_args_var = tk.StringVar(value='expression="2+2*3"')
        ttk.Entry(tool_frame, textvariable=self.tool_args_var, width=50).pack(side=tk.LEFT, padx=4)
        ttk.Button(tool_frame, text="Run", command=self._run_tool).pack(side=tk.LEFT, padx=4)

        self.tool_output = scrolledtext.ScrolledText(self.tools_tab, wrap=tk.WORD, height=12)
        self.tool_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _run_tool(self):
        name = self.tool_name_var.get()
        args_str = self.tool_args_var.get()
        try:
            kwargs = eval(f"dict({args_str})")  # noqa: S307 — local GUI only
            result = self.tools.call(name, **kwargs)
        except Exception as e:
            result = f"Error parsing args: {e}"
        self.tool_output.insert(tk.END, f"[{name}] {result}\n")
        self.tool_output.see(tk.END)

    # ------------------------------------------------------------------ #
    # Logs tab
    # ------------------------------------------------------------------ #
    def _build_logs_tab(self):
        self.logs_display = scrolledtext.ScrolledText(self.logs_tab, wrap=tk.WORD)
        self.logs_display.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Button(self.logs_tab, text="Refresh Logs", command=self._refresh_logs).pack(pady=4)
        self._refresh_logs()

    def _refresh_logs(self):
        self.logs_display.delete("1.0", tk.END)
        log_dir = Path("logs")
        if not log_dir.exists():
            self.logs_display.insert(tk.END, "No logs yet.\n")
            return
        for log_file in sorted(log_dir.glob("*.jsonl")):
            self.logs_display.insert(tk.END, f"=== {log_file.name} ===\n")
            lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            for line in lines[-50:]:
                self.logs_display.insert(tk.END, line + "\n")

    def _load_knowledge_into_index(self):
        for doc in self.knowledge_base.load_all_documents():
            self.search_index.add(doc["id"], doc["content"])

    def run(self):
        self.root.mainloop()


def launch_app():
    app = MyAIApp()
    app.run()


if __name__ == "__main__":
    launch_app()
