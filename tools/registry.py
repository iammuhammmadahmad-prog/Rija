"""
Phase 11: Central registry of tools the AI can invoke.
Add new tools here — you control what's available.
"""

from typing import Callable, Dict, Any
import json

from tools.file_tools import read_file, write_file, list_directory
from tools.calculator import calculate
from tools.code_runner import run_python


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, dict] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register("read_file", read_file, "Read a text file from disk")
        self.register("write_file", write_file, "Write content to a file")
        self.register("list_directory", list_directory, "List files in a directory")
        self.register("calculate", calculate, "Evaluate a math expression safely")
        self.register("run_python", run_python, "Execute approved Python code")

    def register(self, name: str, fn: Callable, description: str) -> None:
        self._tools[name] = {"fn": fn, "description": description}

    def list_tools(self) -> Dict[str, str]:
        return {name: info["description"] for name, info in self._tools.items()}

    def call(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'. Available: {list(self._tools)}"
        try:
            result = self._tools[name]["fn"](**kwargs)
            return str(result)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error running {name}: {e}"

    def call_from_json(self, payload: str) -> str:
        """Accept {"tool": "calculate", "args": {"expression": "2+2"}}"""
        data = json.loads(payload)
        return self.call(data["tool"], **data.get("args", {}))
