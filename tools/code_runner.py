"""Phase 11: Execute approved Python code in a restricted namespace."""

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any


def run_python(code: str, timeout_hint: str = "no timeout enforced") -> str:
    """
    Run Python code with a minimal safe namespace.
    Only use with code you trust — this is not a full sandbox.
    """
    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = {}

    # Allow a small set of builtins
    import builtins
    for name in ("print", "range", "len", "int", "float", "str", "list", "dict", "sum", "min", "max"):
        safe_globals[name] = getattr(builtins, name)

    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, safe_globals, safe_locals)
        out = stdout.getvalue()
        err = stderr.getvalue()
        result = out or "(no output)"
        if err:
            result += f"\n[stderr]\n{err}"
        return result
    except Exception as e:
        return f"Error executing code: {e}"
