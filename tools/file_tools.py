"""Phase 11: Safe file read/write tools."""

from pathlib import Path
from typing import Optional


def read_file(path: str, max_chars: int = 50_000) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"
    content = p.read_text(encoding="utf-8", errors="ignore")
    if len(content) > max_chars:
        return content[:max_chars] + f"\n... (truncated, {len(content):,} total chars)"
    return content


def write_file(path: str, content: str, overwrite: bool = False) -> str:
    p = Path(path)
    if p.exists() and not overwrite:
        return f"Error: file exists (set overwrite=True): {path}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content):,} chars to {path}"


def list_directory(path: str = ".") -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: directory not found: {path}"
    entries = []
    for item in sorted(p.iterdir()):
        kind = "dir" if item.is_dir() else "file"
        entries.append(f"  [{kind}] {item.name}")
    return "\n".join(entries) if entries else "(empty directory)"
