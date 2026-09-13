"""Laptop / CPU topology helpers for v4_wiki training."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def cpu_max_freqs() -> dict[int, int]:
    freqs: dict[int, int] = {}
    root = Path("/sys/devices/system/cpu")
    if not root.exists():
        return freqs
    for child in sorted(root.glob("cpu[0-9]*")):
        name = child.name
        if not name[3:].isdigit():
            continue
        idx = int(name[3:])
        freq = _read_int(child / "cpufreq" / "cpuinfo_max_freq")
        if freq is not None:
            freqs[idx] = freq
    return freqs


def classify_cores(freqs: dict[int, int] | None = None) -> tuple[list[int], list[int], list[int]]:
    """Return (p_cores, e_cores, lpe_cores) from max frequency buckets."""
    freqs = cpu_max_freqs() if freqs is None else freqs
    if not freqs:
        n = os.cpu_count() or 1
        return list(range(n)), [], []
    values = sorted(set(freqs.values()), reverse=True)
    p_cut = values[0]
    lpe_cut = values[-1] if len(values) >= 3 else None
    p_cores = sorted(i for i, f in freqs.items() if f >= p_cut * 0.95)
    lpe = sorted(i for i, f in freqs.items() if lpe_cut is not None and f <= lpe_cut * 1.05)
    e_cores = sorted(i for i in freqs if i not in p_cores and i not in lpe)
    return p_cores, e_cores, lpe


def request_performance_profile() -> str | None:
    """Ask GNOME/power-profiles-daemon for the performance profile (no sudo)."""
    try:
        subprocess.run(
            ["powerprofilesctl", "set", "performance"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "performance"
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


def pin_current_thread(cpus: Iterable[int]) -> bool:
    mask = set(cpus)
    if not mask:
        return False
    try:
        os.sched_setaffinity(0, mask)
        return True
    except (OSError, AttributeError):
        return False


def configure_cpu_runtime() -> dict:
    """
    Pin the training thread to P-cores and size OpenMP to that count.

    On this ThinkPad (Ultra 7 255H) using all 16 cores is much slower than
    the 6 performance cores: E-cores / LPE unbalance GEMM.
    """
    p_cores, e_cores, lpe = classify_cores()
    pin_current_thread(p_cores)
    n = max(1, len(p_cores))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    try:
        import torch

        torch.set_num_interop_threads(1)
        torch.set_num_threads(n)
    except RuntimeError:
        pass
    profile = request_performance_profile()
    return {
        "p_cores": p_cores,
        "e_cores": e_cores,
        "lpe_cores": lpe,
        "threads": n,
        "power_profile": profile,
    }


def has_cpu_bf16_accel() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags
