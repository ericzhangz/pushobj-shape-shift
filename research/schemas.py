"""Small serialization helpers for the PushObj contrast pilot.

The pilot keeps large arrays in ``.pt``/``.npz`` sidecars and stores only
scalar metadata and sidecar paths in JSONL records.  This makes every record
inspectable without silently rounding the tensors used for later analysis.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class JsonlWriter:
    """Append-only JSONL writer that flushes every record."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(jsonable(record), sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(tensor.dtype),
        "min": float(value.min()) if value.numel() else None,
        "max": float(value.max()) if value.numel() else None,
        "mean": float(value.mean()) if value.numel() else None,
        "norm": float(value.norm()) if value.numel() else 0.0,
    }

