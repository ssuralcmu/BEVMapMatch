from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def coarse_metrics(logits: torch.Tensor, labels: torch.Tensor, grid_dim: int = 10) -> dict[str, float]:
    predicted = logits.argmax(1)
    truth = labels.argmax(1)
    row_delta = ((predicted // grid_dim) - (truth // grid_dim)).abs()
    col_delta = ((predicted % grid_dim) - (truth % grid_dim)).abs()
    return {
        "top_1x1": (predicted == truth).float().mean().item(),
        "top_3x3": torch.maximum(row_delta, col_delta).le(1).float().mean().item(),
    }


def localization_metrics(errors_m: list[float]) -> dict[str, float]:
    errors = np.asarray(errors_m, dtype=float)
    if errors.size == 0:
        raise ValueError("No localization errors supplied")
    output = {f"recall_at_{threshold}m": float(np.mean(errors <= threshold)) for threshold in (1, 2, 5, 10)}
    output.update({"count": int(errors.size), "mean_m": float(errors.mean()), "median_m": float(np.median(errors))})
    return output


def load_errors(path: str | Path) -> list[float]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    return [float(row["distance_m"]) for row in rows if row.get("distance_m") is not None]

