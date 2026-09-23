#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bevmapmatch.config import load_config
from bevmapmatch.data import MapDataset
from bevmapmatch.metrics import coarse_metrics
from bevmapmatch.model import CoarseMatcher


def load_checkpoint(model: CoarseMatcher, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    legacy_names = {
        "feature_extractor.": "encoder.",
        "pos_embed": "position",
        "cross_attn.": "attention.",
        "stitched_fusion.": "temporal_fusion.",
        "fc.": "classifier.",
    }
    translated = {}
    for key, value in state.items():
        for old, new in legacy_names.items():
            if key == old or key.startswith(old):
                key = new + key[len(old):]
                break
        translated[key] = value
    missing, unexpected = model.load_state_dict(translated, strict=False)
    unsupported_missing = [key for key in missing if not key.startswith("temporal_fusion.")]
    if unsupported_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint is incompatible: missing={unsupported_missing}, unexpected={unexpected}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/coarse_predictions.json")
    parser.add_argument("--neighbor-frames", type=int, default=None, choices=(0, 1, 3, 7))
    args = parser.parse_args()
    config = load_config(args.config); root = Path(config["data"]["root"]); paths = config["data"]["val"]
    neighbor_frames = config["model"]["neighbor_frames"] if args.neighbor_frames is None else args.neighbor_frames
    data = MapDataset(root / paths["metadata"], root / paths["basemaps"], root / paths["segmentations"], image_size=config["data"]["image_size"], map_size_m=config["data"]["map_size_m"], grid_dim=config["data"]["grid_dim"], neighbor_frames=neighbor_frames, neighbor_window_s=config["model"]["neighbor_window_s"])
    loader = DataLoader(data, batch_size=config["train"]["batch_size"], num_workers=config["train"]["workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoarseMatcher(config["model"]["backbone"], config["data"]["grid_dim"], config["model"]["attention_heads"]).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    rows = []; sums = {"top_1x1": 0.0, "top_3x3": 0.0}; batches = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            logits = model(batch["segmentation"].to(device), batch["basemap"].to(device), batch["neighbors"].to(device))
            metrics = coarse_metrics(logits, batch["label"].to(device), config["data"]["grid_dim"])
            for key in sums: sums[key] += metrics[key]
            batches += 1
            probabilities = logits.softmax(1).cpu(); prediction = probabilities.argmax(1)
            for i, sample_id in enumerate(batch["id"]):
                rows.append({"id": sample_id, "predicted_cell": int(prediction[i]), "ground_truth_cell": int(batch["label"][i].argmax()), "probabilities": probabilities[i].tolist(), "segmentation_path": batch["segmentation_path"][i], "basemap_path": batch["basemap_path"][i], "metadata_path": batch["metadata_path"][i]})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": {key: value / batches for key, value in sums.items()}, "predictions": rows}, indent=2))


if __name__ == "__main__":
    main()
