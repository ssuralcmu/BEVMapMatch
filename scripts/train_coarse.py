#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bevmapmatch.config import load_config
from bevmapmatch.data import MapDataset
from bevmapmatch.losses import paper_loss
from bevmapmatch.metrics import coarse_metrics
from bevmapmatch.model import CoarseMatcher


def dataset(config: dict, split: str) -> MapDataset:
    root = Path(config["data"]["root"])
    paths = config["data"][split]
    return MapDataset(
        root / paths["metadata"], root / paths["basemaps"], root / paths["segmentations"],
        image_size=config["data"]["image_size"], map_size_m=config["data"]["map_size_m"],
        grid_dim=config["data"]["grid_dim"], neighbor_frames=config["model"]["neighbor_frames"],
        neighbor_window_s=config["model"]["neighbor_window_s"],
    )


def run_epoch(model, loader, device, optimizer, config):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "top_1x1": 0.0, "top_3x3": 0.0}
    for batch in tqdm(loader, leave=False):
        inputs = [batch[name].to(device) for name in ("segmentation", "basemap", "neighbors")]
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(inputs[0], inputs[1], inputs[2])
            loss, _ = paper_loss(logits, labels, config["data"]["grid_dim"], config["train"]["distance_loss_weight"], config["train"]["distance_sigma"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        metrics = coarse_metrics(logits.detach(), labels, config["data"]["grid_dim"])
        totals["loss"] += loss.item()
        totals["top_1x1"] += metrics["top_1x1"]
        totals["top_3x3"] += metrics["top_3x3"]
    return {key: value / len(loader) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--output", default="checkpoints/coarse.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoarseMatcher(config["model"]["backbone"], config["data"]["grid_dim"], config["model"]["attention_heads"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    loaders = {split: DataLoader(dataset(config, split), batch_size=config["train"]["batch_size"], shuffle=split == "train", num_workers=config["train"]["workers"], pin_memory=True) for split in ("train", "val")}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    for epoch in range(config["train"]["epochs"]):
        train = run_epoch(model, loaders["train"], device, optimizer, config)
        val = run_epoch(model, loaders["val"], device, None, config)
        history.append({"epoch": epoch + 1, "train": train, "val": val})
        print(json.dumps(history[-1]))
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch + 1}, output)
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()

