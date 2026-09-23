#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from bevmapmatch.config import load_config
from bevmapmatch.data import MapDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["data"]["root"])
    for split in ("train", "val"):
        paths = config["data"][split]
        data = MapDataset(
            root / paths["metadata"],
            root / paths["basemaps"],
            root / paths["segmentations"],
            image_size=config["data"]["image_size"],
            map_size_m=config["data"]["map_size_m"],
            grid_dim=config["data"]["grid_dim"],
            neighbor_frames=config["model"]["neighbor_frames"],
            neighbor_window_s=config["model"]["neighbor_window_s"],
        )
        first = data[0]
        print(f"{split}: {len(data)} samples; first={first['id']}")


if __name__ == "__main__":
    main()

