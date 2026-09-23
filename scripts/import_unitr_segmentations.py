#!/usr/bin/env python3
"""Install UniTR BEV predictions under BEVMapMatch sample filenames."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitr-output", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("train", "val"), required=True)
    args = parser.parse_args()

    source = args.unitr_output / "vis_bevfusion"
    metadata_dir = args.processed_root / args.split / "metadata"
    output_dir = args.processed_root / args.split / "segmentations"
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: list[str] = []
    for metadata_path in sorted(metadata_dir.glob("*_metas.npy")):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        sample_token = metadata["sample_token"]
        prediction = source / f"{sample_token}.png"
        if not prediction.is_file():
            missing.append(sample_token)
            continue
        destination = output_dir / metadata_path.name.replace(
            "_metas.npy", "_generated_map_image.png"
        )
        shutil.copy2(prediction, destination)
        copied += 1

    print(f"Imported {copied} UniTR predictions into {output_dir}")
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"Missing {len(missing)} predictions; first tokens: {preview}")


if __name__ == "__main__":
    main()
