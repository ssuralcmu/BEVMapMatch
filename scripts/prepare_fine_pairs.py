#!/usr/bin/env python3
"""Create the paper's normalized 3x3 crops and ground-truth annotations."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from bevmapmatch.data import perturbation_to_pixel


def crop_bounds(width: int, height: int, cell: int, grid_dim: int = 10) -> tuple[int, int, int, int]:
    row, col = divmod(cell, grid_dim)
    row_start, row_end = max(0, row - 1), min(grid_dim - 1, row + 1)
    col_start, col_end = max(0, col - 1), min(grid_dim - 1, col + 1)
    return (
        int(col_start * width / grid_dim), int(row_start * height / grid_dim),
        int((col_end + 1) * width / grid_dim), int((row_end + 1) * height / grid_dim),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_json")
    parser.add_argument("--output", default="outputs/fine_pairs")
    parser.add_argument("--map-size-m", type=float, default=500.0)
    parser.add_argument("--crop-size", type=int, default=300)
    args = parser.parse_args()
    rows = json.loads(Path(args.predictions_json).read_text())["predictions"]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        sample_id = row["id"]
        basemap = Image.open(row["basemap_path"]).convert("RGB")
        width, height = basemap.size
        bounds = crop_bounds(width, height, row["predicted_cell"])
        reference = output / f"{sample_id}_reference.png"
        basemap.crop(bounds).resize((args.crop_size, args.crop_size), Image.Resampling.BILINEAR).save(reference)
        query = output / f"{sample_id}_query.png"
        shutil.copy2(row["segmentation_path"], query)
        metadata = np.load(row["metadata_path"], allow_pickle=True).item()
        gt_x, gt_y = perturbation_to_pixel(metadata["perturbation"], width / 2, height / 2, width / args.map_size_m)
        left, upper, right, lower = bounds
        gt_crop = [(gt_x - left) * args.crop_size / (right - left), (gt_y - upper) * args.crop_size / (lower - upper)]
        annotation = output / f"{sample_id}_annotation.json"
        annotation.write_text(json.dumps({"id": sample_id, "gt_xy_pixels_crop": gt_crop, "crop_bounds": bounds, "predicted_cell": row["predicted_cell"], "ground_truth_cell": row["ground_truth_cell"]}, indent=2))
        manifest.append({"id": sample_id, "reference": str(reference), "query": str(query), "annotation": str(annotation)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(manifest)} fine-alignment pairs in {output}")


if __name__ == "__main__":
    main()

