#!/usr/bin/env python3
"""Run MatchAnything/EfficientLoFTR on prepared fine-alignment pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from bevmapmatch.fine import map_query_center


def image(path: str) -> np.ndarray:
    value = cv2.imread(path, cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return value[:, :, ::-1].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--matchanything-root", required=True)
    parser.add_argument("--output", default="outputs/distance_errors.json")
    parser.add_argument("--pixels-per-meter", type=float, default=2.0)
    args = parser.parse_args()

    import sys
    root = Path(args.matchanything_root).resolve()
    sys.path.insert(0, str(root))
    from imcui.ui.utils import get_matcher_zoo, load_config, run_matching

    config_path = root / "config" / "config.yaml"
    config = load_config(config_path)
    zoo = get_matcher_zoo(config["matcher_zoo"])
    rows = json.loads(Path(args.manifest).read_text())
    results, skipped = [], []
    for row in tqdm(rows):
        try:
            reference, query = image(row["reference"]), image(row["query"])
            outputs = run_matching(
                image0=reference, image1=query, match_threshold=0.1,
                extract_max_keypoints=1000, keypoint_threshold=0.015,
                key="matchanything_eloftr", ransac_method=config["defaults"]["ransac_method"],
                ransac_reproj_threshold=8.0, ransac_confidence=config["defaults"]["ransac_confidence"],
                ransac_max_iter=config["defaults"]["ransac_max_iter"], choice_geometry_type="Homography",
                matcher_zoo=zoo, force_resize=False, image_width=640, image_height=480,
                use_cached_model=True,
            )
            state = outputs[-2]
            predicted = map_query_center(np.asarray(state["H"]), query.shape[:2], reference.shape[:2])
            truth = json.loads(Path(row["annotation"]).read_text())["gt_xy_pixels_crop"]
            error_pixels = float(np.linalg.norm(np.asarray(predicted) - np.asarray(truth)))
            results.append({"id": row["id"], "predicted_xy": predicted, "ground_truth_xy": truth, "distance_px": error_pixels, "distance_m": error_pixels / args.pixels_per_meter})
        except Exception as error:
            skipped.append({"id": row["id"], "error": f"{type(error).__name__}: {error}"})
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"results": results, "skipped": skipped}, indent=2))
    print(f"Completed {len(results)} pairs; skipped {len(skipped)}")


if __name__ == "__main__":
    main()

