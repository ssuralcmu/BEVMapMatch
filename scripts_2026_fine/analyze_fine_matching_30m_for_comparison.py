#!/usr/bin/env python3
"""
Filter samples to only those whose perturbation radius from (250,250) is <= 30*sqrt(2),
then compute threshold success/failure metrics on localization error.

Inputs:
- main results JSON: list of dicts with fields: id, distance_m (primary error), fallback_error (optional)
- annotation JSONs in a folder:
  scripts_2026_fine/viz_dino_best_3x3crop/<ID>_metas_annotation.json
  containing: "gt_xy_meters": [x, y] where x,y in [0,500]

Perturb radius:
  r = sqrt((x-250)^2 + (y-250)^2)

Filter:
  r <= 30*sqrt(2)

Outputs:
- Prints threshold metrics for error_m at thresholds: 1,2,5,10,20,30 meters
- Saves:
  - filtered_ids.json  (list of kept ids)
  - threshold_metrics_filtered.csv
"""

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_error_map(main_json_path: Path, use_fallback: bool = True) -> Dict[str, float]:
    data = json.loads(main_json_path.read_text())
    id2err: Dict[str, float] = {}
    for i, item in enumerate(data):
        sid = item.get("id", f"idx_{i}")
        dm = item.get("distance_m", None)
        fe = item.get("fallback_error", None)

        if is_finite_number(dm):
            id2err[sid] = float(dm)
        elif use_fallback and is_finite_number(fe):
            id2err[sid] = float(fe)
    if not id2err:
        raise ValueError("No valid errors found in main JSON (distance_m/fallback_error).")
    return id2err


def load_gt_xy_meters(annotation_path: Path) -> Optional[Tuple[float, float]]:
    obj = json.loads(annotation_path.read_text())
    gt = obj.get("gt_xy_meters", None)
    if (
        isinstance(gt, (list, tuple))
        and len(gt) == 2
        and is_finite_number(gt[0])
        and is_finite_number(gt[1])
    ):
        return float(gt[0]), float(gt[1])
    return None


def perturb_radius(xy: Tuple[float, float], center_xy: Tuple[float, float]) -> float:
    dx = xy[0] - center_xy[0]
    dy = xy[1] - center_xy[1]
    return math.sqrt(dx * dx + dy * dy)


def compute_threshold_metrics(sorted_errors: List[float], thresholds: List[float]) -> List[dict]:
    n = len(sorted_errors)
    rows: List[dict] = []
    for t in thresholds:
        k = bisect.bisect_right(sorted_errors, t)  # count <= t
        success = k / n if n else 0.0
        failure = 1.0 - success if n else 1.0
        rows.append(
            {
                "threshold_m": float(t),
                "success_rate": success,
                "failure_rate": failure,
                "success_count": int(k),
                "total": int(n),
            }
        )
    return rows


def save_csv(rows: List[dict], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to save.")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_json", type=str, help="Path to main results JSON containing id + distance_m", default="/home/rtml/shounak_research/bevfusion/match_anything_inference/MatchAnything/out_matchanything_all_evaluate_modelpred/distance_errors.json")
    ap.add_argument(
        "--ann_dir",
        type=str,
        default="viz_dino_best_3x3crop",
        help="Folder containing *_metas_annotation.json files",
    )
    ap.add_argument("--out_dir", type=str, default="out_perturb30sqrt2_modelpred", help="Output directory")
    ap.add_argument("--ignore_fallback", action="store_true", help="Ignore fallback_error in main JSON")
    ap.add_argument("--center_x", type=float, default=250.0)
    ap.add_argument("--center_y", type=float, default=250.0)
    args = ap.parse_args()

    center_xy = (float(args.center_x), float(args.center_y))
    perturb_thresh = 30.0 * math.sqrt(2.0)

    main_json = Path(args.main_json)
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2err = load_error_map(main_json, use_fallback=not args.ignore_fallback)

    ann_paths = sorted(ann_dir.glob("*_metas_annotation.json"))
    if not ann_paths:
        raise FileNotFoundError(f"No *_metas_annotation.json files found in {ann_dir}")

    kept_ids: List[str] = []
    kept_errors: List[float] = []

    missing_err = 0
    missing_gt = 0
    over_thresh = 0

    for p in ann_paths:
        sid = p.name.split("_metas_annotation.json")[0]

        err = id2err.get(sid, None)
        if err is None:
            missing_err += 1
            continue

        gt_xy = load_gt_xy_meters(p)
        if gt_xy is None:
            missing_gt += 1
            continue

        pr = perturb_radius(gt_xy, center_xy)
        if pr <= perturb_thresh:
            kept_ids.append(sid)
            kept_errors.append(err)
        else:
            over_thresh += 1

    kept_errors.sort()

    print(f"Perturb filter: r <= 30*sqrt(2) = {perturb_thresh:.6f} m")
    print(f"Matched & kept: {len(kept_errors)} samples")
    print(f"Dropped (perturb > thresh): {over_thresh}")
    print(f"Missing error for annotation id: {missing_err}")
    print(f"Missing gt_xy_meters in annotation: {missing_gt}")
    if kept_errors:
        print(f"Max kept perturb radius: {perturb_thresh:.3f} m (threshold)")
        print(f"Max kept error: {max(kept_errors):.3f} m")
        print()

    thresholds = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    rows = compute_threshold_metrics(kept_errors, thresholds)

    # Print pretty
    print("Threshold metrics (filtered):")
    print(" threshold_m  success_rate  failure_rate  success_count  total")
    for r in rows:
        print(
            f" {r['threshold_m']:>9.1f}  "
            f"{(100*r['success_rate']):>10.2f}%  "
            f"{(100*r['failure_rate']):>10.2f}%  "
            f"{r['success_count']:>13d}  "
            f"{r['total']:>5d}"
        )

    # Save outputs
    (out_dir / "filtered_ids.json").write_text(json.dumps(kept_ids, indent=2))
    save_csv(rows, out_dir / "threshold_metrics_filtered.csv")

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print("- filtered_ids.json")
    print("- threshold_metrics_filtered.csv")


if __name__ == "__main__":
    main()
