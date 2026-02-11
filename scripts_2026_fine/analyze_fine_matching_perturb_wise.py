#!/usr/bin/env python3
"""
Compute localization error vs. perturbation-from-center (250,250) using:
1) Main results JSON: list of dicts with fields: id, distance_m (primary error), fallback_error (optional)
2) Per-sample annotation JSON files:
   scripts_2026_fine/viz_dino_best_3x3crop/<ID>_metas_annotation.json
   containing: "gt_xy_meters": [x, y]   where x,y in [0,500] (perturbed coords)

We compute:
- perturb_radius_m = sqrt((x-250)^2 + (y-250)^2)
- bucket perturb_radius_m into:
  A) 1m bins for [0, 50]
  B) 5m bins for [0, max]  (max computed from your data)

Then we aggregate localization errors (distance_m) by perturb bin:
- count
- mean error
- median error
- P90 error

Outputs:
- CSVs with per-bin aggregates
- Two plots (if matplotlib works): median error vs perturb bin (and optional mean/P90)
  * Plot 1: 1m bins up to 50m
  * Plot 2: 5m bins up to max

NOTE: This code avoids numpy/pandas. Plotting uses matplotlib only (guarded).
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional


CENTER_XY = (250.0, 250.0)


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_error_map(main_json_path: Path, use_fallback: bool = True) -> Dict[str, float]:
    """
    Returns dict: id -> error_m
    Prefers distance_m; optionally uses fallback_error if distance_m missing/non-finite.
    """
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
        # else: skip (no usable error)

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


def perturb_radius(xy: Tuple[float, float], center_xy: Tuple[float, float] = CENTER_XY) -> float:
    dx = xy[0] - center_xy[0]
    dy = xy[1] - center_xy[1]
    return math.sqrt(dx * dx + dy * dy)


def percentile(sorted_vals: List[float], p: float) -> float:
    """
    Linear-interpolated percentile (like numpy default).
    sorted_vals must be sorted.
    """
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("Empty list")
    if n == 1:
        return sorted_vals[0]
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]

    r = (p / 100.0) * (n - 1)
    lo = int(math.floor(r))
    hi = int(math.ceil(r))
    if lo == hi:
        return sorted_vals[lo]
    w = r - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def bucket_index(value: float, bin_size: float) -> int:
    """
    Bucket by left-closed bins:
      idx = floor(value / bin_size)
    So bin idx corresponds to [idx*bin, (idx+1)*bin)
    """
    return int(math.floor(value / bin_size))


def aggregate_by_bins(
    samples: List[Tuple[float, float]],
    bin_size: float,
    max_x: Optional[float] = None,
) -> List[dict]:
    """
    samples: list of (perturb_radius_m, error_m)
    bin_size: size in meters for x-bins
    max_x: if set, only keep samples with perturb <= max_x and only output bins up to max_x.
           If None, use max perturb in samples.

    Returns list of dict rows:
      bin_left, bin_right, count, mean_err, median_err, p90_err
    """
    if not samples:
        raise ValueError("No (perturb, error) samples to aggregate.")

    if max_x is None:
        max_x = max(p for p, _ in samples)

    # Collect errors per bin
    bins: Dict[int, List[float]] = {}
    for p, e in samples:
        if p < 0:
            continue
        if p > max_x:
            continue
        idx = bucket_index(p, bin_size)
        bins.setdefault(idx, []).append(e)

    # Produce ordered rows for all bins from 0..max_idx (including empty bins)
    max_idx = bucket_index(max_x, bin_size)
    rows: List[dict] = []

    for idx in range(0, max_idx + 1):
        left = idx * bin_size
        right = (idx + 1) * bin_size
        errs = bins.get(idx, [])
        if errs:
            errs_sorted = sorted(errs)
            mean_e = sum(errs_sorted) / len(errs_sorted)
            med_e = statistics.median(errs_sorted)
            p90_e = percentile(errs_sorted, 90.0)
            rows.append(
                {
                    "bin_left_m": left,
                    "bin_right_m": right,
                    "count": len(errs_sorted),
                    "mean_error_m": mean_e,
                    "median_error_m": med_e,
                    "p90_error_m": p90_e,
                }
            )
        else:
            rows.append(
                {
                    "bin_left_m": left,
                    "bin_right_m": right,
                    "count": 0,
                    "mean_error_m": None,
                    "median_error_m": None,
                    "p90_error_m": None,
                }
            )

    return rows


def save_csv(rows: List[dict], out_path: Path) -> None:
    import csv

    if not rows:
        raise ValueError("No rows to save.")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def try_plot(rows: List[dict], out_path: Path, title: str) -> None:
    """
    Plots median error vs bin center (with optional mean and p90).
    Requires matplotlib. If matplotlib fails (e.g., numpy broken), we skip plotting.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as ex:
        print(f"[WARN] Matplotlib not available; skipping plot {out_path.name}. Reason: {ex}")
        return

    xs = []
    med = []
    mean = []
    p90 = []

    for r in rows:
        if r["count"] and r["median_error_m"] is not None:
            x_center = (r["bin_left_m"] + r["bin_right_m"]) / 2.0
            xs.append(x_center)
            med.append(r["median_error_m"])
            mean.append(r["mean_error_m"])
            p90.append(r["p90_error_m"])

    if not xs:
        print(f"[WARN] No non-empty bins to plot for {out_path.name}.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(xs, med, label="Median error")
    plt.plot(xs, mean, label="Mean error")
    plt.plot(xs, p90, label="P90 error")

    plt.xlabel("Perturbation radius from center (m)")
    plt.ylabel("Localization error (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_json", type=str, help="Path to main results JSON containing id + distance_m", default="/home/rtml/shounak_research/bevfusion/match_anything_inference/MatchAnything/out_matchanything_all_evaluate_modelpred/distance_errors.json")
    ap.add_argument(
        "--ann_dir",
        type=str,
        default="viz_dino_best_3x3crop",
        help="Folder containing *_metas_annotation.json files",
    )
    ap.add_argument("--out_dir", type=str, default="out_perturb_analysis_modelpred", help="Output directory")
    ap.add_argument("--ignore_fallback", action="store_true", help="Ignore fallback_error in main JSON")
    ap.add_argument("--center_x", type=float, default=250.0)
    ap.add_argument("--center_y", type=float, default=250.0)
    args = ap.parse_args()

    global CENTER_XY
    CENTER_XY = (float(args.center_x), float(args.center_y))

    main_json = Path(args.main_json)
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2err = load_error_map(main_json, use_fallback=not args.ignore_fallback)

    ann_paths = sorted(ann_dir.glob("*_metas_annotation.json"))
    if not ann_paths:
        raise FileNotFoundError(f"No *_metas_annotation.json files found in {ann_dir}")

    samples: List[Tuple[float, float]] = []
    missing_err = 0
    missing_gt = 0

    for p in ann_paths:
        # id is everything before "_metas_annotation.json"
        sid = p.name.split("_metas_annotation.json")[0]

        err = id2err.get(sid, None)
        if err is None:
            missing_err += 1
            continue

        gt_xy = load_gt_xy_meters(p)
        if gt_xy is None:
            missing_gt += 1
            continue

        pr = perturb_radius(gt_xy, CENTER_XY)
        samples.append((pr, err))

    if not samples:
        raise ValueError(
            "No matched samples. Check that annotation filenames match ids in main JSON and gt_xy_meters exists."
        )

    max_perturb = max(pr for pr, _ in samples)
    print(f"Matched samples: {len(samples)}")
    print(f"Missing error for annotation id: {missing_err}")
    print(f"Missing gt_xy_meters in annotation: {missing_gt}")
    print(f"Max perturb radius observed: {max_perturb:.3f} m")

    # --- Aggregation 1: 1m bins up to 50m ---
    rows_1m_50 = aggregate_by_bins(samples, bin_size=1.0, max_x=50.0)
    save_csv(rows_1m_50, out_dir / "error_vs_perturb_1m_bins_upto_50m.csv")
    try_plot(
        rows_1m_50,
        out_dir / "error_vs_perturb_1m_bins_upto_50m.png",
        title="Localization error vs perturbation radius (1m bins, 0–50m)",
    )

    # --- Aggregation 2: 5m bins up to max ---
    rows_5m_max = aggregate_by_bins(samples, bin_size=5.0, max_x=max_perturb)
    save_csv(rows_5m_max, out_dir / "error_vs_perturb_5m_bins_upto_max.csv")
    try_plot(
        rows_5m_max,
        out_dir / "error_vs_perturb_5m_bins_upto_max.png",
        title="Localization error vs perturbation radius (5m bins, 0–max)",
    )

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print("- error_vs_perturb_1m_bins_upto_50m.csv (+ .png if matplotlib works)")
    print("- error_vs_perturb_5m_bins_upto_max.csv (+ .png if matplotlib works)")


if __name__ == "__main__":
    main()
