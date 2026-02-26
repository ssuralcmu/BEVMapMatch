#!/usr/bin/env python3
"""Plot median localization error vs perturbation radius for multiple runs.

This script:
- accepts multiple `distance_errors.json` files,
- filters samples to keep only errors < 5m (configurable),
- buckets perturbation radius in 0.1m bins (configurable),
- plots all runs together using median error only.
"""

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# Global font sizes for all plots
plt.rcParams.update({
    "font.size": 22,          # base font
    "axes.titlesize": 22,     # plot title
    "axes.labelsize": 22,     # x/y labels
    "xtick.labelsize": 20,    # x tick labels
    "ytick.labelsize": 20,    # y tick labels
    "legend.fontsize": 20,    # legend text
    "legend.title_fontsize": 20,
})

CENTER_XY = (250.0, 250.0)


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_error_map(main_json_path: Path, use_fallback: bool = True) -> Dict[str, float]:
    """Load id -> error_m from a distance-errors JSON list."""
    data = json.loads(main_json_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {main_json_path}, got {type(data).__name__}")

    id2err: Dict[str, float] = {}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        sid = str(item.get("id", f"idx_{i}"))
        dm = item.get("distance_m")
        fe = item.get("fallback_error")

        if is_finite_number(dm):
            id2err[sid] = float(dm)
        elif use_fallback and is_finite_number(fe):
            id2err[sid] = float(fe)

    if not id2err:
        raise ValueError(f"No valid errors found in {main_json_path}")
    return id2err


def load_gt_xy_meters(annotation_path: Path) -> Optional[Tuple[float, float]]:
    """Read gt_xy_meters from one annotation file."""
    obj = json.loads(annotation_path.read_text())
    gt = obj.get("gt_xy_meters")
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


def bucket_index(value: float, bin_size: float) -> int:
    return int(math.floor(value / bin_size))


def aggregate_by_bins(
    samples: List[Tuple[float, float]],
    bin_size: float,
    max_x: Optional[float] = None,
) -> List[dict]:
    """Aggregate median error per perturbation-radius bin."""
    if not samples:
        raise ValueError("No (perturb, error) samples to aggregate.")

    if max_x is None:
        max_x = max(p for p, _ in samples)

    bins: Dict[int, List[float]] = {}
    for perturb_m, err_m in samples:
        if perturb_m < 0 or perturb_m > max_x:
            continue
        idx = bucket_index(perturb_m, bin_size)
        bins.setdefault(idx, []).append(err_m)

    max_idx = bucket_index(max_x, bin_size)
    rows: List[dict] = []

    for idx in range(max_idx + 1):
        left = idx * bin_size
        right = (idx + 1) * bin_size
        errs = bins.get(idx, [])

        rows.append(
            {
                "bin_left_m": left,
                "bin_right_m": right,
                "count": len(errs),
                "median_error_m": statistics.median(errs) if errs else None,
            }
        )

    return rows


def save_csv(rows: List[dict], out_path: Path) -> None:
    import csv

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_name(label: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return label or "run"


def try_plot_multi(run_rows: List[Tuple[str, List[dict]]], out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 7))

    # NEW: pick a distinct color per run (tab10 gives up to 10 nice distinct colors)
    cmap = plt.get_cmap("Set2")
    n = max(1, len(run_rows))

    any_line = False
    for k, (run_name, rows) in enumerate(run_rows):
        xs, ys = [], []
        for row in rows:
            if row["count"] > 0 and row["median_error_m"] is not None:
                x_center = (row["bin_left_m"] + row["bin_right_m"]) / 2.0
                xs.append(x_center)
                ys.append(row["median_error_m"])

        if xs:
            any_line = True
            plt.plot(xs, ys, label=run_name, color=cmap(k % 10), linewidth=4.0)  # NEW

    if not any_line:
        print(f"[WARN] No non-empty bins to plot for {out_path.name}.")
        plt.close()
        return

    plt.xlabel("Perturbation radius from center (m)")
    plt.ylabel("Median localization error (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--main_jsons",
        nargs="+",
        required=True,
        help="One or more distance_errors.json paths",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for --main_jsons (same length). Default: each JSON stem",
    )
    ap.add_argument(
        "--ann_dir",
        type=str,
        default="viz_dino_best_3x3crop",
        help="Directory containing *_metas_annotation.json files",
    )
    ap.add_argument("--out_dir", type=str, default="out_perturb_analysis_bevfusion", help="Output directory")
    ap.add_argument("--ignore_fallback", action="store_true", help="Ignore fallback_error if distance_m missing")
    ap.add_argument("--center_x", type=float, default=250.0)
    ap.add_argument("--center_y", type=float, default=250.0)
    ap.add_argument("--max_error_m", type=float, default=1000.0, help="Keep only errors strictly less than this")
    ap.add_argument("--bin_size_m", type=float, default=20, help="Perturbation-radius bin size in meters")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.main_jsons):
        raise ValueError("--labels must have the same number of entries as --main_jsons")
    if args.bin_size_m <= 0:
        raise ValueError("--bin_size_m must be > 0")

    global CENTER_XY
    CENTER_XY = (float(args.center_x), float(args.center_y))

    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_paths = sorted(ann_dir.glob("*_metas_annotation.json"))
    if not ann_paths:
        raise FileNotFoundError(f"No *_metas_annotation.json files found in {ann_dir}")

    collected_runs: List[Tuple[str, List[Tuple[float, float]]]] = []
    global_max_perturb = 0.0

    for i, json_path_str in enumerate(args.main_jsons):
        json_path = Path(json_path_str)
        run_name = args.labels[i] if args.labels else json_path.stem

        id2err = load_error_map(json_path, use_fallback=not args.ignore_fallback)

        samples: List[Tuple[float, float]] = []
        missing_err = 0
        missing_gt = 0
        filtered_over_threshold = 0

        for ann_path in ann_paths:
            sample_id = ann_path.name.split("_metas_annotation.json")[0]
            err = id2err.get(sample_id)
            if err is None:
                missing_err += 1
                continue
            if err >= args.max_error_m:
                filtered_over_threshold += 1
                continue

            gt_xy = load_gt_xy_meters(ann_path)
            if gt_xy is None:
                missing_gt += 1
                continue

            samples.append((perturb_radius(gt_xy, CENTER_XY), err))

        if not samples:
            print(f"[WARN] No usable samples for run '{run_name}' from {json_path}")
            continue

        run_max_perturb = max(p for p, _ in samples)
        global_max_perturb = max(global_max_perturb, run_max_perturb)
        collected_runs.append((run_name, samples))

        print(f"\nRun: {run_name}")
        print(f"  Source JSON: {json_path}")
        print(f"  Matched samples (< {args.max_error_m}m error): {len(samples)}")
        print(f"  Filtered by error threshold: {filtered_over_threshold}")
        print(f"  Missing error for annotation id: {missing_err}")
        print(f"  Missing gt_xy_meters in annotation: {missing_gt}")
        print(f"  Max perturb radius observed: {run_max_perturb:.3f} m")

    if not collected_runs:
        raise ValueError("No runs had usable samples after filtering.")

    all_rows: List[Tuple[str, List[dict]]] = []
    for run_name, samples in collected_runs:
        rows = aggregate_by_bins(samples, bin_size=args.bin_size_m, max_x=global_max_perturb)
        all_rows.append((run_name, rows))
        save_csv(rows, out_dir / f"median_error_vs_perturb_{safe_name(run_name)}.csv")

    try_plot_multi(
        all_rows,
        out_dir / "median_error_vs_perturb_all_runs.png",
        title=(
            "Median localization error vs perturbation radius\n"
            #f"(error < {args.max_error_m}m, bin={args.bin_size_m}m)"
        ),
    )

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print("- median_error_vs_perturb_<label>.csv (one per run)")
    print("- median_error_vs_perturb_all_runs.png")


if __name__ == "__main__":
    main()
