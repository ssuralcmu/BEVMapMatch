#!/usr/bin/env python3
"""
Plot distance error distributions for multiple distance_errors.json files on one graph.

- Uses 0.5m bins over [0, 10]m.
- Plots normalized histograms as smooth curves (Gaussian-smoothed).
- Handles JSON as:
  1) list of dicts with key `distance_m`
  2) list of numbers

Example:
python scripts_2026_fine/plot_distance_error_distribution_multi.py \
  --json_glob "match_anything_inference/MatchAnything/out_*/distance_errors.json" \
  --out_png scripts_2026_fine/out/distance_error_distribution_0_10m.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Global font sizes for all plots
plt.rcParams.update({
    "font.size": 14,          # base font
    "axes.titlesize": 18,     # plot title
    "axes.labelsize": 16,     # x/y labels
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 16,    # legend text
    "legend.title_fontsize": 16,
})

def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_distance_errors(json_path: Path) -> np.ndarray:
    data = json.loads(json_path.read_text())
    values: List[float] = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                dm = item.get("distance_m", None)
                if is_finite_number(dm):
                    values.append(float(dm))
            elif is_finite_number(item):
                values.append(float(item))

    if not values:
        raise ValueError(f"No valid distance values found in: {json_path}")

    return np.asarray(values, dtype=np.float64)


def gaussian_kernel1d(sigma_bins: float, radius: int | None = None) -> np.ndarray:
    if sigma_bins <= 0:
        return np.asarray([1.0], dtype=np.float64)

    if radius is None:
        radius = int(math.ceil(3.0 * sigma_bins))

    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma_bins * sigma_bins))
    k /= k.sum()
    return k


def smooth_histogram(hist: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return hist
    kernel = gaussian_kernel1d(sigma_bins)
    return np.convolve(hist, kernel, mode="same")


def build_label(path: Path, mode: str = "parent") -> str:
    if mode == "file":
        return path.stem
    if mode == "full":
        return str(path)
    return path.parent.name


def resolve_inputs(json_files: Sequence[str], json_glob: str | None) -> List[Path]:
    paths: List[Path] = [Path(p) for p in json_files]
    if json_glob:
        paths.extend(sorted(Path().glob(json_glob)))

    # de-duplicate while preserving order
    seen = set()
    unique_paths: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)

    existing = [p for p in unique_paths if p.exists()]
    missing = [p for p in unique_paths if not p.exists()]
    for m in missing:
        print(f"[WARN] Missing file, skipping: {m}")

    if not existing:
        raise FileNotFoundError("No valid JSON files found. Provide --json_files and/or --json_glob.")

    return existing


def plot_distributions(
    inputs: Iterable[Path],
    out_png: Path,
    out_csv: Path | None,
    min_m: float,
    max_m: float,
    bin_size_m: float,
    smooth_sigma_bins: float,
    label_mode: str,
    custom_labels: Sequence[str] | None = None,
) -> None:
    bins = np.arange(min_m, max_m + bin_size_m, bin_size_m, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(10, 6))

    csv_rows: List[Tuple[str, float, float, float]] = []

    for idx, p in enumerate(inputs):
        values = load_distance_errors(p)
        in_range = values[(values >= min_m) & (values <= max_m)]
        if in_range.size == 0:
            print(f"[WARN] No values in [{min_m}, {max_m}] for {p}, skipping")
            continue

        hist, _ = np.histogram(in_range, bins=bins, density=True)
        hist_s = smooth_histogram(hist, smooth_sigma_bins)

        if custom_labels is not None and idx < len(custom_labels):
            label = custom_labels[idx]
        else:
            label = build_label(p, mode=label_mode)

        plt.plot(centers, hist_s, linewidth=2, label=f"{label}")

        if out_csv is not None:
            for x, y_raw, y_smooth in zip(centers, hist, hist_s):
                csv_rows.append((label, float(x), float(y_raw), float(y_smooth)))

    plt.title(f"Localization Error Distribution ({min_m:.0f}–{max_m:.0f} m)")
    plt.xlabel("Localization Error (m)")
    plt.ylabel("Density")
    plt.xlim(min_m, max_m)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved plot: {out_png}")

    if out_csv is not None and csv_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w") as f:
            f.write("series,bin_center_m,density_raw,density_smoothed\n")
            for row in csv_rows:
                f.write(f"{row[0]},{row[1]:.3f},{row[2]:.10f},{row[3]:.10f}\n")
        print(f"Saved curves CSV: {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json_files",
        nargs="*",
        default=[],
        help="Explicit distance_errors.json paths",
    )
    ap.add_argument(
        "--json_glob",
        type=str,
        default=None,
        help="Glob pattern for distance_errors.json files",
    )
    ap.add_argument(
        "--out_png",
        type=str,
        default="scripts_2026_fine/out/distance_error_distribution_0_5m.png",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="scripts_2026_fine/out/distance_error_distribution_0_5m.csv",
        help="Optional CSV output with per-bin raw/smoothed densities",
    )
    ap.add_argument("--min_m", type=float, default=0.0)
    ap.add_argument("--max_m", type=float, default=5.0)
    ap.add_argument("--bin_size_m", type=float, default=0.05)
    ap.add_argument(
        "--smooth_sigma_bins",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma in units of bins (0 disables smoothing)",
    )
    ap.add_argument(
        "--label_mode",
        choices=["parent", "file", "full"],
        default="parent",
        help="How to label each curve",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Custom labels for curves (must match the number/order of resolved input files)",
    )

    args = ap.parse_args()

    inputs = resolve_inputs(args.json_files, args.json_glob)

    if args.labels is not None and len(args.labels) != len(inputs):
        raise ValueError(
            f"--labels has {len(args.labels)} entries, but resolved {len(inputs)} input files. "
            "Provide exactly one label per input file (in the same order)."
        )
    out_csv = Path(args.out_csv) if args.out_csv else None

    plot_distributions(
        inputs=inputs,
        out_png=Path(args.out_png),
        out_csv=out_csv,
        min_m=float(args.min_m),
        max_m=float(args.max_m),
        bin_size_m=float(args.bin_size_m),
        smooth_sigma_bins=float(args.smooth_sigma_bins),
        label_mode=args.label_mode,
        custom_labels=args.labels,
    )


if __name__ == "__main__":
    main()
