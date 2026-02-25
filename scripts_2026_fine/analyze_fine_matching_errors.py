
"""
Compute localization error metrics + plots from a JSON list of dicts like:
[
  {"id": "...", "distance_m": 1.23, "fallback_error": null, ...},
  ...
]

Outputs:
- CDF plot (optionally with tail inset)
- Percentile table (P50/P75/P90/P95/P99)
- Failure-rate / success-rate metrics at thresholds (1m, 2m, 5m, 10m)

Usage:
  python eval_localization_errors.py --json_path results.json --out_dir out

Notes:
- By default, uses distance_m if present/finite.
- If distance_m is missing but fallback_error is present/finite, uses fallback_error.
- You can disable fallback usage with --ignore_fallback.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def load_errors_m(json_path: Path, use_fallback: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: id, error_m, used_field
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for i, item in enumerate(data):
        sid = item.get("id", f"idx_{i}")

        dm = item.get("distance_m", None)
        fe = item.get("fallback_error", None)

        if _is_finite_number(dm):
            rows.append({"id": sid, "error_m": float(dm), "used_field": "distance_m"})
        elif use_fallback and _is_finite_number(fe):
            rows.append({"id": sid, "error_m": float(fe), "used_field": "fallback_error"})
        else:
            # Skip if no usable error
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "No valid error values found. Check that 'distance_m' (or 'fallback_error') exists and is numeric."
        )
    return df


def compute_percentiles(errors: np.ndarray, percentiles=(50, 75, 90, 95, 99)) -> pd.DataFrame:
    vals = np.percentile(errors, list(percentiles))
    out = pd.DataFrame({"percentile": [f"P{p}" for p in percentiles], "error_m": vals})
    return out

def compute_cdf_auc(errors: np.ndarray, T: float = 5.0) -> float:
    """
    Normalized area under the empirical CDF on [0, T].
    Returns a score in [0, 1], where higher is better.

    Interpretation:
      - 1.0 means all errors are ~0 (CDF rises immediately)
      - lower values mean errors are spread further right within [0, T]
      - errors > T are clipped at T (so they contribute poorly to early CDF rise)
    """
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return float("nan")
    if T <= 0:
        raise ValueError("T must be > 0")

    # Clip to [0, T] so catastrophic errors don't dominate this "good-case" metric
    x = np.clip(errors, 0.0, T)
    x.sort()
    n = x.size

    # Build a right-continuous empirical CDF over [0, T]
    # x_points:   0, x1, x2, ..., xn, T
    # cdf_vals:   0, 1/n, 2/n, ..., 1, 1
    x_points = np.concatenate(([0.0], x, [T]))
    cdf_vals = np.concatenate(([0.0], np.arange(1, n + 1) / n, [1.0]))

    auc = np.trapz(cdf_vals, x_points)   # area under CDF from 0..T
    return float(auc / T)                # normalize to [0, 1]

def compute_threshold_metrics(errors: np.ndarray, thresholds=(1, 2, 5, 10)) -> pd.DataFrame:
    """
    Returns success and failure rates at thresholds:
      success@t = P(error <= t)
      failure@t = P(error > t)
    """
    n = len(errors)
    rows = []
    for t in thresholds:
        succ = float(np.sum(errors <= t)) / n
        fail = 1.0 - succ
        rows.append(
            {
                "threshold_m": float(t),
                "success_rate": succ,
                "failure_rate": fail,
                "success_count": int(np.sum(errors <= t)),
                "total": int(n),
            }
        )
    return pd.DataFrame(rows)


def plot_cdf(errors: np.ndarray, out_path: Path, xmax_main: float = 50.0, tail_xlim=(50.0, 500.0)):
    """
    Saves a CDF plot up to xmax_main, with an optional tail inset showing tail_xlim.
    """
    errors = np.asarray(errors, dtype=np.float64)
    errors_sorted = np.sort(errors)
    n = len(errors_sorted)
    cdf = np.arange(1, n + 1) / n

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Main CDF (clipped view)
    ax.plot(errors_sorted, cdf)
    ax.set_xlim(0, xmax_main)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Localization error (m)")
    ax.set_ylabel("CDF  P(error ≤ x)")
    ax.set_title(f"CDF of Localization Error (0–{xmax_main:g} m view)")
    ax.grid(True, alpha=0.3)

    # Tail inset (only if there are points beyond tail start)
    tail_start = tail_xlim[0]
    if np.max(errors_sorted) > tail_start:
        # Inset axes [left, bottom, width, height] in figure fraction
        inset = fig.add_axes([0.55, 0.20, 0.38, 0.35])
        inset.plot(errors_sorted, cdf)
        inset.set_xlim(tail_xlim[0], tail_xlim[1])
        inset.set_ylim(0.95, 1.0)  # focus on tail; tweak if needed
        inset.set_title("Tail zoom", fontsize=10)
        inset.set_xlabel("m", fontsize=9)
        inset.set_ylabel("CDF", fontsize=9)
        inset.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ccdf_tail(errors: np.ndarray, out_path: Path, xlim=(0.0, 500.0)):
    """
    CCDF plot: P(error > x) vs x. Great for visualizing rare catastrophic failures.
    """
    errors = np.asarray(errors, dtype=np.float64)
    errors_sorted = np.sort(errors)
    n = len(errors_sorted)
    cdf = np.arange(1, n + 1) / n
    ccdf = 1.0 - cdf

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(errors_sorted, ccdf)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel("Localization error (m)")
    ax.set_ylabel("CCDF  P(error > x)")
    ax.set_title("Tail View (CCDF)")
    ax.grid(True, alpha=0.3)

    # Optional: log-scale y helps when tail probabilities are tiny
    # ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, help="Path to JSON results file", default="../match_anything_inference/MatchAnything/out_matchanything_all_evaluate/distance_errors.json")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--ignore_fallback", action="store_true", default=True, help="Ignore fallback_error even if present")
    parser.add_argument("--cdf_xmax", type=float, default=50.0, help="Max x for main CDF view")
    parser.add_argument("--tail_xmax", type=float, default=500.0, help="Max x for tail view")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_errors_m(json_path, use_fallback=not args.ignore_fallback)
    errors = df["error_m"].to_numpy()

    # Basic summary
    n = len(errors)
    max_err = float(np.max(errors))
    med = float(np.median(errors))
    print(f"Loaded {n} samples")
    print(f"Median error: {med:.3f} m | Max error: {max_err:.3f} m")
    print()

    # Percentile table
    pct_df = compute_percentiles(errors, percentiles=(50, 75, 90, 95, 99))
    print("Percentiles:")
    print(pct_df.to_string(index=False, justify="left", formatters={"error_m": "{:.3f}".format}))
    print()

    # Threshold metrics (1,2,5,10m)
    thr_df = compute_threshold_metrics(errors, thresholds=(1, 2, 5, 10, 20, 50, 100, 166))
    print("Threshold metrics:")
    pretty_thr = thr_df.copy()
    pretty_thr["success_rate"] = pretty_thr["success_rate"].map(lambda x: f"{100*x:.2f}%")
    pretty_thr["failure_rate"] = pretty_thr["failure_rate"].map(lambda x: f"{100*x:.2f}%")
    print(pretty_thr.to_string(index=False, justify="left"))
    print()

    # CDF-AUC scores (good-case concentration metrics)
    cdf_auc_1 = compute_cdf_auc(errors, T=1.0)
    cdf_auc_2 = compute_cdf_auc(errors, T=2.0)
    cdf_auc_5 = compute_cdf_auc(errors, T=5.0)

    print("CDF-AUC (normalized, higher is better):")
    print(f"  AUC@1m: {cdf_auc_1:.4f}")
    print(f"  AUC@2m: {cdf_auc_2:.4f}")
    print(f"  AUC@5m: {cdf_auc_5:.4f}")
    print()

    # Save CSVs
    df.to_csv(out_dir / "errors_clean.csv", index=False)
    pct_df.to_csv(out_dir / "percentiles.csv", index=False)
    thr_df.to_csv(out_dir / "threshold_metrics.csv", index=False)

    # Plots
    plot_cdf(
        errors,
        out_dir / "cdf_0_to_xmax.png",
        xmax_main=args.cdf_xmax,
        tail_xlim=(args.cdf_xmax, args.tail_xmax),
    )
    plot_ccdf_tail(errors, out_dir / "ccdf_0_to_tailmax.png", xlim=(0.0, args.tail_xmax))

    print(f"Saved outputs to: {out_dir.resolve()}")
    print(f"- {out_dir / 'cdf_0_to_xmax.png'}")
    print(f"- {out_dir / 'ccdf_0_to_tailmax.png'}")
    print(f"- {out_dir / 'percentiles.csv'} / .tex")
    print(f"- {out_dir / 'threshold_metrics.csv'} / .tex")


if __name__ == "__main__":
    main()
