"""
Compute localization error metrics + plots from a JSON list of dicts like:
[
  {"id": "...", "distance_m": 1.23, "fallback_error": null, ...},
  ...
]

Outputs:
- CDF plot (optionally with tail inset)
- Percentile table (P50/P75/P90/P95/P99)
- Threshold metrics (success/failure) for:
  (a) absolute thresholds (e.g., 1m,2m,5m,10m)
  (b) perturbation-normalized equivalent thresholds (e.g., MapLocNet->ours)
  (c) fractional thresholds of perturbation bound (optional)

Usage:
  python eval_localization_errors.py --json_path results.json --out_dir out

Example for your setup:
  python eval_localization_errors.py \
      --json_path distance_errors.json \
      --out_dir out_eval \
      --perturb_bound_m 200 \
      --map_side_m 500 \
      --ref_perturb_bound_m 30 \
      --ref_thresholds_m 1 2 5 10

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
    """
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return float("nan")
    if T <= 0:
        raise ValueError("T must be > 0")

    x = np.clip(errors, 0.0, T)
    x.sort()
    n = x.size

    x_points = np.concatenate(([0.0], x, [T]))
    cdf_vals = np.concatenate(([0.0], np.arange(1, n + 1) / n, [1.0]))

    auc = np.trapz(cdf_vals, x_points)
    return float(auc / T)


def compute_threshold_metrics(errors: np.ndarray, thresholds, label: str = "absolute") -> pd.DataFrame:
    """
    Returns success and failure rates at thresholds:
      success@t = P(error <= t)
      failure@t = P(error > t)
    """
    errors = np.asarray(errors, dtype=np.float64)
    n = len(errors)
    rows = []
    for t in thresholds:
        t = float(t)
        succ_count = int(np.sum(errors <= t))
        succ = float(succ_count) / n
        fail = 1.0 - succ
        rows.append(
            {
                "metric_group": label,
                "threshold_m": t,
                "success_rate": succ,
                "failure_rate": fail,
                "success_count": succ_count,
                "total": int(n),
            }
        )
    return pd.DataFrame(rows)


def compute_fractional_threshold_metrics(errors: np.ndarray, perturb_bound_m: float, fractions, label="perturb_fraction"):
    """
    Fractions are relative to perturbation bound (e.g., 0.0333 means 3.33% of bound).
    Converts to meter thresholds and computes success/failure rates.
    """
    thresholds_m = [float(f) * float(perturb_bound_m) for f in fractions]
    df = compute_threshold_metrics(errors, thresholds_m, label=label)
    df["fraction_of_perturb_bound"] = list(map(float, fractions))
    # Reorder columns for readability
    cols = [
        "metric_group",
        "fraction_of_perturb_bound",
        "threshold_m",
        "success_rate",
        "failure_rate",
        "success_count",
        "total",
    ]
    return df[cols]


def make_equivalent_thresholds(ref_thresholds_m, ref_perturb_bound_m: float, target_perturb_bound_m: float):
    """
    Scale reference thresholds by perturbation-bound ratio.
    Example: ref ±30m to target ±200m => scale factor 200/30.
    """
    if ref_perturb_bound_m <= 0 or target_perturb_bound_m <= 0:
        raise ValueError("Perturbation bounds must be > 0")

    scale = float(target_perturb_bound_m) / float(ref_perturb_bound_m)
    eq = [float(t) * scale for t in ref_thresholds_m]
    return eq, scale


def add_normalized_error_columns(df: pd.DataFrame, perturb_bound_m: float = None, map_side_m: float = None) -> pd.DataFrame:
    df = df.copy()
    if perturb_bound_m is not None and perturb_bound_m > 0:
        df["error_over_perturb_bound"] = df["error_m"] / float(perturb_bound_m)
    if map_side_m is not None and map_side_m > 0:
        df["error_over_map_side"] = df["error_m"] / float(map_side_m)
    return df


def plot_cdf(errors: np.ndarray, out_path: Path, xmax_main: float = 50.0, tail_xlim=(50.0, 500.0)):
    errors = np.asarray(errors, dtype=np.float64)
    errors_sorted = np.sort(errors)
    n = len(errors_sorted)
    cdf = np.arange(1, n + 1) / n

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(errors_sorted, cdf)
    ax.set_xlim(0, xmax_main)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Localization error (m)")
    ax.set_ylabel("CDF  P(error ≤ x)")
    ax.set_title(f"CDF of Localization Error (0–{xmax_main:g} m view)")
    ax.grid(True, alpha=0.3)

    tail_start = tail_xlim[0]
    if np.max(errors_sorted) > tail_start:
        inset = fig.add_axes([0.55, 0.20, 0.38, 0.35])
        inset.plot(errors_sorted, cdf)
        inset.set_xlim(tail_xlim[0], tail_xlim[1])
        inset.set_ylim(0.95, 1.0)
        inset.set_title("Tail zoom", fontsize=10)
        inset.set_xlabel("m", fontsize=9)
        inset.set_ylabel("CDF", fontsize=9)
        inset.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ccdf_tail(errors: np.ndarray, out_path: Path, xlim=(0.0, 500.0)):
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_latex_table(df: pd.DataFrame, out_path: Path, float_cols=None):
    """
    Saves a basic LaTeX table for quick Overleaf copy.
    """
    tab = df.copy()
    if float_cols is None:
        float_cols = []
    for c in float_cols:
        if c in tab.columns:
            tab[c] = tab[c].map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x)
    latex = tab.to_latex(index=False, escape=False)
    out_path.write_text(latex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        default="../match_anything_inference/MatchAnything/out_matchanything_all_evaluate_modelpred_rerun_unitr_fixed/distance_errors.json",
        help="Path to JSON results file",
    )
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")

    # Fallback behavior
    parser.add_argument(
        "--ignore_fallback",
        action="store_true",
        help="If set, ignore fallback_error even if present",
    )

    # Plot ranges
    parser.add_argument("--cdf_xmax", type=float, default=50.0, help="Max x for main CDF view")
    parser.add_argument("--tail_xmax", type=float, default=500.0, help="Max x for tail view")

    # Absolute thresholds
    parser.add_argument(
        "--abs_thresholds_m",
        type=float,
        nargs="+",
        default=[1, 2, 5, 10],
        help="Absolute threshold metrics to report (meters)",
    )

    # Your setup (for normalized metrics)
    parser.add_argument(
        "--perturb_bound_m",
        type=float,
        default=200.0,
        help="Your translation perturbation bound in meters (e.g., 200 for ±200m)",
    )
    parser.add_argument(
        "--map_side_m",
        type=float,
        default=500.0,
        help="Your map side length in meters (e.g., 500 for 500x500m map)",
    )

    # Reference setup (e.g., MapLocNet)
    parser.add_argument(
        "--ref_perturb_bound_m",
        type=float,
        default=30.0,
        help="Reference method perturbation bound in meters (e.g., 30 for ±30m)",
    )
    parser.add_argument(
        "--ref_thresholds_m",
        type=float,
        nargs="+",
        default=[1, 2, 5, 10],
        help="Reference thresholds to map into your perturbation setting",
    )

    # Fractions to report directly (optional but useful)
    parser.add_argument(
        "--fraction_thresholds",
        type=float,
        nargs="+",
        default=[1/30, 2/30, 5/30, 10/30],  # matches MapLocNet 1,2,5,10 w.r.t ±30m
        help="Fractions of perturbation bound for normalized recall (e.g., 0.0333 0.0667 ...)",
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and enrich
    df = load_errors_m(json_path, use_fallback=not args.ignore_fallback)
    df = add_normalized_error_columns(
        df,
        perturb_bound_m=args.perturb_bound_m,
        map_side_m=args.map_side_m,
    )
    errors = df["error_m"].to_numpy(dtype=np.float64)

    # Basic summary
    n = len(errors)
    max_err = float(np.max(errors))
    med = float(np.median(errors))
    mean_err = float(np.mean(errors))
    print(f"Loaded {n} samples")
    print(f"Mean error:   {mean_err:.3f} m")
    print(f"Median error: {med:.3f} m")
    print(f"Max error:    {max_err:.3f} m")
    print()

    # Setup summary
    map_area = float(args.map_side_m) * float(args.map_side_m)
    ref_map_area = None  # unknown unless you want to add a CLI arg
    print("Evaluation settings:")
    print(f"  Your perturbation bound: ±{args.perturb_bound_m:.1f} m")
    print(f"  Your map size: {args.map_side_m:.1f} x {args.map_side_m:.1f} m  (area={map_area:.1f} m^2)")
    print(f"  Reference perturbation bound: ±{args.ref_perturb_bound_m:.1f} m")
    print()

    # Percentiles (absolute)
    pct_df = compute_percentiles(errors, percentiles=(50, 75, 90, 95, 99))
    print("Percentiles (absolute error in meters):")
    print(pct_df.to_string(index=False, justify="left", formatters={"error_m": "{:.3f}".format}))
    print()

    # Percentiles (normalized by perturbation/map)
    norm_pct_rows = []
    if "error_over_perturb_bound" in df.columns:
        vals = np.percentile(df["error_over_perturb_bound"].to_numpy(), [50, 75, 90, 95, 99])
        for p, v in zip([50, 75, 90, 95, 99], vals):
            norm_pct_rows.append({"percentile": f"P{p}", "metric": "error/perturb_bound", "value": float(v)})
    if "error_over_map_side" in df.columns:
        vals = np.percentile(df["error_over_map_side"].to_numpy(), [50, 75, 90, 95, 99])
        for p, v in zip([50, 75, 90, 95, 99], vals):
            norm_pct_rows.append({"percentile": f"P{p}", "metric": "error/map_side", "value": float(v)})
    norm_pct_df = pd.DataFrame(norm_pct_rows)
    if not norm_pct_df.empty:
        print("Percentiles (normalized):")
        print(norm_pct_df.to_string(index=False, justify="left", formatters={"value": "{:.4f}".format}))
        print()

    # A) Absolute threshold metrics
    abs_thr_df = compute_threshold_metrics(errors, thresholds=args.abs_thresholds_m, label="absolute")
    print("Absolute threshold metrics:")
    pretty_abs = abs_thr_df.copy()
    pretty_abs["success_rate"] = pretty_abs["success_rate"].map(lambda x: f"{100*x:.2f}%")
    pretty_abs["failure_rate"] = pretty_abs["failure_rate"].map(lambda x: f"{100*x:.2f}%")
    print(pretty_abs.to_string(index=False, justify="left"))
    print()

    # B) Reference-equivalent thresholds via perturbation scaling
    eq_thresholds, scale = make_equivalent_thresholds(
        ref_thresholds_m=args.ref_thresholds_m,
        ref_perturb_bound_m=args.ref_perturb_bound_m,
        target_perturb_bound_m=args.perturb_bound_m,
    )
    eq_thr_df = compute_threshold_metrics(errors, thresholds=eq_thresholds, label="perturbation_equivalent")
    eq_thr_df["ref_threshold_m"] = list(map(float, args.ref_thresholds_m))
    eq_thr_df["scale_factor_target_over_ref"] = float(scale)

    print("Perturbation-normalized equivalent thresholds:")
    print(f"  Scale factor = target/ref = {args.perturb_bound_m:.3f}/{args.ref_perturb_bound_m:.3f} = {scale:.6f}")
    print("  Mapping (ref -> target equivalent):")
    for rt, et in zip(args.ref_thresholds_m, eq_thresholds):
        print(f"    {rt:.3f} m  ->  {et:.3f} m")
    print()

    pretty_eq = eq_thr_df.copy()
    pretty_eq["success_rate"] = pretty_eq["success_rate"].map(lambda x: f"{100*x:.2f}%")
    pretty_eq["failure_rate"] = pretty_eq["failure_rate"].map(lambda x: f"{100*x:.2f}%")
    cols = [
        "metric_group",
        "ref_threshold_m",
        "threshold_m",
        "success_rate",
        "failure_rate",
        "success_count",
        "total",
        "scale_factor_target_over_ref",
    ]
    print(pretty_eq[cols].to_string(index=False, justify="left"))
    print()

    # C) Fraction-of-perturbation metrics (direct normalized reporting)
    frac_thr_df = compute_fractional_threshold_metrics(
        errors,
        perturb_bound_m=args.perturb_bound_m,
        fractions=args.fraction_thresholds,
        label="fraction_of_perturb_bound",
    )
    print("Fraction-of-perturbation threshold metrics:")
    pretty_frac = frac_thr_df.copy()
    pretty_frac["success_rate"] = pretty_frac["success_rate"].map(lambda x: f"{100*x:.2f}%")
    pretty_frac["failure_rate"] = pretty_frac["failure_rate"].map(lambda x: f"{100*x:.2f}%")
    print(pretty_frac.to_string(index=False, justify="left"))
    print()

    # CDF-AUC scores
    cdf_auc_1 = compute_cdf_auc(errors, T=1.0)
    cdf_auc_2 = compute_cdf_auc(errors, T=2.0)
    cdf_auc_5 = compute_cdf_auc(errors, T=5.0)

    # Optional normalized CDF-AUC windows
    T_eq_1 = float(eq_thresholds[0]) if len(eq_thresholds) > 0 else None
    T_eq_2 = float(eq_thresholds[1]) if len(eq_thresholds) > 1 else None

    print("CDF-AUC (normalized, higher is better):")
    print(f"  AUC@1m: {cdf_auc_1:.4f}")
    print(f"  AUC@2m: {cdf_auc_2:.4f}")
    print(f"  AUC@5m: {cdf_auc_5:.4f}")
    if T_eq_1 is not None:
        print(f"  AUC@equiv({args.ref_thresholds_m[0]}m): AUC@{T_eq_1:.3f}m = {compute_cdf_auc(errors, T=T_eq_1):.4f}")
    if T_eq_2 is not None:
        print(f"  AUC@equiv({args.ref_thresholds_m[1]}m): AUC@{T_eq_2:.3f}m = {compute_cdf_auc(errors, T=T_eq_2):.4f}")
    print()

    # Save CSVs
    df.to_csv(out_dir / "errors_clean.csv", index=False)
    pct_df.to_csv(out_dir / "percentiles.csv", index=False)
    if not norm_pct_df.empty:
        norm_pct_df.to_csv(out_dir / "percentiles_normalized.csv", index=False)
    abs_thr_df.to_csv(out_dir / "threshold_metrics_absolute.csv", index=False)
    eq_thr_df.to_csv(out_dir / "threshold_metrics_equivalent.csv", index=False)
    frac_thr_df.to_csv(out_dir / "threshold_metrics_fractional.csv", index=False)

    # Save simple combined table too
    combined_thr_df = pd.concat([abs_thr_df, eq_thr_df, frac_thr_df], ignore_index=True, sort=False)
    combined_thr_df.to_csv(out_dir / "threshold_metrics_all.csv", index=False)

    # Save quick LaTeX tables
    save_latex_table(pct_df, out_dir / "percentiles.tex", float_cols=["error_m"])
    if not norm_pct_df.empty:
        save_latex_table(norm_pct_df, out_dir / "percentiles_normalized.tex", float_cols=["value"])
    save_latex_table(abs_thr_df, out_dir / "threshold_metrics_absolute.tex", float_cols=["threshold_m", "success_rate", "failure_rate"])
    save_latex_table(eq_thr_df, out_dir / "threshold_metrics_equivalent.tex",
                     float_cols=["ref_threshold_m", "threshold_m", "success_rate", "failure_rate", "scale_factor_target_over_ref"])
    save_latex_table(frac_thr_df, out_dir / "threshold_metrics_fractional.tex",
                     float_cols=["fraction_of_perturb_bound", "threshold_m", "success_rate", "failure_rate"])

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
    print(f"- {out_dir / 'percentiles_normalized.csv'} / .tex")
    print(f"- {out_dir / 'threshold_metrics_absolute.csv'} / .tex")
    print(f"- {out_dir / 'threshold_metrics_equivalent.csv'} / .tex")
    print(f"- {out_dir / 'threshold_metrics_fractional.csv'} / .tex")
    print(f"- {out_dir / 'threshold_metrics_all.csv'}")


if __name__ == "__main__":
    main()