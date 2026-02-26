"""
Compute localization error metrics + plots from a JSON list of dicts and additionally
report the same outputs for weather/time contexts:
- night
- rain
- night+rain
- not night + not rain

Outputs (for overall + each context subset):
- CDF plot (optionally with tail inset)
- Percentile table (P50/P75/P90/P95/P99)
- Failure-rate / success-rate metrics at thresholds
- Cleaned per-sample CSV

Usage:
  python scripts_2026_fine/analyze_fine_matching_errors_with_context.py \
      --json_path results.json --out_dir out_with_context \
      --scene_json_path /data1/data/nuscenes/v1.0-trainval/scene.json \
      --sample_json_path /data1/data/nuscenes/v1.0-trainval/sample.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTEXTS = ("all", "night", "rain", "night_rain", "not_night_not_rain")


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def build_sample_token_condition_lookup(scene_json_path: Path, sample_json_path: Path) -> dict:
    with open(scene_json_path, "r") as f:
        scenes = json.load(f)
    with open(sample_json_path, "r") as f:
        samples = json.load(f)

    scene_flags = {}
    for scene in scenes:
        scene_flags[scene["token"]] = {
            "night": str(scene.get("N", "0")) == "1",
            "rain": str(scene.get("R", "0")) == "1",
        }

    lookup = {}
    for sample in samples:
        scene_token = sample.get("scene_token")
        if scene_token in scene_flags:
            lookup[sample["token"]] = scene_flags[scene_token]
    return lookup


def sample_matches_condition(flags: dict, subset_name: str) -> bool:
    night = flags["night"]
    rain = flags["rain"]
    if subset_name == "night":
        return night
    if subset_name == "rain":
        return rain
    if subset_name == "night_rain":
        return night and rain
    if subset_name == "not_night_not_rain":
        return (not night) and (not rain)
    return True


def extract_sample_token_from_id(sample_id: str) -> str:
    """
    Best-effort extraction of nuScenes sample token from id.
    Expected common format: '<timestamp>-<sample_token>'.
    Falls back to full id if no '-' exists.
    """
    if not isinstance(sample_id, str):
        return ""
    if "-" in sample_id:
        return sample_id.split("-")[-1]
    return sample_id


def load_errors_m(json_path: Path, use_fallback: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: id, sample_token, error_m, used_field
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for i, item in enumerate(data):
        sid = item.get("id", f"idx_{i}")

        dm = item.get("distance_m", None)
        fe = item.get("fallback_error", None)

        if _is_finite_number(dm):
            rows.append(
                {
                    "id": sid,
                    "sample_token": extract_sample_token_from_id(sid),
                    "error_m": float(dm),
                    "used_field": "distance_m",
                }
            )
        elif use_fallback and _is_finite_number(fe):
            rows.append(
                {
                    "id": sid,
                    "sample_token": extract_sample_token_from_id(sid),
                    "error_m": float(fe),
                    "used_field": "fallback_error",
                }
            )

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


def compute_threshold_metrics(errors: np.ndarray, thresholds=(1, 2, 5, 10, 20, 50, 100, 166)) -> pd.DataFrame:
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


def write_subset_outputs(df_subset: pd.DataFrame, subset_name: str, out_dir: Path, cdf_xmax: float, tail_xmax: float):
    errors = df_subset["error_m"].to_numpy()

    pct_df = compute_percentiles(errors, percentiles=(50, 75, 90, 95, 99))
    thr_df = compute_threshold_metrics(errors)

    cdf_auc_1 = compute_cdf_auc(errors, T=1.0)
    cdf_auc_2 = compute_cdf_auc(errors, T=2.0)
    cdf_auc_5 = compute_cdf_auc(errors, T=5.0)

    subset_out_dir = out_dir / subset_name
    subset_out_dir.mkdir(parents=True, exist_ok=True)

    df_subset.to_csv(subset_out_dir / "errors_clean.csv", index=False)
    pct_df.to_csv(subset_out_dir / "percentiles.csv", index=False)
    thr_df.to_csv(subset_out_dir / "threshold_metrics.csv", index=False)

    plot_cdf(
        errors,
        subset_out_dir / "cdf_0_to_xmax.png",
        xmax_main=cdf_xmax,
        tail_xlim=(cdf_xmax, tail_xmax),
    )
    plot_ccdf_tail(errors, subset_out_dir / "ccdf_0_to_tailmax.png", xlim=(0.0, tail_xmax))

    print(f"\n[{subset_name}] Loaded {len(errors)} samples")
    print(f"[{subset_name}] Median error: {np.median(errors):.3f} m | Max error: {np.max(errors):.3f} m")
    print(f"[{subset_name}] Percentiles:")
    print(pct_df.to_string(index=False, justify="left", formatters={"error_m": "{:.3f}".format}))
    print(f"[{subset_name}] CDF-AUC: AUC@1m={cdf_auc_1:.4f}, AUC@2m={cdf_auc_2:.4f}, AUC@5m={cdf_auc_5:.4f}")

    pretty_thr = thr_df.copy()
    pretty_thr["success_rate"] = pretty_thr["success_rate"].map(lambda x: f"{100*x:.2f}%")
    pretty_thr["failure_rate"] = pretty_thr["failure_rate"].map(lambda x: f"{100*x:.2f}%")
    print(f"[{subset_name}] Threshold metrics:")
    print(pretty_thr.to_string(index=False, justify="left"))

    return {
        "subset": subset_name,
        "num_samples": int(len(errors)),
        "median_error_m": float(np.median(errors)),
        "max_error_m": float(np.max(errors)),
        "auc_1m": cdf_auc_1,
        "auc_2m": cdf_auc_2,
        "auc_5m": cdf_auc_5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="../match_anything_inference/MatchAnything/out_matchanything_all_evaluate/distance_errors.json", help="Path to JSON results file")
    parser.add_argument("--out_dir", type=str, default="out_with_context", help="Output directory")
    parser.add_argument("--scene_json_path", type=str, default="/data1/data/nuscenes/v1.0-trainval/scene.json", help="Path to nuScenes scene.json")
    parser.add_argument("--sample_json_path", type=str, default="/data1/data/nuscenes/v1.0-trainval/sample.json", help="Path to nuScenes sample.json")
    parser.add_argument("--ignore_fallback", action="store_true", default=True, help="Ignore fallback_error even if present")
    parser.add_argument("--cdf_xmax", type=float, default=50.0, help="Max x for main CDF view")
    parser.add_argument("--tail_xmax", type=float, default=500.0, help="Max x for tail view")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_errors_m(Path(args.json_path), use_fallback=not args.ignore_fallback)

    sample_flag_lookup = build_sample_token_condition_lookup(Path(args.scene_json_path), Path(args.sample_json_path))
    flags_df = pd.DataFrame.from_dict(sample_flag_lookup, orient="index").reset_index().rename(columns={"index": "sample_token"})
    df_with_flags = df.merge(flags_df, on="sample_token", how="left")

    summary_rows = []
    for subset in CONTEXTS:
        if subset == "all":
            subset_df = df_with_flags.copy()
        else:
            valid_df = df_with_flags[df_with_flags["night"].notna() & df_with_flags["rain"].notna()].copy()
            mask = valid_df.apply(
                lambda row: sample_matches_condition(
                    {"night": bool(row["night"]), "rain": bool(row["rain"])},
                    subset,
                ),
                axis=1,
            )
            subset_df = valid_df[mask]

        subset_df = subset_df.drop(columns=[c for c in ["night", "rain"] if c in subset_df.columns])

        if subset_df.empty:
            print(f"[{subset}] No samples matched; skipping output generation.")
            continue

        summary_rows.append(write_subset_outputs(subset_df, subset, out_dir, args.cdf_xmax, args.tail_xmax))

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "context_summary.csv", index=False)
        print(f"\nSaved context summary to: {out_dir / 'context_summary.csv'}")


if __name__ == "__main__":
    main()
