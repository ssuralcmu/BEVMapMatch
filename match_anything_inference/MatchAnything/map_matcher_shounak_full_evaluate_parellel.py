import argparse
import json
from pathlib import Path
import multiprocessing as mp
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from imcui.ui.utils import get_matcher_zoo, load_config, run_matching

_WORKER_CONTEXT = {}

def imread_rgb(p: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    return img_bgr[:, :, ::-1].copy()  # BGR -> RGB

def center_image0_from_image(image0: np.ndarray):
    h0, w0 = image0.shape[:2]
    return (w0 - 1) / 2.0, (h0 - 1) / 2.0


def imwrite_rgb(p: Path, img_rgb: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img_rgb[:, :, ::-1])  # RGB -> BGR


def draw_center_on_image0(state_cache, x0, y0, out_path="image0_with_image1_center.png"):
    """
    Draws the mapped center of image1 on image0 (original resolution).
    """
    img0 = state_cache["image0_orig"].copy()  # RGB

    # Safety clamp (in case point is slightly outside bounds)
    h, w = img0.shape[:2]
    x0i = int(np.clip(round(x0), 0, w - 1))
    y0i = int(np.clip(round(y0), 0, h - 1))

    cv2.drawMarker(
        img0,
        (x0i, y0i),
        color=(0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=25,
        thickness=2
    )

    # Save (OpenCV expects BGR)
    cv2.imwrite(out_path, img0[:, :, ::-1])
    print(f"Saved visualization to: {out_path}")


def map_point_H(H, x, y):
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-12:
        raise RuntimeError("Homography projection has near-zero scale.")
    return float(q[0] / q[2]), float(q[1] / q[2])

def center_image0(state_cache):
    h0, w0 = state_cache["image0_orig"].shape[:2]
    return (w0 - 1) / 2.0, (h0 - 1) / 2.0

def center_image0_from_image(image0: np.ndarray):
    h0, w0 = image0.shape[:2]
    return (w0 - 1) / 2.0, (h0 - 1) / 2.0

def center_image1_in_image0_from_H(state_cache, invert_if_needed=True):
    try:
        H = np.asarray(state_cache["H"], dtype=np.float64)

        h1, w1 = state_cache["image1_orig"].shape[:2]
        cx1, cy1 = (w1 - 1) / 2.0, (h1 - 1) / 2.0

        # First try as-is
        x0, y0 = map_point_H(H, cx1, cy1)

        if not invert_if_needed:
            return x0, y0

        # Sanity-check: mapped corners should form a sensible quad in image0
        corners1 = np.array(
            [[[0, 0]], [[w1 - 1, 0]], [[w1 - 1, h1 - 1]], [[0, h1 - 1]]], dtype=np.float32
        )
        corners0 = cv2.perspectiveTransform(corners1, H).reshape(-1, 2)

        h0, w0 = state_cache["image0_orig"].shape[:2]
        # If all corners are way outside, you probably need inv(H)
        outside = np.sum(
            (corners0[:, 0] < -0.5 * w0)
            | (corners0[:, 0] > 1.5 * w0)
            | (corners0[:, 1] < -0.5 * h0)
            | (corners0[:, 1] > 1.5 * h0)
        )

        if outside >= 3:
            Hinv = np.linalg.inv(H)
            x0, y0 = map_point_H(Hinv, cx1, cy1)

        return x0, y0
    except Exception:
        return center_image0(state_cache)

def load_annotation(annotation_path: Path, key="gt_xy_pixels_top_pred_3x3"):
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"Missing annotation key '{key}' in {annotation_path}")
    xy = data[key]
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        raise ValueError(f"Invalid annotation value for '{key}' in {annotation_path}")
    return float(xy[0]), float(xy[1])


def collect_pairs(data_dir: Path, img0_suffix: str, img1_suffix: str, annotation_suffix: str):
    for img0_path in sorted(data_dir.glob(f"*{img0_suffix}")):
        base = img0_path.name[: -len(img0_suffix)]
        img1_path = data_dir / f"{base}{img1_suffix}"
        annotation_path = data_dir / f"{base}{annotation_suffix}"
        yield base, img0_path, img1_path, annotation_path


def plot_histograms(distances_m, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    distances = np.array(distances_m, dtype=np.float64)

    if distances.size == 0:
        print("No distances to plot.")
        return

    # 1m buckets up to 50m
    bins_1m = np.arange(0, 51, 1)
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=bins_1m, edgecolor="black")
    plt.title("Distance Error Histogram (1m bins up to 50m)")
    plt.xlabel("Distance error (m)")
    plt.ylabel("Count")
    plt.xlim(0, 50)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    hist_1m_path = out_dir / "distance_histogram_1m.png"
    plt.savefig(hist_1m_path)
    plt.close()

    # 5m buckets up to max
    max_distance = float(np.max(distances))
    max_bin = int(np.ceil(max_distance / 5.0) * 5)
    bins_5m = np.arange(0, max_bin + 5, 5)
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=bins_5m, edgecolor="black")
    plt.title("Distance Error Histogram (5m bins up to max)")
    plt.xlabel("Distance error (m)")
    plt.ylabel("Count")
    plt.xlim(0, max_bin)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    hist_5m_path = out_dir / "distance_histogram_5m.png"
    plt.savefig(hist_5m_path)
    plt.close()

    print(f"Saved histograms to: {hist_1m_path} and {hist_5m_path}")

def init_worker(config_path: str, settings: dict):
    cfg = load_config(Path(config_path))
    matcher_zoo = get_matcher_zoo(cfg["matcher_zoo"])
    _WORKER_CONTEXT["matcher_zoo"] = matcher_zoo
    _WORKER_CONTEXT["settings"] = settings
    _WORKER_CONTEXT["ransac_method"] = cfg["defaults"]["ransac_method"]
    _WORKER_CONTEXT["ransac_confidence"] = cfg["defaults"]["ransac_confidence"]
    _WORKER_CONTEXT["ransac_max_iter"] = cfg["defaults"]["ransac_max_iter"]


def process_pair(pair):
    """
    Robust per-pair worker:
    - never raises (returns {"skipped": (...)} on any error)
    - records fallback_error for debugging
    - safe even if run_matching / homography logic fails
    """
    base, img0_path, img1_path, annotation_path = pair

    try:
        # ---- Existence checks ----
        if not img0_path.exists() or not img1_path.exists():
            return {"skipped": (base, "missing_images")}
        if annotation_path is not None and not annotation_path.exists():
            return {"skipped": (base, "missing_annotation")}

        settings = _WORKER_CONTEXT["settings"]

        # ---- Load images (can throw FileNotFoundError or other I/O issues) ----
        image0 = imread_rgb(img0_path)
        image1 = imread_rgb(img1_path)

        fallback_error = None

        # ---- Matching + center mapping ----
        try:
            (
                _output_keypoints,
                _output_matches_raw,
                _output_matches_ransac,
                _num_matches,
                _configs,
                _geom_info,
                _output_wrapped,
                state_cache,
                _pkl_path,
            ) = run_matching(
                image0=image0,
                image1=image1,
                match_threshold=settings["match_threshold"],
                extract_max_keypoints=settings["max_features"],
                keypoint_threshold=settings["keypoint_threshold"],
                key=settings["matcher"],
                ransac_method=_WORKER_CONTEXT["ransac_method"],
                ransac_reproj_threshold=settings["ransac_reproj_threshold"],
                ransac_confidence=_WORKER_CONTEXT["ransac_confidence"],
                ransac_max_iter=_WORKER_CONTEXT["ransac_max_iter"],
                choice_geometry_type=settings["geometry"],
                matcher_zoo=_WORKER_CONTEXT["matcher_zoo"],
                force_resize=settings["force_resize"],
                image_width=settings["width"],
                image_height=settings["height"],
                use_cached_model=True,
            )

            # Map center of image1 into image0 via H (with robust invert-if-needed logic)
            x0, y0 = center_image1_in_image0_from_H(state_cache)

        except cv2.error as exc:
            # OpenCV threw, use fallback (center of image0) but keep reason
            fallback_error = f"opencv_error: {exc}"
            x0, y0 = center_image0_from_image(image0)

        except Exception as exc:
            # Any other matching-related exception: fallback but keep reason
            fallback_error = f"matching_exception: {type(exc).__name__}: {exc}"
            x0, y0 = center_image0_from_image(image0)

        # ---- If no annotation, just return prediction ----
        if annotation_path is None:
            return {
                "result": {
                    "id": base,
                    "pred_center_xy": [float(x0), float(y0)],
                    "distance_px": None,
                    "distance_m": None,
                    "fallback_error": fallback_error,
                }
            }

        # ---- Load GT + compute distance ----
        try:
            gt_x, gt_y = load_annotation(annotation_path, key=settings["annotation_key"])
        except Exception as exc:
            return {"skipped": (base, f"bad_annotation: {type(exc).__name__}: {exc}")}

        dist_px = float(np.linalg.norm(np.array([x0, y0], dtype=np.float64) -
                                       np.array([gt_x, gt_y], dtype=np.float64)))
        dist_m = dist_px / 2.0

        return {
            "result": {
                "id": base,
                "pred_center_xy": [float(x0), float(y0)],
                "gt_center_xy": [float(gt_x), float(gt_y)],
                "distance_px": dist_px,
                "distance_m": float(dist_m),
                "fallback_error": fallback_error,
            }
        }

    except Exception as exc:
        # Absolute catch-all so the worker never kills the pool
        return {"skipped": (base, f"process_pair_exception: {type(exc).__name__}: {exc}")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../../scripts_2026_fine/viz_dino_best_3x3crop")
    ap.add_argument("--img0", type=str)
    ap.add_argument("--img1", type=str)
    ap.add_argument("--out_dir", type=str, default="out_matchanything_all_evaluate")
    ap.add_argument("--img0_suffix", type=str, default="_metas_top_pred_3x3.png")
    ap.add_argument("--img1_suffix", type=str, default="_metas_stitched.png")
    ap.add_argument("--annotation_suffix", type=str, default="_metas_annotation.json")
    ap.add_argument("--annotation_key", type=str, default="gt_xy_pixels_top_pred_3x3")
    # Match the demo defaults:
    ap.add_argument("--matcher", type=str, default="matchanything_eloftr")
    ap.add_argument("--match_threshold", type=float, default=0.1)
    ap.add_argument("--max_features", type=int, default=1000)
    ap.add_argument("--keypoint_threshold", type=float, default=0.015)
    ap.add_argument("--geometry", type=str, default="Homography")  # demo default

    # Keep demo behavior: no forced resize unless you explicitly enable it
    ap.add_argument("--force_resize", action="store_true")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=min(30, os.cpu_count() or 1),
        help="Number of multiprocessing workers (default: min(10, cpu_count)).",
    )
    ap.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
        help="Multiprocessing start method.",
    )
    ap.add_argument("--pair_timeout_s", type=float, default=60.0,
                help="Timeout (seconds) per pair. Timed-out pairs are skipped.")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = repo_root / "config" / "config.yaml"
    cfg = load_config(config_path)
    matcher_zoo = get_matcher_zoo(cfg["matcher_zoo"])

    # RANSAC defaults come from config.yaml (demo uses these for method/conf/iters;
    # UI sets reproj_threshold slider default to 8.0)
    ransac_method = cfg["defaults"]["ransac_method"]
    ransac_confidence = cfg["defaults"]["ransac_confidence"]
    ransac_max_iter = cfg["defaults"]["ransac_max_iter"]
    ransac_reproj_threshold = 8.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = []

    if args.img0 and args.img1:
        pairs = [("single", Path(args.img0), Path(args.img1), None)]
    else:
        data_dir = Path(args.data_dir)
        pairs = list(collect_pairs(data_dir, args.img0_suffix, args.img1_suffix, args.annotation_suffix))
        if not pairs:
            raise FileNotFoundError(f"No image pairs found in {data_dir} with suffix {args.img0_suffix}")

    settings = {
        "matcher": args.matcher,
        "match_threshold": args.match_threshold,
        "max_features": args.max_features,
        "keypoint_threshold": args.keypoint_threshold,
        "geometry": args.geometry,
        "force_resize": args.force_resize,
        "width": args.width,
        "height": args.height,
        "ransac_reproj_threshold": ransac_reproj_threshold,
        "annotation_key": args.annotation_key,
    }

    if args.num_workers <= 1 or len(pairs) <= 1:
        _WORKER_CONTEXT.update(
            {
                "matcher_zoo": matcher_zoo,
                "settings": settings,
                "ransac_method": ransac_method,
                "ransac_confidence": ransac_confidence,
                "ransac_max_iter": ransac_max_iter,
            }
        )

        for pair in tqdm(pairs, desc="Processing pairs"):
            outcome = process_pair(pair)
            if "skipped" in outcome:
                skipped.append(outcome["skipped"])
            else:
                results.append(outcome["result"])

    else:
        ctx = mp.get_context(args.start_method)
        with ctx.Pool(
            processes=args.num_workers,
            initializer=init_worker,
            initargs=(str(config_path), settings),
        ) as pool:

            # submit all jobs
            jobs = [(pair[0], pool.apply_async(process_pair, (pair,))) for pair in pairs]

            # collect with timeout per job
            for base, job in tqdm(jobs, total=len(jobs), desc="Processing pairs"):
                try:
                    outcome = job.get(timeout=args.pair_timeout_s)
                except mp.context.TimeoutError:
                    skipped.append((base, f"timeout>{args.pair_timeout_s}s"))
                    continue
                except Exception as e:
                    skipped.append((base, f"worker_exception: {type(e).__name__}: {e}"))
                    continue

                if "skipped" in outcome:
                    skipped.append(outcome["skipped"])
                else:
                    results.append(outcome["result"])

                    
                    
    results_path = out_dir / "distance_errors.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    distances_m = [r["distance_m"] for r in results if r["distance_m"] is not None]
    if distances_m:
        mean_distance = float(np.mean(distances_m))
        print(f"Mean distance error (m): {mean_distance:.3f}")
    else:
        mean_distance = None
        print("No distance errors computed.")

    plot_histograms(distances_m, out_dir)

    if skipped:
        skipped_path = out_dir / "skipped_pairs.json"
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2)
        print(f"Skipped {len(skipped)} pairs. Details saved to {skipped_path}.")

    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
