import argparse
from pathlib import Path

import cv2
import numpy as np

from imcui.ui.utils import load_config, get_matcher_zoo, run_matching


def imread_rgb(p: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    return img_bgr[:, :, ::-1].copy()  # BGR -> RGB


def imwrite_rgb(p: Path, img_rgb: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img_rgb[:, :, ::-1])  # RGB -> BGR

import cv2
import numpy as np

def draw_center_on_image0(state_cache, x0, y0, out_path="image0_with_image1_center.png"):
    """
    Draws the mapped center of image1 on image0 (original resolution).
    """
    img0 = state_cache["image0_orig"].copy()  # RGB

    # Safety clamp (in case point is slightly outside bounds)
    h, w = img0.shape[:2]
    x0i = int(np.clip(round(x0), 0, w - 1))
    y0i = int(np.clip(round(y0), 0, h - 1))


    # Optional: crosshair for precision
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

import numpy as np

def center_image1_in_image0_from_matches(state_cache, k=50, use_mm=True):
    """
    Returns (x0, y0) in image0 pixel coords corresponding to the center of image1.
    Uses matched keypoints in ORIGINAL (un-resized) coordinates.
    """
    h1, w1 = state_cache["image1_orig"].shape[:2]
    center1 = np.array([w1 / 2.0, h1 / 2.0], dtype=np.float32)

    if use_mm and "mmkeypoints1_orig" in state_cache and state_cache["mmkeypoints1_orig"].shape[0] > 0:
        p0 = state_cache["mmkeypoints0_orig"].astype(np.float32)  # (N,2) in img0 orig
        p1 = state_cache["mmkeypoints1_orig"].astype(np.float32)  # (N,2) in img1 orig
        conf = state_cache["mmconf"].astype(np.float32)           # (N,)
    else:
        p0 = state_cache["mkeypoints0_orig"].astype(np.float32)
        p1 = state_cache["mkeypoints1_orig"].astype(np.float32)
        conf = state_cache["mconf"].astype(np.float32)

    if p0.shape[0] == 0:
        raise RuntimeError("No matched keypoints available.")

    # nearest matches in image1 to the center
    d = np.linalg.norm(p1 - center1[None, :], axis=1)
    idx = np.argsort(d)[:min(k, len(d))]

    # confidence-weighted average of corresponding points in image0
    w = conf[idx]
    w = w / (w.sum() + 1e-8)
    center0 = (p0[idx] * w[:, None]).sum(axis=0)

    return float(center0[0]), float(center0[1])


import numpy as np
import cv2

def map_point_H(H, x, y):
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-12:
        raise RuntimeError("Homography projection has near-zero scale.")
    return float(q[0] / q[2]), float(q[1] / q[2])

def center_image1_in_image0_from_H(state_cache, invert_if_needed=True):
    H = np.asarray(state_cache["H"], dtype=np.float64)

    h1, w1 = state_cache["image1_orig"].shape[:2]
    cx1, cy1 = (w1 - 1) / 2.0, (h1 - 1) / 2.0

    # First try as-is
    x0, y0 = map_point_H(H, cx1, cy1)

    if not invert_if_needed:
        return x0, y0

    # Sanity-check: mapped corners should form a sensible quad in image0
    corners1 = np.array([[[0,0]], [[w1-1,0]], [[w1-1,h1-1]], [[0,h1-1]]], dtype=np.float32)
    corners0 = cv2.perspectiveTransform(corners1, H).reshape(-1, 2)

    h0, w0 = state_cache["image0_orig"].shape[:2]
    # If all corners are way outside, you probably need inv(H)
    outside = np.sum(
        (corners0[:,0] < -0.5*w0) | (corners0[:,0] > 1.5*w0) |
        (corners0[:,1] < -0.5*h0) | (corners0[:,1] > 1.5*h0)
    )

    if outside >= 3:
        Hinv = np.linalg.inv(H)
        x0, y0 = map_point_H(Hinv, cx1, cy1)

    return x0, y0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img0", type=str, required=True)
    ap.add_argument("--img1", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out_matchanything")

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

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_config(repo_root / "config" / "config.yaml")
    matcher_zoo = get_matcher_zoo(cfg["matcher_zoo"])

    # RANSAC defaults come from config.yaml (demo uses these for method/conf/iters;
    # UI sets reproj_threshold slider default to 8.0)
    ransac_method = cfg["defaults"]["ransac_method"]
    ransac_confidence = cfg["defaults"]["ransac_confidence"]
    ransac_max_iter = cfg["defaults"]["ransac_max_iter"]
    ransac_reproj_threshold = 8.0

    image0 = imread_rgb(Path(args.img0))
    image1 = imread_rgb(Path(args.img1))

    (
        output_keypoints,
        output_matches_raw,
        output_matches_ransac,
        num_matches,
        configs,
        geom_info,
        output_wrapped,   # visualization pair (like the demo panel)
        state_cache,      # contains "wrapped_image" (the actual warped image)
        pkl_path,
    ) = run_matching(
        image0=image0,
        image1=image1,
        match_threshold=args.match_threshold,
        extract_max_keypoints=args.max_features,
        keypoint_threshold=args.keypoint_threshold,
        key=args.matcher,
        ransac_method=ransac_method,
        ransac_reproj_threshold=ransac_reproj_threshold,
        ransac_confidence=ransac_confidence,
        ransac_max_iter=ransac_max_iter,
        choice_geometry_type=args.geometry,
        matcher_zoo=matcher_zoo,
        force_resize=args.force_resize,
        image_width=args.width,
        image_height=args.height,
        use_cached_model=False,
    )

    print("geom_info keys:", geom_info.keys() if isinstance(geom_info, dict) else type(geom_info))
    print("state_cache keys:", list(state_cache.keys()))


    x0, y0 = center_image1_in_image0_from_H(state_cache)
    print(f"H-mapped center: (x={x0:.2f}, y={y0:.2f})")
    draw_center_on_image0(state_cache, x0, y0, out_path="image0_with_center_of_image1.png")


    out_dir = Path(args.out_dir)
    imwrite_rgb(out_dir / "keypoints.png", output_keypoints)
    imwrite_rgb(out_dir / "matches_raw.png", output_matches_raw)
    imwrite_rgb(out_dir / "matches_ransac.png", output_matches_ransac)

    # This is the "wrapped image pair" visualization (Image0 + warped Image1)
    if output_wrapped is not None:
        imwrite_rgb(out_dir / "wrapped_pair.png", output_wrapped)

    # This is the ACTUAL warped image (Image1 warped into Image0 frame), same as the demo’s stored "wrapped_image"
    warped = state_cache.get("wrapped_image", None)
    if warped is not None:
        imwrite_rgb(out_dir / "warped_image1_into_image0.png", warped)

    print("Saved outputs to:", out_dir.resolve())
    print("num_matches:", num_matches)
    print("pkl:", pkl_path)


if __name__ == "__main__":
    main()
