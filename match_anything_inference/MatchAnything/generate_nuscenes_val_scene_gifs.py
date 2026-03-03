
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

MAP_LAYER_COLORS = {
    "drivable_area": np.array([230, 230, 230], dtype=np.uint8),
    "road_segment": np.array([170, 170, 170], dtype=np.uint8),
    "lane": np.array([255, 220, 120], dtype=np.uint8),
    "walkway": np.array([120, 200, 255], dtype=np.uint8),
    "ped_crossing": np.array([255, 120, 120], dtype=np.uint8),
}


@dataclass
class NuScenesTables:
    scene: List[dict]
    sample: List[dict]
    sample_data: List[dict]
    ego_pose: List[dict]
    log: List[dict]
    calibrated_sensor: List[dict]
    sensor: List[dict]


class TableIndex:
    def __init__(self, rows: Iterable[dict]):
        self._by_token = {r["token"]: r for r in rows}

    def get(self, token: str) -> dict:
        return self._by_token[token]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate day/night val-scene GIFs with camera+lidar+map layout.")
    p.add_argument("--dataroot", type=Path, default=Path("/data1/data/nuscenes/"))
    p.add_argument("--version", type=str, default="v1.0-trainval")
    p.add_argument("--distance-errors", type=Path, default=Path("/data1/match_anything_inference/MatchAnything/out_matchanything_all_evaluate/distance_errors.json"))
    p.add_argument("--output-dir", type=Path, default=Path("/data1/match_anything_inference/MatchAnything/out_matchanything_all_evaluate/scene_gifs"))
    p.add_argument("--day-scenes", type=int, default=10)
    p.add_argument("--night-scenes", type=int, default=10)
    p.add_argument("--max-samples-per-scene", type=int, default=0)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--camera-tile-w", type=int, default=360)
    p.add_argument("--camera-tile-h", type=int, default=200)
    p.add_argument("--map-size", type=int, default=700)
    p.add_argument("--map-patch-m", type=float, default=80.0)
    p.add_argument("--lidar-max-range", type=float, default=60.0)
    return p.parse_args()


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def load_tables(dataroot: Path, version: str) -> NuScenesTables:
    root = dataroot / version
    required = ["scene.json", "sample.json", "sample_data.json", "ego_pose.json", "log.json", "calibrated_sensor.json", "sensor.json"]
    missing = [x for x in required if not (root / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing NuScenes metadata files under {root}: {missing}")

    return NuScenesTables(
        scene=load_json(root / "scene.json"),
        sample=load_json(root / "sample.json"),
        sample_data=load_json(root / "sample_data.json"),
        ego_pose=load_json(root / "ego_pose.json"),
        log=load_json(root / "log.json"),
        calibrated_sensor=load_json(root / "calibrated_sensor.json"),
        sensor=load_json(root / "sensor.json"),
    )


def load_val_scene_names() -> set:
    from nuscenes.utils.splits import val

    return set(val)


def load_distance_error_index(path: Path) -> Dict[str, dict]:
    rows = load_json(path)
    out = {}
    for row in rows:
        out[str(row["id"]).split("-")[-1]] = row
    return out


def is_night_scene(scene_row: dict) -> bool:
    desc = (scene_row.get("description") or "").lower()
    return "(n)" in desc or "night" in desc


def read_image(path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img)


def resolve_nuscenes_file(dataroot: Path, filename: str, version: Optional[str] = None) -> Optional[Path]:
    """Resolve NuScenes sample_data filename across common storage layouts."""
    cand = Path(filename)

    # 1) Already absolute.
    if cand.is_absolute() and cand.exists():
        return cand

    # Potential roots observed in different local copies.
    roots = [dataroot]
    if version:
        roots.append(dataroot / version)
    roots.extend([
        dataroot.parent,
        dataroot.parent / "nuscenes",
        dataroot.parent / "data" / "nuscenes",
    ])

    # Normalize path variants.
    parts = cand.parts
    stripped_variants = [cand]

    # Remove possible leading anchors such as data/nuscenes or version name.
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "nuscenes":
        stripped_variants.append(Path(*parts[2:]))
    if version and len(parts) >= 1 and parts[0] == version:
        stripped_variants.append(Path(*parts[1:]))

    # Keep only from known content anchors if present.
    for anchor in ("samples", "sweeps", "maps"):
        if anchor in parts:
            idx = parts.index(anchor)
            stripped_variants.append(Path(*parts[idx:]))

    # Try all combinations.
    tried = set()
    for root in roots:
        for rel in stripped_variants:
            probe = (root / rel).resolve()
            if probe in tried:
                continue
            tried.add(probe)
            if probe.exists():
                return probe

    return None



def build_camera_mosaic(sample_row: dict, sample_data_by_sample_and_channel: Dict[Tuple[str, str], str], sd_index: TableIndex, dataroot: Path, tile_hw: Tuple[int, int], version: str) -> np.ndarray:
    tile_h, tile_w = tile_hw
    rows = []
    for r in range(2):
        cols = []
        for c in range(3):
            cam = CAMERA_ORDER[r * 3 + c]
            token = sample_data_by_sample_and_channel.get((sample_row["token"], cam))
            if token is None:
                cols.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                continue
            sd = sd_index.get(token)
            fp = resolve_nuscenes_file(dataroot, sd["filename"], version)
            cols.append(read_image(fp, (tile_h, tile_w)) if fp is not None else np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def render_lidar_panel(sample_row: dict, sample_data_by_sample_and_channel: Dict[Tuple[str, str], str], sd_index: TableIndex, dataroot: Path, size: Tuple[int, int], max_range: float, version: str) -> np.ndarray:
    h, w = size
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = np.array([10, 10, 10], dtype=np.uint8)

    token = sample_data_by_sample_and_channel.get((sample_row["token"], "LIDAR_TOP"))
    if token is None:
        return panel
    sd = sd_index.get(token)
    fp = resolve_nuscenes_file(dataroot, sd["filename"], version)
    if fp is None:
        return panel

    pts = np.fromfile(fp, dtype=np.float32)
    if pts.size < 5:
        return panel
    pts = pts.reshape(-1, 5)
    x, y = pts[:, 0], pts[:, 1]
    r = np.sqrt(x**2 + y**2)
    keep = r <= max_range
    x, y, r = x[keep], y[keep], r[keep]

    px = ((x / (2 * max_range)) + 0.5) * (w - 1)
    py = (0.5 - (y / (2 * max_range))) * (h - 1)
    px = np.clip(px.astype(np.int32), 0, w - 1)
    py = np.clip(py.astype(np.int32), 0, h - 1)

    intensity = np.clip((1 - r / max_range) * 255, 0, 255).astype(np.uint8)
    panel[py, px, 1] = np.maximum(panel[py, px, 1], intensity)
    panel[py, px, 2] = np.maximum(panel[py, px, 2], 255 - intensity)

    return panel


def render_map_panel(sample_row: dict, scene_row: dict, sample_data_by_sample_and_channel: Dict[Tuple[str, str], str], sd_index: TableIndex, ego_index: TableIndex, log_index: TableIndex, dataroot: Path, map_size: int, patch_m: float, map_cache: Dict[str, object]) -> np.ndarray:
    from nuscenes.map_expansion.map_api import NuScenesMap

    lidar_token = sample_data_by_sample_and_channel.get((sample_row["token"], "LIDAR_TOP"))
    if lidar_token is None:
        return np.zeros((map_size, map_size, 3), dtype=np.uint8)
    sd = sd_index.get(lidar_token)
    ego = ego_index.get(sd["ego_pose_token"])
    x, y = float(ego["translation"][0]), float(ego["translation"][1])

    log_row = log_index.get(scene_row["log_token"])
    map_name = log_row["location"]
    if map_name not in map_cache:
        map_cache[map_name] = NuScenesMap(dataroot=str(dataroot), map_name=map_name)
    nusc_map = map_cache[map_name]

    layers = list(MAP_LAYER_COLORS.keys())
    mask = nusc_map.get_map_mask((x, y, patch_m, patch_m), 0, layers, (map_size, map_size))

    rgb = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    for i, layer in enumerate(layers):
        rgb[mask[i].astype(bool)] = MAP_LAYER_COLORS[layer]

    c = map_size // 2
    rgb[max(0, c - 6): min(map_size, c + 7), max(0, c - 6): min(map_size, c + 7)] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def build_sample_data_lookup(
    sample_data_rows: List[dict],
    calibrated_sensor_rows: List[dict],
    sensor_rows: List[dict],
) -> Dict[Tuple[str, str], str]:
    """Map (sample_token, channel) -> sample_data_token from raw NuScenes tables.

    Note: `sample_data.json` does not include `channel`; channel comes from
    sample_data.calibrated_sensor_token -> calibrated_sensor.sensor_token -> sensor.channel.
    """
    cal_to_sensor = {r["token"]: r.get("sensor_token") for r in calibrated_sensor_rows}
    sensor_to_channel = {r["token"]: r.get("channel") for r in sensor_rows}

    lookup: Dict[Tuple[str, str], str] = {}
    for row in sample_data_rows:
        sample_token = row.get("sample_token")
        cal_token = row.get("calibrated_sensor_token")
        sensor_token = cal_to_sensor.get(cal_token)
        channel = sensor_to_channel.get(sensor_token)
        token = row.get("token")
        if sample_token and channel and token:
            lookup[(sample_token, channel)] = token
    return lookup


def select_target_scenes(all_scenes: List[dict], val_names: set, day_k: int, night_k: int) -> List[dict]:
    val_scenes = sorted([s for s in all_scenes if s["name"] in val_names], key=lambda s: s["name"])
    night = [s for s in val_scenes if is_night_scene(s)]
    day = [s for s in val_scenes if not is_night_scene(s)]
    if len(day) < day_k or len(night) < night_k:
        raise ValueError(f"Insufficient val scenes: day={len(day)} night={len(night)}")
    return day[:day_k] + night[:night_k]


def walk_scene_samples(scene_row: dict, sample_index: TableIndex, limit: int = 0) -> List[dict]:
    out = []
    tok = scene_row["first_sample_token"]
    while tok:
        s = sample_index.get(tok)
        out.append(s)
        tok = s["next"]
        if limit > 0 and len(out) >= limit:
            break
    return out


def concat_h(imgs: List[np.ndarray]) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    padded = []
    for i in imgs:
        if i.shape[0] < h:
            pad = np.zeros((h - i.shape[0], i.shape[1], 3), dtype=np.uint8)
            padded.append(np.concatenate([i, pad], axis=0))
        else:
            padded.append(i)
    return np.concatenate(padded, axis=1)


def add_header(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    out = Image.new("RGB", (img.width, img.height + 32), (0, 0, 0))
    out.paste(img, (0, 32))
    dr = ImageDraw.Draw(out)
    dr.text((8, 8), text, fill=(255, 255, 255))
    return np.asarray(out)


def stack_frame(cam: np.ndarray, lidar: np.ndarray, map_img: np.ndarray, header: str) -> np.ndarray:
    left = np.concatenate([cam, lidar], axis=0)
    map_resized = np.asarray(Image.fromarray(map_img).resize((map_img.shape[1], left.shape[0]), Image.BILINEAR))
    combo = concat_h([left, map_resized])
    return add_header(combo, header)


def save_gif(path: Path, frames: List[np.ndarray], fps: int) -> None:
    duration_ms = max(1, int(1000 / max(1, fps)))
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0)


def print_diagnostics(dataroot: Path, version: str, samples: List[dict], sample_data_by_sample_and_channel: Dict[Tuple[str, str], str], sd_index: TableIndex) -> None:
    cams_total = 0
    cams_found = 0
    lidar_total = 0
    lidar_found = 0

    unresolved_examples = []

    for s in samples:
        st = s["token"]
        for cam in CAMERA_ORDER:
            cams_total += 1
            tok = sample_data_by_sample_and_channel.get((st, cam))
            if tok is None:
                continue
            sd = sd_index.get(tok)
            if resolve_nuscenes_file(dataroot, sd["filename"], version) is not None:
                cams_found += 1
            elif len(unresolved_examples) < 5:
                unresolved_examples.append(sd["filename"])

        lidar_total += 1
        tok = sample_data_by_sample_and_channel.get((st, "LIDAR_TOP"))
        if tok is None:
            continue
        sd = sd_index.get(tok)
        if resolve_nuscenes_file(dataroot, sd["filename"], version) is not None:
            lidar_found += 1
        elif len(unresolved_examples) < 5:
            unresolved_examples.append(sd["filename"])

    print(f"[diag] camera files found: {cams_found}/{cams_total} | lidar files found: {lidar_found}/{lidar_total}")
    if unresolved_examples:
        print("[diag] unresolved filename examples:")
        for ex in unresolved_examples:
            print(f"  - {ex}")



def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(args.dataroot, args.version)
    sample_index = TableIndex(tables.sample)
    sd_index = TableIndex(tables.sample_data)
    ego_index = TableIndex(tables.ego_pose)
    log_index = TableIndex(tables.log)
    sample_data_by_sample_and_channel = build_sample_data_lookup(
        tables.sample_data,
        tables.calibrated_sensor,
        tables.sensor,
    )

    val_names = load_val_scene_names()
    scenes = select_target_scenes(tables.scene, val_names, args.day_scenes, args.night_scenes)
    dist_by_sample = load_distance_error_index(args.distance_errors)
    print(f"[diag] sample/channel lookup entries: {len(sample_data_by_sample_and_channel)}")
    map_cache: Dict[str, object] = {}

    report = []
    for i, scene in enumerate(scenes, 1):
        scene_name = scene["name"]
        scene_type = "night" if is_night_scene(scene) else "day"
        samples = walk_scene_samples(scene, sample_index, args.max_samples_per_scene)
        print(f"[{i:02d}/{len(scenes)}] {scene_name} ({scene_type}) samples={len(samples)}")
        if i == 1:
            print_diagnostics(args.dataroot, args.version, samples, sample_data_by_sample_and_channel, sd_index)

        frames = []
        linked = 0
        for s in samples:
            tok = s["token"]
            err = dist_by_sample.get(tok)
            if err:
                linked += 1

            cam = build_camera_mosaic(s, sample_data_by_sample_and_channel, sd_index, args.dataroot, (args.camera_tile_h, args.camera_tile_w), args.version)
            lidar = render_lidar_panel(s, sample_data_by_sample_and_channel, sd_index, args.dataroot, (args.camera_tile_h * 2, args.camera_tile_w * 3), args.lidar_max_range, args.version)
            map_img = render_map_panel(s, scene, sample_data_by_sample_and_channel, sd_index, ego_index, log_index, args.dataroot, args.map_size, args.map_patch_m, map_cache)

            msg = f"{scene_name} | {tok[:8]}"
            if err:
                msg += f" | dist_m={err['distance_m']:.2f}"
            frames.append(stack_frame(cam, lidar, map_img, msg))

        out_path = args.output_dir / f"{scene_name}_{scene_type}.gif"
        save_gif(out_path, frames, args.fps)
        report.append({"scene_name": scene_name, "scene_type": scene_type, "num_samples": len(samples), "num_linked_distance_errors": linked, "gif": str(out_path)})

    with (args.output_dir / "scene_gif_report.json").open("w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
