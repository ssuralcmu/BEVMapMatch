#!/usr/bin/env python3
"""Create BEVMapMatch basemaps, local semantic maps and metadata from nuScenes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm


LAYERS = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "road_divider",
    "lane_divider",
]

PALETTE = np.asarray(
    [
        [255, 255, 255],
        [128, 128, 128],
        [255, 0, 0],
        [0, 180, 0],
        [255, 165, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 255, 0],
    ],
    dtype=np.uint8,
)


def colorize(masks: np.ndarray) -> Image.Image:
    canvas = np.zeros(masks.shape[1:], dtype=np.uint8)
    for index, mask in enumerate(masks, start=1):
        canvas[np.asarray(mask, dtype=bool)] = index
    return Image.fromarray(PALETTE[canvas], mode="RGB")


def render_map(
    map_api: NuScenesMap,
    center_x: float,
    center_y: float,
    size_m: float,
    size_px: int,
    angle_degrees: float,
) -> Image.Image:
    masks = map_api.get_map_mask(
        patch_box=(center_x, center_y, size_m, size_m),
        patch_angle=angle_degrees,
        layer_names=LAYERS,
        canvas_size=(size_px, size_px),
    )
    return colorize(masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--basemap-size-m", type=float, default=500.0)
    parser.add_argument("--basemap-size-px", type=int, default=1000)
    parser.add_argument("--local-size-m", type=float, default=100.0)
    parser.add_argument("--local-size-px", type=int, default=300)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    output = Path(args.output) / args.split
    directories = {name: output / name for name in ("metadata", "basemaps", "segmentations")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=True)
    split_scenes = set(create_splits_scenes()[args.split])
    scene_names = {scene["token"]: scene["name"] for scene in nusc.scene}
    log_locations = {log["token"]: log["location"] for log in nusc.log}
    scene_locations = {
        scene["token"]: log_locations[scene["log_token"]]
        for scene in nusc.scene
    }
    maps = {location: NuScenesMap(dataroot=str(dataroot), map_name=location) for location in set(scene_locations.values())}
    samples = [sample for sample in nusc.sample if scene_names[sample["scene_token"]] in split_scenes]
    samples.sort(key=lambda sample: (scene_names[sample["scene_token"]], sample["timestamp"]))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    rng = np.random.default_rng(args.seed)

    for sample in tqdm(samples, desc=f"Preparing {args.split}"):
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", lidar["ego_pose_token"])
        true_x, true_y = pose["translation"][:2]
        yaw = Quaternion(pose["rotation"]).yaw_pitch_roll[0]
        perturbation = rng.uniform(-200.0, 200.0, size=2)
        base_x, base_y = true_x + perturbation[0], true_y + perturbation[1]
        location = scene_locations[sample["scene_token"]]
        map_api = maps[location]
        identifier = f'{sample["timestamp"]}-{sample["token"]}'

        basemap = render_map(map_api, base_x, base_y, args.basemap_size_m, args.basemap_size_px, 0.0)
        local = render_map(map_api, true_x, true_y, args.local_size_m, args.local_size_px, np.degrees(yaw))
        basemap.save(directories["basemaps"] / f"{identifier}_base_map_image.png")
        local.save(directories["segmentations"] / f"{identifier}_generated_map_image.png")
        np.save(
            directories["metadata"] / f"{identifier}_metas.npy",
            {
                "perturbation": perturbation.astype(np.float32),
                "timestamp": sample["timestamp"],
                "scene_id": sample["scene_token"],
                "sample_token": sample["token"],
                "location": location,
                "ego_translation": pose["translation"],
                "ego_rotation": pose["rotation"],
            },
            allow_pickle=True,
        )
    print(f"Prepared {len(samples)} samples in {output}")


if __name__ == "__main__":
    main()

