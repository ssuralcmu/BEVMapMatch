#!/usr/bin/env python3
"""Fix orientation of UniTR stitched outputs in-place.

For every file matching ``*_metas_stitched.png`` in the target directory,
this script applies:
1) 90-degree clockwise rotation
2) left-to-right mirror

The transformed image is written back to the same path.
"""

from __future__ import annotations
from tqdm import tqdm
import argparse
from pathlib import Path


def fix_image_in_place(image_path: Path, image_module) -> None:
    with image_module.open(image_path) as img:
        # Pillow compatibility: newer versions use Image.Transpose.*, older use Image.ROTATE_*/FLIP_*
        try:
            rotate_270 = image_module.Transpose.ROTATE_270   # 90 deg clockwise
            flip_lr = image_module.Transpose.FLIP_LEFT_RIGHT
        except AttributeError:
            rotate_270 = image_module.ROTATE_270
            flip_lr = image_module.FLIP_LEFT_RIGHT

        fixed = img.transpose(rotate_270).transpose(flip_lr)
        fixed.save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply 90-degree clockwise rotation and left-right mirroring "
            "to all *_metas_stitched.png files in a folder (in-place)."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing *_metas_stitched.png files.",
    )
    args = parser.parse_args()

    target_dir = args.directory.expanduser().resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {target_dir}")

    files = sorted(target_dir.glob("*_metas_stitched.png"))
    if not files:
        print(f"No matching files found in: {target_dir}")
        return

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required to run this script. Install with: pip install pillow"
        ) from exc

    for image_path in tqdm(files, desc="Fixing image orientations"):
        fix_image_in_place(image_path, Image)

    print(f"Updated {len(files)} files in-place in: {target_dir}")


if __name__ == "__main__":
    main()