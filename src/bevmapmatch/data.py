from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class Sample:
    segmentation: Path
    basemap: Path
    metadata: Path
    sequence: str
    timestamp: float | None


def _timestamp_seconds(value: float) -> float:
    magnitude = abs(value)
    if magnitude >= 1e17:
        return value * 1e-9
    if magnitude >= 1e14:
        return value * 1e-6
    if magnitude >= 1e11:
        return value * 1e-3
    return value


def perturbation_to_pixel(
    perturbation: np.ndarray | list[float], center_x: float, center_y: float, pixels_per_meter: float
) -> tuple[float, float]:
    """Convert the map perturbation vector to image coordinates."""
    return (
        center_x - float(perturbation[1]) * pixels_per_meter,
        center_y - float(perturbation[0]) * pixels_per_meter,
    )


class MapDataset(Dataset):
    """Pairs generated BEV segmentations, 500 m basemaps and metadata.

    Files are joined by the stem before the dataset suffix. Temporal neighbors
    are selected only inside the same sequence, avoiding accidental scene leaks.
    """

    def __init__(
        self,
        metadata_dir: str | Path,
        basemap_dir: str | Path,
        segmentation_dir: str | Path,
        *,
        image_size: int = 224,
        map_size_m: float = 500.0,
        grid_dim: int = 10,
        neighbor_frames: int = 0,
        neighbor_window_s: float = 2.0,
    ) -> None:
        self.metadata_dir = Path(metadata_dir)
        self.basemap_dir = Path(basemap_dir)
        self.segmentation_dir = Path(segmentation_dir)
        self.image_size = image_size
        self.map_size_m = map_size_m
        self.grid_dim = grid_dim
        self.neighbor_frames = neighbor_frames
        self.neighbor_window_s = neighbor_window_s
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.samples = self._discover()
        self.neighbors = self._build_neighbors()

    @staticmethod
    def _key(path: Path, suffix: str) -> str:
        return path.name[: -len(suffix)]

    @staticmethod
    def _metadata(path: Path) -> dict:
        return np.load(path, allow_pickle=True).item()

    def _discover(self) -> list[Sample]:
        meta = {self._key(p, "_metas.npy"): p for p in self.metadata_dir.glob("*_metas.npy")}
        bases = {
            self._key(p, "_base_map_image.png"): p
            for p in self.basemap_dir.glob("*_base_map_image.png")
        }
        generated = {}
        for path in self.segmentation_dir.glob("*.png"):
            suffix = "_generated_map_image.png"
            key = self._key(path, suffix) if path.name.endswith(suffix) else path.stem
            generated[key] = path

        samples: list[Sample] = []
        for key in sorted(generated.keys() & bases.keys() & meta.keys()):
            values = self._metadata(meta[key])
            raw_time = next(
                (values[name] for name in ("timestamp", "sample_timestamp", "time", "ts") if name in values),
                None,
            )
            timestamp = _timestamp_seconds(float(raw_time)) if raw_time is not None else None
            sequence = str(
                next(
                    (values[name] for name in ("scene_id", "sequence_id", "log_id", "route_id") if name in values),
                    key.rsplit("-", 1)[0],
                )
            )
            samples.append(Sample(generated[key], bases[key], meta[key], sequence, timestamp))
        if not samples:
            raise FileNotFoundError("No matched segmentation/basemap/metadata triplets found")
        return samples

    def _build_neighbors(self) -> list[list[int]]:
        by_sequence: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for index, sample in enumerate(self.samples):
            if sample.timestamp is not None:
                by_sequence[sample.sequence].append((sample.timestamp, index))
        result: list[list[int]] = []
        for index, sample in enumerate(self.samples):
            candidates = []
            if sample.timestamp is not None:
                for timestamp, other in by_sequence[sample.sequence]:
                    delta = abs(timestamp - sample.timestamp)
                    if other != index and delta <= self.neighbor_window_s:
                        candidates.append((delta, other))
            candidates.sort()
            chosen = [other for _, other in candidates[: self.neighbor_frames]]
            chosen += [index] * (self.neighbor_frames - len(chosen))
            result.append(chosen)
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        segmentation = self.transform(Image.open(sample.segmentation).convert("RGB"))
        basemap = self.transform(Image.open(sample.basemap).convert("RGB"))
        neighbors = torch.stack(
            [self.transform(Image.open(self.samples[i].segmentation).convert("RGB")) for i in self.neighbors[index]]
        ) if self.neighbor_frames else torch.empty((0, 3, self.image_size, self.image_size))

        values = self._metadata(sample.metadata)
        x, y = perturbation_to_pixel(
            values["perturbation"], self.image_size / 2, self.image_size / 2, self.image_size / self.map_size_m
        )
        cell = self.image_size / self.grid_dim
        col = int(np.clip(x // cell, 0, self.grid_dim - 1))
        row = int(np.clip(y // cell, 0, self.grid_dim - 1))
        label = torch.zeros(self.grid_dim * self.grid_dim)
        label[row * self.grid_dim + col] = 1.0
        return {
            "segmentation": segmentation,
            "neighbors": neighbors,
            "basemap": basemap,
            "label": label,
            "id": sample.metadata.stem.removesuffix("_metas"),
            "metadata_path": str(sample.metadata),
            "segmentation_path": str(sample.segmentation),
            "basemap_path": str(sample.basemap),
        }
