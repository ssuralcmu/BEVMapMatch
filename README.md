# BEVMapMatch - clean reproduction code

This repository isolates the code path used for the BEVMapMatch paper results:

1. Generate 100 m x 100 m BEV semantic maps with CAF/UniTR.
2. Retrieve one cell from a 10 x 10 grid over a 500 m x 500 m basemap using a
   frozen DINOv2-Large encoder and 8-head cross-attention.
3. Expand the prediction to its 3 x 3 neighborhood.
4. Align the generated BEV map inside that crop with MatchAnything's
   EfficientLoFTR model and a RANSAC homography.
5. Report exact-cell, 3 x 3, and absolute localization recall metrics.

The code was distilled from the newest paper-specific internal scripts rather
than copied from the older public README. See `docs/PROVENANCE.md` for the exact
mapping and known reproducibility boundary.

## Installation

Python 3.10+ and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

DINOv2 weights are downloaded by Hugging Face on first use. For the final
alignment stage, clone and install MatchAnything separately, then use its
`matchanything_eloftr` matcher. This project deliberately does not vendor that
third-party repository or its weights.

## Data layout

Set `data.root` in `configs/paper.yaml`. The expected prepared-data layout is:

```text
data/processed/
├── all_train_metas_v3_modelpred/*_metas.npy
├── all_train_basemaps_segmented_v3_modelpred/*_base_map_image.png
├── all_train_maps_segmented_v3_modelpred_UNITR/*.png
├── all_val_metas_v3_modelpred/*_metas.npy
├── all_val_basemaps_segmented_v3_modelpred/*_base_map_image.png
└── all_val_maps_segmented_v3_modelpred_UNITR/*.png
```

Each metadata file must be a pickled NumPy dictionary containing
`perturbation: [dx_m, dy_m]`; timestamps and scene IDs are strongly recommended
for multi-frame experiments. Basemaps represent 500 m x 500 m. Generated maps
are CAF/UniTR semantic segmentation outputs for the local 100 m x 100 m region.

The original experiments used 28,130 training and 6,019 validation NuScenes
samples with independent X/Y perturbations in [-200 m, 200 m].

## Coarse retrieval

Paper defaults are already in `configs/paper.yaml`. `neighbor_frames: 0`, `1`,
`3`, and `7` correspond to total frame counts of 1, 2, 4, and 8.

```bash
python scripts/train_coarse.py \
  --config configs/paper.yaml \
  --output checkpoints/coarse-4frame.pt

python scripts/infer_coarse.py \
  --config configs/paper.yaml \
  --checkpoint checkpoints/coarse-4frame.pt \
  --output outputs/coarse-4frame.json
```

The loss is the paper's BCE-with-logits plus a Gaussian distance-aware soft
cross-entropy term (`weight=0.1`, `sigma=0.8` grid cells).

## Fine alignment

Prepare the predicted cell plus its immediate neighbors, normalized to the
paper's 300 x 300 fine-matching input:

```bash
python scripts/prepare_fine_pairs.py outputs/coarse-4frame.json \
  --output outputs/fine_pairs
```

Then run MatchAnything/EfficientLoFTR (the supplied path is a separate checkout):

```bash
python scripts/run_fine_matchanything.py outputs/fine_pairs/manifest.json \
  --matchanything-root /path/to/MatchAnything \
  --output outputs/distance_errors.json
```

The matcher uses these paper settings:

```text
matcher: matchanything_eloftr
geometry: Homography
match threshold: 0.1
keypoint threshold: 0.015
maximum features: 1000
RANSAC reprojection threshold: 8.0
```

The script projects the query center through the estimated homography,
automatically handles transform direction, and converts pixels to meters at the
paper's 2 pixels/m scale.

## Metrics

```bash
python scripts/evaluate.py outputs/distance_errors.json
```

This reports absolute recall at 1, 2, 5 and 10 m plus mean and median error.
The paper's headline 4-frame CAF results are 53.19% exact-cell accuracy, 88.90%
3 x 3 accuracy, 39.82% Recall@1m and 54.04% Recall@2m.

## Validation

```bash
pytest -q
python -m compileall -q src scripts
```

## Not included

- NuScenes data or derived prepared samples
- CAF/UniTR, coarse matcher, DINOv2 or MatchAnything weights
- Generated outputs and plots
- The legacy BEVFusion/MMDetection3D training tree

These are intentionally excluded so the repository contains clean source code
instead of datasets, binary checkpoints and historical experiments.
