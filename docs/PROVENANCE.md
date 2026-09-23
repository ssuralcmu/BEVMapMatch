# Code provenance and paper mapping

This repository was distilled from commit `fc99400` (2026-06-14) of the
internal checkout at `/home/rtml/shounak_research/bevfusion`. Its configured
remote was `https://github.com/ssuralcmu/BEVMapMatch.git`.

The paper is *BEVMapMatch: Multimodal BEV Neural Map Matching for Robust
Re-Localization of Autonomous Vehicles* (arXiv:2603.25963v1).

## Internal sources used

| Paper component | Internal source examined | Clean implementation |
|---|---|---|
| 1/2/4/8-frame coarse matching | `scripts_2026_paper/neural_map_matcher_v9_faster_dinov2_large_extraloss_trainon_modelpred_UNITR_k_frames_input_consume_together.py` | `data.py`, `model.py`, `losses.py`, training/inference scripts |
| Backbone ablation | `scripts_2026_paper/ablations/ablation_{resnet34,resnet50,dino_base,dinov3}.py` | backbone is a configuration value |
| 3x3 crop generation | `scripts_2026_fine/neural_fine_matcher_v2_3x3crop_unitr_k_frames.py` | coarse prediction JSON plus documented crop contract |
| EfficientLoFTR/MatchAnything matching | `match_anything_inference/MatchAnything/map_matcher_shounak_full_evaluate_parallel.py` | `fine.py`; external matcher remains an explicit dependency |
| Tables IV/V and Figures 3/4 | `scripts_2026_fine/analyze_*.py` | `metrics.py` and `scripts/evaluate.py` |

## Intentional cleanup

- Absolute machine paths were replaced by one YAML configuration.
- Duplicated 900-1,300 line variants were consolidated.
- Temporal neighbors are constrained to the same sequence. The internal script
  searched a global timestamp list, which could pair adjacent scenes.
- The paper metrics are named directly: exact cell is `top_1x1`; Chebyshev
  distance at most one cell is `top_3x3`.
- Checkpoints, datasets, generated figures and result JSON are excluded.
- The segmentation stage remains compatible with CAF/UniTR outputs rather than
  vendoring the full legacy MMDetection3D stack.

## Reproducibility boundary

The internal repository did not contain a single end-to-end command. It used
pre-generated CAF/UniTR segmentation PNGs, metadata arrays, basemap PNGs,
separate coarse checkpoints, and a nested MatchAnything checkout. Therefore an
exact numerical reproduction additionally needs those artifacts and model
weights. This clean repository makes that boundary explicit instead of claiming
that raw NuScenes can be reproduced with one command.

