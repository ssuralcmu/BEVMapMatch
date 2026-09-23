# Model manifest

| File | Purpose | Size (bytes) | SHA-256 |
|---|---|---:|---|
| `bevfusion_segmentation_baseline.pth` | BEVFusion semantic segmentation baseline | 482376381 | `6be4a6879aed3b1d195c6af1b617449bc627b8cdfb65107517ca76b44f7f2133` |
| `coarse_1frame_caf.pth` | One-frame coarse retrieval | 1270724327 | `614159460a5c755273c1013d709e4a846851c35b8c0f52249f98a8adcea6ca7e` |
| `coarse_2frame_caf.pth` | Two-frame coarse retrieval | 1295933519 | `efabc54aa2ab44c552d9a452c7e80991683e58f5b80cfab66ba058a5fa66e044` |
| `coarse_4frame_caf.pth` | Four-frame coarse retrieval | 1295933775 | `f77c54774a3334d94fa36733798f4d70e18efb97b630a943430b25033dfccbf7` |
| `coarse_8frame_caf.pth` | Eight-frame coarse retrieval | 1295931727 | `7c9ae6bb326960fea9312c1d3d9c0ace3d1119e5ee9dc8bb2d269113e3de3372` |
| `matchanything_eloftr.ckpt` | MatchAnything EfficientLoFTR fine alignment | 64366723 | `b394fd368488e5e3136906daac825fadd3dd5376755dcdf5cb6ede0de20f6e7e` |
| `unitr_map_lss.pth` | UniTR+LSS CAF BEV map segmentation | 483299733 | `96f45ebccadb7110211ba33f6baa2ac4c05699cc66943ed458b8c1464ecad509` |

Download: https://drive.google.com/drive/folders/1p133pqV2i6RiZHF30qR6LCDdoSuAWILv

Inference examples:

```bash
python scripts/infer_coarse.py --config configs/paper.yaml --checkpoint models/coarse_1frame_caf.pth --neighbor-frames 0 --output outputs/coarse_1frame_predictions.json
python scripts/infer_coarse.py --config configs/paper.yaml --checkpoint models/coarse_2frame_caf.pth --neighbor-frames 1 --output outputs/coarse_2frame_predictions.json
python scripts/infer_coarse.py --config configs/paper.yaml --checkpoint models/coarse_4frame_caf.pth --neighbor-frames 3 --output outputs/coarse_4frame_predictions.json
python scripts/infer_coarse.py --config configs/paper.yaml --checkpoint models/coarse_8frame_caf.pth --neighbor-frames 7 --output outputs/coarse_8frame_predictions.json
```
