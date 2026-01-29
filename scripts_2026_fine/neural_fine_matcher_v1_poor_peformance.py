# coarse_to_fine_map_matcher.py
#
# What this script does (INFER MODE):
# 1) loads your coarse checkpoint
# 2) runs coarse inference -> top-1 grid cell (10x10)
# 3) takes the 3x3 neighborhood on the BASemap => 150x150 crop (3 cells * 50px if basemap is 500px wide)
# 4) takes a 100x100 CENTER crop of the GENERATED map (stitched)  (assumes center is the anchor)
# 5) runs "Baseline 1": rotation sweep + phase correlation + DT (Chamfer-like) verification
# 6) writes JSON with coarse + fine results + optional debug images
#
# Notes/assumptions:
# - Your "1 pixel = 1 meter" implies basemap is ~500x500 for 500m, generated is ~100x100 for 100m.
# - We compute cell_size_px from the *raw* basemap image width / GRID_DIM.
# - If crops hit image boundaries, we pad with white.
#
# Dependencies:
# - Requires OpenCV: pip install opencv-python
#
# Usage example:
#   python coarse_to_fine_map_matcher.py \
#     --mode infer --checkpoint /path/to/best_map_location_model_val_....pth \
#     --infer_split val --batch_size 16 \
#     --fine_out fine_outputs.json --fine_viz --fine_viz_dir viz_fine

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from torchvision.ops import sigmoid_focal_loss
from transformers import AutoModel

import matplotlib.pyplot as plt

# ---- OpenCV (required for Baseline 1) ----
try:
    import cv2
except Exception as e:
    cv2 = None

# --------------------
# Globals / constants
# --------------------
GRID_DIM = 10
TARGET_BLOCK = 1
POSITIVE_CELLS = TARGET_BLOCK * TARGET_BLOCK
BASEMAP_INPUT_SIZE = 224  # coarse model input size

# Fine stage crop sizes (pixel=meter in your setup)
FINE_BASE_CROP_SIZE = 150  # basemap crop = 3x3 neighborhood => 150x150 (if cell size is 50)
FINE_GEN_CROP_SIZE  = 100  # generated crop = 100x100 (center crop)

# Fine stage search params
THETA_COARSE_STEP_DEG = 5.0
THETA_REFINE_STEP_DEG = 1.0
THETA_REFINE_WINDOW_DEG = 6.0  # refine around best +/- window
CANNY1, CANNY2 = 50, 150


def perturbation_to_pixel(perturbation, center_x, center_y, pixels_per_meter):
    return (
        center_x - perturbation[1] * pixels_per_meter,
        center_y - perturbation[0] * pixels_per_meter,
    )


# --------------------
# Dataset
# --------------------
class MapDataset(Dataset):
    def __init__(self, metas_folder, basemap_folder, stitched_folder, transform_base=None, transform_gen=None):
        self.basemap_folder = basemap_folder
        self.stitched_folder = stitched_folder
        self.metas_folder = metas_folder
        self.transform_base = transform_base
        self.transform_gen = transform_gen

        stitched_files = os.listdir(stitched_folder)
        self.file_triplets = []
        for stitched_file in stitched_files:
            if stitched_file.endswith("_generated_map_image.png"):
                prefix = stitched_file.split("_generated_map_image.png")[0]
                basemap_file = f"{prefix}_base_map_image.png"
                metas_file = f"{prefix}_metas.npy"
                if os.path.exists(os.path.join(basemap_folder, basemap_file)) and \
                   os.path.exists(os.path.join(metas_folder, metas_file)):
                    self.file_triplets.append((stitched_file, basemap_file, metas_file))

    def __len__(self):
        return len(self.file_triplets)

    def __getitem__(self, idx):
        stitched_file, basemap_file, metas_file = self.file_triplets[idx]

        stitched_img_path = os.path.join(self.stitched_folder, stitched_file)
        basemap_img_path = os.path.join(self.basemap_folder, basemap_file)
        metas_path = os.path.join(self.metas_folder, metas_file)

        stitched_img = Image.open(stitched_img_path).convert("RGB")
        basemap_img = Image.open(basemap_img_path).convert("RGB")

        if self.transform_gen:
            stitched_img_t = self.transform_gen(stitched_img)
        else:
            stitched_img_t = transforms.ToTensor()(stitched_img)

        if self.transform_base:
            basemap_img_t = self.transform_base(basemap_img)
        else:
            basemap_img_t = transforms.ToTensor()(basemap_img)

        metas = np.load(metas_path, allow_pickle=True).item()

        # Convert coordinates to grid labels using the *coarse* basemap tensor size (224)
        center_x, center_y = basemap_img_t.shape[1] // 2, basemap_img_t.shape[2] // 2
        basemap_size_px = basemap_img_t.shape[1]
        meters_per_patch = 500.0
        pixels_per_meter = basemap_size_px / meters_per_patch

        x_val, y_val = perturbation_to_pixel(
            metas["perturbation"], center_x, center_y, pixels_per_meter
        )

        grid_size = basemap_size_px / GRID_DIM
        grid_x = int(x_val // grid_size)
        grid_y = int(y_val // grid_size)

        # single-cell label (for exact top1)
        single_cell_label = torch.zeros(GRID_DIM, GRID_DIM)
        grid_x_cl = int(np.clip(grid_x, 0, GRID_DIM - 1))
        grid_y_cl = int(np.clip(grid_y, 0, GRID_DIM - 1))
        single_cell_label[grid_y_cl, grid_x_cl] = 1.0

        # target block label (TARGET_BLOCK x TARGET_BLOCK)
        grid_label = torch.zeros(GRID_DIM, GRID_DIM)
        max_anchor = GRID_DIM - TARGET_BLOCK
        i0 = min(grid_y_cl, max_anchor)
        j0 = min(grid_x_cl, max_anchor)
        grid_label[i0:i0 + TARGET_BLOCK, j0:j0 + TARGET_BLOCK] = 1.0

        return (
            stitched_img_t,
            basemap_img_t,
            grid_label.flatten().float(),
            single_cell_label.flatten().float(),
            stitched_img_path,
            basemap_img_path,
            metas_path,
        )


# --------------------
# Model (same as yours)
# --------------------
class DINOv2Backbone(nn.Module):
    """
    DINOv2 via HuggingFace (py3.8 friendly).
    Returns patch embeddings as a spatial feature map (B, D, H, W).
    """
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, x):
        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:]  # (B, N, D)
        batch, num_tokens, dim = tokens.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise ValueError(f"Expected square number of tokens, got {num_tokens}.")
        return tokens.transpose(1, 2).reshape(batch, dim, side, side)


class GridClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = DINOv2Backbone(model_name="facebook/dinov2-large")
        self.embed_dim = self.feature_extractor.embed_dim

        self.grid_pool = nn.AdaptiveAvgPool2d((10, 10))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            batch_first=True,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 100, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.attn_mask = None
        self.fc = nn.Linear(self.embed_dim, 100)

    def forward(self, stitched, basemap):
        stitched_feat = self.feature_extractor(stitched)
        basemap_feat = self.feature_extractor(basemap)

        grid = self.grid_pool(basemap_feat)  # (B, D, 10, 10)

        grid_3x3 = F.unfold(grid, kernel_size=3, padding=1)  # (B, D*9, 100)
        grid_3x3 = grid_3x3.view(grid_3x3.size(0), self.embed_dim, 9, 100).mean(dim=2)  # (B, D, 100)

        query = F.adaptive_avg_pool2d(stitched_feat, (1, 1)).flatten(1).unsqueeze(1)  # (B, 1, D)
        key_value = grid_3x3.permute(0, 2, 1) + self.pos_embed  # (B, 100, D)

        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=self.attn_mask,
        )

        scores = self.fc(attn_out.squeeze(1))  # (B, 100)
        return scores


# --------------------
# Utilities
# --------------------
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12800"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def create_dataloader(rank, world_size, dataset, batch_size=16, num_workers=10, prefetch_factor=2):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# --------------------
# Fine matching helpers (Baseline 1)
# --------------------
def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for the fine baseline. Install with: pip install opencv-python"
        )


def crop_with_padding_rgb(img_rgb: np.ndarray, x0: int, y0: int, w: int, h: int, pad_value: int = 255):
    """
    img_rgb: HxWx3 uint8
    Returns: crop_rgb (h,w,3), info dict with:
      - x0_req,y0_req,w,h
      - x0_src,y0_src,x1_src,y1_src (clamped)
      - pad_left, pad_top
    """
    H, W = img_rgb.shape[:2]
    x1 = x0 + w
    y1 = y0 + h

    x0_src = max(0, x0)
    y0_src = max(0, y0)
    x1_src = min(W, x1)
    y1_src = min(H, y1)

    crop = np.full((h, w, 3), pad_value, dtype=np.uint8)

    pad_left = x0_src - x0
    pad_top = y0_src - y0

    crop_y0 = pad_top
    crop_x0 = pad_left
    crop_y1 = crop_y0 + (y1_src - y0_src)
    crop_x1 = crop_x0 + (x1_src - x0_src)

    if (x1_src > x0_src) and (y1_src > y0_src):
        crop[crop_y0:crop_y1, crop_x0:crop_x1] = img_rgb[y0_src:y1_src, x0_src:x1_src]

    info = dict(
        x0_req=int(x0),
        y0_req=int(y0),
        w=int(w),
        h=int(h),
        x0_src=int(x0_src),
        y0_src=int(y0_src),
        x1_src=int(x1_src),
        y1_src=int(y1_src),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
    )
    return crop, info


def resize_rgb(img_rgb: np.ndarray, out_size: int, pad_value: int = 255):
    """
    Resize the *entire* image to (out_size, out_size).
    Returns resized image and a small info dict.
    """
    _require_cv2()
    H, W = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    info = {
        "method": "resize_full_image",
        "orig_hw": [int(H), int(W)],
        "out_hw": [int(out_size), int(out_size)],
    }
    return resized, info



def rgb_to_structure(img_rgb: np.ndarray):
    """
    Convert segmentation-style RGB to a "structure channel" for matching.
    We do: grayscale -> canny edges.
    Returns:
      edge_uint8: HxW uint8 in {0,255}
      edge_bin:   HxW uint8 in {0,1}
      gray:       HxW uint8
    """
    _require_cv2()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    # connect small gaps a bit
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edge_bin = (edges > 0).astype(np.uint8)
    return edges, edge_bin, gray


def distance_transform_from_edges(edge_bin: np.ndarray):
    """
    edge_bin: 0/1 uint8
    returns DT float32 where DT[y,x] = distance to nearest edge pixel.
    """
    _require_cv2()
    # cv2.distanceTransform expects non-zero pixels as "foreground" to compute dist to zero pixels.
    # We want distance to edges => invert: edges=1 => zeros, background=0 => ones
    inv = (1 - edge_bin).astype(np.uint8) * 255
    dt = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    return dt.astype(np.float32)


def rotate_image(img: np.ndarray, angle_deg: float, out_size=None, border_value=0):
    """
    Rotate around center. If out_size is None, keeps same size.
    """
    _require_cv2()
    H, W = img.shape[:2]
    if out_size is None:
        out_w, out_h = W, H
    else:
        out_w, out_h = int(out_size[0]), int(out_size[1])

    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    interp = cv2.INTER_NEAREST if img.dtype == np.uint8 else cv2.INTER_LINEAR
    rotated = cv2.warpAffine(
        img,
        M,
        (out_w, out_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


def phase_corr_shift(a_f32: np.ndarray, b_f32: np.ndarray):
    """
    Return shift (dx,dy) so that shifting b aligns to a, using cv2.phaseCorrelate.
    a_f32, b_f32: float32, same shape.
    """
    _require_cv2()
    (dx, dy), resp = cv2.phaseCorrelate(a_f32, b_f32)
    return float(dx), float(dy), float(resp)


def shift_image(img: np.ndarray, dx: float, dy: float, border_value=0):
    _require_cv2()
    H, W = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    interp = cv2.INTER_NEAREST if img.dtype == np.uint8 else cv2.INTER_LINEAR
    out = cv2.warpAffine(
        img,
        M,
        (W, H),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return out


def dt_score(dt_base: np.ndarray, edge_bin_aligned: np.ndarray):
    """
    Lower is better.
    dt_base: float32 HxW
    edge_bin_aligned: 0/1 uint8 HxW
    """
    pts = edge_bin_aligned.astype(bool)
    if pts.sum() < 10:
        return float("inf")
    return float(dt_base[pts].mean())


def fine_match_baseline1(
    basemap_crop_150_rgb: np.ndarray,
    gen_crop_100_rgb: np.ndarray,
    theta_coarse_step=THETA_COARSE_STEP_DEG,
    theta_refine_step=THETA_REFINE_STEP_DEG,
    theta_refine_window=THETA_REFINE_WINDOW_DEG,
):
    """
    Baseline 1:
    - edges + DT on basemap crop
    - for each theta: rotate gen edges (padded to 150) -> phase correlation -> shift -> DT score
    Returns dict with best dx,dy,theta, scores and debug images (optional).
    """
    _require_cv2()

    # structure channels
    _, base_edge_bin, _ = rgb_to_structure(basemap_crop_150_rgb)
    _, gen_edge_bin_100, _ = rgb_to_structure(gen_crop_100_rgb)

    # DT on base
    dt_base = distance_transform_from_edges(base_edge_bin)

    # Prepare a 150x150 canvas for gen
    H, W = base_edge_bin.shape
    assert (H, W) == (FINE_BASE_CROP_SIZE, FINE_BASE_CROP_SIZE), (H, W)

    def place_gen_on_center_canvas(gen_100_bin):
        canvas = np.zeros((H, W), dtype=np.uint8)
        y0 = (H - FINE_GEN_CROP_SIZE) // 2
        x0 = (W - FINE_GEN_CROP_SIZE) // 2
        canvas[y0:y0 + FINE_GEN_CROP_SIZE, x0:x0 + FINE_GEN_CROP_SIZE] = gen_100_bin
        return canvas

    # Convert base to float32 for phase correlation
    base_f32 = base_edge_bin.astype(np.float32)

    best = dict(score=float("inf"), theta=None, dx=None, dy=None, resp=None)

    # coarse sweep
    thetas = np.arange(0.0, 360.0, float(theta_coarse_step), dtype=np.float32)
    for th in thetas:
        # rotate gen in 100x100, then place into 150x150, then rotate whole canvas? (better: rotate after placement)
        gen_placed = place_gen_on_center_canvas(gen_edge_bin_100)
        gen_rot = rotate_image(gen_placed, float(th), out_size=(W, H), border_value=0)

        gen_f32 = gen_rot.astype(np.float32)
        dx, dy, resp = phase_corr_shift(base_f32, gen_f32)

        # apply shift to gen_rot and score by DT
        gen_aligned = shift_image(gen_rot, dx, dy, border_value=0)
        s = dt_score(dt_base, gen_aligned.astype(np.uint8))

        if s < best["score"]:
            best = dict(score=s, theta=float(th), dx=float(dx), dy=float(dy), resp=float(resp))

    # refine sweep around best theta
    th0 = best["theta"]
    refine_thetas = np.arange(
        th0 - theta_refine_window,
        th0 + theta_refine_window + 1e-6,
        float(theta_refine_step),
        dtype=np.float32,
    )
    for th in refine_thetas:
        thn = float(th % 360.0)
        gen_placed = place_gen_on_center_canvas(gen_edge_bin_100)
        gen_rot = rotate_image(gen_placed, thn, out_size=(W, H), border_value=0)
        gen_f32 = gen_rot.astype(np.float32)

        dx, dy, resp = phase_corr_shift(base_f32, gen_f32)
        gen_aligned = shift_image(gen_rot, dx, dy, border_value=0)
        s = dt_score(dt_base, gen_aligned.astype(np.uint8))

        if s < best["score"]:
            best = dict(score=s, theta=thn, dx=float(dx), dy=float(dy), resp=float(resp))

    # compute predicted center location in basemap crop coordinates
    # center of the gen is the center of the 150x150 canvas, then shifted by (dx,dy)
    cx = (W / 2.0)
    cy = (H / 2.0)
    pred_center_x = cx + best["dx"]
    pred_center_y = cy + best["dy"]

    return {
        "theta_deg": best["theta"],
        "dx_px": best["dx"],
        "dy_px": best["dy"],
        "phase_corr_resp": best["resp"],
        "dt_score": best["score"],
        "pred_center_in_150": [float(pred_center_x), float(pred_center_y)],
    }


def save_fine_debug_images(
    out_dir: Path,
    sample_id: str,
    base150_rgb: np.ndarray,
    gen100_rgb: np.ndarray,
    fine_result: dict,
):
    _require_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)

    # rebuild aligned gen overlay for visualization
    _, base_edge_bin, _ = rgb_to_structure(base150_rgb)
    _, gen_edge_bin_100, _ = rgb_to_structure(gen100_rgb)

    H, W = base_edge_bin.shape
    canvas = np.zeros((H, W), dtype=np.uint8)
    y0 = (H - FINE_GEN_CROP_SIZE) // 2
    x0 = (W - FINE_GEN_CROP_SIZE) // 2
    canvas[y0:y0 + FINE_GEN_CROP_SIZE, x0:x0 + FINE_GEN_CROP_SIZE] = gen_edge_bin_100

    gen_rot = rotate_image(canvas, fine_result["theta_deg"], out_size=(W, H), border_value=0)
    gen_aligned = shift_image(gen_rot, fine_result["dx_px"], fine_result["dy_px"], border_value=0)

    # overlay edges: base edges in red, aligned gen edges in blue
    overlay = np.stack([base_edge_bin * 255, np.zeros_like(base_edge_bin), gen_aligned * 255], axis=-1).astype(np.uint8)

    # mark predicted center
    px, py = fine_result["pred_center_in_150"]
    overlay2 = overlay.copy()
    cv2.circle(overlay2, (int(round(px)), int(round(py))), 3, (0, 255, 0), -1)

    Image.fromarray(base150_rgb).save(out_dir / f"{sample_id}_base150.png")
    Image.fromarray(gen100_rgb).save(out_dir / f"{sample_id}_gen100.png")
    Image.fromarray(overlay2).save(out_dir / f"{sample_id}_overlay_edges.png")


# --------------------
# Coarse viz (kept)
# --------------------
def visualize_pred_gt_grid(pred_mask_10x10, gt_mask_10x10, save_path):
    if torch.is_tensor(pred_mask_10x10):
        pred = pred_mask_10x10.detach().cpu().numpy()
    else:
        pred = pred_mask_10x10
    if torch.is_tensor(gt_mask_10x10):
        gt = gt_mask_10x10.detach().cpu().numpy()
    else:
        gt = gt_mask_10x10

    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    img = np.ones((10, 10, 3), dtype=np.float32)
    overlap = (pred == 1) & (gt == 1)
    gt_only = (pred == 0) & (gt == 1)
    pred_only = (pred == 1) & (gt == 0)

    img[gt_only] = np.array([1.0, 0.2, 0.2])
    img[pred_only] = np.array([0.2, 0.2, 1.0])
    img[overlap] = np.array([0.6, 0.2, 0.8])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(img, interpolation="nearest")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.grid(True, linewidth=0.5)
    plt.title("Pred (Blue) vs GT (Red) | Overlap (Purple)")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    plt.close()


# --------------------
# Inference (coarse + fine)
# --------------------
def run_inference_with_fine(
    model,
    dataset,
    checkpoint_path,
    batch_size=64,
    num_workers=10,
    device=None,
    viz=False,
    viz_dir="viz_grids",
    output_json="inference_outputs.json",
    fine_out_json="fine_outputs.json",
    fine_viz=False,
    fine_viz_dir="viz_fine",
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    viz_dir = Path(viz_dir)
    fine_viz_dir = Path(fine_viz_dir)

    # Coarse metrics (kept)
    correct_topk = 0
    exact_top1_correct = 0
    topk_correct = {1: 0, 2: 0, 3: 0}
    total = 0
    total_iou = 0.0
    distance_values = []
    fine_error_values_m = []

    coarse_outputs = []
    fine_outputs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference (coarse + fine)")
        for stitched, basemap, labels, single_labels, stitched_img_path, basemap_img_path, metas_path in pbar:
            stitched = stitched.to(device, non_blocking=True)
            basemap = basemap.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            single_labels = single_labels.to(device, non_blocking=True)

            logits = model(stitched, basemap)  # (B,100)

            # coarse topk
            _, topk = torch.topk(logits, POSITIVE_CELLS, dim=1)
            batch_hits = (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
            correct_topk += batch_hits

            for k in topk_correct:
                _, topk_k = torch.topk(logits, k, dim=1)
                topk_correct[k] += (labels.gather(1, topk_k).sum(dim=1) > 0).sum().item()

            _, top1_idx = torch.topk(logits, 1, dim=1)
            exact_top1_correct += (single_labels.gather(1, top1_idx).sum(dim=1) > 0).sum().item()

            total += labels.size(0)

            pred_idx = top1_idx.squeeze(1)
            gt_idx = single_labels.argmax(dim=1)
            pred_row = pred_idx // GRID_DIM
            pred_col = pred_idx % GRID_DIM
            gt_row = gt_idx // GRID_DIM
            gt_col = gt_idx % GRID_DIM

            distances = torch.sqrt((pred_row - gt_row).float() ** 2 + (pred_col - gt_col).float() ** 2)
            distance_values.extend(distances.detach().cpu().tolist())

            # coarse IoU
            _, topk_iou = torch.topk(logits, POSITIVE_CELLS, dim=1)
            preds = torch.zeros_like(labels)
            preds.scatter_(1, topk_iou, 1)

            intersection = (preds * labels).sum(dim=1)
            union = ((preds + labels) > 0).sum(dim=1)
            iou_percentage = (intersection / union * 100.0).mean().item()
            total_iou += iou_percentage

            # coarse viz (optional)
            if viz:
                for b in range(labels.size(0)):
                    gt_10x10 = labels[b].view(10, 10)
                    pred_10x10 = preds[b].view(10, 10)
                    sample_id = Path(metas_path[b]).stem
                    save_path = viz_dir / f"{sample_id}_grid.png"
                    visualize_pred_gt_grid(pred_10x10, gt_10x10, save_path)

            # ---------- Fine stage ----------
            for b in range(labels.size(0)):
                sample_id = Path(metas_path[b]).stem

                # load RAW images (no resize, no normalization)
                base_rgb = np.array(Image.open(basemap_img_path[b]).convert("RGB"))
                gen_rgb  = np.array(Image.open(stitched_img_path[b]).convert("RGB"))

                metas = np.load(metas_path[b], allow_pickle=True).item()
                H0, W0 = base_rgb.shape[:2]
                center_x_raw, center_y_raw = (W0 / 2.0), (H0 / 2.0)
                meters_per_patch = 500.0
                pixels_per_meter_raw = W0 / meters_per_patch  # assumes square scale using width
                gt_x_raw, gt_y_raw = perturbation_to_pixel(
                    metas["perturbation"], center_x_raw, center_y_raw, pixels_per_meter_raw
                )

                H0, W0 = base_rgb.shape[:2]

                # compute cell size in raw basemap
                cell_w = W0 / GRID_DIM
                cell_h = H0 / GRID_DIM
                # assume square-ish; we use integer rounding per boundary
                # predicted coarse cell -> 3x3 neighborhood crop in basemap
                pr = int(pred_row[b].item())
                pc = int(pred_col[b].item())

                # 3x3 neighborhood bounds in cell coordinates: (pr-1..pr+1), (pc-1..pc+1)
                # convert to pixel bounds
                x0 = int(np.floor((pc - 1) * cell_w))
                y0 = int(np.floor((pr - 1) * cell_h))
                x1 = int(np.floor((pc + 2) * cell_w))
                y1 = int(np.floor((pr + 2) * cell_h))

                # Force exact 150x150 if your basemap is 500x500 (cell_w=50),
                # otherwise we take whatever cell geometry implies and then resize to 150.
                crop_w = x1 - x0
                crop_h = y1 - y0
                base_crop, crop_info = crop_with_padding_rgb(base_rgb, x0, y0, crop_w, crop_h, pad_value=255)

                # resize basemap crop to 150x150 for the fine matcher
                if cv2 is None:
                    raise RuntimeError("cv2 is required for fine stage. pip install opencv-python")
                base150 = cv2.resize(base_crop, (FINE_BASE_CROP_SIZE, FINE_BASE_CROP_SIZE), interpolation=cv2.INTER_NEAREST)

                # generated 100x100 crop: CENTER CROP (assumes its center is the anchor)
                gen100, gen_info = resize_rgb(gen_rgb, FINE_GEN_CROP_SIZE)

                fine_res = fine_match_baseline1(base150, gen100)

                # predicted center in *raw basemap* coordinates:
                # fine_res gives center in 150x150 coords.
                # map that back to base_crop coords (before resizing), then to full basemap.
                px150, py150 = fine_res["pred_center_in_150"]

                # convert 150->base_crop scale
                sx = base_crop.shape[1] / float(FINE_BASE_CROP_SIZE)
                sy = base_crop.shape[0] / float(FINE_BASE_CROP_SIZE)

                px_in_base_crop = px150 * sx
                py_in_base_crop = py150 * sy

                # base_crop top-left in full basemap is crop_info x0_req,y0_req with padding accounted for:
                # Our crop_with_padding constructed crop such that crop pixel (pad_left,pad_top) corresponds to (x0_src,y0_src).
                # So crop pixel (0,0) corresponds to (x0_req,y0_req) in "requested coords".
                # Thus full basemap coords of a point in crop are:
                full_x = crop_info["x0_req"] + px_in_base_crop
                full_y = crop_info["y0_req"] + py_in_base_crop

                fine_err_m = float(np.sqrt((full_x - gt_x_raw) ** 2 + (full_y - gt_y_raw) ** 2))
                fine_error_values_m.append(fine_err_m)

                fine_out = {
                    "sample_id": sample_id,
                    "basemap_img_path": basemap_img_path[b],
                    "stitched_img_path": stitched_img_path[b],
                    "metas_path": metas_path[b],
                    "coarse_pred_cell": int(pred_idx[b].item()),
                    "coarse_pred_rowcol": [int(pr), int(pc)],
                    "coarse_gt_cell": int(gt_idx[b].item()),
                    "fine_theta_deg": float(fine_res["theta_deg"]),
                    "fine_dx_px_in_150": float(fine_res["dx_px"]),
                    "fine_dy_px_in_150": float(fine_res["dy_px"]),
                    "fine_dt_score": float(fine_res["dt_score"]),
                    "fine_phase_resp": float(fine_res["phase_corr_resp"]),
                    "fine_pred_center_in_150": fine_res["pred_center_in_150"],
                    "fine_pred_center_in_full_basemap_xy": [float(full_x), float(full_y)],
                    "fine_gt_xy_full_basemap": [float(gt_x_raw), float(gt_y_raw)],
                    "fine_error_m": float(fine_err_m),
                    "basemap_crop_info": crop_info,
                    "gen_resize_info": gen_info,
                }
                fine_outputs.append(fine_out)


                if fine_viz:
                    save_fine_debug_images(fine_viz_dir, sample_id, base150, gen100, fine_res)

                coarse_outputs.append({
                    "sample_id": sample_id,
                    "pred_idx": int(pred_idx[b].item()),
                    "gt_idx": int(gt_idx[b].item()),
                    "distance_cells": float(distances[b].item()),
                    "topk_idx": topk[b].detach().cpu().numpy().tolist(),
                    "probs": torch.sigmoid(logits[b]).detach().cpu().numpy().tolist(),
                })

            # progress bar
            avg_acc = correct_topk / max(1, total)
            avg_iou = total_iou / max(1, (pbar.n + 1))
            avg_topk_acc = {k: topk_correct[k] / max(1, total) for k in topk_correct}
            avg_exact_top1_acc = exact_top1_correct / max(1, total)
            avg_distance = sum(distance_values) / max(1, len(distance_values))
            pbar.set_postfix({
                "topk_acc": f"{avg_acc:.3f}",
                "top1/2/3": f"{avg_topk_acc[1]:.3f}/{avg_topk_acc[2]:.3f}/{avg_topk_acc[3]:.3f}",
                "exact_top1": f"{avg_exact_top1_acc:.3f}",
                "iou%": f"{avg_iou:.2f}",
                "dist": f"{avg_distance:.2f}",
            })

    # write coarse summary JSON (kept similar)
    final_topk = correct_topk / max(1, total)
    final_topk_acc = {k: topk_correct[k] / max(1, total) for k in topk_correct}
    final_exact_top1_acc = exact_top1_correct / max(1, total)
    final_iou = total_iou / max(1, len(loader))
    mean_distance = sum(distance_values) / max(1, len(distance_values))


    # ---- Fine summary (global) ----
    if len(fine_error_values_m) > 0:
        arr = np.array(fine_error_values_m, dtype=np.float32)
        mean_fine_error_m = float(arr.mean())
        median_fine_error_m = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        print(f"[Fine] Mean error (m): {mean_fine_error_m:.2f}")
        print(f"[Fine] Median/P90/P95 (m): {median_fine_error_m:.2f}/{p90:.2f}/{p95:.2f}")
    else:
        print("[Fine] No fine errors computed.")


    with open(output_json, "w") as f:
        json.dump({
            "topk_acc": final_topk,
            "mean_iou_percent": final_iou,
            "mean_distance_cells": mean_distance,
            "topk_acc_1_2_3": final_topk_acc,
            "exact_top1_acc": final_exact_top1_acc,
            "coarse_outputs": coarse_outputs,
        }, f, indent=2)

    with open(fine_out_json, "w") as f:
        json.dump({
            "fine_outputs": fine_outputs,
            "fine_params": {
                "base_crop_size": FINE_BASE_CROP_SIZE,
                "gen_crop_size": FINE_GEN_CROP_SIZE,
                "theta_coarse_step": THETA_COARSE_STEP_DEG,
                "theta_refine_step": THETA_REFINE_STEP_DEG,
                "theta_refine_window": THETA_REFINE_WINDOW_DEG,
                "canny": [CANNY1, CANNY2],
            }
        }, f, indent=2)

    print(f"[Coarse] Top-{POSITIVE_CELLS} Acc: {final_topk:.2%}, Mean IoU%: {final_iou:.2f}")
    print(f"[Coarse] Top-1/2/3 Acc: {final_topk_acc[1]:.2%}/{final_topk_acc[2]:.2%}/{final_topk_acc[3]:.2%}, "
          f"Exact Top-1 Acc: {final_exact_top1_acc:.2%}")
    print(f"[Coarse] Mean distance (cells): {mean_distance:.2f}")
    print(f"[Coarse] Saved outputs to: {output_json}")
    print(f"[Fine]   Saved fine outputs to: {fine_out_json}")
    if fine_viz:
        print(f"[Fine]   Saved fine debug images to: {str(fine_viz_dir)}")


# --------------------
# Training (unchanged)
# --------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs, targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="mean"
        )


def train_model(rank, world_size, num_epochs, model, criterion, optimizer, scheduler,
               train_dataset, val_dataset, batch_size, lr, version, fraction,
               checkpoint_path, seed, validate_every, amp):
    setup(rank, world_size)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = nn.BCEWithLogitsLoss().to(device)

    if fraction >= 1.0:
        if checkpoint_path is None:
            start_epoch = 0
            losses = {"train": [], "val": []}
            subset_indices = torch.randperm(len(train_dataset))
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            losses = checkpoint["losses"]
            subset_indices = checkpoint["subset_indices"]
        else:
            raise ValueError("Checkpoint file not found")
    else:
        if checkpoint_path is None:
            num_samples = int(len(train_dataset) * fraction)
            subset_indices = torch.randperm(len(train_dataset))[:num_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
            start_epoch = 0
            losses = {"train": [], "val": []}
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            losses = checkpoint["losses"]
            subset_indices = checkpoint["subset_indices"]
            train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

    train_loader = create_dataloader(rank, world_size, train_dataset, batch_size)
    val_loader = create_dataloader(rank, world_size, val_dataset, batch_size)

    unique_name = f"grid_v{version}-lr{lr}-bs{batch_size}-frac{fraction}-seed{seed}"

    best_val_loss = min((loss for _, loss, _, _ in losses["val"]), default=float("inf"))
    best_train_loss = min((loss for _, loss, _, _ in losses["train"]), default=float("inf"))

    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            train_loader.sampler.set_epoch(epoch)
            total_loss = 0.0
            correct = 0
            topk_correct = {1: 0, 2: 0, 3: 0}
            exact_top1_correct = 0
            total = 0
            total_iou = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
            for stitched, basemap, labels, single_labels, *_ in pbar:
                stitched = stitched.to(device, non_blocking=True)
                basemap = basemap.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                single_labels = single_labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs = model(stitched, basemap)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

                _, topk = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                correct += (labels.gather(1, topk).sum(dim=1) > 0).sum().item()

                for k in topk_correct:
                    _, topk_k = torch.topk(outputs, k, dim=1)
                    topk_correct[k] += (labels.gather(1, topk_k).sum(dim=1) > 0).sum().item()

                _, top1_idx = torch.topk(outputs, 1, dim=1)
                exact_top1_correct += (single_labels.gather(1, top1_idx).sum(dim=1) > 0).sum().item()

                total += labels.size(0)

                _, topk_iou = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                predictions = torch.zeros_like(labels)
                predictions.scatter_(1, topk_iou, 1)

                intersection = (predictions * labels).sum(dim=1)
                union = ((predictions + labels) > 0).sum(dim=1)
                iou_percentage = (intersection / union * 100).mean().item()
                total_iou += iou_percentage

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            train_topk_acc = {k: topk_correct[k] / total for k in topk_correct}
            train_exact_top1_acc = exact_top1_correct / total
            train_iou = total_iou / len(train_loader)
            losses["train"].append([epoch, train_loss, train_acc, train_iou])

            if rank == 0:
                best_train_loss = min(best_train_loss, train_loss)
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "subset_indices": subset_indices,
                }
                torch.save(checkpoint, f"latest_map_location_model_train_{unique_name}.pth")

            # Validation
            if epoch % validate_every == 0 or epoch == start_epoch:
                model.eval()
                val_loss = 0.0
                correct = 0
                topk_correct = {1: 0, 2: 0, 3: 0}
                exact_top1_correct = 0
                total = 0
                total_iou = 0.0

                with torch.no_grad():
                    pbar = tqdm(val_loader, desc="Validation", disable=rank != 0)
                    for stitched, basemap, labels, single_labels, *_ in pbar:
                        stitched = stitched.to(device, non_blocking=True)
                        basemap = basemap.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        single_labels = single_labels.to(device, non_blocking=True)

                        with torch.cuda.amp.autocast(enabled=amp):
                            outputs = model(stitched, basemap)
                            val_loss += criterion(outputs, labels).item()

                        if rank == 0:
                            pbar.set_postfix({"val_loss": val_loss / (pbar.n + 1)})

                        _, topk = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                        correct += (labels.gather(1, topk).sum(dim=1) > 0).sum().item()

                        for k in topk_correct:
                            _, topk_k = torch.topk(outputs, k, dim=1)
                            topk_correct[k] += (labels.gather(1, topk_k).sum(dim=1) > 0).sum().item()

                        _, top1_idx = torch.topk(outputs, 1, dim=1)
                        exact_top1_correct += (single_labels.gather(1, top1_idx).sum(dim=1) > 0).sum().item()

                        total += labels.size(0)

                        _, topk_iou = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                        predictions = torch.zeros_like(labels)
                        predictions.scatter_(1, topk_iou, 1)

                        intersection = (predictions * labels).sum(dim=1)
                        union = ((predictions + labels) > 0).sum(dim=1)
                        iou_percentage = (intersection / union * 100).mean().item()
                        total_iou += iou_percentage

                val_loss /= len(val_loader)
                val_acc = correct / total
                val_topk_acc = {k: topk_correct[k] / total for k in topk_correct}
                val_exact_top1_acc = exact_top1_correct / total
                val_iou = total_iou / len(val_loader)
                losses["val"].append([epoch, val_loss, val_acc, val_iou])

                scheduler.step()

                if rank == 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "losses": losses,
                        "subset_indices": subset_indices,
                    }
                    torch.save(checkpoint, f"best_map_location_model_val_{unique_name}.pth")

                if rank == 0:
                    log_line = (
                        f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Train Loss: {train_loss:.4f}, Top-{POSITIVE_CELLS} Train Acc: {train_acc:.2%}, "
                        f"Top-1/2/3 Train Acc: {train_topk_acc[1]:.2%}/{train_topk_acc[2]:.2%}/{train_topk_acc[3]:.2%}, "
                        f"Val Loss: {val_loss:.4f}, Top-{POSITIVE_CELLS} Val Acc: {val_acc:.2%}, "
                        f"Top-1/2/3 Val Acc: {val_topk_acc[1]:.2%}/{val_topk_acc[2]:.2%}/{val_topk_acc[3]:.2%}, "
                        f"Train IoU: {train_iou:.2f}, Val IoU: {val_iou:.2f}, "
                        f"Exact Top-1 Train Acc: {train_exact_top1_acc:.2%}, "
                        f"Exact Top-1 Val Acc: {val_exact_top1_acc:.2%}"
                    )
                    print(log_line)
                    with open(f"loss_{unique_name}.txt", "a") as f:
                        f.write(log_line + "\n")

        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
        except Exception as e:
            print("Train loop exception:", e)

    cleanup()


# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="9_dinov2_faster_2by2")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument("--viz", action="store_true", help="Save 10x10 coarse Pred vs GT grid visualizations")
    parser.add_argument("--validate_every", type=int, default=1)
    parser.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision training/inference")
    parser.add_argument("--viz_dir", type=str, default="viz_grids")
    parser.add_argument("--infer_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--infer_out", type=str, default="inference_outputs.json")

    # fine args
    parser.add_argument("--fine_out", type=str, default="fine_outputs.json")
    parser.add_argument("--fine_viz", action="store_true", help="Save fine debug overlays (requires cv2)")
    parser.add_argument("--fine_viz_dir", type=str, default="viz_fine")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data transforms (coarse)
    transform_base = transforms.Compose([
        transforms.Resize((BASEMAP_INPUT_SIZE, BASEMAP_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_gen = transforms.Compose([
        transforms.Resize((BASEMAP_INPUT_SIZE, BASEMAP_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    base_folder = "/data1/"
    train_dataset = MapDataset(
        base_folder + "all_train_metas_v3",
        base_folder + "all_train_basemaps_segmented_v3",
        base_folder + "all_train_maps_segmented_gt_v3/map/",
        transform_base, transform_gen
    )
    val_dataset = MapDataset(
        base_folder + "all_val_metas_v3",
        base_folder + "all_val_basemaps_segmented_v3",
        base_folder + "all_val_maps_segmented_gt_v3/map/",
        transform_base, transform_gen
    )

    model = GridClassifier()
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    if args.mode == "infer":
        if args.checkpoint is None or not os.path.exists(args.checkpoint):
            raise ValueError("For --mode infer, you must provide a valid --checkpoint path via --checkpoint.")

        infer_dataset = val_dataset if args.infer_split == "val" else train_dataset

        # If they asked for fine viz, ensure cv2 exists early with a clear error.
        if args.fine_viz and cv2 is None:
            raise RuntimeError(
                "You passed --fine_viz but OpenCV (cv2) is not available. Install with: pip install opencv-python"
            )

        run_inference_with_fine(
            model=model,
            dataset=infer_dataset,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            num_workers=10,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            viz=args.viz,
            viz_dir=args.viz_dir,
            output_json=args.infer_out,
            fine_out_json=args.fine_out,
            fine_viz=args.fine_viz,
            fine_viz_dir=args.fine_viz_dir,
        )
        return

    # ---------------------------
    # TRAIN (DDP as before)
    # ---------------------------
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found. This training script expects at least 1 GPU for DDP.")

    # Criterion is set inside train_model (kept consistent with your original script)
    criterion = None

    mp.spawn(
        train_model,
        args=(
            world_size,
            args.num_epochs,
            model,
            criterion,
            optimizer,
            scheduler,
            train_dataset,
            val_dataset,
            args.batch_size,
            args.lr,
            args.version,
            args.train_fraction,
            args.checkpoint,
            args.seed,
            args.validate_every,
            args.amp,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()