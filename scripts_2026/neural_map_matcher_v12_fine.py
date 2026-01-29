# This code is for coarse matching- to find the best grid.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import json
import argparse
from torchvision.ops import sigmoid_focal_loss
from transformers import AutoModel

from pathlib import Path
import matplotlib.pyplot as plt

GRID_DIM = 10
TARGET_BLOCK = 1
POSITIVE_CELLS = TARGET_BLOCK * TARGET_BLOCK
BASEMAP_INPUT_SIZE = 224
REGRESSION_WEIGHT = 1
METERS_PER_PATCH = 500.0
def perturbation_to_pixel(perturbation, center_x, center_y, pixels_per_meter):
    return (
        center_x - perturbation[1] * pixels_per_meter,
        center_y - perturbation[0] * pixels_per_meter,
    )

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
                if os.path.exists(os.path.join(basemap_folder, basemap_file)):
                    if os.path.exists(os.path.join(metas_folder, metas_file)):
                        self.file_triplets.append((stitched_file, basemap_file, metas_file))

    def __len__(self):
        return len(self.file_triplets)

    def __getitem__(self, idx):
        # try:
        # import pdb; pdb.set_trace()
        stitched_file, basemap_file, metas_file = self.file_triplets[idx]
        
        stitched_img_path = os.path.join(self.stitched_folder, stitched_file)
        basemap_img_path = os.path.join(self.basemap_folder, basemap_file)
        metas_path = os.path.join(self.metas_folder, metas_file)

        stitched_img = Image.open(stitched_img_path).convert('RGB')
        basemap_img = Image.open(basemap_img_path).convert('RGB')
    
        if self.transform_gen:
            stitched_img = self.transform_gen(stitched_img)
        if self.transform_base:
            basemap_img = self.transform_base(basemap_img)

        metas = np.load(metas_path, allow_pickle=True).item()
        
        # Convert coordinates to grid labels
        center_x, center_y = basemap_img.shape[1] // 2, basemap_img.shape[2] // 2
        # print("center_x: ", center_x)
        # print("center_y: ", center_y)
        basemap_size_px = basemap_img.shape[1]
        pixels_per_meter = basemap_size_px / METERS_PER_PATCH
        x_val, y_val = perturbation_to_pixel(
            metas['perturbation'],
            center_x,
            center_y,
            pixels_per_meter,
        )
        
        grid_size = basemap_size_px / GRID_DIM
        grid_x = int(x_val // grid_size)
        grid_y = int(y_val // grid_size)
                
        grid_dim = GRID_DIM
        target_block = TARGET_BLOCK
    
        # Clamp grid indices to valid range
        grid_x = int(np.clip(grid_x, 0, grid_dim - 1))
        grid_y = int(np.clip(grid_y, 0, grid_dim - 1))

        single_cell_label = torch.zeros(grid_dim, grid_dim)
        single_cell_label[grid_y, grid_x] = 1.0

        # Create 10x10 grid mask (2x2 target)
        grid_label = torch.zeros(grid_dim, grid_dim)

        max_anchor = grid_dim - target_block
        i0 = min(grid_y, max_anchor)  # ensure i0+1 <= 9
        j0 = min(grid_x, max_anchor)  # ensure j0+1 <= 9

        grid_label[i0:i0+target_block, j0:j0+target_block] = 1.0

        gt_pixel = torch.tensor([x_val, y_val], dtype=torch.float32)

        return (
            stitched_img,
            basemap_img,
            grid_label.flatten().float(),
            single_cell_label.flatten().float(),
            gt_pixel,
            stitched_img_path,
            basemap_img_path,
            metas_path,
        )
                
        # except Exception as e:
        #     return torch.zeros(3, 100, 100), torch.zeros(3, 1000, 1000), torch.zeros(100), "", "", ""

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
        """
        x: torch float tensor, shape (B,3,H,W), already normalized in your pipeline.
        HF processor normally expects raw pixels, but we can bypass it:
        - DINOv2 expects ImageNet normalization; you already do that.
        - So we feed pixel_values=x directly.
        """
        out = self.model(pixel_values=x)
        # last_hidden_state: (B, 1+N, D); drop CLS token
        tokens = out.last_hidden_state[:, 1:]  # (B, N, D)
        batch, num_tokens, dim = tokens.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise ValueError(
                f"Expected square number of tokens, got {num_tokens}."
            )
        return tokens.transpose(1, 2).reshape(batch, dim, side, side)

class GridClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = DINOv2Backbone(model_name="facebook/dinov2-large")
        self.embed_dim = self.feature_extractor.embed_dim

        # # Freeze feature extractor
        # for p in self.feature_extractor.parameters():
        #     p.requires_grad = False

        # Pool basemap to 10x10
        self.grid_pool = nn.AdaptiveAvgPool2d((10, 10))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            batch_first=True
        )

        # Positional encoding (UPDATED): batch-safe + ViT-style init
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # No mask needed if you want full attention
        self.attn_mask = None

        self.fc = nn.Linear(self.embed_dim, 100)
        self.regression_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 2),
        )

    def forward(self, stitched, basemap):
        # Feature extraction
        stitched_feat = self.feature_extractor(stitched)   # (B,512,h1,w1)
        basemap_feat  = self.feature_extractor(basemap)    # (B,512,h2,w2)

        # Basemap -> 10x10 grid
        grid = self.grid_pool(basemap_feat)                # (B,512,10,10)

        grid_3x3 = F.unfold(grid, kernel_size=3, padding=1)  # (B,D*9,100)
        grid_3x3 = grid_3x3.view(grid_3x3.size(0), self.embed_dim, 9, 100).mean(dim=2)  # (B,D,100)

        # Attention inputs
        query = F.adaptive_avg_pool2d(stitched_feat, (1, 1)).flatten(1)      # (B,512)
        query = query.unsqueeze(1)                                           # (B,1,512)

        key_value = grid_3x3.permute(0, 2, 1)                                 # (B,100,D)
        key = value = key_value + self.pos_embed                              # (B,100,D)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=query,   # (B,1,D)
            key=key,       # (B,100,D)
            value=value,   # (B,100,D)
            attn_mask=self.attn_mask
        )

        scores = self.fc(attn_out.squeeze(1))  # (B,100)

        deltas = self.regression_head(attn_out.squeeze(1))  # (B,2), in cell units
        return scores, deltas


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
            reduction="mean"  # Required for DDP compatibility
        )

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10112'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_dataloader(rank, world_size, dataset, batch_size=16, num_workers=10, prefetch_factor=2):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                    num_workers=num_workers, pin_memory=True, persistent_workers=True,
                    prefetch_factor=prefetch_factor)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def strip_module_prefix(state_dict):
    """Handle checkpoints saved from DDP (keys start with 'module.')."""
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_coarse_weights(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    state = strip_module_prefix(state)
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    missing = set(model_state.keys()) - set(filtered.keys())
    if missing:
        print(f"[Coarse Load] Skipping {len(missing)} unmatched keys.")
    model.load_state_dict(filtered, strict=False)


def freeze_coarse_backbone(model):
    for name, param in model.named_parameters():
        if name.startswith((
            "feature_extractor",
            "grid_pool",
            "cross_attn",
            "pos_embed",
            "fc",
        )):
            param.requires_grad = False


def visualize_pred_gt_grid(pred_mask_10x10, gt_mask_10x10, save_path):
    """
    pred_mask_10x10, gt_mask_10x10: torch or numpy, shape (10,10), values {0,1}
    Colors:
      - GT only:   Red
      - Pred only: Blue
      - Overlap:   Purple
      - None:      White
    """
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

    # Build RGB image
    img = np.ones((10, 10, 3), dtype=np.float32)  # white background

    overlap = (pred == 1) & (gt == 1)
    gt_only = (pred == 0) & (gt == 1)
    pred_only = (pred == 1) & (gt == 0)

    # Red for GT only
    img[gt_only] = np.array([1.0, 0.2, 0.2])
    # Blue for Pred only
    img[pred_only] = np.array([0.2, 0.2, 1.0])
    # Purple for overlap
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

def visualize_basemap_topk(basemap_path, metas_path, topk_idx, topk_probs, save_path):
    basemap_img = Image.open(basemap_path).convert("RGB")
    basemap_img = basemap_img.resize((BASEMAP_INPUT_SIZE, BASEMAP_INPUT_SIZE), Image.BILINEAR)
    basemap_np = np.array(basemap_img)
    height, width = basemap_np.shape[0], basemap_np.shape[1]
    cell_width = width / GRID_DIM
    cell_height = height / GRID_DIM

    metas = np.load(metas_path, allow_pickle=True).item()
    center_x, center_y = width // 2, height // 2
    meters_per_patch = 500.0
    pixels_per_meter = width / meters_per_patch
    gt_x, gt_y = perturbation_to_pixel(
        metas['perturbation'],
        center_x,
        center_y,
        pixels_per_meter,
    )
    topk_idx = np.array(topk_idx)
    topk_probs = np.array(topk_probs)
    pred_x = (topk_idx % GRID_DIM + 0.5) * cell_width
    pred_y = (topk_idx // GRID_DIM + 0.5) * cell_height

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(basemap_np)
    plt.axis("off")

    plt.scatter(gt_x, gt_y, s=220, c="blue", edgecolors="white", linewidths=1.5, label="GT")

    plt.scatter(pred_x, pred_y, s=80, c="green", edgecolors="white", linewidths=1.0, label="Pred Top-k")

    for x, y, prob in zip(pred_x, pred_y, topk_probs):
        plt.text(
            x + 4,
            y - 4,
            f"{prob:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none")
        )

    plt.legend(loc="upper right", framealpha=0.8)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close()



def run_inference(model, dataset, checkpoint_path, batch_size=64, num_workers=10,
                  device=None, viz=False, viz_dir="viz_grids", output_json="inference_outputs.json"):
    """
    Runs inference on dataset. Computes:
      - top-k "hit" accuracy (k matches the number of positive cells)
      - IoU based on top-k predicted cells vs GT positives
    Optionally saves 10x10 visualization grids per sample.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)  # allow raw state_dict too
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    correct_topk = 0
    exact_top1_correct = 0
    topk_correct = {1: 0, 2: 0, 3: 0}
    total = 0
    total_iou = 0.0
    distance_values = []
    total_within_1px = 0
    total_within_5px = 0
    total_within_10px = 0
    meters_error_values = []
    cell_error_values = []

    viz_dir = Path(viz_dir)
    outputs_list = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference")
        for stitched, basemap, labels, single_labels, gt_pixels, stitched_img_path, basemap_img_path, metas_path in pbar:
            stitched = stitched.to(device, non_blocking=True)
            basemap = basemap.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)  # (B,100)
            single_labels = single_labels.to(device, non_blocking=True)
            gt_pixels = gt_pixels.to(device, non_blocking=True)

            logits, deltas = model(stitched, basemap)  # (B,100), (B,2)

            # Top-k hit accuracy (k matches the number of positives)
            _, topk = torch.topk(logits, POSITIVE_CELLS, dim=1)  # (B,k)
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
            distances = torch.sqrt(
                (pred_row - gt_row).float() ** 2 + (pred_col - gt_col).float() ** 2
            )
            distance_values.extend(distances.detach().cpu().tolist())

            cell_size = BASEMAP_INPUT_SIZE / GRID_DIM
            pred_center_x = (pred_col.float() + 0.5) * cell_size
            pred_center_y = (pred_row.float() + 0.5) * cell_size
            pred_x = pred_center_x + deltas[:, 0] * cell_size
            pred_y = pred_center_y + deltas[:, 1] * cell_size

            gt_x = gt_pixels[:, 0]
            gt_y = gt_pixels[:, 1]
            pixel_error = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            cell_error = (pixel_error / cell_size).detach().cpu().tolist()

            within_1px = (pixel_error <= 1.0).sum().item()
            within_5px = (pixel_error <= 5.0).sum().item()
            within_10px = (pixel_error <= 10.0).sum().item()
            pixels_per_meter = BASEMAP_INPUT_SIZE / METERS_PER_PATCH
            meters_error = (pixel_error / pixels_per_meter).detach().cpu().tolist()
            total_within_1px += within_1px
            total_within_5px += within_5px
            total_within_10px += within_10px
            meters_error_values.extend(meters_error)
            cell_error_values.extend(cell_error)

            # IoU with top-k predicted indices
            _, topk_iou = torch.topk(logits, POSITIVE_CELLS, dim=1)  # (B,k)
            preds = torch.zeros_like(labels)
            preds.scatter_(1, topk_iou, 1)

            intersection = (preds * labels).sum(dim=1)
            union = ((preds + labels) > 0).sum(dim=1)
            iou_percentage = (intersection / union * 100.0).mean().item()
            total_iou += iou_percentage

            # Optional visualization per-sample
            if viz:
                for b in range(labels.size(0)):
                    gt_10x10 = labels[b].view(10, 10)
                    pred_10x10 = preds[b].view(10, 10)

                    # Use metas filename as ID
                    sample_id = Path(metas_path[b]).stem
                    save_path = viz_dir / f"{sample_id}_grid.png"
                    visualize_pred_gt_grid(pred_10x10, gt_10x10, save_path)
                    topk_probs = torch.sigmoid(logits[b]).gather(0, topk[b]).detach().cpu().numpy()
                    basemap_save_path = viz_dir / f"{sample_id}_basemap_topk.png"
                    visualize_basemap_topk(
                        basemap_img_path[b],
                        metas_path[b],
                        topk[b].detach().cpu().numpy(),
                        topk_probs,
                        basemap_save_path
                    )
                    stitched_save_path = viz_dir / f"{sample_id}_stitched.png"
                    stitched_img = Image.open(stitched_img_path[b]).convert("RGB")
                    stitched_save_path.parent.mkdir(parents=True, exist_ok=True)
                    stitched_img.save(stitched_save_path)

            # Save raw outputs (optional but useful)
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            topk_idx = topk.detach().cpu().numpy().tolist()

            for b in range(labels.size(0)):
                outputs_list.append({
                    "stitched_img_path": stitched_img_path[b],
                    "basemap_img_path": basemap_img_path[b],
                    "metas_path": metas_path[b],
                    "pred_idx": pred_idx[b].item(),
                    "gt_idx": gt_idx[b].item(),
                    "distance_cells": distances[b].item(),
                    "pixel_error": pixel_error[b].item(),
                    "cell_error": cell_error[b],
                    "meters_error": meters_error[b],
                    "topk_idx": topk_idx[b],
                    "probs": probs[b],
                })

            avg_acc = correct_topk / max(1, total)
            avg_iou = total_iou / max(1, (pbar.n + 1))
            avg_topk_acc = {k: topk_correct[k] / max(1, total) for k in topk_correct}
            avg_exact_top1_acc = exact_top1_correct / max(1, total)
            avg_distance = sum(distance_values) / max(1, len(distance_values))
            avg_within_1px = total_within_1px / max(1, total)
            avg_within_5px = total_within_5px / max(1, total)
            avg_within_10px = total_within_10px / max(1, total)
            avg_meters_error = sum(meters_error_values) / max(1, len(meters_error_values))
            avg_cell_error = sum(cell_error_values) / max(1, len(cell_error_values))
            pbar.set_postfix({
                "topk_acc": f"{avg_acc:.3f}",
                "top1/2/3": f"{avg_topk_acc[1]:.3f}/{avg_topk_acc[2]:.3f}/{avg_topk_acc[3]:.3f}",
                "exact_top1": f"{avg_exact_top1_acc:.3f}",
                "iou%": f"{avg_iou:.2f}",
                "dist": f"{avg_distance:.2f}",
                "px<=1/5/10": f"{avg_within_1px:.2f}/{avg_within_5px:.2f}/{avg_within_10px:.2f}",
                "m_err": f"{avg_meters_error:.2f}",
                "cell_err": f"{avg_cell_error:.2f}",
            })

    final_topk = correct_topk / max(1, total)
    final_topk_acc = {k: topk_correct[k] / max(1, total) for k in topk_correct}
    final_exact_top1_acc = exact_top1_correct / max(1, total)
    final_iou = total_iou / len(loader)
    final_within_1px = total_within_1px / max(1, total)
    final_within_5px = total_within_5px / max(1, total)
    final_within_10px = total_within_10px / max(1, total)
    mean_meters_error = sum(meters_error_values) / max(1, len(meters_error_values))
    mean_cell_error = sum(cell_error_values) / max(1, len(cell_error_values))

    mean_distance = sum(distance_values) / max(1, len(distance_values))

    histogram_path = f"{checkpoint_path}_histogram.png"
    meters_histogram_1m_path = f"{checkpoint_path}_meters_histogram_1m.png"
    meters_histogram_5m_path = f"{checkpoint_path}_meters_histogram_5m.png"
    if distance_values:
        hist_values = np.array(distance_values)
        unique_distances, counts = np.unique(hist_values, return_counts=True)
        order = np.argsort(unique_distances)
        unique_distances = unique_distances[order]
        counts = counts[order]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(unique_distances)), counts, color="steelblue")
        labels = [f"{value:.2f}" for value in unique_distances]
        plt.xticks(range(len(unique_distances)), labels, rotation=45, ha="right")
        plt.xlabel("Distance (cells)")
        plt.ylabel("Count")
        plt.title("Prediction Distance Error Distribution")
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=200, bbox_inches="tight")
        plt.close()

    if meters_error_values:
        meters_values = np.array(meters_error_values)
        meters_values_50 = meters_values[meters_values <= 50.0]
        if meters_values_50.size > 0:
            bins_1m = np.arange(0, 51 + 1, 1)
            plt.figure(figsize=(10, 6))
            plt.hist(meters_values_50, bins=bins_1m, color="seagreen", edgecolor="black")
            plt.xlabel("Distance Error (meters)")
            plt.ylabel("Count")
            plt.title("Distance Error Distribution (1m bins, up to 50m)")
            plt.tight_layout()
            plt.savefig(meters_histogram_1m_path, dpi=200, bbox_inches="tight")
            plt.close()

        max_meter_error = meters_values.max()
        if max_meter_error > 0:
            bins_5m = np.arange(0, max_meter_error + 5, 5)
            plt.figure(figsize=(10, 6))
            plt.hist(meters_values, bins=bins_5m, color="mediumpurple", edgecolor="black")
            plt.xlabel("Distance Error (meters)")
            plt.ylabel("Count")
            plt.title("Distance Error Distribution (5m bins)")
            plt.tight_layout()
            plt.savefig(meters_histogram_5m_path, dpi=200, bbox_inches="tight")
            plt.close()

    with open(output_json, "w") as f:
        json.dump({
            "topk_acc": final_topk,
            "mean_iou_percent": final_iou,
            "mean_distance_cells": mean_distance,
            "within_1px_acc": final_within_1px,
            "within_5px_acc": final_within_5px,
            "within_10px_acc": final_within_10px,
            "mean_error_meters": mean_meters_error,
            "mean_error_cells": mean_cell_error,
            "outputs": outputs_list
        }, f)

    print(f"[Inference] Top-{POSITIVE_CELLS} Acc: {final_topk:.2%}, Mean IoU%: {final_iou:.2f}")
    print(
        "[Inference] "
        f"Top-1/2/3 Acc: {final_topk_acc[1]:.2%}/{final_topk_acc[2]:.2%}/{final_topk_acc[3]:.2%}, "
        f"Exact Top-1 Acc: {final_exact_top1_acc:.2%}"
    )
    print(f"[Inference] Mean distance (cells): {mean_distance:.2f}")
    print(
        "[Inference] "
        f"Pixel Acc<=1/5/10: {final_within_1px:.2%}/{final_within_5px:.2%}/{final_within_10px:.2%}, "
        f"Mean error (m): {mean_meters_error:.2f}, "
        f"Mean error (cells): {mean_cell_error:.2f}"
    )
    print(f"[Inference] Saved outputs to: {output_json}")
    if distance_values:
        print(f"[Inference] Saved distance histogram to: {histogram_path}")
    if meters_error_values:
        if Path(meters_histogram_1m_path).exists():
            print(f"[Inference] Saved meters histogram (1m bins) to: {meters_histogram_1m_path}")
        if Path(meters_histogram_5m_path).exists():
            print(f"[Inference] Saved meters histogram (5m bins) to: {meters_histogram_5m_path}")
    if viz:
        print(f"[Inference] Saved visualizations to: {str(viz_dir)}")


def train_model(rank, world_size, num_epochs, model, criterion, optimizer, scheduler,
               train_dataset, val_dataset, batch_size, lr, version, fraction, 
               checkpoint_path, seed, validate_every, amp):
    setup(rank, world_size)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # pos_weight = torch.ones(100) * (91 / 9)  # Shape: (100,) and 9 positive out of 100
    # pos_weight = pos_weight.to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    # criterion = DiceLoss(
    #     mode="binary",        # Binary segmentation task
    #     from_logits=True,     # Model outputs raw logits (before sigmoid)
    #     smooth=1e-6,          # Small value to prevent division by zero
    #     eps=1e-7              # Numerical stability
    # )
    # criterion = FocalBCEWithLogitsLoss(alpha=1, gamma=0)

    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(alpha=0.9, gamma=2)
    # criterion = lambda inputs, targets: sigmoid_focal_loss(
    #     inputs, targets, alpha=0.9, gamma=2.5, reduction="mean"
    # )

    criterion = criterion.to(device)
    if fraction >= 1.0:
        print("FULL TRAINING")
        #Use all the training data, no changes required for subset. Takes the full dataset from args
        if checkpoint_path is None:
            print("Starting from scratch")
            start_epoch = 0
            losses = {"train": [], "val": []}
            subset_indices = torch.randperm(len(train_dataset))
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            print("Resuming training")
            checkpoint = torch.load(checkpoint_path)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint['losses']
            subset_indices = checkpoint['subset_indices']  # Load the indices
        else:
            raise ValueError("Checkpoint file not found")

    else:
        print("FRACTION TRAINING")
        if checkpoint_path is None:
            print("Starting from scratch")
            num_samples = int(len(train_dataset) * fraction)
            subset_indices = torch.randperm(len(train_dataset))[:num_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
            start_epoch = 0
            losses = {"train": [], "val": []}

        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            print("Resuming training")
            checkpoint = torch.load(checkpoint_path)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint['losses']
            subset_indices = checkpoint['subset_indices']  # Load the indices
            train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

    train_loader = create_dataloader(rank, world_size, train_dataset, batch_size)
    val_loader = create_dataloader(rank, world_size, val_dataset, batch_size)
    
    unique_name = f"grid_v{version}-lr{lr}-bs{batch_size}-frac{fraction}-seed{seed}"

    best_val_loss = min(loss for epoch, loss, _, _ in losses["val"]) if losses["val"] else float('inf')
    best_train_loss = min(loss for epoch, loss, _, _ in losses["train"]) if losses["train"] else float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            train_loader.sampler.set_epoch(epoch)
            total_loss = 0.0
            total_loss_cls = 0.0
            total_loss_reg = 0.0
            correct = 0
            topk_correct = {1: 0, 2: 0, 3: 0}
            exact_top1_correct = 0
            total = 0
            total_iou = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
            total_within_1px = 0
            total_within_5px = 0
            total_within_10px = 0
            meters_error_values = []
            cell_error_values = []
            for stitched, basemap, labels, single_labels, gt_pixels, stitched_img_path, basemap_img_path, metas_path in pbar:
                stitched = stitched.to(device)
                basemap = basemap.to(device)
                labels = labels.to(device)
                single_labels = single_labels.to(device)
                gt_pixels = gt_pixels.to(device)

                optimizer.zero_grad()
                gt_idx = single_labels.argmax(dim=1)
                # import pdb; pdb.set_trace()

                with torch.cuda.amp.autocast(enabled=amp):
                    outputs, deltas = model(stitched, basemap)
                    loss_cls = criterion(outputs, labels)
                    pred_idx = outputs.argmax(dim=1)
                    pred_row = pred_idx // GRID_DIM
                    pred_col = pred_idx % GRID_DIM
                    cell_size = BASEMAP_INPUT_SIZE / GRID_DIM
                    pred_center_x = (pred_col.float() + 0.5) * cell_size
                    pred_center_y = (pred_row.float() + 0.5) * cell_size
                    target_delta_x = (gt_pixels[:, 0] - pred_center_x) / cell_size
                    target_delta_y = (gt_pixels[:, 1] - pred_center_y) / cell_size
                    target_deltas = torch.stack([target_delta_x, target_delta_y], dim=1)
                    loss_reg = F.smooth_l1_loss(deltas, target_deltas)
                    loss = loss_cls + REGRESSION_WEIGHT * loss_reg
                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                total_loss_cls += loss_cls.item()
                total_loss_reg += loss_reg.item()
                pbar.set_postfix({'loss': total_loss/(pbar.n+1)})

                _, topk = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                correct += (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
                for k in topk_correct:
                    _, topk_k = torch.topk(outputs, k, dim=1)
                    topk_correct[k] += (labels.gather(1, topk_k).sum(dim=1) > 0).sum().item()

                _, top1_idx = torch.topk(outputs, 1, dim=1)
                exact_top1_correct += (single_labels.gather(1, top1_idx).sum(dim=1) > 0).sum().item()

                total += labels.size(0)

                # Calculate IOU
                _, topk_iou = torch.topk(outputs, POSITIVE_CELLS, dim=1)  # Shape: (B, k)
                # Create binary predictions tensor based on top-k indices
                predictions = torch.zeros_like(labels)
                predictions.scatter_(1, topk_iou, 1)

                # Calculate Intersection over Union (IoU)
                intersection = (predictions * labels).sum(dim=1)  # Element-wise multiplication and sum
                union = ((predictions + labels) > 0).sum(dim=1)  # Union is the count of non-zero elements

                # IoU as a percentage
                iou_percentage = (intersection / union * 100).mean().item()
                total_iou += iou_percentage

                pred_idx = outputs.argmax(dim=1)
                pred_row = pred_idx // GRID_DIM
                pred_col = pred_idx % GRID_DIM

                cell_size = BASEMAP_INPUT_SIZE / GRID_DIM
                pred_center_x = (pred_col.float() + 0.5) * cell_size
                pred_center_y = (pred_row.float() + 0.5) * cell_size
                pred_x = pred_center_x + deltas[:, 0] * cell_size
                pred_y = pred_center_y + deltas[:, 1] * cell_size
                gt_x = gt_pixels[:, 0]
                gt_y = gt_pixels[:, 1]
                pixel_error = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                cell_error = (pixel_error / cell_size).detach().cpu().tolist()

                total_within_1px += (pixel_error <= 1.0).sum().item()
                total_within_5px += (pixel_error <= 5.0).sum().item()
                total_within_10px += (pixel_error <= 10.0).sum().item()
                pixels_per_meter = BASEMAP_INPUT_SIZE / METERS_PER_PATCH
                meters_error_values.extend((pixel_error / pixels_per_meter).detach().cpu().tolist())
                cell_error_values.extend(cell_error)
            
            train_loss = total_loss / len(train_loader)
            train_loss_cls = total_loss_cls / len(train_loader)
            train_loss_reg = total_loss_reg / len(train_loader)
            train_acc = correct / total
            train_topk_acc = {k: topk_correct[k] / total for k in topk_correct}
            train_exact_top1_acc = exact_top1_correct / total
            train_iou = total_iou / len(train_loader)
            train_within_1px = total_within_1px / max(1, total)
            train_within_5px = total_within_5px / max(1, total)
            train_within_10px = total_within_10px / max(1, total)
            train_mean_meters_error = sum(meters_error_values) / max(1, len(meters_error_values))
            train_mean_cell_error = sum(cell_error_values) / max(1, len(cell_error_values))
            losses["train"].append([epoch, train_loss, train_acc, train_iou])

            if rank == 0:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'subset_indices': subset_indices  # Save the indices
                }
                torch.save(checkpoint, 'latest_map_location_model_train_'+unique_name+'.pth')

            # Validation
            if epoch % validate_every == 0 or epoch == start_epoch:
                model.eval()
                val_loss = 0.0
                val_loss_cls = 0.0
                val_loss_reg = 0.0
                correct = 0
                topk_correct = {1: 0, 2: 0, 3: 0}
                exact_top1_correct = 0
                total = 0
                total_iou = 0.0
                with torch.no_grad():
                    pbar = tqdm(val_loader, desc=f"Validation", disable=rank != 0)
                    total_within_1px = 0
                    total_within_5px = 0
                    total_within_10px = 0
                    meters_error_values = []
                    cell_error_values = []
                    for stitched, basemap, labels, single_labels, gt_pixels, stitched_img_path, basemap_img_path, metas_path in pbar:
                        stitched = stitched.to(device)
                        basemap = basemap.to(device)
                        labels = labels.to(device)
                        single_labels = single_labels.to(device)
                        gt_pixels = gt_pixels.to(device)

                        with torch.cuda.amp.autocast(enabled=amp):
                            outputs, deltas = model(stitched, basemap)
                            loss_cls = criterion(outputs, labels)
                            pred_idx = outputs.argmax(dim=1)
                            pred_row = pred_idx // GRID_DIM
                            pred_col = pred_idx % GRID_DIM
                            cell_size = BASEMAP_INPUT_SIZE / GRID_DIM
                            pred_center_x = (pred_col.float() + 0.5) * cell_size
                            pred_center_y = (pred_row.float() + 0.5) * cell_size
                            target_delta_x = (gt_pixels[:, 0] - pred_center_x) / cell_size
                            target_delta_y = (gt_pixels[:, 1] - pred_center_y) / cell_size
                            target_deltas = torch.stack([target_delta_x, target_delta_y], dim=1)
                            loss_reg = F.smooth_l1_loss(deltas, target_deltas)
                            val_loss += (loss_cls + REGRESSION_WEIGHT * loss_reg).item()
                            val_loss_cls += loss_cls.item()
                            val_loss_reg += loss_reg.item()
                        if rank == 0:
                            pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})
                        # Calculate top-k accuracy
                        _, topk = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                        correct += (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
                        for k in topk_correct:
                            _, topk_k = torch.topk(outputs, k, dim=1)
                            topk_correct[k] += (labels.gather(1, topk_k).sum(dim=1) > 0).sum().item()
                        _, top1_idx = torch.topk(outputs, 1, dim=1)
                        exact_top1_correct += (single_labels.gather(1, top1_idx).sum(dim=1) > 0).sum().item()

                        total += labels.size(0)

                        # Calculate IOU
                        _, topk_iou = torch.topk(outputs, POSITIVE_CELLS, dim=1)  # Shape: (B, k)
                        # Create binary predictions tensor based on top-k indices
                        predictions = torch.zeros_like(labels)
                        predictions.scatter_(1, topk_iou, 1)

                        # Calculate Intersection over Union (IoU)
                        intersection = (predictions * labels).sum(dim=1)  # Element-wise multiplication and sum
                        union = ((predictions + labels) > 0).sum(dim=1)  # Union is the count of non-zero elements

                        # IoU as a percentage
                        iou_percentage = (intersection / union * 100).mean().item()
                        total_iou += iou_percentage

                        pred_idx = outputs.argmax(dim=1)
                        pred_row = pred_idx // GRID_DIM
                        pred_col = pred_idx % GRID_DIM

                        cell_size = BASEMAP_INPUT_SIZE / GRID_DIM
                        pred_center_x = (pred_col.float() + 0.5) * cell_size
                        pred_center_y = (pred_row.float() + 0.5) * cell_size
                        pred_x = pred_center_x + deltas[:, 0] * cell_size
                        pred_y = pred_center_y + deltas[:, 1] * cell_size
                        gt_x = gt_pixels[:, 0]
                        gt_y = gt_pixels[:, 1]
                        pixel_error = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                        cell_error = (pixel_error / cell_size).detach().cpu().tolist()

                        total_within_1px += (pixel_error <= 1.0).sum().item()
                        total_within_5px += (pixel_error <= 5.0).sum().item()
                        total_within_10px += (pixel_error <= 10.0).sum().item()
                        pixels_per_meter = BASEMAP_INPUT_SIZE / METERS_PER_PATCH
                        meters_error_values.extend((pixel_error / pixels_per_meter).detach().cpu().tolist())
                        cell_error_values.extend(cell_error)
                
                val_loss /= len(val_loader)
                val_loss_cls /= len(val_loader)
                val_loss_reg /= len(val_loader)
                val_acc = correct / total
                val_topk_acc = {k: topk_correct[k] / total for k in topk_correct}
                val_exact_top1_acc = exact_top1_correct / total
                val_iou = total_iou / len(val_loader)
                val_within_1px = total_within_1px / max(1, total)
                val_within_5px = total_within_5px / max(1, total)
                val_within_10px = total_within_10px / max(1, total)
                val_mean_meters_error = sum(meters_error_values) / max(1, len(meters_error_values))
                val_mean_cell_error = sum(cell_error_values) / max(1, len(cell_error_values))
                losses["val"].append([epoch, val_loss, val_acc, val_iou])

                scheduler.step()

                if rank == 0:
                    print("LR now:", optimizer.param_groups[0]['lr'])
                
                if rank == 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'subset_indices': subset_indices  # Save the indices
                    }
                    torch.save(checkpoint, 'best_map_location_model_val_'+unique_name+'.pth')
        
            if rank == 0:
                log_line = (
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {train_loss:.4f}, Top-{POSITIVE_CELLS} Train Acc: {train_acc:.2%}, "
                    f"Top-1/2/3 Train Acc: {train_topk_acc[1]:.2%}/{train_topk_acc[2]:.2%}/{train_topk_acc[3]:.2%}, "
                    f"Val Loss: {val_loss:.4f}, Top-{POSITIVE_CELLS} Val Acc: {val_acc:.2%}, "
                    f"Top-1/2/3 Val Acc: {val_topk_acc[1]:.2%}/{val_topk_acc[2]:.2%}/{val_topk_acc[3]:.2%}, "
                    f"Train IoU: {train_iou:.2f}, Val IoU: {val_iou:.2f}"
                )
                log_line += (
                    f", Exact Top-1 Train Acc: {train_exact_top1_acc:.2%}, "
                    f"Exact Top-1 Val Acc: {val_exact_top1_acc:.2%}"
                )
                log_line += (
                    f", Train Cls/Reg Loss: {train_loss_cls:.4f}/{train_loss_reg:.4f}, "
                    f"Val Cls/Reg Loss: {val_loss_cls:.4f}/{val_loss_reg:.4f}, "
                    f"Train Px<=1/5/10: {train_within_1px:.2%}/{train_within_5px:.2%}/{train_within_10px:.2%}, "
                    f"Val Px<=1/5/10: {val_within_1px:.2%}/{val_within_5px:.2%}/{val_within_10px:.2%}, "
                    f"Train Mean Err (m): {train_mean_meters_error:.2f}, "
                    f"Train Mean Err (cells): {train_mean_cell_error:.2f}, "
                    f"Val Mean Err (m): {val_mean_meters_error:.2f}, "
                    f"Val Mean Err (cells): {val_mean_cell_error:.2f}"
                )
                print(log_line)
                with open('loss_'+unique_name+'.txt', 'a') as f:
                    f.write(log_line + "\n")
        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=str, default="12_fine")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "infer"])
    parser.add_argument('--viz', action='store_true', help="Save 10x10 Pred vs GT grid visualizations")
    parser.add_argument('--validate_every', type=str, default=1)
    parser.add_argument('--amp', action='store_true', default=True, help="Enable mixed precision training/inference")
    parser.add_argument('--viz_dir', type=str, default="viz_grids")
    parser.add_argument('--infer_split', type=str, default="val", choices=["train", "val"])
    parser.add_argument('--infer_out', type=str, default="inference_outputs.json")
    parser.add_argument('--coarse_checkpoint', type=str, default=None,
                        help="Optional coarse matcher checkpoint to initialize weights.")
    parser.add_argument('--freeze_coarse', action='store_true',
                        help="Freeze coarse matcher layers and train only fine regression head.")

    args = parser.parse_args()

    # Data transforms
    transform_base = transforms.Compose([
        transforms.Resize((BASEMAP_INPUT_SIZE, BASEMAP_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_gen = transforms.Compose([
        transforms.Resize((BASEMAP_INPUT_SIZE, BASEMAP_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    base_folder = "/data1/"
    train_dataset = MapDataset(
        base_folder+'all_train_metas_v3',
        base_folder+'all_train_basemaps_segmented_v3',
        base_folder+'all_train_maps_segmented_gt_v3/map/',
        transform_base, transform_gen
    )
    val_dataset = MapDataset(
        base_folder+'all_val_metas_v3',
        base_folder+'all_val_basemaps_segmented_v3', 
        base_folder+'all_val_maps_segmented_gt_v3/map/',
        transform_base, transform_gen
    )

    # Model and training setup
    model = GridClassifier()
    if args.coarse_checkpoint:
        if not os.path.exists(args.coarse_checkpoint):
            raise ValueError(f"Coarse checkpoint not found: {args.coarse_checkpoint}")
        load_coarse_weights(model, args.coarse_checkpoint)
        if args.freeze_coarse:
            freeze_coarse_backbone(model)
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    criterion=None


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,   # usually total epochs
        eta_min=1e-6
    )

    if args.mode == "infer":
        if args.checkpoint is None or not os.path.exists(args.checkpoint):
            raise ValueError("For --mode infer, you must provide a valid --checkpoint path.")

        infer_dataset = val_dataset if args.infer_split == "val" else train_dataset

        # Single-GPU inference (no DDP)
        run_inference(
            model=model,
            dataset=infer_dataset,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            num_workers=10,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            viz=args.viz,
            viz_dir=args.viz_dir,
            output_json=args.infer_out
        )
        return

    # Otherwise, TRAIN (DDP as before)
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args.num_epochs, model, criterion, optimizer, scheduler,
              train_dataset, val_dataset, args.batch_size, args.lr,
              args.version, args.train_fraction, args.checkpoint, args.seed,
              args.validate_every, args.amp),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
