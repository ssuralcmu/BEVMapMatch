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
        
        basemap_files = [f for f in os.listdir(basemap_folder) if f.endswith("_base_map_image.png")]
        metas_files = [f for f in os.listdir(metas_folder) if f.endswith("_metas.npy")]

        basemap_index = self._build_suffix_index(basemap_files, "_base_map_image.png")
        metas_index = self._build_suffix_index(metas_files, "_metas.npy")

        stitched_files = os.listdir(stitched_folder)
        self.file_triplets = []
        for stitched_file in stitched_files:
            if stitched_file.endswith("_generated_map_image.png"):
                stitched_key = stitched_file[:-len("_generated_map_image.png")]
            elif stitched_file.endswith(".png"):
                stitched_key = os.path.splitext(stitched_file)[0]
            else:
                continue

            basemap_file = self._resolve_file(stitched_key, basemap_index)
            metas_file = self._resolve_file(stitched_key, metas_index)

            if basemap_file is not None and metas_file is not None:
                self.file_triplets.append((stitched_file, basemap_file, metas_file))

    @staticmethod
    def _build_suffix_index(files, suffix):
        index = {}
        for file_name in files:
            if not file_name.endswith(suffix):
                continue

            full_prefix = file_name[:-len(suffix)]
            short_prefix = full_prefix.split("-")[-1]

            index.setdefault(full_prefix, []).append(file_name)
            if short_prefix != full_prefix:
                index.setdefault(short_prefix, []).append(file_name)
        return index

    @staticmethod
    def _resolve_file(stitched_key, index):
        matches = index.get(stitched_key, [])
        if len(matches) == 1:
            return matches[0]

        stitched_suffix = stitched_key.split("-")[-1]
        suffix_matches = [
            match for match in matches
            if match.split("-")[-1].startswith(stitched_suffix)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        fallback_matches = index.get(stitched_suffix, [])
        if len(fallback_matches) == 1:
            return fallback_matches[0]

        return None
    
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
        meters_per_patch = 500.0
        pixels_per_meter = basemap_size_px / meters_per_patch
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
    
        # Create 10x10 grid mask (2x2 target)
        grid_label = torch.zeros(grid_dim, grid_dim)

        # Clamp grid indices to valid range
        grid_x = int(np.clip(grid_x, 0, grid_dim - 1))
        grid_y = int(np.clip(grid_y, 0, grid_dim - 1))

        max_anchor = grid_dim - target_block
        i0 = min(grid_y, max_anchor)  # ensure i0+1 <= 9
        j0 = min(grid_x, max_anchor)  # ensure j0+1 <= 9

        grid_label[i0:i0+target_block, j0:j0+target_block] = 1.0

        return stitched_img, basemap_img, grid_label.flatten().float(), stitched_img_path, basemap_img_path, metas_path
            
        # except Exception as e:
        #     return torch.zeros(3, 100, 100), torch.zeros(3, 1000, 1000), torch.zeros(100), "", "", ""

class DINOv2Backbone(nn.Module):
    """
    DINOv2 via HuggingFace (py3.8 friendly).
    Returns patch embeddings as a spatial feature map (B, D, H, W).
    """
    def __init__(self, model_name="facebook/dinov2-large"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch float tensor, shape (B,3,H,W), already normalized in your pipeline.
        """
        out = self.model(pixel_values=x)
        # last_hidden_state: (B, 1+N, D); drop CLS token
        tokens = out.last_hidden_state[:, 1:]  # (B, N, D)
        b, n, d = tokens.shape
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError(f"Expected square number of tokens, got {n}.")
        return tokens.transpose(1, 2).reshape(b, d, side, side)  # (B, D, H, W)


class GridClassifier(nn.Module):
    """
    V1 GridClassifier with ONLY Option A:
      - Replace fc(attn_out) with temperature-scaled dot-product scoring
        between the attended query vector and the per-cell key/value tokens.
    Everything else matches your original GridClassifier:
      - single query slot
      - same 10x10 pooling
      - same 3x3 mean neighborhood
      - same pos_embed
      - same cross-attn module
    """
    def __init__(self, dino_name="facebook/dinov2-large", num_heads=8, init_logit_scale=10.0):
        super().__init__()

        self.feature_extractor = DINOv2Backbone(model_name=dino_name)
        self.embed_dim = self.feature_extractor.embed_dim

        # Pool basemap to 10x10
        self.grid_pool = nn.AdaptiveAvgPool2d((10, 10))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,  # keep deterministic unless you explicitly add dropout
        )

        # Positional encoding (same as V1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.attn_mask = None

        # Option A: temperature / logit scale (learnable)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

        # Note: we keep self.fc for backwards compatibility if you want to compare,
        # but it is not used in forward for Option A.
        self.fc = nn.Linear(self.embed_dim, 100)

    def forward(self, stitched: torch.Tensor, basemap: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        stitched_feat = self.feature_extractor(stitched)   # (B, D, hs, ws)
        basemap_feat  = self.feature_extractor(basemap)    # (B, D, hb, wb)

        # Basemap -> 10x10 grid
        grid = self.grid_pool(basemap_feat)                # (B, D, 10, 10)

        # Same as V1: 3x3 mean neighborhood per cell
        grid_3x3 = F.unfold(grid, kernel_size=3, padding=1)  # (B, D*9, 100)
        grid_3x3 = grid_3x3.view(grid_3x3.size(0), self.embed_dim, 9, 100).mean(dim=2)  # (B, D, 100)

        # Attention inputs
        query = F.adaptive_avg_pool2d(stitched_feat, (1, 1)).flatten(1).unsqueeze(1)  # (B, 1, D)

        key_value = grid_3x3.permute(0, 2, 1)               # (B, 100, D)
        kv = key_value + self.pos_embed                     # (B, 100, D)

        # Cross-attention: query attends to kv tokens
        attn_out, _ = self.cross_attn(
            query=query,   # (B, 1, D)
            key=kv,        # (B, 100, D)
            value=kv,      # (B, 100, D)
            attn_mask=self.attn_mask,
            need_weights=False,
        )  # (B, 1, D)

        # Option A scoring: cosine-similarity-style dot product with temperature
        q = attn_out.squeeze(1)                             # (B, D)

        q = F.normalize(q, dim=-1)
        kvn = F.normalize(kv, dim=-1)

        scores = torch.einsum("bd,bnd->bn", q, kvn)         # (B, 100)

        # Scale logits (higher => sharper distribution). Clamp for stability.
        scores = scores * self.logit_scale.clamp(1.0, 100.0)

        return scores


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

def build_soft_targets(labels, grid_dim, sigma):
    batch_size = labels.size(0)
    labels_flat = labels.view(batch_size, -1)
    center_idx = labels_flat.argmax(dim=1)
    center_y = (center_idx // grid_dim).float()
    center_x = (center_idx % grid_dim).float()
    centers = torch.stack([center_y, center_x], dim=1)

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_dim, device=labels.device),
            torch.arange(grid_dim, device=labels.device),
            indexing="ij",
        ),
        dim=-1,
    ).view(-1, 2)

    diff = coords.unsqueeze(0) - centers.unsqueeze(1)
    dist2 = (diff ** 2).sum(dim=2)
    soft_targets = torch.exp(-dist2 / (2 * sigma ** 2))
    soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)
    return soft_targets

def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return (-soft_targets * log_probs).sum(dim=1).mean()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12015'
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
    total = 0
    total_iou = 0.0

    viz_dir = Path(viz_dir)
    outputs_list = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference")
        for stitched, basemap, labels, stitched_img_path, basemap_img_path, metas_path in pbar:
            stitched = stitched.to(device, non_blocking=True)
            basemap = basemap.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)  # (B,100)

            logits = model(stitched, basemap)  # (B,100)

            # Top-k hit accuracy (k matches the number of positives)
            _, topk = torch.topk(logits, POSITIVE_CELLS, dim=1)  # (B,k)
            batch_hits = (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
            correct_topk += batch_hits
            total += labels.size(0)

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
                    "topk_idx": topk_idx[b],
                    "probs": probs[b],
                })

            avg_acc = correct_topk / max(1, total)
            avg_iou = total_iou / max(1, (pbar.n + 1))
            pbar.set_postfix({"topk_acc": f"{avg_acc:.3f}", "iou%": f"{avg_iou:.2f}"})

    final_topk = correct_topk / max(1, total)
    final_iou = total_iou / len(loader)

    with open(output_json, "w") as f:
        json.dump({
            "topk_acc": final_topk,
            "mean_iou_percent": final_iou,
            "outputs": outputs_list
        }, f)

    print(f"[Inference] Top-{POSITIVE_CELLS} Acc: {final_topk:.2%}, Mean IoU%: {final_iou:.2f}")
    print(f"[Inference] Saved outputs to: {output_json}")
    if viz:
        print(f"[Inference] Saved visualizations to: {str(viz_dir)}")


def train_model(rank, world_size, num_epochs, model, criterion, optimizer, scheduler,
               train_dataset, val_dataset, batch_size, lr, version, fraction, 
               checkpoint_path, seed, validate_every, amp, distance_loss_weight,
               distance_sigma):
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
    
    unique_name = f"modelpred_grid_v{version}-lr{lr}-bs{batch_size}-frac{fraction}-seed{seed}"

    best_val_loss = min(loss for epoch, loss, _, _ in losses["val"]) if losses["val"] else float('inf')
    best_train_loss = min(loss for epoch, loss, _, _ in losses["train"]) if losses["train"] else float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            train_loader.sampler.set_epoch(epoch)
            total_loss = 0.0
            base_loss_total = 0.0
            distance_loss_total = 0.0
            correct = 0
            total = 0
            total_iou = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
            for stitched, basemap, labels, stitched_img_path, basemap_img_path, metas_path in pbar:
                stitched = stitched.to(device)
                basemap = basemap.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                # import pdb; pdb.set_trace()

                with torch.cuda.amp.autocast(enabled=amp):
                    outputs = model(stitched, basemap)
                    loss = criterion(outputs, labels)

                    if distance_loss_weight > 0:
                        soft_targets = build_soft_targets(labels, GRID_DIM, distance_sigma)
                        distance_loss = soft_cross_entropy(outputs, soft_targets)
                        loss = loss + distance_loss_weight * distance_loss

                base_loss = loss.detach() if distance_loss_weight <= 0 else criterion(outputs, labels).detach()
                distance_loss = torch.tensor(0.0, device=device) if distance_loss_weight <= 0 else distance_loss.detach()

                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                base_loss_total += base_loss.item()
                distance_loss_total += distance_loss.item()
                pbar.set_postfix({'loss': total_loss/(pbar.n+1)})

                _, topk = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                correct += (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
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
            
            train_loss = total_loss / len(train_loader)
            train_base_loss = base_loss_total / len(train_loader)
            train_distance_loss = distance_loss_total / len(train_loader)
            train_acc = correct / total
            train_iou = total_iou / len(train_loader)
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
                val_base_loss_total = 0.0
                val_distance_loss_total = 0.0                
                correct = 0
                total = 0
                total_iou = 0.0
                with torch.no_grad():
                    pbar = tqdm(val_loader, desc=f"Validation", disable=rank != 0)
                    for stitched, basemap, labels, stitched_img_path, basemap_img_path, metas_path in pbar:
                        stitched = stitched.to(device)
                        basemap = basemap.to(device)
                        labels = labels.to(device)
                        
                        with torch.cuda.amp.autocast(enabled=amp):
                            outputs = model(stitched, basemap)
                            base_loss = criterion(outputs, labels)
                            loss = base_loss
                            if distance_loss_weight > 0:
                                soft_targets = build_soft_targets(labels, GRID_DIM, distance_sigma)
                                distance_loss = soft_cross_entropy(outputs, soft_targets)
                                loss = loss + distance_loss_weight * distance_loss
                            else:
                                distance_loss = torch.tensor(0.0, device=device)
                            val_loss += loss.item()
                            val_base_loss_total += base_loss.item()
                            val_distance_loss_total += distance_loss.item()
                        if rank == 0:
                            pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})
                        # Calculate top-k accuracy
                        _, topk = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                        correct += (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
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
                
                val_loss /= len(val_loader)
                val_base_loss = val_base_loss_total / len(val_loader)
                val_distance_loss = val_distance_loss_total / len(val_loader)                
                val_acc = correct / total
                val_iou = total_iou / len(val_loader)
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
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {train_loss:.4f} "
                    f"(base {train_base_loss:.4f}, dist {train_distance_loss:.4f}), "
                    f"Top-{POSITIVE_CELLS} Train Acc: {train_acc:.2%}, "
                    f"Val Loss: {val_loss:.4f} "
                    f"(base {val_base_loss:.4f}, dist {val_distance_loss:.4f}), "
                    f"Top-{POSITIVE_CELLS} Val Acc: {val_acc:.2%}, "
                    f"Train IoU: {train_iou:.2f}, Val IoU: {val_iou:.2f}"
                )
                with open('loss_'+unique_name+'.json', 'a') as f:
                    json.dump(losses, f)
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
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=str, default="improve_v2")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "infer"])
    parser.add_argument('--viz', action='store_true', help="Save 10x10 Pred vs GT grid visualizations")
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--amp', action='store_true', default=True, help="Enable mixed precision training/inference")
    parser.add_argument('--viz_dir', type=str, default="viz_grids")
    parser.add_argument('--infer_split', type=str, default="val", choices=["train", "val"])
    parser.add_argument('--infer_out', type=str, default="inference_outputs.json")
    parser.add_argument('--distance_loss_weight', type=float, default=0.1,
                        help="Weight for distance-aware soft target loss.")
    parser.add_argument('--distance_sigma', type=float, default=0.8,
                        help="Gaussian sigma (grid cells) for distance-aware loss.")
    
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
        base_folder+'all_train_metas_v3_modelpred',
        base_folder+'all_train_basemaps_segmented_v3_modelpred',
        base_folder+'all_train_maps_segmented_v3_modelpred_UNITR',
        transform_base, transform_gen
    )
    val_dataset = MapDataset(
        base_folder+'all_val_metas_v3_modelpred',
        base_folder+'all_val_basemaps_segmented_v3_modelpred', 
        base_folder+'all_val_maps_segmented_v3_modelpred_UNITR',
        transform_base, transform_gen
    )

    # Model and training setup
    model = GridClassifier()
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
              args.validate_every, args.amp, args.distance_loss_weight,
              args.distance_sigma),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()