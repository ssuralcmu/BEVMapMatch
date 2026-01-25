# This code is for coarse matching- to find the best grid.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import json
import argparse
from torchvision.ops import sigmoid_focal_loss
# from segmentation_models_pytorch.losses import DiceLoss

from pathlib import Path
import matplotlib.pyplot as plt

GRID_DIM = 10
TARGET_BLOCK = 1
POSITIVE_CELLS = TARGET_BLOCK * TARGET_BLOCK
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
        meters_per_patch = 500.0
        pixels_per_meter = basemap_size_px / meters_per_patch
        x_val, y_val = perturbation_to_pixel(
            metas['perturbation'],
            center_x,
            center_y,
            pixels_per_meter,
        )
        
        grid_size = 100  # Each grid is 100x100 pixels
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GridClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        # # Freeze feature extractor
        # for p in self.feature_extractor.parameters():
        #     p.requires_grad = False

        # Pool basemap to 10x10
        self.grid_pool = nn.AdaptiveAvgPool2d((10, 10))

        # Cross-attention (embed_dim matches ResNet features: 512)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )

        # Positional encoding (UPDATED): batch-safe + ViT-style init
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, 512))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # No mask needed if you want full attention
        self.attn_mask = None

        self.fc = nn.Linear(512, 100)

    def forward(self, stitched, basemap):
        # Feature extraction
        stitched_feat = self.feature_extractor(stitched)   # (B,512,h1,w1)
        basemap_feat  = self.feature_extractor(basemap)    # (B,512,h2,w2)

        # Basemap -> 10x10 grid
        grid = self.grid_pool(basemap_feat)                # (B,512,10,10)

        # 3x3 local context aggregation per grid cell
        grid_3x3 = F.unfold(grid, kernel_size=3, padding=1)                  # (B,512*9,100)
        grid_3x3 = grid_3x3.view(grid_3x3.size(0), 512, 9, 100).mean(dim=2)  # (B,512,100)

        # Attention inputs
        query = F.adaptive_avg_pool2d(stitched_feat, (1, 1)).flatten(1)      # (B,512)
        query = query.unsqueeze(1)                                           # (B,1,512)

        key_value = grid_3x3.permute(0, 2, 1)                                 # (B,100,512)
        key = value = key_value + self.pos_embed                              # (B,100,512)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=query,   # (B,1,512)
            key=key,       # (B,100,512)
            value=value,   # (B,100,512)
            attn_mask=self.attn_mask
        )

        scores = self.fc(attn_out.squeeze(1))  # (B,100)
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '19453'
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
                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
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
                            val_loss += criterion(outputs, labels).item()
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Top-{POSITIVE_CELLS} Train Acc: {train_acc:.2%}, Val Loss: {val_loss:.4f}, Top-{POSITIVE_CELLS} Val Acc: {val_acc:.2%}, Train IoU: {train_iou:.2f}, Val IoU: {val_iou:.2f}")
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=str, default="9_faster")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "infer"])
    parser.add_argument('--viz', action='store_true', help="Save 10x10 Pred vs GT grid visualizations")
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--amp', action='store_true', default=True, help="Enable mixed precision training/inference")
    parser.add_argument('--viz_dir', type=str, default="viz_grids")
    parser.add_argument('--infer_split', type=str, default="val", choices=["train", "val"])
    parser.add_argument('--infer_out', type=str, default="inference_outputs.json")

    args = parser.parse_args()

    # Data transforms
    transform_base = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_gen = transforms.Compose([
        transforms.Resize((100, 100)),
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