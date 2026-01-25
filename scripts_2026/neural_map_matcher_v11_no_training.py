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
        basemap_size_px = basemap_img.shape[1]
        meters_per_patch = 500.0
        pixels_per_meter = basemap_size_px / meters_per_patch

        x_val, y_val = perturbation_to_pixel(
            metas['perturbation'],
            center_x,
            center_y,
            pixels_per_meter,
        )

        grid_size = 100  # Each grid is 100x100 pixels (given basemap is resized to 1000x1000)
        grid_x = int(x_val // grid_size)
        grid_y = int(y_val // grid_size)

        grid_dim = GRID_DIM
        target_block = TARGET_BLOCK

        grid_label = torch.zeros(grid_dim, grid_dim)

        grid_x = int(np.clip(grid_x, 0, grid_dim - 1))
        grid_y = int(np.clip(grid_y, 0, grid_dim - 1))

        max_anchor = grid_dim - target_block
        i0 = min(grid_y, max_anchor)
        j0 = min(grid_x, max_anchor)

        grid_label[i0:i0+target_block, j0:j0+target_block] = 1.0

        return stitched_img, basemap_img, grid_label.flatten().float(), stitched_img_path, basemap_img_path, metas_path


# ============================================================
# OPTION C: Frozen ResNet-18 retrieval matcher (NO TRAINING)
# ============================================================

class ResNet18Embedder(nn.Module):
    """
    Frozen ResNet-18 -> global pooled embedding (512-d).
    Input should already be ImageNet-normalized (your transforms already do this).
    """
    def __init__(self):
        super().__init__()
        rn = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(rn.children())[:-2])  # (B,512,h,w)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))                  # (B,512,1,1)

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        feat = self.backbone(x)
        emb = self.pool(feat).flatten(1)          # (B,512)
        emb = F.normalize(emb, dim=1)             # cosine space
        return emb


def strip_module_prefix(state_dict):
    """Handle checkpoints saved from DDP (keys start with 'module.')."""
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


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

    img = np.ones((10, 10, 3), dtype=np.float32)  # white

    overlap = (pred == 1) & (gt == 1)
    gt_only = (pred == 0) & (gt == 1)
    pred_only = (pred == 1) & (gt == 0)

    img[gt_only] = np.array([1.0, 0.2, 0.2])   # red
    img[pred_only] = np.array([0.2, 0.2, 1.0]) # blue
    img[overlap] = np.array([0.6, 0.2, 0.8])   # purple

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


def visualize_basemap_topk(basemap_path, metas_path, topk_idx, topk_scores, save_path):
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
    topk_scores = np.array(topk_scores)

    pred_x = (topk_idx % GRID_DIM + 0.5) * cell_width
    pred_y = (topk_idx // GRID_DIM + 0.5) * cell_height

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(basemap_np)
    plt.axis("off")

    plt.scatter(gt_x, gt_y, s=220, c="blue", edgecolors="white", linewidths=1.5, label="GT")
    plt.scatter(pred_x, pred_y, s=80, c="green", edgecolors="white", linewidths=1.0, label="Pred Top-k")

    for x, y, sc in zip(pred_x, pred_y, topk_scores):
        plt.text(
            x + 4, y - 4, f"{sc:.3f}",
            color="white", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none")
        )

    plt.legend(loc="upper right", framealpha=0.8)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def run_inference_resnet_retrieval(
    embedder,
    dataset,
    batch_size=64,
    num_workers=10,
    device=None,
    viz=False,
    viz_dir="viz_grids",
    output_json="inference_outputs.json",
    amp=True,
):
    """
    NO TRAINING / NO CHECKPOINT.

    For each sample:
      - Compute stitched embedding (1 vector)
      - Split basemap into 10x10 cells (each 100x100), compute embedding per cell (100 vectors)
      - Similarity = cosine(stitched_emb, cell_emb) = dot product (since normalized)
      - Predict top-k indices and compute your same metrics (Top-k hit acc, IoU)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    embedder = embedder.to(device)
    embedder.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    correct_topk = 0
    total = 0
    total_iou = 0.0

    viz_dir = Path(viz_dir)
    outputs_list = []

    cell_px = 100  # because basemap resized to 1000x1000 and GRID_DIM=10

    pbar = tqdm(loader, desc="Inference (ResNet18 Retrieval)")
    for stitched, basemap, labels, stitched_img_path, basemap_img_path, metas_path in pbar:
        stitched = stitched.to(device, non_blocking=True)
        basemap = basemap.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)  # (B,100)

        B = stitched.size(0)

        # 1) stitched embeddings
        with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
            stitched_emb = embedder(stitched)  # (B,512), normalized

        # 2) basemap cell embeddings: build (B,100,3,100,100)
        #    We slice normalized basemap tensor directly.
        cells = []
        for gy in range(GRID_DIM):
            y0, y1 = gy * cell_px, (gy + 1) * cell_px
            for gx in range(GRID_DIM):
                x0, x1 = gx * cell_px, (gx + 1) * cell_px
                cells.append(basemap[:, :, y0:y1, x0:x1])  # (B,3,100,100)
        basemap_cells = torch.stack(cells, dim=1)         # (B,100,3,100,100)
        basemap_cells_flat = basemap_cells.view(B * 100, 3, cell_px, cell_px)

        # embed all cells in one go
        with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
            cell_emb_flat = embedder(basemap_cells_flat)  # (B*100,512), normalized

        cell_emb = cell_emb_flat.view(B, 100, -1)         # (B,100,512)

        # 3) cosine similarity (dot product because normalized)
        # sims[b,i] = dot(stitched_emb[b], cell_emb[b,i])
        sims = torch.bmm(cell_emb, stitched_emb.unsqueeze(2)).squeeze(2)  # (B,100)

        # Top-k hit accuracy (k = POSITIVE_CELLS)
        _, topk = torch.topk(sims, POSITIVE_CELLS, dim=1)  # (B,k)
        batch_hits = (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
        correct_topk += batch_hits
        total += B

        # IoU with top-k predicted indices
        preds = torch.zeros_like(labels)
        preds.scatter_(1, topk, 1)

        intersection = (preds * labels).sum(dim=1)
        union = ((preds + labels) > 0).sum(dim=1)
        iou_percentage = (intersection / union * 100.0).mean().item()
        total_iou += iou_percentage

        # Optional visualization per-sample
        if viz:
            for b in range(B):
                gt_10x10 = labels[b].view(10, 10)
                pred_10x10 = preds[b].view(10, 10)

                sample_id = Path(metas_path[b]).stem
                save_path = viz_dir / f"{sample_id}_grid.png"
                visualize_pred_gt_grid(pred_10x10, gt_10x10, save_path)

                topk_scores = sims[b].gather(0, topk[b]).detach().cpu().numpy()
                basemap_save_path = viz_dir / f"{sample_id}_basemap_topk.png"
                visualize_basemap_topk(
                    basemap_img_path[b],
                    metas_path[b],
                    topk[b].detach().cpu().numpy(),
                    topk_scores,
                    basemap_save_path
                )

                stitched_save_path = viz_dir / f"{sample_id}_stitched.png"
                stitched_img = Image.open(stitched_img_path[b]).convert("RGB")
                stitched_save_path.parent.mkdir(parents=True, exist_ok=True)
                stitched_img.save(stitched_save_path)

        # Save raw outputs
        sims_list = sims.detach().cpu().numpy().tolist()
        topk_idx = topk.detach().cpu().numpy().tolist()

        for b in range(B):
            outputs_list.append({
                "stitched_img_path": stitched_img_path[b],
                "basemap_img_path": basemap_img_path[b],
                "metas_path": metas_path[b],
                "topk_idx": topk_idx[b],
                "sims": sims_list[b],  # 100 similarity scores
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

    print(f"[Inference-ResNet18] Top-{POSITIVE_CELLS} Acc: {final_topk:.2%}, Mean IoU%: {final_iou:.2f}")
    print(f"[Inference-ResNet18] Saved outputs to: {output_json}")
    if viz:
        print(f"[Inference-ResNet18] Saved visualizations to: {str(viz_dir)}")


# ============================================================
# (Your training code left as-is; not used in retrieval infer)
# ============================================================

class GridClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        self.grid_pool = nn.AdaptiveAvgPool2d((10, 10))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 100, 512))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.attn_mask = None
        self.fc = nn.Linear(512, 100)

    def forward(self, stitched, basemap):
        stitched_feat = self.feature_extractor(stitched)
        basemap_feat  = self.feature_extractor(basemap)

        grid = self.grid_pool(basemap_feat)

        grid_3x3 = F.unfold(grid, kernel_size=3, padding=1)
        grid_3x3 = grid_3x3.view(grid_3x3.size(0), 512, 9, 100).mean(dim=2)

        query = F.adaptive_avg_pool2d(stitched_feat, (1, 1)).flatten(1)
        query = query.unsqueeze(1)

        key_value = grid_3x3.permute(0, 2, 1)
        key = value = key_value + self.pos_embed

        attn_out, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=self.attn_mask
        )

        scores = self.fc(attn_out.squeeze(1))
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
            reduction="mean"
        )


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '19003'
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


def train_model(rank, world_size, num_epochs, model, criterion, optimizer, scheduler,
               train_dataset, val_dataset, batch_size, lr, version, fraction,
               checkpoint_path, seed, validate_every, amp):
    # (UNCHANGED TRAINING CODE – kept for structure compatibility)
    setup(rank, world_size)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = nn.BCEWithLogitsLoss().to(device)

    if fraction >= 1.0:
        print("FULL TRAINING")
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
            subset_indices = checkpoint['subset_indices']
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
            subset_indices = checkpoint['subset_indices']
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
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs = model(stitched, basemap)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss/(pbar.n+1)})

                _, topk = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                correct += (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
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
                    'subset_indices': subset_indices
                }
                torch.save(checkpoint, 'latest_map_location_model_train_'+unique_name+'.pth')

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

                        _, topk = torch.topk(outputs, POSITIVE_CELLS, dim=1)
                        correct += (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
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
                        'subset_indices': subset_indices
                    }
                    torch.save(checkpoint, 'best_map_location_model_val_'+unique_name+'.pth')

            if rank == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
                      f"Top-{POSITIVE_CELLS} Train Acc: {train_acc:.2%}, Val Loss: {val_loss:.4f}, "
                      f"Top-{POSITIVE_CELLS} Val Acc: {val_acc:.2%}, Train IoU: {train_iou:.2f}, Val IoU: {val_iou:.2f}")
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
    parser.add_argument('--version', type=str, default="11")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "infer"])
    parser.add_argument('--viz', action='store_true', help="Save 10x10 Pred vs GT grid visualizations")
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--amp', action='store_true', default=True, help="Enable mixed precision training/inference")
    parser.add_argument('--viz_dir', type=str, default="viz_grids")
    parser.add_argument('--infer_split', type=str, default="val", choices=["train", "val"])
    parser.add_argument('--infer_out', type=str, default="inference_outputs.json")
    parser.add_argument('--infer_method', type=str, default="retrieval",
                        choices=["retrieval", "checkpoint_model"],
                        help="retrieval = Option C (frozen ResNet18, no training). "
                             "checkpoint_model = use your trained GridClassifier checkpoint.")

    args = parser.parse_args()

    # Data transforms (unchanged)
    transform_base = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_gen = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets (unchanged)
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

    if args.mode == "infer":
        infer_dataset = val_dataset if args.infer_split == "val" else train_dataset

        if args.infer_method == "retrieval":
            # OPTION C: NO CHECKPOINT NEEDED
            embedder = ResNet18Embedder()
            run_inference_resnet_retrieval(
                embedder=embedder,
                dataset=infer_dataset,
                batch_size=args.batch_size,
                num_workers=10,
                device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                viz=args.viz,
                viz_dir=args.viz_dir,
                output_json=args.infer_out,
                amp=args.amp
            )
            return

        # Otherwise run your old checkpoint-based model (kept for compatibility)
        if args.checkpoint is None or not os.path.exists(args.checkpoint):
            raise ValueError("For --mode infer --infer_method checkpoint_model, you must provide a valid --checkpoint.")

        # Build original model (unchanged behavior)
        model = GridClassifier()
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        state = strip_module_prefix(state)
        model.load_state_dict(state, strict=True)

        # Reuse your old logits-based inference by quick wrapper:
        # NOTE: This branch is optional; you can delete if you only want retrieval.
        def run_inference_logits(model, dataset, batch_size=64, num_workers=10,
                                device=None, viz=False, viz_dir="viz_grids", output_json="inference_outputs.json", amp=True):
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device).eval()

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
            correct_topk = 0
            total = 0
            total_iou = 0.0
            viz_dir = Path(viz_dir)
            outputs_list = []

            with torch.no_grad():
                pbar = tqdm(loader, desc="Inference (Checkpoint Model)")
                for stitched, basemap, labels, stitched_img_path, basemap_img_path, metas_path in pbar:
                    stitched = stitched.to(device, non_blocking=True)
                    basemap = basemap.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
                        logits = model(stitched, basemap)

                    _, topk = torch.topk(logits, POSITIVE_CELLS, dim=1)
                    batch_hits = (labels.gather(1, topk).sum(dim=1) > 0).sum().item()
                    correct_topk += batch_hits
                    total += labels.size(0)

                    preds = torch.zeros_like(labels)
                    preds.scatter_(1, topk, 1)

                    intersection = (preds * labels).sum(dim=1)
                    union = ((preds + labels) > 0).sum(dim=1)
                    iou_percentage = (intersection / union * 100.0).mean().item()
                    total_iou += iou_percentage

                    if viz:
                        for b in range(labels.size(0)):
                            gt_10x10 = labels[b].view(10, 10)
                            pred_10x10 = preds[b].view(10, 10)
                            sample_id = Path(metas_path[b]).stem
                            save_path = viz_dir / f"{sample_id}_grid.png"
                            visualize_pred_gt_grid(pred_10x10, gt_10x10, save_path)

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

        run_inference_logits(
            model=model,
            dataset=infer_dataset,
            checkpoint_path=args.checkpoint if args.checkpoint else None,
            batch_size=args.batch_size,
            num_workers=10,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            viz=args.viz,
            viz_dir=args.viz_dir,
            output_json=args.infer_out,
            amp=args.amp
        )
        return

    # TRAIN path (unchanged)
    model = GridClassifier()
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args.num_epochs, model, None, optimizer, scheduler,
              train_dataset, val_dataset, args.batch_size, args.lr,
              args.version, args.train_fraction, args.checkpoint, args.seed,
              args.validate_every, args.amp),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
