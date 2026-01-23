# Correlation-based map matcher with rotation sweep.
# Derived from neural_map_matcher_v8_cross_attention.py.
import os
import json
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchvision import models, transforms
from torchvision.transforms import functional as TF


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

        basemap_img = np.array(basemap_img)
        basemap_img = np.flipud(basemap_img)
        basemap_img = Image.fromarray(basemap_img)

        if self.transform_gen:
            stitched_img = self.transform_gen(stitched_img)
        if self.transform_base:
            basemap_img = self.transform_base(basemap_img)

        metas = np.load(metas_path, allow_pickle=True).item()
        return stitched_img, basemap_img, metas, stitched_img_path, basemap_img_path, metas_path


def collate_map_batch(batch):
    stitched, basemap, metas, stitched_path, basemap_path, metas_path = zip(*batch)
    stitched = torch.stack(stitched)
    basemap = torch.stack(basemap)
    return stitched, basemap, list(metas), list(stitched_path), list(basemap_path), list(metas_path)


def _grouped_correlation(feature_map, template):
    """Compute per-sample correlation via grouped convolution.

    feature_map: (B, C, H, W)
    template: (B, C, h, w)
    Returns: (B, 1, H-h+1, W-w+1)
    """
    batch, channels, height, width = feature_map.shape
    _, _, t_h, t_w = template.shape

    feature_map = feature_map.reshape(1, batch * channels, height, width)
    weights = template.reshape(batch, channels, t_h, t_w)
    corr = F.conv2d(feature_map, weights, groups=batch)
    return corr.reshape(batch, 1, corr.shape[-2], corr.shape[-1])


def _normalized_correlation(feature_map, template, eps=1e-6):
    """L2-normalized correlation for stability."""
    template_norm = template / (template.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + eps)
    feature_norm = feature_map / (feature_map.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + eps)
    return _grouped_correlation(feature_norm, template_norm)


class CorrRotationMatcher(nn.Module):
    def __init__(self, angles=None, freeze_backbone=False, normalize_corr=False):
        super().__init__()
        self.angles = angles or list(range(0, 360, 10))
        self.normalize_corr = normalize_corr

        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, stitched, basemap):
        # Extract features
        base_feat = self.backbone(basemap)   # (B, C, Hb, Wb)

        best_corr = None
        best_angle = None
        for angle in self.angles:
            rotated = TF.rotate(stitched, angle, interpolation=TF.InterpolationMode.BILINEAR)
            query_feat = self.backbone(rotated)  # (B, C, Hq, Wq)

            if self.normalize_corr:
                corr = _normalized_correlation(base_feat, query_feat)  # (B, 1, Ho, Wo)
            else:
                corr = _grouped_correlation(base_feat, query_feat)
            if best_corr is None:
                best_corr = corr
                best_angle = torch.full((stitched.size(0),), angle, device=stitched.device)
            else:
                better = corr > best_corr
                best_angle = torch.where(better.flatten(1).any(dim=1),
                                         torch.full_like(best_angle, angle),
                                         best_angle)
                best_corr = torch.maximum(best_corr, corr)

        return best_corr.squeeze(1), best_angle


def build_target_heatmap(metas, output_h, output_w, input_h=1000, input_w=1000, patch_size=100, sigma=2.5):
    """Create a gaussian target heatmap centered at the patch top-left position."""
    batch_size = len(metas)
    heatmaps = torch.zeros(batch_size, output_h, output_w)

    for i in range(batch_size):
        center_x, center_y = input_w // 2, input_h // 2
        x_val = center_x - metas[i]['perturbation'][0]
        y_val = center_y - metas[i]['perturbation'][1]

        top_left_x = x_val - patch_size / 2
        top_left_y = y_val - patch_size / 2

        scale_x = output_w / (input_w - patch_size)
        scale_y = output_h / (input_h - patch_size)

        cx = int(np.clip(top_left_x * scale_x, 0, output_w - 1))
        cy = int(np.clip(top_left_y * scale_y, 0, output_h - 1))

        ys = torch.arange(output_h).float()
        xs = torch.arange(output_w).float()
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        heatmaps[i] = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    return heatmaps


def contrastive_info_nce(logits, targets, temperature=0.1):
    """InfoNCE loss on flattened heatmap logits with a single positive per sample."""
    batch, height, width = logits.shape
    logits = logits.reshape(batch, -1) / temperature
    targets = targets.reshape(batch, -1)

    positive_idx = targets.argmax(dim=1)
    loss = F.cross_entropy(logits, positive_idx)
    return loss


def compute_gt_indices(metas, output_h, output_w, input_h=1000, input_w=1000, patch_size=100):
    gt_x = []
    gt_y = []
    for meta in metas:
        center_x, center_y = input_w // 2, input_h // 2
        x_val = center_x - meta['perturbation'][0]
        y_val = center_y - meta['perturbation'][1]

        top_left_x = x_val - patch_size / 2
        top_left_y = y_val - patch_size / 2

        scale_x = output_w / (input_w - patch_size)
        scale_y = output_h / (input_h - patch_size)

        cx = int(np.clip(top_left_x * scale_x, 0, output_w - 1))
        cy = int(np.clip(top_left_y * scale_y, 0, output_h - 1))
        gt_x.append(cx)
        gt_y.append(cy)

    return torch.tensor(gt_x), torch.tensor(gt_y)


def compute_localization_metrics(logits, metas):
    batch_size, output_h, output_w = logits.shape
    gt_x, gt_y = compute_gt_indices(metas, output_h, output_w)
    gt_x = gt_x.to(logits.device)
    gt_y = gt_y.to(logits.device)

    flat = logits.flatten(1)
    pred_idx = flat.argmax(dim=1)
    pred_y = torch.div(pred_idx, output_w, rounding_mode='trunc')
    pred_x = pred_idx % output_w

    top1 = ((pred_x == gt_x) & (pred_y == gt_y)).float().mean().item()

    dx = (pred_x - gt_x).float()
    dy = (pred_y - gt_y).float()
    dist_px = torch.sqrt(dx ** 2 + dy ** 2).mean().item()

    top9 = torch.topk(flat, 9, dim=1).indices
    preds = torch.zeros_like(flat)
    preds.scatter_(1, top9, 1)
    preds = preds.view(batch_size, output_h, output_w)

    gt_mask = torch.zeros_like(preds)
    for i in range(batch_size):
        x0 = max(0, gt_x[i].item() - 1)
        x1 = min(output_w, gt_x[i].item() + 2)
        y0 = max(0, gt_y[i].item() - 1)
        y1 = min(output_h, gt_y[i].item() + 2)
        gt_mask[i, y0:y1, x0:x1] = 1

    intersection = (preds * gt_mask).sum(dim=(1, 2))
    union = ((preds + gt_mask) > 0).sum(dim=(1, 2))
    iou = (intersection / union).mean().item()

    return top1, iou, dist_px


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '17739'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def create_dataloader(rank, world_size, dataset, batch_size=16, num_workers=10):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_map_batch,
    )


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_model(rank, world_size, num_epochs, model, optimizer, scheduler,
               train_dataset, val_dataset, batch_size, lr, version, fraction,
               checkpoint_path, seed, heatmap_weight=1.0, contrastive_weight=0.5,
               contrastive_temp=0.1, target_sigma=2.5):
    setup(rank, world_size)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

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

    unique_name = f"corr_rotation_v{version}-lr{lr}-bs{batch_size}-frac{fraction}-seed{seed}"

    best_val_loss = min(loss for epoch, loss, _ in losses["val"]) if losses["val"] else float('inf')
    best_train_loss = min(loss for epoch, loss, _ in losses["train"]) if losses["train"] else float('inf')

    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            train_loader.sampler.set_epoch(epoch)
            total_loss = 0.0
            total_top1 = 0.0
            total_iou = 0.0
            total_dist = 0.0
            total_count = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
            for stitched, basemap, metas, *_ in pbar:
                stitched = stitched.to(device)
                basemap = basemap.to(device)

                corr_map, _ = model(stitched, basemap)
                target = build_target_heatmap(
                    metas, corr_map.shape[-2], corr_map.shape[-1], sigma=target_sigma
                ).to(device)

                heatmap_loss = criterion(corr_map, target)
                contrastive_loss = contrastive_info_nce(corr_map, target, temperature=contrastive_temp)
                loss = heatmap_weight * heatmap_loss + contrastive_weight * contrastive_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                top1, iou, dist_px = compute_localization_metrics(corr_map.detach(), metas)
                batch_size = stitched.size(0)
                total_top1 += top1 * batch_size
                total_iou += iou * batch_size
                total_dist += dist_px * batch_size
                total_count += batch_size

                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

            metrics_tensor = torch.tensor(
                [total_loss, total_top1, total_iou, total_dist, total_count],
                device=device
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

            train_loss = metrics_tensor[0].item() / len(train_loader)
            train_top1 = metrics_tensor[1].item() / max(1, metrics_tensor[4].item())
            train_iou = metrics_tensor[2].item() / max(1, metrics_tensor[4].item())
            train_dist = metrics_tensor[3].item() / max(1, metrics_tensor[4].item())
            losses["train"].append([epoch, train_loss, train_top1, train_iou, train_dist])

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
                torch.save(checkpoint, 'latest_map_location_model_train_' + unique_name + '.pth')

            if epoch % 1 == 0 or epoch == start_epoch:
                model.eval()
                val_loss = 0.0
                val_top1 = 0.0
                val_iou = 0.0
                val_dist = 0.0
                val_count = 0
                with torch.no_grad():
                    pbar = tqdm(val_loader, desc="Validation", disable=rank != 0)
                    for stitched, basemap, metas, *_ in pbar:
                        stitched = stitched.to(device)
                        basemap = basemap.to(device)
                        corr_map, _ = model(stitched, basemap)
                        target = build_target_heatmap(
                            metas, corr_map.shape[-2], corr_map.shape[-1], sigma=target_sigma
                        ).to(device)
                        heatmap_loss = criterion(corr_map, target)
                        contrastive_loss = contrastive_info_nce(corr_map, target, temperature=contrastive_temp)
                        val_loss += (heatmap_weight * heatmap_loss + contrastive_weight * contrastive_loss).item()
                        top1, iou, dist_px = compute_localization_metrics(corr_map, metas)
                        batch_size = stitched.size(0)
                        val_top1 += top1 * batch_size
                        val_iou += iou * batch_size
                        val_dist += dist_px * batch_size
                        val_count += batch_size
                        if rank == 0:
                            pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})

                val_metrics = torch.tensor(
                    [val_loss, val_top1, val_iou, val_dist, val_count],
                    device=device
                )
                dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
                val_loss = val_metrics[0].item() / len(val_loader)
                val_top1 = val_metrics[1].item() / max(1, val_metrics[4].item())
                val_iou = val_metrics[2].item() / max(1, val_metrics[4].item())
                val_dist = val_metrics[3].item() / max(1, val_metrics[4].item())
                losses["val"].append([epoch, val_loss, val_top1, val_iou, val_dist])

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
                    torch.save(checkpoint, 'best_map_location_model_val_' + unique_name + '.pth')

            if rank == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {train_loss:.4f}, Train Top1: {train_top1:.2%}, "
                    f"Train IoU: {train_iou:.2%}, Train Dist(px): {train_dist:.2f}, "
                    f"Val Loss: {val_loss:.4f}, Val Top1: {val_top1:.2%}, "
                    f"Val IoU: {val_iou:.2%}, Val Dist(px): {val_dist:.2f}"
                )
                with open('loss_' + unique_name + '.json', 'a') as f:
                    json.dump(losses, f)
        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
    cleanup()


def run_inference(model, loader, device, output_json):
    model.eval()
    outputs = []
    with torch.no_grad():
        for stitched, basemap, metas, stitched_path, basemap_path, metas_path in tqdm(loader, desc="Infer"):
            stitched = stitched.to(device)
            basemap = basemap.to(device)
            corr_map, best_angle = model(stitched, basemap)
            corr_flat = corr_map.flatten(1)
            max_idx = corr_flat.argmax(dim=1)
            outputs.extend([
                {
                    "stitched_img_path": stitched_path[i],
                    "basemap_img_path": basemap_path[i],
                    "metas_path": metas_path[i],
                    "best_angle": float(best_angle[i].item()),
                    "best_index": int(max_idx[i].item()),
                    "score": float(corr_flat[i, max_idx[i]].item()),
                }
                for i in range(len(stitched_path))
            ])

    with open(output_json, "w") as f:
        json.dump({"outputs": outputs}, f, indent=2)
    print(f"Saved outputs to {output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=str, default="10_corr_rotation")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "infer"])
    parser.add_argument('--infer_split', type=str, default="val", choices=["train", "val"])
    parser.add_argument('--infer_out', type=str, default="corr_rotation_outputs.json")
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--angle_step', type=int, default=10)
    parser.add_argument('--normalize_corr', action='store_true')
    parser.add_argument('--contrastive_weight', type=float, default=0.5)
    parser.add_argument('--contrastive_temp', type=float, default=0.1)
    parser.add_argument('--target_sigma', type=float, default=2.5)

    args = parser.parse_args()

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

    base_folder = "/data1/"
    train_dataset = MapDataset(
        base_folder + 'all_train_metas_v3',
        base_folder + 'all_train_basemaps_segmented_v3',
        base_folder + 'all_train_maps_segmented_gt_v3/map/',
        transform_base, transform_gen
    )
    val_dataset = MapDataset(
        base_folder + 'all_val_metas_v3',
        base_folder + 'all_val_basemaps_segmented_v3',
        base_folder + 'all_val_maps_segmented_gt_v3/map/',
        transform_base, transform_gen
    )

    if args.train_fraction < 1.0:
        num_samples = int(len(train_dataset) * args.train_fraction)
        subset_indices = torch.randperm(len(train_dataset))[:num_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

    angles = list(range(0, 360, args.angle_step))
    model = CorrRotationMatcher(
        angles=angles,
        freeze_backbone=args.freeze_backbone,
        normalize_corr=args.normalize_corr
    )

    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    if args.mode == "train":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        world_size = torch.cuda.device_count()
        mp.spawn(
            train_model,
            args=(world_size, args.num_epochs, model, optimizer, scheduler,
                  train_dataset, val_dataset, args.batch_size, args.lr,
                  args.version, args.train_fraction, args.checkpoint, args.seed,
                  1.0, args.contrastive_weight, args.contrastive_temp, args.target_sigma),
            nprocs=world_size,
            join=True
        )
    else:
        if args.checkpoint is None or not os.path.exists(args.checkpoint):
            raise ValueError("For --mode infer, you must provide a valid --checkpoint path.")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.to(device)
        infer_dataset = val_dataset if args.infer_split == "val" else train_dataset
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_map_batch,
        )
        run_inference(model, infer_loader, device, args.infer_out)


if __name__ == '__main__':
    main()
