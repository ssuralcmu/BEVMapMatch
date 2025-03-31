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

# Dataset Class (unchanged)
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
        try:
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

            location_x = metas['perturbation'][0]
            location_y = metas['perturbation'][1]

            location_normalized = torch.tensor([location_x / 500.0, location_y / 500.0], dtype=torch.float32)

            return stitched_img, basemap_img, location_normalized
        except Exception as e:
            return torch.zeros(3, 500, 500), torch.zeros(3, 500, 500), torch.zeros(2)


# Focal Loss for Sparse Supervision (added)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target_heatmap):
        pred_prob = torch.sigmoid(pred_logits)
        pt = target_heatmap * pred_prob + (1 - target_heatmap) * (1 - pred_prob)
        focal_weight = (1 - pt).pow(self.gamma)
        loss_bce = F.binary_cross_entropy_with_logits(pred_logits, target_heatmap, reduction='none')
        loss_focal = focal_weight * loss_bce
        return loss_focal.mean()


import torch.nn.functional as F

class LocationModel(nn.Module):
    def __init__(self):
        super(LocationModel, self).__init__()
        
        # Feature extraction backbone (ResNet18)
        resnet_stitched = models.resnet18(pretrained=True)
        resnet_basemap = models.resnet18(pretrained=True)
        
        # Shared feature extraction layers for BEV and Base Maps
        self.stitched_features = nn.Sequential(*list(resnet_stitched.children())[:-2])
        self.basemap_features = nn.Sequential(*list(resnet_basemap.children())[:-2])
        
        # Projection layer to unify feature dimensions
        self.feature_projection = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, stitched_map, base_map):
        # Extract features from both inputs using shared backbone layers
        stitched_features = self.stitched_features(stitched_map)  # [B x C x H_s x W_s]
        base_features = self.basemap_features(base_map)          # [B x C x H_b x W_b]
        
        # Project features to a unified dimension for correlation computation
        stitched_proj = F.normalize(self.feature_projection(stitched_features), p=2, dim=1)  # Normalize features
        base_proj = F.normalize(self.feature_projection(base_features), p=2, dim=1)          # Normalize features
        
        # Compute correlation using matrix multiplication (efficient implementation)
        B, C, H_s, W_s = stitched_proj.shape
        _, _, H_b, W_b = base_proj.shape
        
        stitched_flattened = stitched_proj.view(B, C, -1)   # [B x C x (H_s * W_s)]
        base_flattened = base_proj.view(B, C, -1)           # [B x C x (H_b * W_b)]
        
        correlation_matrix = torch.bmm(stitched_flattened.transpose(1, 2), base_flattened)  # [B x (H_s*W_s) x (H_b*W_b)]
        
        # Reshape correlation matrix into score maps matching heatmap_targets
        score_maps = correlation_matrix.view(B, H_s * W_s, H_b, W_b).mean(dim=1)  # [B x H_b x W_b]
        
        # Upsample score_maps to match heatmap_targets dimensions
        score_maps_upsampled = F.interpolate(score_maps.unsqueeze(1), size=(base_map.shape[2], base_map.shape[3]), mode='bilinear', align_corners=False)
        
        return score_maps_upsampled.squeeze(1)  # [B x H x W]


# Location Model with Matmul-Based Correlation and Heatmaps (updated)
# class LocationModel(nn.Module):
#     def __init__(self):
#         super(LocationModel, self).__init__()
        
#         # Feature extraction backbone (ResNet18)
#         resnet_stitched = models.resnet18(pretrained=True)
#         resnet_basemap = models.resnet18(pretrained=True)
        
#         # Shared feature extraction layers for BEV and Base Maps
#         self.stitched_features = nn.Sequential(*list(resnet_stitched.children())[:-2])
#         self.basemap_features = nn.Sequential(*list(resnet_basemap.children())[:-2])
        
#         # Projection layer to unify feature dimensions
#         self.feature_projection = nn.Conv2d(512, 256, kernel_size=1)

#     def forward(self, stitched_map, base_map):
#         # Extract features from both inputs using shared backbone layers
#         stitched_features = self.stitched_features(stitched_map)  # [B x C x H_s x W_s]
#         base_features = self.basemap_features(base_map)          # [B x C x H_b x W_b]
        
#         # Project features to a unified dimension for correlation computation
#         stitched_proj = F.normalize(self.feature_projection(stitched_features), p=2, dim=1)  # Normalize features
#         base_proj = F.normalize(self.feature_projection(base_features), p=2, dim=1)          # Normalize features
        
#         # Compute correlation using matrix multiplication (efficient implementation)
#         B, C, H_s, W_s = stitched_proj.shape
#         _, _, H_b, W_b = base_proj.shape
        
#         stitched_flattened = stitched_proj.view(B, C, -1)   # [B x C x (H_s * W_s)]
#         base_flattened = base_proj.view(B, C, -1)           # [B x C x (H_b * W_b)]
        
#         correlation_matrix = torch.bmm(stitched_flattened.transpose(1, 2), base_flattened)  # [B x (H_s*W_s) x (H_b*W_b)]
        
#         # Reshape correlation matrix into score maps matching heatmap_targets
#         score_maps = correlation_matrix.view(B, H_s * W_s, H_b, W_b).mean(dim=1)  # [B x H_b x W_b]
        
#         return score_maps



def generate_heatmaps(locations_normalized: torch.Tensor, map_shape: tuple, sigma: float = 3.0, device: torch.device = "cuda"):
    """
    Generate Gaussian heatmaps centered at normalized locations.

    Args:
        locations_normalized (torch.Tensor): Normalized [x, y] coordinates (on device).
        map_shape (tuple): Shape of the map (height and width).
        sigma (float): Standard deviation of Gaussian.
        device (torch.device): Device to place tensors on.

    Returns:
        torch.Tensor: Heatmap tensor on the specified device.
    """
    height, width = map_shape
    x_coords_normed, y_coords_normed = locations_normalized[:, 0], locations_normalized[:, 1]
    
    # Convert normalized coordinates to pixel indices
    x_coords_pixel = (x_coords_normed * width).long().to(device)
    y_coords_pixel = (y_coords_normed * height).long().to(device)

    heatmaps_batch = []
    
    # Create meshgrid directly on the specified device
    yy_grid, xx_grid = torch.meshgrid(
        torch.arange(height, device=device), 
        torch.arange(width, device=device), 
        indexing='ij'
    )
    
    for i in range(len(x_coords_pixel)):
        x_center_pixel, y_center_pixel = x_coords_pixel[i], y_coords_pixel[i]
        
        # Compute squared distances and Gaussian heatmap
        dist_sq_grid = (xx_grid - x_center_pixel) ** 2 + (yy_grid - y_center_pixel) ** 2
        gaussian_heatmap_single_sample = torch.exp(-dist_sq_grid / (2 * sigma ** 2))
        
        heatmaps_batch.append(gaussian_heatmap_single_sample.unsqueeze(0))

    return torch.cat(heatmaps_batch)



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12378'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_dataloader(rank, world_size, dataset, batch_size=16, num_workers=10):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_model(rank, world_size, num_epochs, model, criterion, optimizer, train_dataset, val_dataset, batch_size, lr, version, fraction, checkpoint_path, seed):
    setup(rank, world_size)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    print("Checkpoint path: ", checkpoint_path)

    # Handle training fraction and checkpoint loading
    if fraction >= 1.0:
        print("FULL TRAINING")
        if checkpoint_path is None:
            print("Starting from scratch")
            start_epoch = 0
            losses = {"train": [], "val": []}
            subset_indices = torch.randperm(len(train_dataset))
        elif os.path.exists(checkpoint_path):
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
        elif os.path.exists(checkpoint_path):
            print("Resuming training")
            checkpoint = torch.load(checkpoint_path)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint['losses']
            subset_indices = checkpoint['subset_indices']
            train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

    train_dataloader = create_dataloader(rank, world_size, train_dataset, batch_size)
    val_dataloader = create_dataloader(rank, world_size, val_dataset, batch_size)

    unique_name = f"v{version}-lr{lr}-bs{batch_size}-frac{fraction}-seed{seed}"

    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_dataloader.sampler.set_epoch(epoch)

        pbar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
        for stitched_imgs, basemap_imgs, locations in pbar_train:
            stitched_imgs = stitched_imgs.to(device)
            basemap_imgs = basemap_imgs.to(device)
            locations_normalized = locations.to(device)

            # Generate target heatmap
            heatmap_targets = generate_heatmaps(locations_normalized, basemap_imgs.shape[-2:], sigma=3.0, device=device)

            optimizer.zero_grad()

            # Forward pass
            score_maps = model(stitched_imgs, basemap_imgs)

            # Compute focal loss using heatmaps
            loss_focal = criterion(score_maps.view(-1), heatmap_targets.view(-1))
            
            # Backward pass and optimization
            loss_focal.backward()
            optimizer.step()

            running_loss += loss_focal.item()
            
            if rank == 0:
                pbar_train.set_postfix({'train_loss': running_loss / (pbar_train.n + 1)})

        train_loss_avg = running_loss / len(train_dataloader)
        losses["train"].append([epoch, train_loss_avg])

        # Validation every two epochs or at the start
        if epoch % 20 == 0 or epoch == start_epoch:
            model.eval()
            val_loss_total = 0.0

            with torch.no_grad():
                pbar_val = tqdm(val_dataloader, desc="Validation", disable=rank != 0)
                for stitched_imgs_val, basemap_imgs_val, locations_val in pbar_val:
                    stitched_imgs_val = stitched_imgs_val.to(device)
                    basemap_imgs_val = basemap_imgs_val.to(device)
                    locations_normalized_val = locations_val.to(device)

                    # Generate target heatmap for validation
                    heatmap_targets_val = generate_heatmaps(locations_normalized_val,
                                                            basemap_imgs_val.shape[-2:], sigma=3.0).to(device)

                    # Forward pass for validation
                    score_maps_val = model(stitched_imgs_val, basemap_imgs_val)

                    # Compute validation focal loss
                    loss_focal_val_batch = criterion(score_maps_val.view(-1), heatmap_targets_val.view(-1))
                    val_loss_total += loss_focal_val_batch.item()

                    if rank == 0:
                        pbar_val.set_postfix({'val_loss': val_loss_total / (pbar_val.n + 1)})

                val_loss_avg = val_loss_total / len(val_dataloader)
                losses["val"].append([epoch, val_loss_avg])

                # Save best validation model
                if rank == 0 and val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'losses': losses,
                        'subset_indices': subset_indices,
                    }, f'best_map_location_model_{unique_name}.pth')

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file to resume training')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--version', type=int, default=2, help='Version of the model')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world_size = torch.cuda.device_count()
    
    base_folder = "/data1/"

    transform_base = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
    ])

    transform_gen = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
    ])

    # Load training and validation datasets
    train_dataset = MapDataset(
        os.path.join(base_folder, 'all_train_metas_v2_500m'),
        os.path.join(base_folder, 'all_train_basemaps_v2_500m'),
        os.path.join(base_folder, 'all_train_maps_gt_v2_500m/map/'),
        transform_base=transform_base,
        transform_gen=transform_gen
    )

    val_dataset = MapDataset(
        os.path.join(base_folder, 'all_val_metas_v2_500m'),
        os.path.join(base_folder, 'all_val_basemaps_v2_500m'),
        os.path.join(base_folder, 'all_val_maps_gt_v2_500m/map/'),
        transform_base=transform_base,
        transform_gen=transform_gen
    )

    # Initialize the model
    model = LocationModel()
    
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    # Use focal loss for sparse supervision
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Start distributed training using multiple GPUs
    mp.spawn(
        train_model,
        args=(world_size,
              args.num_epochs,
              model,
              criterion,
              optimizer,
              train_dataset,
              val_dataset,
              args.batch_size,
              args.lr,
              args.version,
              args.train_fraction,
              args.checkpoint,
              args.seed),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
