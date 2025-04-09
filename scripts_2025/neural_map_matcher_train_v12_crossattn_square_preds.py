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
from segmentation_models_pytorch.losses import DiceLoss

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
            # import pdb; pdb.set_trace()
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
            
            # Convert coordinates to grid labels
            center_x, center_y = basemap_img.shape[1] // 2, basemap_img.shape[2] // 2
            x_val = center_x - metas['perturbation'][0]
            y_val = center_y - metas['perturbation'][1]
            
            grid_size = 100  # Each grid is 100x100 pixels
            grid_x = int(x_val // grid_size)
            grid_y = int(y_val // grid_size)
            
            # Create 10x10 grid mask for multi-label classification
            grid_label = torch.zeros(10, 10)
            
            # Mark overlapping 2x2 grids. So basically if the center of the map is at (x_val, y_val),
            # we need to find the grid cell it falls into and mark that cell and its surrounding cells.
            # This will create a 3x3 area around the grid cell that contains the center. 
            # This is the only way to have a unique label for each grid cell and its surrounding cells.
            min_i = max(0, grid_x - 1)
            max_i = min(9, grid_x + 2)
            min_j = max(0, grid_y - 1)
            max_j = min(9, grid_y + 2)
            
            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    grid_label[i, j] = 1
            
            # grid_label[grid_x][grid_y] = 3
            
            return stitched_img, basemap_img, grid_label.flatten().float(), stitched_img_path, basemap_img_path, metas_path
            
        except Exception as e:
            return torch.zeros(3, 100, 100), torch.zeros(3, 1000, 1000), torch.zeros(100), "", "", ""


class GridClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extractor
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # 3x3 processing
        self.grid_pool = nn.AdaptiveAvgPool2d((10, 10))
        
        # Cross-attention with local constraints
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=8,
            batch_first=True
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(100, 512)*0.02)
        
        # 3x3 attention mask
        self.register_buffer('attn_mask', self.create_local_mask())
        
        self.fc = nn.Linear(512, 100) 

    def create_local_mask(self):
        """Create (1, 100) mask allowing queries to see all 3x3 regions"""
        # Allow full attention since we pre-processed 3x3 context
        return torch.zeros(1, 100, dtype=torch.bool)  # No masking

    def forward(self, stitched, basemap):
        # Feature extraction
        stitched_feat = self.feature_extractor(stitched)  # (B,512,7,7)
        basemap_feat = self.feature_extractor(basemap)    # (B,512,25,25)
        
        # Process basemap to 10x10 grid with 3x3 context
        grid = self.grid_pool(basemap_feat)  # (B,512,10,10)
        grid_3x3 = F.unfold(grid, kernel_size=3, padding=1)  # (B,512*9,100)
        grid_3x3 = grid_3x3.view(-1, 512, 9, 100).mean(dim=2)  # (B,512,100)
        
        # Prepare attention inputs
        query = F.adaptive_avg_pool2d(stitched_feat, (1,1)).flatten(1)  # (B,512)
        key = value = grid_3x3.permute(0,2,1) + self.pos_embed  # (B,100,512)
        
        # Cross-attention with corrected mask
        scores, _ = self.cross_attn(
            query=query.unsqueeze(1),  # (B,1,512)
            key=key,                   # (B,100,512)
            value=value,               # (B,100,512)
            attn_mask=self.attn_mask   # (1,100)
        )

        scores = self.fc(scores.squeeze(1))
        
        return scores  # (B,100)


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

class SpatialContinuityLoss(nn.Module):
    def __init__(self, grid_size=10, lambda_spatial=1, ideal_distance=1):
        super().__init__()
        self.grid_size = grid_size
        self.lambda_spatial = lambda_spatial
        self.ideal_distance = ideal_distance
        
    def forward(self, outputs, labels=None):
        batch_size = outputs.size(0)
        device = outputs.device
        total_loss = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            # Get top 9 indices
            _, top9_indices = torch.topk(outputs[i], 9)
            
            # Convert flat indices to 2D coordinates
            row_indices = top9_indices // self.grid_size
            col_indices = top9_indices % self.grid_size
            
            # Calculate centroid
            center_row = row_indices.float().mean()
            center_col = col_indices.float().mean()
            
            # Calculate distances from each point to centroid
            distances_to_center = torch.sqrt((row_indices.float() - center_row)**2 + 
                                             (col_indices.float() - center_col)**2)
            
            # Calculate pairwise distances between points
            pairwise_distances = torch.cdist(torch.stack([row_indices.float(), col_indices.float()], dim=1),
                                             torch.stack([row_indices.float(), col_indices.float()], dim=1))
            
            # Calculate loss based on deviation from ideal configuration
            center_loss = self._calculate_center_loss(distances_to_center, device)
            spread_loss = self._calculate_spread_loss(pairwise_distances, device)
            
            spatial_loss = center_loss + spread_loss
            total_loss += spatial_loss
            
        return self.lambda_spatial * (total_loss / batch_size)
    
    def _calculate_center_loss(self, distances_to_center, device):
        # Penalize points that are too far from or too close to the center
        ideal_distances = torch.tensor([0, self.ideal_distance, self.ideal_distance * 2**0.5], device=device)
        expanded_distances = distances_to_center.unsqueeze(1).expand(-1, 3)
        min_deviations = torch.min(torch.abs(expanded_distances - ideal_distances), dim=1)[0]
        return min_deviations.sum()
    
    def _calculate_spread_loss(self, pairwise_distances, device):
        # Penalize configurations where points are not evenly spread
        ideal_distances = torch.tensor([self.ideal_distance, self.ideal_distance * 2**0.5, 2 * self.ideal_distance], device=device)
        expanded_distances = pairwise_distances.unsqueeze(2).expand(-1, -1, 3)
        min_deviations = torch.min(torch.abs(expanded_distances - ideal_distances), dim=2)[0]
        # Mask out diagonal elements (distance to self)
        mask = ~torch.eye(pairwise_distances.size(0), dtype=torch.bool, device=device)
        return min_deviations[mask].sum() / 2  # Divide by 2 to avoid double counting


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12738'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_dataloader(rank, world_size, dataset, batch_size=16, num_workers=10):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                     num_workers=num_workers, pin_memory=True)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_model(rank, world_size, num_epochs, model, criterion, optimizer, 
               train_dataset, val_dataset, batch_size, lr, version, fraction, 
               checkpoint_path, seed):
    setup(rank, world_size)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
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

    criterion = FocalLoss(alpha=0.9, gamma=2) #nn.BCEWithLogitsLoss()
    criterion_spatial = SpatialContinuityLoss(grid_size=10, lambda_spatial=0.5)
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
            total_bce_loss = 0.0
            total_spatial_loss = 0.0
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

                outputs = model(stitched, basemap)
                loss_bce = criterion(outputs, labels)
                loss_spatial = criterion_spatial(outputs)
                loss = loss_bce + loss_spatial

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_bce_loss += loss_bce.item()
                total_spatial_loss += loss_spatial.item()
                pbar.set_postfix({'loss': total_loss/(pbar.n+1), 'loss_bce': total_bce_loss/(pbar.n+1), 'loss_spatial': total_spatial_loss/(pbar.n+1)})

                _, top3 = torch.topk(outputs, 3, dim=1)
                correct += (labels.gather(1, top3).sum(dim=1) > 0).sum().item()
                total += labels.size(0)

                #Calculate IOU
                _, top9 = torch.topk(outputs, 9, dim=1)  # Shape: (B, 9)
                # Create binary predictions tensor based on top-9 indices
                predictions = torch.zeros_like(labels)
                predictions.scatter_(1, top9, 1)

                # Calculate Intersection over Union (IoU)
                intersection = (predictions * labels).sum(dim=1)  # Element-wise multiplication and sum
                union = ((predictions + labels) > 0).sum(dim=1)  # Union is the count of non-zero elements

                # IoU as a percentage
                iou_percentage = (intersection / union * 100).mean().item()
                total_iou += iou_percentage
            
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            train_iou = total_iou / len(train_loader)
            train_spatial_loss = total_spatial_loss / len(train_loader)
            train_bce_loss = total_bce_loss / len(train_loader)
            if rank == 0:
                losses["train"].append([{'epoch':epoch, 'train_loss':train_loss, 'train_acc':train_acc, 'train_iou': train_iou, 'total_bce_loss': train_bce_loss, 'total_spatial_loss': train_spatial_loss}])

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
            if epoch % 20 == 0 or epoch==start_epoch:
                model.eval()
                val_loss = 0.0
                val_bce_loss = 0.0
                val_spatial_loss = 0.0
                correct = 0
                total = 0
                total_iou = 0.0
                with torch.no_grad():
                    pbar = tqdm(val_loader, desc=f"Validation", disable=rank != 0)
                    for stitched, basemap, labels, stitched_img_path, basemap_img_path, metas_path in pbar:
                        stitched = stitched.to(device)
                        basemap = basemap.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(stitched, basemap)
                        loss_bce = criterion(outputs, labels)
                        loss_spatial = criterion_spatial(outputs)
                        loss = loss_bce + loss_spatial
                        
                        val_loss += loss.item()
                        val_bce_loss += loss_bce.item()
                        val_spatial_loss += loss_spatial.item()
                        if rank == 0:
                            pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1), 'val_loss_bce': val_bce_loss/(pbar.n+1), 'val_loss_spatial': val_spatial_loss/(pbar.n+1)})

                        # Calculate top-3 accuracy
                        _, top3 = torch.topk(outputs, 3, dim=1)
                        correct += (labels.gather(1, top3).sum(dim=1) > 0).sum().item()
                        total += labels.size(0)

                        #Calculate IOU
                        _, top9 = torch.topk(outputs, 9, dim=1)  # Shape: (B, 9)
                        # Create binary predictions tensor based on top-9 indices
                        predictions = torch.zeros_like(labels)
                        predictions.scatter_(1, top9, 1)

                        # Calculate Intersection over Union (IoU)
                        intersection = (predictions * labels).sum(dim=1)  # Element-wise multiplication and sum
                        union = ((predictions + labels) > 0).sum(dim=1)  # Union is the count of non-zero elements

                        # IoU as a percentage
                        iou_percentage = (intersection / union * 100).mean().item()
                        total_iou += iou_percentage
                
                val_loss /= len(val_loader)
                val_acc = correct / total
                val_iou = total_iou / len(val_loader)
                val_spatial_loss /= len(val_loader)
                val_bce_loss /= len(val_loader)
                if rank == 0:
                    losses["val"].append([{'epoch':epoch, 'val_loss':val_loss, 'val_acc':val_acc, 'val_iou': val_iou, 'val_bce_loss': val_bce_loss, 'val_spatial_loss': val_spatial_loss}])
                
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Top-3 Train Acc: {train_acc:.2%}, Train BCE Loss: {train_bce_loss:.4f}, Train Spatial Loss: {train_spatial_loss:.4f}, Val Loss: {val_loss:.4f}, Top-3 Val Acc: {val_acc:.2%}, Train IoU: {train_iou:.2f}, Val IoU: {val_iou:.2f}, Val BCE Loss: {val_bce_loss:.4f}, Val Spatial Loss: {val_spatial_loss:.4f}")
                with open('loss_'+unique_name+'.json', 'a') as f:
                    json.dump(losses, f)
        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
    cleanup()


# class FocalBCEWithLogitsLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.bce = nn.BCEWithLogitsLoss()  # Base loss

#     def forward(self, inputs, targets):
#         bce_loss = self.bce(inputs, targets)
#         pt = torch.sigmoid(inputs)
#         p_t = pt * targets + (1 - pt) * (1 - targets)  # p for true class
        
#         focal_weight = (1 - p_t).pow(self.gamma)
#         alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
#         loss = alpha_factor * focal_weight * bce_loss
            
#         return loss
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=str, default="12_Focal_Spatial")
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
        base_folder+'all_train_metas_v2',
        base_folder+'all_train_basemaps_v2',
        base_folder+'all_train_maps_gt_v2/map/',
        transform_base, transform_gen
    )
    val_dataset = MapDataset(
        base_folder+'all_val_metas_v2',
        base_folder+'all_val_basemaps_v2', 
        base_folder+'all_val_maps_gt_v2/map/',
        transform_base, transform_gen
    )

    # Model and training setup
    model = GridClassifier()
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    criterion=None


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Distributed training
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args.num_epochs, model, criterion, optimizer,
              train_dataset, val_dataset, args.batch_size, args.lr,
              args.version, args.train_fraction, args.checkpoint, args.seed),
        nprocs=world_size,
        join=True
    )
if __name__ == '__main__':
    main()
