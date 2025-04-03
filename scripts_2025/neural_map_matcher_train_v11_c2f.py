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
            
            # Normalize coordinates to [0, 1] range for regression
            norm_x = x_val / basemap_img.shape[1]
            norm_y = y_val / basemap_img.shape[2]
            exact_coords = torch.tensor([norm_x, norm_y], dtype=torch.float32)
            
            grid_size = 100  # Each grid is 100x100 pixels
            grid_x = int(x_val // grid_size)
            grid_y = int(y_val // grid_size)
            
            # Create 10x10 grid mask for multi-label classification
            grid_label = torch.zeros(10, 10)
            
            # Mark overlapping 3x3 grids
            min_i = max(0, grid_x - 1)
            max_i = min(9, grid_x + 2)
            min_j = max(0, grid_y - 1)
            max_j = min(9, grid_y + 2)
            
            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    grid_label[i, j] = 1
            
            return stitched_img, basemap_img, grid_label.flatten().float(), exact_coords, stitched_img_path, basemap_img_path, metas_path
            
        except Exception as e:
            return torch.zeros(3, 100, 100), torch.zeros(3, 1000, 1000), torch.zeros(100), torch.zeros(2), "", "", ""


class GridClassifierWithRegression(nn.Module):
    def __init__(self, dropout_rate=0.3, pretrained=True):
        super().__init__()
        
        # Feature extractor
        resnet = models.resnet18(pretrained=pretrained)
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
        
        # Classification head
        self.fc = nn.Linear(512, 100)
        
        # Regression head for exact coordinates
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 2)  # (x, y) coordinates
        )

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
        
        features = scores.squeeze(1)  # (B,512)
        
        # Classification output
        grid_scores = self.fc(features)  # (B,100)
        
        # Regression output
        coordinates = self.regression_head(features)  # (B,2)
        
        return grid_scores, coordinates

class CombinedLoss(nn.Module):
    def __init__(self, grid_loss_weight=1.0, coord_loss_weight=1.0):
        super().__init__()
        self.grid_loss_weight = grid_loss_weight
        self.coord_loss_weight = coord_loss_weight
        self.grid_loss = nn.BCEWithLogitsLoss()
        self.coord_loss = nn.SmoothL1Loss()  # Huber loss for regression
        
    def forward(self, grid_outputs, coord_outputs, grid_targets, coord_targets):
        grid_loss = self.grid_loss(grid_outputs, grid_targets)
        coord_loss = self.coord_loss(coord_outputs, coord_targets)
        return self.grid_loss_weight * grid_loss + self.coord_loss_weight * coord_loss, grid_loss, coord_loss


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
               checkpoint_path, seed, grid_loss_weight=1.0, coord_loss_weight=1.0):
    setup(rank, world_size)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    # Combined loss for both grid classification and coordinate regression
    criterion = CombinedLoss(grid_loss_weight=grid_loss_weight, coord_loss_weight=coord_loss_weight).to(device)
    
    if fraction >= 1.0:
        print("FULL TRAINING")
        if checkpoint_path is None:
            print("Starting from scratch")
            start_epoch = 0
            losses = {"train": [], "val": []}
            subset_indices = torch.randperm(len(train_dataset))
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            print("Loading checkpoint and adding regression head")
            checkpoint = torch.load(checkpoint_path)
            
            # Load only the matching parameters from the checkpoint
            model_dict = model.module.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                              if k in model_dict and 'regression_head' not in k}
            model_dict.update(pretrained_dict)
            model.module.load_state_dict(model_dict, strict=False)
            
            # Don't load optimizer state since we're adding new parameters
            start_epoch = 0
            losses = {"train": [], "val": []}
            subset_indices = checkpoint['subset_indices'] if 'subset_indices' in checkpoint else torch.randperm(len(train_dataset))
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
            print("Loading checkpoint and adding regression head")
            checkpoint = torch.load(checkpoint_path)
            
            # Load only the matching parameters from the checkpoint
            model_dict = model.module.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                              if k in model_dict and 'regression_head' not in k}
            model_dict.update(pretrained_dict)
            model.module.load_state_dict(model_dict, strict=False)
            
            # Don't load optimizer state since we're adding new parameters
            start_epoch = 0
            losses = {"train": [], "val": []}
            subset_indices = checkpoint['subset_indices']
            train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

    train_loader = create_dataloader(rank, world_size, train_dataset, batch_size)
    val_loader = create_dataloader(rank, world_size, val_dataset, batch_size)
    
    unique_name = f"grid_regression_v{version}-lr{lr}-bs{batch_size}-frac{fraction}-seed{seed}-grid{grid_loss_weight}-coord{coord_loss_weight}"

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            train_loader.sampler.set_epoch(epoch)
            total_loss = 0.0
            total_grid_loss = 0.0
            total_coord_loss = 0.0
            correct = 0
            total = 0
            total_iou = 0.0
            total_coord_error = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
            for stitched, basemap, grid_labels, coord_labels, stitched_img_path, basemap_img_path, metas_path in pbar:
                stitched = stitched.to(device)
                basemap = basemap.to(device)
                grid_labels = grid_labels.to(device)
                coord_labels = coord_labels.to(device)
                
                optimizer.zero_grad()

                grid_outputs, coord_outputs = model(stitched, basemap)
                loss, grid_loss, coord_loss = criterion(grid_outputs, coord_outputs, grid_labels, coord_labels)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_grid_loss += grid_loss.item()
                total_coord_loss += coord_loss.item()
                
                pbar.set_postfix({
                    'loss': total_loss/(pbar.n+1), 
                    'grid_loss': total_grid_loss/(pbar.n+1),
                    'coord_loss': total_coord_loss/(pbar.n+1)
                })

                # Grid accuracy metrics
                _, top3 = torch.topk(grid_outputs, 3, dim=1)
                correct += (grid_labels.gather(1, top3).sum(dim=1) > 0).sum().item()
                total += grid_labels.size(0)

                # Grid IoU metrics
                _, top9 = torch.topk(grid_outputs, 9, dim=1)
                predictions = torch.zeros_like(grid_labels)
                predictions.scatter_(1, top9, 1)
                intersection = (predictions * grid_labels).sum(dim=1)
                union = ((predictions + grid_labels) > 0).sum(dim=1)
                iou_percentage = (intersection / union * 100).mean().item()
                total_iou += iou_percentage
                
                # Coordinate error metrics
                coord_error = torch.sqrt(((coord_outputs - coord_labels) ** 2).sum(dim=1)).mean().item()
                total_coord_error += coord_error
            
            train_loss = total_loss / len(train_loader)
            train_grid_loss = total_grid_loss / len(train_loader)
            train_coord_loss = total_coord_loss / len(train_loader)
            train_acc = correct / total
            train_iou = total_iou / len(train_loader)
            train_coord_error = total_coord_error / len(train_loader)
            
            losses["train"].append([epoch, train_loss, train_acc, train_iou, train_grid_loss, train_coord_loss, train_coord_error])

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
                torch.save(checkpoint, f'latest_{unique_name}.pth')

            # Validation
            if epoch % 1 == 0 or epoch==start_epoch:
                model.eval()
                val_loss = 0.0
                val_grid_loss = 0.0
                val_coord_loss = 0.0
                correct = 0
                total = 0
                total_iou = 0.0
                total_coord_error = 0.0
                
                with torch.no_grad():
                    pbar = tqdm(val_loader, desc=f"Validation", disable=rank != 0)
                    for stitched, basemap, grid_labels, coord_labels, stitched_img_path, basemap_img_path, metas_path in pbar:
                        stitched = stitched.to(device)
                        basemap = basemap.to(device)
                        grid_labels = grid_labels.to(device)
                        coord_labels = coord_labels.to(device)
                        
                        grid_outputs, coord_outputs = model(stitched, basemap)
                        loss, grid_loss, coord_loss = criterion(grid_outputs, coord_outputs, grid_labels, coord_labels)
                        
                        val_loss += loss.item()
                        val_grid_loss += grid_loss.item()
                        val_coord_loss += coord_loss.item()
                        
                        if rank == 0:
                            pbar.set_postfix({
                                'val_loss': val_loss/(pbar.n+1),
                                'val_grid_loss': val_grid_loss/(pbar.n+1),
                                'val_coord_loss': val_coord_loss/(pbar.n+1)
                            })
                        
                        # Grid accuracy metrics
                        _, top3 = torch.topk(grid_outputs, 3, dim=1)
                        correct += (grid_labels.gather(1, top3).sum(dim=1) > 0).sum().item()
                        total += grid_labels.size(0)

                        # Grid IoU metrics
                        _, top9 = torch.topk(grid_outputs, 9, dim=1)
                        predictions = torch.zeros_like(grid_labels)
                        predictions.scatter_(1, top9, 1)
                        intersection = (predictions * grid_labels).sum(dim=1)
                        union = ((predictions + grid_labels) > 0).sum(dim=1)
                        iou_percentage = (intersection / union * 100).mean().item()
                        total_iou += iou_percentage
                        
                        # Coordinate error metrics
                        coord_error = torch.sqrt(((coord_outputs - coord_labels) ** 2).sum(dim=1)).mean().item()
                        total_coord_error += coord_error
                
                val_loss /= len(val_loader)
                val_grid_loss /= len(val_loader)
                val_coord_loss /= len(val_loader)
                val_acc = correct / total
                val_iou = total_iou / len(val_loader)
                val_coord_error = total_coord_error / len(val_loader)
                
                losses["val"].append([epoch, val_loss, val_acc, val_iou, val_grid_loss, val_coord_loss, val_coord_error])
                
                if rank == 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'losses': losses,
                        'subset_indices': subset_indices
                    }
                    torch.save(checkpoint, f'best_{unique_name}.pth')
        
            if rank == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Grid Loss: {train_grid_loss:.4f}, Coord Loss: {train_coord_loss:.4f}, "
                      f"Train Acc: {train_acc:.2%}, Train IoU: {train_iou:.2f}, Coord Error: {train_coord_error:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, Val IoU: {val_iou:.2f}, Val Coord Error: {val_coord_error:.4f}")
                
                with open(f'loss_{unique_name}.json', 'w') as f:
                    json.dump(losses, f)
                    
        except Exception as e:
            print(f"Error during training: {e}")
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=str, default="11_c2f")
    parser.add_argument('--grid_loss_weight', type=float, default=1.0)
    parser.add_argument('--coord_loss_weight', type=float, default=1.0)
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
    model = GridClassifierWithRegression()
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    # Using AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Distributed training
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args.num_epochs, model, None, optimizer,
              train_dataset, val_dataset, args.batch_size, args.lr,
              args.version, args.train_fraction, args.checkpoint, args.seed,
              args.grid_loss_weight, args.coord_loss_weight),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
