# This code is for coarse matching- to find the best grid (regression version).
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

# =========================
# DATASETS
# =========================

class AugmentedMapDataset(Dataset):
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
                if os.path.exists(os.path.join(basemap_folder, basemap_file)) and os.path.exists(os.path.join(metas_folder, metas_file)):
                    self.file_triplets.append((stitched_file, basemap_file, metas_file))

    def __len__(self):
        # Multiply by 4 to account for original + 3 rotations
        return len(self.file_triplets) * 4

    def __getitem__(self, idx):
        # Determine which augmentation to apply based on the index
        original_idx = torch.div(idx, 4, rounding_mode='trunc')
        rotation_idx = idx % 4

        stitched_file, basemap_file, metas_file = self.file_triplets[original_idx]
        
        stitched_img_path = os.path.join(self.stitched_folder, stitched_file)
        basemap_img_path = os.path.join(self.basemap_folder, basemap_file)
        metas_path = os.path.join(self.metas_folder, metas_file)

        # Load images and metadata
        stitched_img = Image.open(stitched_img_path).convert('RGB')
        basemap_img = Image.open(basemap_img_path).convert('RGB')
        
        basemap_img = np.array(basemap_img)
        basemap_img = np.flipud(basemap_img)
        basemap_img = Image.fromarray(basemap_img)
        
        # Rotate the stitched image based on rotation_idx
        if rotation_idx == 1:
            stitched_img = stitched_img.rotate(90)
        elif rotation_idx == 2:
            stitched_img = stitched_img.rotate(180)
        elif rotation_idx == 3:
            stitched_img = stitched_img.rotate(270)

        # Apply transformations if provided
        if self.transform_gen:
            stitched_img = self.transform_gen(stitched_img)
        if self.transform_base:
            basemap_img = self.transform_base(basemap_img)

        # Load metadata and process grid labels (unchanged)
        metas = np.load(metas_path, allow_pickle=True).item()
        
        center_x, center_y = basemap_img.shape[1] // 2, basemap_img.shape[2] // 2
        x_val = center_x - metas['perturbation'][0]
        y_val = center_y - metas['perturbation'][1]
        
        grid_size = 100  # Each grid is 100x100 pixels
        grid_x = x_val / grid_size #int(x_val // grid_size)
        grid_y = y_val / grid_size #int(y_val // grid_size)

        # Regression target: integer grid coordinates (no sub-grid precision)
        coord = torch.tensor([grid_x, grid_y], dtype=torch.float32)

        return stitched_img, basemap_img, coord, stitched_img_path, basemap_img_path, metas_path

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
            
            center_x, center_y = basemap_img.shape[1] // 2, basemap_img.shape[2] // 2
            x_val = center_x - metas['perturbation'][0]
            y_val = center_y - metas['perturbation'][1]
            
            grid_size = 100  # Each grid is 100x100 pixels
            grid_x = x_val / grid_size
            grid_y = y_val / grid_size

            coord = torch.tensor([grid_x, grid_y], dtype=torch.float32)

            return stitched_img, basemap_img, coord, stitched_img_path, basemap_img_path, metas_path
            
        except Exception as e:
            return torch.zeros(3, 100, 100), torch.zeros(3, 1000, 1000), torch.zeros(2), "", "", ""

# =========================
# MODEL
# =========================

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
        self.pos_embed = nn.Parameter(torch.randn(100, 512)*0.02)
        self.register_buffer('attn_mask', self.create_local_mask())
        self.fc = nn.Linear(512, 2)  # Output 2 coordinates

    def create_local_mask(self):
        return torch.zeros(1, 100, dtype=torch.bool)  # No masking

    def forward(self, stitched, basemap):
        stitched_feat = self.feature_extractor(stitched)  # (B,512,7,7)
        basemap_feat = self.feature_extractor(basemap)    # (B,512,25,25)
        grid = self.grid_pool(basemap_feat)  # (B,512,10,10)
        grid_3x3 = F.unfold(grid, kernel_size=3, padding=1)  # (B,512*9,100)
        grid_3x3 = grid_3x3.view(-1, 512, 9, 100).mean(dim=2)  # (B,512,100)
        query = F.adaptive_avg_pool2d(stitched_feat, (1,1)).flatten(1)  # (B,512)
        key = value = grid_3x3.permute(0,2,1) + self.pos_embed  # (B,100,512)
        scores, _ = self.cross_attn(
            query=query.unsqueeze(1),  # (B,1,512)
            key=key,                   # (B,100,512)
            value=value,               # (B,100,512)
            attn_mask=self.attn_mask   # (1,100)
        )
        scores = self.fc(scores.squeeze(1))  # (B,2)
        return scores

# =========================
# TRAINING UTILS
# =========================

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '17733'
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

# =========================
# TRAINING LOOP
# =========================

def train_model(rank, world_size, num_epochs, model, criterion, optimizer, 
               train_dataset, val_dataset, batch_size, lr, version, fraction, 
               checkpoint_path, seed):
    setup(rank, world_size)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss().to(device)  # Use MSE loss for regression

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

    best_val_mae = min(loss[2] for loss in losses["val"]) if losses["val"] else float('inf')
    best_train_mae = min(loss[2] for loss in losses["train"]) if losses["train"] else float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            train_loader.sampler.set_epoch(epoch)
            total_loss = 0.0
            total_mae = 0.0
            total_mse = 0.0
            total = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
            for stitched, basemap, labels, *_ in pbar:
                stitched = stitched.to(device)
                basemap = basemap.to(device)
                labels = labels.to(device)  # (B,2)
                
                optimizer.zero_grad()
                outputs = model(stitched, basemap)  # (B,2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                mae = torch.abs(outputs - labels).mean().item()
                mse = torch.pow(outputs - labels, 2).mean().item()
                total_mae += mae
                total_mse += mse
                total += 1
                pbar.set_postfix({'loss': total_loss/(pbar.n+1), 'mae': total_mae/(pbar.n+1), 'mse': total_mse/(pbar.n+1)})

            train_loss = total_loss / total
            train_mae = total_mae / total
            train_mse = total_mse / total
            losses["train"].append([epoch, train_loss, train_mae, train_mse])

            if rank == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'subset_indices': subset_indices
                }
                torch.save(checkpoint, 'latest_map_location_model_train_'+unique_name+'.pth')

            # Validation
            if epoch % 1 == 0 or epoch == start_epoch:
                model.eval()
                val_loss = 0.0
                val_mae = 0.0
                val_mse = 0.0
                val_total = 0
                with torch.no_grad():
                    pbar = tqdm(val_loader, desc=f"Validation", disable=rank != 0)
                    for stitched, basemap, labels, *_ in pbar:
                        stitched = stitched.to(device)
                        basemap = basemap.to(device)
                        labels = labels.to(device)
                        outputs = model(stitched, basemap)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        mae = torch.abs(outputs - labels).mean().item()
                        mse = torch.pow(outputs - labels, 2).mean().item()
                        val_mae += mae
                        val_mse += mse
                        val_total += 1
                        if rank == 0:
                            pbar.set_postfix({'val_loss': val_loss/(pbar.n+1), 'val_mae': val_mae/(pbar.n+1), 'val_mse': val_mse/(pbar.n+1)})

                val_loss /= val_total
                val_mae /= val_total
                val_mse /= val_total
                losses["val"].append([epoch, val_loss, val_mae, val_mse])
                
                if rank == 0 and val_mae < best_val_mae:
                    best_val_mae = val_mae
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'losses': losses,
                        'subset_indices': subset_indices
                    }
                    torch.save(checkpoint, 'best_map_location_model_val_'+unique_name+'.pth')
        
            if rank == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train MSE: {train_mse:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val MSE: {val_mse:.4f}")
                with open('loss_'+unique_name+'.json', 'w') as f:
                    json.dump(losses, f)
        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
    cleanup()

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=str, default="18_MSE_Fine")
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
    train_dataset = AugmentedMapDataset(
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

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Distributed training
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args.num_epochs, model, None, optimizer,
              train_dataset, val_dataset, args.batch_size, args.lr,
              args.version, args.train_fraction, args.checkpoint, args.seed),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
