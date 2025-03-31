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
            
            return stitched_img, basemap_img, grid_label.flatten().float(), stitched_img_path, basemap_img_path, metas_path
            
        except Exception as e:
            return torch.zeros(3, 100, 100), torch.zeros(3, 1000, 1000), torch.zeros(100)

class GridClassifier(nn.Module):
    def __init__(self):
        super(GridClassifier, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Grid processing
        self.grid_pool = nn.AdaptiveAvgPool2d((10, 10))  # For 1000x1000 -> 10x10 grids
        self.stitched_pool = nn.AdaptiveAvgPool2d((1, 1))

    # def forward(self, stitched, basemap):
    #     # Feature extraction
    #     stitched_feat = self.feature_extractor(stitched)  # (B,512,7,7)
    #     basemap_feat = self.feature_extractor(basemap)    # (B,512,25,25)
        
    #     # Grid processing (vectorized)
    #     grid_features = self.grid_pool(basemap_feat)  # (B,512,10,10)
    #     unfolded = F.unfold(grid_features, kernel_size=3, padding=1)  # (B,512*9,100)
        
    #     # Reshape and average pool windows
    #     window_feats = unfolded.view(-1, 512, 9, 100).mean(dim=2)  # (B,512,100)
    #     window_feats = window_feats.permute(0, 2, 1)  # (B,100,512)
        
    #     # Process stitched features
    #     stitched_feat = self.stitched_pool(stitched_feat).squeeze()  # (B,512)
        
    #     # Vectorized cosine similarity
    #     stitched_feat = F.normalize(stitched_feat, p=2, dim=1)
    #     window_feats = F.normalize(window_feats, p=2, dim=2)
    #     scores = torch.bmm(window_feats, stitched_feat.unsqueeze(2)).squeeze()  # (B,100)
        
    #     return scores

    def forward(self, stitched, basemap):

        stitched_feat = self.feature_extractor(stitched)  # (B,512,7,7)
        basemap_feat = self.feature_extractor(basemap)    # (B,512,25,25)
        
        grid_features = self.grid_pool(basemap_feat)      # (B,512,10,10)
        grid_features = F.unfold(grid_features, kernel_size=3, padding=1)  # (B,512*9,100)
        
        # Process stitched features
        stitched_feat = self.stitched_pool(stitched_feat)  # (B,512,1,1)
        
        # Calculate similarity for 3x3 regions
        batch_size = stitched_feat.size(0)
        scores = torch.zeros(batch_size, 100).to(stitched.device)  # 100 possible centers
        
        # Compare each 3x3 window
        for i in range(10):
            for j in range(10):
                # Get 3x3 window (handles edge cases)
                window = grid_features[:, :, i*10 + j]  # (B, 512*9)
                window = window.view(batch_size, 512, 3, 3)  # (B,512,3,3)
                
                # Average pool window features
                window_feat = F.adaptive_avg_pool2d(window, (1,1))  # (B,512,1,1)
                
                # Calculate similarity
                sim = F.cosine_similarity(stitched_feat, window_feat, dim=1)
                scores[:, i*10 + j] = sim.squeeze()
        
        return scores


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12338'
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

    best_val_loss = min(loss for epoch, loss in losses["val"]) if losses["val"] else float('inf')
    best_train_loss = min(loss for epoch, loss in losses["train"]) if losses["train"] else float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
        for stitched, basemap, labels, stitched_img_path, basemap_img_path, metas_path in pbar:
            stitched = stitched.to(device)
            basemap = basemap.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()

            outputs = model(stitched, basemap)
            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss/(pbar.n+1)})

            _, top3 = torch.topk(outputs, 3, dim=1)
            correct += (labels.gather(1, top3).sum(dim=1) > 0).sum().item()
            total += labels.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        losses["train"].append([epoch, train_loss])

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
            correct = 0
            total = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Validation", disable=rank != 0)
                for stitched, basemap, labels, stitched_img_path, basemap_img_path, metas_path in pbar:
                    stitched = stitched.to(device)
                    basemap = basemap.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(stitched, basemap)
                    val_loss += criterion(outputs, labels).item()
                    if rank == 0:
                        pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})
                    # Calculate top-3 accuracy
                    _, top3 = torch.topk(outputs, 3, dim=1)
                    correct += (labels.gather(1, top3).sum(dim=1) > 0).sum().item()
                    total += labels.size(0)
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            losses["val"].append([epoch, val_loss])
            
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Top-3 Train Acc: {train_acc:.2%}, Val Loss: {val_loss:.4f}, Top-3 Val Acc: {val_acc:.2%}")
            with open('loss_'+unique_name+'.json', 'w') as f:
                json.dump(losses, f)

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=int, default=7)
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

    # criterion = nn.BCEWithLogitsLoss()
    criterion = lambda inputs, targets: sigmoid_focal_loss(
        inputs, targets, alpha=0.9, gamma=2.5, reduction="mean"
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Distributed training
    world_size = torch.cuda.device_count()
    # mp.spawn(
    #     train_model,
    #     args=(world_size, args.num_epochs, model, criterion, optimizer,
    #           train_dataset, val_dataset, args.batch_size, args.lr,
    #           args.version, args.train_fraction, args.checkpoint, args.seed),
    #     nprocs=world_size,
    #     join=True
    # )
    train_model(
        rank=0,  # For single GPU run, we can set rank to 0
        world_size=1,  # Set to 1 for single GPU
        num_epochs=args.num_epochs,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        version=args.version,
        fraction=args.train_fraction,
        checkpoint_path=args.checkpoint,
        seed=args.seed
    )

if __name__ == '__main__':
    main()
