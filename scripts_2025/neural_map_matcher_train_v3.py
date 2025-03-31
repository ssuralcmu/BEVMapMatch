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
from torch.profiler import profile, record_function, ProfilerActivity

import dill

class MapDataset(Dataset):
    def __init__(self, metas_folder, basemap_folder, stitched_folder, transform_base=None, transform_gen=None, grid_size=10):
        self.basemap_folder = basemap_folder
        self.stitched_folder = stitched_folder
        self.metas_folder = metas_folder
        self.transform_base = transform_base
        self.transform_gen = transform_gen
        self.grid_size = grid_size  # Add grid size as an argument
        
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
            basemap_img = np.flipud(basemap_img)  # Flip vertically if needed
            basemap_img = Image.fromarray(basemap_img)

            if self.transform_gen:
                stitched_img = self.transform_gen(stitched_img)
            if self.transform_base:
                basemap_img = self.transform_base(basemap_img)

            metas = np.load(metas_path, allow_pickle=True).item()

            # Extract location information (center of smaller image)
            center_x, center_y = basemap_img.shape[1] // 2, basemap_img.shape[2] // 2

            x_val = center_x - metas['perturbation'][0]/2
            y_val = center_y - metas['perturbation'][1]/2

            location = torch.tensor([x_val, y_val], dtype=torch.float32)

            # Compute target cell index
            height, width = 500, 500  # Assuming resized images are 500x500
            grid_h, grid_w = height // self.grid_size, width // self.grid_size

            row = int(x_val // grid_h)
            col = int(y_val // grid_w)
            target_cell = row * self.grid_size + col

            return stitched_img, basemap_img, location, target_cell
        except Exception as e:
            return torch.zeros(3, 500, 500), torch.zeros(3, 500, 500), torch.zeros(2), 0


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from e2cnn import gspaces, nn as enn  # Group-equivariant convolutions library

# Rotation-Invariant Feature Extractor with Group-Equivariant Convolutions
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotationInvariantFeatureExtractor(nn.Module):
    def __init__(self):
        super(RotationInvariantFeatureExtractor, self).__init__()
        
        # Define standard convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False)
        
        # Final layer to reduce dimensions for downstream tasks
        self.conv1x1 = nn.Conv2d(128, 256, kernel_size=1)

    def rotation_pooling(self, x):
        """
        Perform rotation pooling by applying rotations to the input and aggregating the responses.
        """
        rotations = [0, 90, 180, 270]  # Discrete rotations in degrees
        pooled_responses = []
        
        for angle in rotations:
            # Rotate input tensor
            rotated_x = self.rotate_tensor(x, angle)
            pooled_responses.append(rotated_x)
        
        # Aggregate responses (e.g., max pooling across rotations)
        return torch.max(torch.stack(pooled_responses), dim=0)[0]

    def rotate_tensor(self, x, angle):
        """
        Rotate the tensor by a given angle (in degrees).
        """
        theta = math.radians(angle)
        rotation_matrix = torch.tensor([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0]
        ], dtype=torch.float32).unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)

        grid = F.affine_grid(rotation_matrix, x.size(), align_corners=False)
        rotated_x = F.grid_sample(x, grid, align_corners=False)
        
        return rotated_x

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = F.relu(x)
        
        # Apply rotation pooling to enforce rotation invariance
        x = self.rotation_pooling(x)
        
        # Second convolutional layer
        x = self.conv2(x)
        x = F.relu(x)

        # Final dimensionality reduction layer
        x = self.conv1x1(x)
        
        return x


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        
        # Localization network with adaptive pooling
        self.localization = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4))  # Ensures fixed spatial output size (4x4)
        )
        
        # Fully connected layers for affine transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 32),  # Input size is now fixed
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # Initialize weights to represent an identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Pass through localization network
        xs = self.localization(x)
        
        # Flatten for fully connected layer
        xs = xs.view(xs.size(0), -1)
        
        # Pass through fully connected layers
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Generate sampling grid and apply transformation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x




# Correlation Layer for matching basemap and BEV features
class CorrelationLayer(nn.Module):
    def __init__(self):
        super(CorrelationLayer, self).__init__()

    def forward(self, basemap_features, bev_features):
        b, c, h, w = basemap_features.size()
        
        # Resize BEV features to match basemap dimensions
        bev_features = F.interpolate(bev_features, size=(h, w), mode='bilinear', align_corners=False)
        
        # Compute correlation using Einstein summation notation for efficiency
        correlation = torch.einsum('bchw,bchw->bc', basemap_features, bev_features)
        
        return correlation

# Classification Head for coarse localization (cell probabilities)
class ClassificationHead(nn.Module):
    def __init__(self, num_classes=100):
        super(ClassificationHead, self).__init__()
        
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Regression Head for fine-grained localization (coordinates within cells)
class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        
        # Replace Conv2d with Linear for flattened inputs
        self.fc1 = nn.Linear(256, 64)  # Assuming input size is 256
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)   # Output: (x_offset, y_offset)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Main Localization Model combining all components
class LocationModel(nn.Module):
    def __init__(self, grid_size=10, num_cells=100):
        super(LocationModel, self).__init__()
        
        self.grid_size = grid_size
        self.num_cells = num_cells
        
        # Submodules of the model
        self.feature_extractor = RotationInvariantFeatureExtractor()
        self.stn = SpatialTransformerNetwork()
        self.correlation_layer = CorrelationLayer()
        
        # Heads for classification and regression tasks
        self.classification_head = ClassificationHead(num_classes=num_cells)
        self.regression_head = RegressionHead()

    def extract_grid_features(self, features):
        b, c, h, w = features.size()
        
        # Extract patches of size corresponding to grid cells
        grid_h, grid_w = h // self.grid_size, w // self.grid_size
        if h % self.grid_size != 0 or w % self.grid_size != 0:
            # Pad input if dimensions are not divisible by grid size
            pad_h = (self.grid_size - h % self.grid_size) % self.grid_size
            pad_w = (self.grid_size - w % self.grid_size) % self.grid_size
            features = F.pad(features, (0, pad_w, 0, pad_h))
            h, w = features.size(2), features.size(3)
            grid_h, grid_w = h // self.grid_size, w // self.grid_size

        # Unfold the feature map into grid-sized patches
        x = features.unfold(2, grid_h, grid_h).unfold(3, grid_w, grid_w)
        x = x.contiguous().view(b, c, -1, grid_h, grid_w)
        x = x.transpose(1, 2)  # Move the grid dimension to the batch axis
        return x

    def custom_activation(self, x):
        # Custom activation function to scale outputs
        return 250 + 250 * torch.tanh(x)

    def forward(self, stitched, basemap):
        # Extract rotation-invariant features from basemap and stitched input
        basemap_features = self.feature_extractor(basemap)
        bev_features = self.feature_extractor(stitched)
        
        # Apply Spatial Transformer Network (STN) to BEV features
        transformed_bev = self.stn(bev_features)
        
        # Extract grid features from the basemap
        grid_features = self.extract_grid_features(basemap_features)
        
        # Compute correlation for each grid cell
        correlations = []
        for i in range(self.num_cells):
            cell_correlation = self.correlation_layer(grid_features[:, i], transformed_bev)
            correlations.append(cell_correlation)
        
        # Stack correlations into a correlation map
        correlation_map = torch.stack(correlations, dim=1)
        
        # Classification stage: predict probabilities for each cell
        cell_probs = self.classification_head(correlation_map)
        
        # Get top 4 most probable cells
        _, top_indices = torch.topk(cell_probs, k=4, dim=1)
        
        # Regression stage: refine localization within top cells
        top_correlations = torch.gather(
            correlation_map,
            1,
            top_indices.unsqueeze(2).expand(-1, -1, correlation_map.size(2))  # Match actual dimensions
        )
        
        locations = []
        for i in range(4):
            cell_location = self.regression_head(top_correlations[:, i])  # No reshaping needed
            locations.append(cell_location)

        
        # Combine regression results for the top cells
        locations = torch.stack(locations, dim=1)  # Shape: (batch_size, 4, 2)
        
        # Combine classification probabilities with regression results
        cell_probs_top4 = torch.gather(cell_probs, 1, top_indices)  # Shape: (batch_size, 4)
        weighted_locations = (locations * cell_probs_top4.unsqueeze(2)).sum(dim=1)  # Weighted average
        
        # Apply custom activation function to final location predictions
        final_location = self.custom_activation(weighted_locations)
        
        return final_location, cell_probs


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12308'
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
    
    train_dataloader = create_dataloader(rank, world_size, train_dataset, batch_size)
    val_dataloader = create_dataloader(rank, world_size, val_dataset, batch_size)
    
    classification_loss_weight=10000
    

    unique_name=f"v{version}-lr{lr}-bs{batch_size}-frac{fraction}-seed{seed}"

    best_val_loss = min(loss for epoch, loss in losses["val"]) if losses["val"] else float('inf')
    best_train_loss = min(loss for epoch, loss in losses["train"]) if losses["train"] else float('inf')
    best_model_state = None
    best_train_model_state = None
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_dataloader.sampler.set_epoch(epoch)
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
        for stitched_imgs, basemap_imgs, locations, target_cells in pbar:
            stitched_imgs = stitched_imgs.to(device)
            basemap_imgs = basemap_imgs.to(device)
            locations = locations.to(device)
            target_cells = target_cells.to(device)


            optimizer.zero_grad()
            # print("Before model", flush=True)
            predicted_locations, cell_probs = model(stitched_imgs, basemap_imgs)
            # Compute losses
            # print("After model", flush=True)

            # print("Predicted locations shape: ", predicted_locations.shape)
            # print("Locations shape: ", locations.shape)
            # print("Cell probs shape: ", cell_probs)
            # print("Target cells shape: ", target_cells)
            regression_loss = criterion(predicted_locations, locations)  # Fine-grained localization loss
            classification_loss = F.cross_entropy(cell_probs, target_cells)  # Coarse localization loss

            # print("Regression loss: ", regression_loss)
            # print("Classification loss: ", classification_loss)

            # Combine losses (weighted sum)
            loss = regression_loss + classification_loss_weight * classification_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        train_loss = running_loss / len(train_dataloader)
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

        if epoch % 10 == 0 or epoch == start_epoch:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                pbar = tqdm(val_dataloader, desc=f"Validation", disable=rank != 0)
                for stitched_imgs, basemap_imgs, locations, target_cells in pbar:
                    stitched_imgs = stitched_imgs.to(device)
                    basemap_imgs = basemap_imgs.to(device)
                    locations = locations.to(device)
                    target_cells = target_cells.to(device)

                    predicted_locations, cell_probs= model(stitched_imgs, basemap_imgs)


                    # Compute losses
                    regression_loss = criterion(predicted_locations, locations)  # Fine-grained localization loss
                    classification_loss = F.cross_entropy(cell_probs, target_cells)  # Coarse localization loss
                    # Combine losses (weighted sum)
                    loss = regression_loss + classification_loss_weight * classification_loss
                    
                    val_loss += loss.item()
                    if rank == 0:
                        pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})

            val_loss /= len(val_dataloader)
            losses["val"].append([epoch, val_loss])

            val_loss=0

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
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            with open('loss_'+unique_name+'.json', 'w') as f:
                json.dump(losses, f)            
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file to resume training')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--version', type=int, default=3, help='Version of the model')
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

    train_dataset = MapDataset(base_folder+'all_train_metas_v2_500m', base_folder+'all_train_basemaps_v2_500m', base_folder+'all_train_maps_gt_v2_500m/map/', transform_base=transform_base, transform_gen=transform_gen)
    

    val_dataset = MapDataset(base_folder+'all_val_metas_v2_500m', base_folder+'all_val_basemaps_v2_500m', base_folder+'all_val_maps_gt_v2_500m/map/', transform_base=transform_base, transform_gen=transform_gen)

    model = LocationModel()
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    num_epochs = args.num_epochs

    mp.spawn(
        train_model,
        args=(world_size, num_epochs, model, criterion, optimizer, train_dataset, val_dataset, args.batch_size, args.lr, args.version, args.train_fraction, args.checkpoint, args.seed),
        nprocs=world_size,
        join=True
    )

    

if __name__ == '__main__':
    main()
