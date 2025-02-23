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
import torchvision.models as models

class MapDataset(Dataset):
    def __init__(self, metas_folder, basemap_folder, stitched_folder, transform_base=None, transform_gen=None):
        self.basemap_folder = basemap_folder
        self.stitched_folder = stitched_folder
        self.metas_folder = metas_folder
        self.transform_base = transform_base
        self.transform_gen = transform_gen
        
        # Get all stitched image files
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
        # print("Length of dataset")
        # print(len(self.file_triplets))
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

            # print(stitched_img.size)

            if self.transform_gen:
                stitched_img = self.transform_gen(stitched_img)
            if self.transform_base:
                basemap_img = self.transform_base(basemap_img)

            # Load metas
            metas = np.load(metas_path, allow_pickle=True).item()
            # Extract the bounding box coordinates

            # # Calculate the center of the basemap image
            center_x, center_y = basemap_img.shape[1] // 2, basemap_img.shape[2] // 2

            # # print(basemap_img.shape)
            
            # # Calculate the bounding box coordinates (500x500 pixels around the center)
            # x1 = center_x - 80
            # y1 = center_y - 80
            # x2 = center_x + 80
            # y2 = center_y + 80

            x_val=center_x - metas['perturbation'][0]/2 # Size mult by 2 and then divided by 4 for resizing to (500,500)
            y_val=center_y - metas['perturbation'][1]/2

            location = torch.tensor([x_val, y_val], dtype=torch.float32)

            return stitched_img, basemap_img, location
        except Exception as e:
            return torch.zeros(3, 50, 50), torch.zeros(3, 500, 500), torch.zeros(2)

# class LocationModel(nn.Module):
#     def __init__(self):
#         super(LocationModel, self).__init__()
#         self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(984064, 1024)
#         self.fc2 = nn.Linear(1024, 2)  # 4 outputs for bbox coordinates
#         self.relu = nn.ReLU()

#     def forward(self, stitched, basemap):
#         x = torch.cat((stitched, basemap), dim=1)
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)  # Flatten while preserving batch size
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x



import torch
import torch.nn as nn
from torchvision import models

class LocationModel(nn.Module):
    def __init__(self):
        super(LocationModel, self).__init__()
        # Load pretrained ResNet models
        resnet18 = models.resnet18(pretrained=True)
        self.stitched_features = nn.Sequential(*list(resnet18.children())[:-2])
        
        resnet34 = models.resnet34(pretrained=True)
        self.basemap_features = nn.Sequential(*list(resnet34.children())[:-2])
        
        # Freeze the pretrained layers
        for param in self.stitched_features.parameters():
            param.requires_grad = False
        for param in self.basemap_features.parameters():
            param.requires_grad = False
        
        # Combined processing
        self.conv_combined = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, stitched, basemap):
        x_stitched = self.stitched_features(stitched)
        x_basemap = self.basemap_features(basemap)
        
        # Ensure x_stitched and x_basemap have the same spatial dimensions
        x_stitched = nn.functional.adaptive_avg_pool2d(x_stitched, x_basemap.shape[2:])
        
        # Concatenate features from both inputs
        combined = torch.cat((x_stitched, x_basemap), dim=1)
        
        x = self.conv_combined(combined)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12378'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_dataloader(rank, world_size, dataset, batch_size=256, num_workers=10):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_model(rank, world_size, num_epochs, model, criterion, optimizer, train_dataset, val_dataset):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    train_dataloader = create_dataloader(rank, world_size, train_dataset)
    val_dataloader = create_dataloader(rank, world_size, val_dataset)
    
    best_val_loss = float('inf')
    best_model_state = None

    best_train_loss = float('inf')
    best_train_model_state = None

    losses = {"train": [], "val": []}  # Dictionary to store losses
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_dataloader.sampler.set_epoch(epoch)
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0)
        for stitched_imgs, basemap_imgs, locations in pbar:
            # try:
            stitched_imgs = stitched_imgs.to(device)
            basemap_imgs = basemap_imgs.to(device)
            locations = locations.to(device)

            optimizer.zero_grad()

            predicted_locations = model(stitched_imgs, basemap_imgs)
            loss = criterion(predicted_locations, locations)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            # except Exception as e:
            #     #print(e)
            #     continue
            # except KeyboardInterrupt:
            #     print("KeyboardInterrupt")
            #     cleanup()

        print("Train done")
        print("Train loss: ", running_loss / len(train_dataloader))

        train_loss = running_loss / len(train_dataloader)
        losses["train"].append([epoch,train_loss])
        if rank == 0 and train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_model_state = model.module.state_dict()
            torch.save(best_train_model_state, 'best_map_location_model_train_v1.pth')

        # Validation step
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                pbar = tqdm(val_dataloader, desc=f"Validation", disable=rank != 0)
                for stitched_imgs, basemap_imgs, locations in pbar:
                    # try:
                    stitched_imgs = stitched_imgs.to(device)
                    basemap_imgs = basemap_imgs.to(device)
                    locations = locations.to(device)

                    predicted_locations = model(stitched_imgs, basemap_imgs)
                    loss = criterion(predicted_locations, locations)
                    val_loss += loss.item()
                    if rank == 0:
                        pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})
                    # except Exception as e:
                    #     print(e)
                    #     continue
                    # except KeyboardInterrupt:
                    #     print("KeyboardInterrupt")
                    #     cleanup()
                    #     exit()
            val_loss /= len(val_dataloader)
            losses["val"].append([epoch,val_loss])
            # Save the best model
            if rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.module.state_dict()
                torch.save(best_model_state, 'best_map_location_model_val_v1.pth')

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save losses to JSON file after each epoch
            with open('loss_v1.json', 'w') as f:
                json.dump(losses, f)

    cleanup()

def main():

    world_size = torch.cuda.device_count()
    base_folder = "/data1/"

    transform_base = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
    ])

    transform_gen = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
    ])

    train_dataset = MapDataset(base_folder+'all_train_metas_v2', base_folder+'all_train_basemaps_v2', base_folder+'all_train_maps_gt_v2/map/', transform_base=transform_base, transform_gen=transform_gen)
    val_dataset = MapDataset(base_folder+'all_val_metas_v2', base_folder+'all_val_basemaps_v2', base_folder+'all_val_maps_gt_v2/map/', transform_base=transform_base, transform_gen=transform_gen)

    model = LocationModel()
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Total parameters:", count_total_parameters(model))

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0002)
    num_epochs = 300

    mp.spawn(
        train_model,
        args=(world_size, num_epochs, model, criterion, optimizer, train_dataset, val_dataset),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()