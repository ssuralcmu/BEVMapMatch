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

            x_val = center_x - metas['perturbation'][0]/2
            y_val = center_y - metas['perturbation'][1]/2

            location = torch.tensor([x_val, y_val], dtype=torch.float32)

            return stitched_img, basemap_img, location
        except Exception as e:
            return torch.zeros(3, 500, 500), torch.zeros(3, 500, 500), torch.zeros(2)


class LocationModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(LocationModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.stitched_features = nn.Sequential(*list(resnet.children())[:-2])
        self.basemap_features = nn.Sequential(*list(resnet.children())[:-2])
        
        # for param in self.stitched_features.parameters():
        #     param.requires_grad = False
        # for param in self.basemap_features.parameters():
        #     param.requires_grad = False
        
        self.conv_combined = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def custom_activation(self, x):
        return 250 + 250 * torch.tanh(x)

    def forward(self, stitched, basemap):
        x_stitched = self.stitched_features(stitched)
        x_basemap = self.basemap_features(basemap)
        
        combined = torch.cat((x_stitched, x_basemap), dim=1)
        
        x = self.conv_combined(combined)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.custom_activation(x)
        
        return x

# class LocationModel(nn.Module):
#     def __init__(self):
#         super(LocationModel, self).__init__()
#         resnet = models.resnet18(pretrained=True)
#         self.stitched_features = nn.Sequential(*list(resnet.children())[:-2])
        
#         # resnet34 = models.resnet34(pretrained=True)
#         self.basemap_features = nn.Sequential(*list(resnet.children())[:-2])
        
#         for param in self.stitched_features.parameters():
#             param.requires_grad = False
#         for param in self.basemap_features.parameters():
#             param.requires_grad = False
        
#         self.conv_combined = nn.Sequential(
#             nn.Conv2d(1024, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),  # Add BatchNorm here
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
        
#         self.fc = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),  # Add BatchNorm here
#             nn.ReLU(),
#             nn.Linear(256, 2)
#         )

#     def custom_activation(self, x):
#         return 250 + 250 * torch.tanh(x)


#     def forward(self, stitched, basemap):
        
#         #Save stitched and basemap features for visualization
#         # torch.save(stitched, 'stitched_img.pth')
#         # torch.save(basemap, 'basemap_img.pth')

#         # print("Stitched: ", stitched.shape)
#         # print("Basemap: ", basemap.shape)
        
#         x_stitched = self.stitched_features(stitched)
#         x_basemap = self.basemap_features(basemap)


#         #Save stitched and basemap features for visualization
#         # torch.save(x_stitched, 'stitched_features.pth')
#         # torch.save(x_basemap, 'basemap_features.pth')

#         # print("Stitched Features: ", x_stitched.shape)
#         # print("Basemap Features: ", x_basemap.shape)
        
#         #x_stitched = nn.functional.adaptive_avg_pool2d(x_stitched, x_basemap.shape[2:])
        
#         combined = torch.cat((x_stitched, x_basemap), dim=1)
        
#         # print("Combined:", combined.shape)

#         x = self.conv_combined(combined)

#         # print("Conv Combined:", x.shape)
#         x = x.view(x.size(0), -1)
#         # print("Flattened:", x.shape)
#         x = self.fc(x)
#         x = self.custom_activation(x)
#         # print("FC:", x.shape)
        
#         return x

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
        for stitched_imgs, basemap_imgs, locations in pbar:
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

        if epoch % 2 == 0 or epoch == start_epoch:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                pbar = tqdm(val_dataloader, desc=f"Validation", disable=rank != 0)
                for stitched_imgs, basemap_imgs, locations in pbar:
                    stitched_imgs = stitched_imgs.to(device)
                    basemap_imgs = basemap_imgs.to(device)
                    locations = locations.to(device)

                    predicted_locations = model(stitched_imgs, basemap_imgs)
                    loss = criterion(predicted_locations, locations)
                    val_loss += loss.item()
                    if rank == 0:
                        pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})

            val_loss /= len(val_dataloader)
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # checkpoint = {
            #     'epoch': epoch,
            #     'model_state_dict': model.module.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'losses': losses
            # }
            # torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
            
            with open('loss_'+unique_name+'.json', 'w') as f:
                json.dump(losses, f)

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

    train_dataset = MapDataset(base_folder+'all_train_metas_v2', base_folder+'all_train_basemaps_v2', base_folder+'all_train_maps_gt_v2/map/', transform_base=transform_base, transform_gen=transform_gen)
    

    val_dataset = MapDataset(base_folder+'all_val_metas_v2', base_folder+'all_val_basemaps_v2', base_folder+'all_val_maps_gt_v2/map/', transform_base=transform_base, transform_gen=transform_gen)

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
