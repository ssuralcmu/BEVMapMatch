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
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from scripts_2025.neural_map_matcher_train_v2_resnet_with_resume_train import MapDataset, LocationModel

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

def save_results(stitched_imgs, basemap_imgs, true_poses, pred_poses, output_dir, start_index):
    for i in range(stitched_imgs.shape[0]):
        stitched_pil = transforms.ToPILImage()(stitched_imgs[i])
        basemap_pil = transforms.ToPILImage()(basemap_imgs[i])
        
        draw = ImageDraw.Draw(basemap_pil)

        draw.ellipse((true_poses[i][0]-3, true_poses[i][1]-3, true_poses[i][0]+3, true_poses[i][1]+3), fill="red")
        draw.ellipse((pred_poses[i][0]-3, pred_poses[i][1]-3, pred_poses[i][0]+3, pred_poses[i][1]+3), fill="blue")
        
        # Save images
        stitched_pil.save(os.path.join(output_dir, f'stitched_{start_index + i}.png'))
        basemap_pil.save(os.path.join(output_dir, f'basemap_with_bbox_{start_index + i}.png'))

def train_model(rank, world_size, num_epochs, model, criterion, optimizer, train_dataset, val_dataset, batch_size, lr, version, fraction, checkpoint_path, seed):
    setup(rank, world_size)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    if fraction < 1.0 and checkpoint_path is None:
        num_samples = int(len(train_dataset) * fraction)
        subset_indices = torch.randperm(len(train_dataset))[:num_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
        start_epoch = 0
        losses = {"train": [], "val": []}

    elif fraction < 1.0 and checkpoint_path is not None and os.path.exists(checkpoint_path):
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
    
    # for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    # train_dataloader.sampler.set_epoch(epoch)
    pbar = tqdm(train_dataloader, desc=f"Eval", disable=rank != 0)
    count=0
    output_dir = 'eval_results_'+unique_name
    os.makedirs(output_dir, exist_ok=True)
    for stitched_imgs, basemap_imgs, locations in pbar:
        count+=1
        stitched_imgs = stitched_imgs.to(device)
        basemap_imgs = basemap_imgs.to(device)
        locations = locations.to(device)

        optimizer.zero_grad()

        predicted_locations = model(stitched_imgs, basemap_imgs)
        loss = criterion(predicted_locations, locations)

        stitched_imgs = stitched_imgs.cpu()
        basemap_imgs = basemap_imgs.cpu()

        true_poses = locations.cpu().numpy()
        pred_poses = predicted_locations.detach().cpu().numpy()

        save_results(stitched_imgs, basemap_imgs, true_poses, pred_poses, output_dir, count * train_dataloader.batch_size)

        running_loss += loss.item()
        if rank == 0:
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    train_loss = running_loss / len(train_dataloader)
    print("Loss: ",train_loss)
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file to resume training')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Fraction of training data to use (0.0 to 1.0)')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs

    mp.spawn(
        train_model,
        args=(world_size, num_epochs, model, criterion, optimizer, train_dataset, val_dataset, args.batch_size, args.lr, args.version, args.train_fraction, args.checkpoint, args.seed),
        nprocs=world_size,
        join=True
    )

    

if __name__ == '__main__':
    main()



# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# import os

# # Import the MapDataset and LocationModel from your training script
#


# def save_results(stitched_imgs, basemap_imgs, true_poses, pred_poses, output_dir, start_index):
#     for i in range(stitched_imgs.shape[0]):
#         stitched_pil = transforms.ToPILImage()(stitched_imgs[i])
#         basemap_pil = transforms.ToPILImage()(basemap_imgs[i])
        
#         draw = ImageDraw.Draw(basemap_pil)
        
#         # Convert NumPy array to a tuple of tuples
#         # print(true_poses[i])
#         # print(pred_poses[i])
#         # print(stitched_pil.size)
#         # print(basemap_pil.size)

#         # true_poses[i] = (int(true_poses[i][0] / 6), int(true_poses[i][1] / 6))
#         # pred_poses[i] = (int(pred_poses[i][0] / 6), int(pred_poses[i][1] / 6))

#         # print(stitched_imgs[i].shape)
#         # print(basemap_imgs[i].shape)

#         draw.ellipse((true_poses[i][0]-3, true_poses[i][1]-3, true_poses[i][0]+3, true_poses[i][1]+3), fill="red")
#         draw.ellipse((pred_poses[i][0]-3, pred_poses[i][1]-3, pred_poses[i][0]+3, pred_poses[i][1]+3), fill="blue")
        
#         # Save images
#         stitched_pil.save(os.path.join(output_dir, f'stitched_{start_index + i}.png'))
#         basemap_pil.save(os.path.join(output_dir, f'basemap_with_bbox_{start_index + i}.png'))


# # Set up data transformations (same as in training)
# transform_base = transforms.Compose([
#     transforms.Resize((500, 500)),
#     transforms.ToTensor(),
# ])
# transform_gen = transforms.Compose([
#     transforms.Resize((500, 500)),
#     transforms.ToTensor(),
# ])

# base_folder="/data1/"


# # Load the trained model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LocationModel().to(device)
# torch.manual_seed(seed)
#     np.random.seed(seed)
    
# model = DDP(model, device_ids=[rank])

# unique_id = 'train_v2-lr0.001-bs256-frac0.005-seed42'
# checkpoint = torch.load('latest_map_location_model_'+unique_id+'.pth')
# model.module.load_state_dict(checkpoint['model_state_dict'])
# # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch'] + 1
# losses = checkpoint['losses']
# subset_indices = checkpoint['subset_indices']  # Load the indices
# model.eval()


# # Create dataset and dataloader
# print("Creating dataset ...")
# dataset = MapDataset(base_folder+'all_train_metas_v2', base_folder+'all_train_basemaps_v2', base_folder+'all_train_maps_gt_v2/map/', transform_base=transform_base, transform_gen=transform_gen)
# dataset = torch.utils.data.Subset(dataset, subset_indices)
# print("dataset created")
# dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=10)

# # Create output directory
# output_dir = 'evaluation_results'+unique_id
# os.makedirs(output_dir, exist_ok=True)

# # Evaluation loop
# with torch.no_grad():
#     #Calculate MSE loss too
#     total_loss = 0
#     total_samples = 0
#     criterion = torch.nn.MSELoss()
#     for idx, (stitched_imgs, basemap_imgs, true_poses) in enumerate(dataloader):
#         stitched_imgs = stitched_imgs.to(device)
#         basemap_imgs = basemap_imgs.to(device)
#         true_poses = true_poses.to(device)

#         pred_poses = model(stitched_imgs, basemap_imgs)

#         # Calculate loss
#         loss = criterion(pred_poses, true_poses)
#         total_loss += loss.item()

#         print(stitched_imgs.shape)
#         print(basemap_imgs.shape)

#         stitched_imgs = stitched_imgs.cpu()
#         basemap_imgs = basemap_imgs.cpu()
#         # stitched_imgs = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#         #                                      std=[1/0.229, 1/0.224, 1/0.225])(stitched_imgs)
#         # basemap_imgs = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#         #                                     std=[1/0.229, 1/0.224, 1/0.225])(basemap_imgs)

#         # Convert tensors to numpy for saving
#         true_poses = true_poses.cpu().numpy()
#         pred_poses = pred_poses.cpu().numpy()

#         # Save results
#         save_results(stitched_imgs, basemap_imgs, true_poses, pred_poses, output_dir, idx * dataloader.batch_size)

#         print(f"Processed and saved image batch {idx+1}")
    
#     print(f"Mean squared error: {total_loss / len(dataloader)}")


# print("Evaluation complete. Results saved in the 'evaluation_results' directory.")

