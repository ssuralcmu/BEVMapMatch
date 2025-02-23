import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

# Import the MapDataset and LocationModel from your training script
from scripts_2025.neural_map_matcher_train_v1_resnet import MapDataset, LocationModel


def save_results(stitched_imgs, basemap_imgs, true_poses, output_dir, start_index):
    for i in range(stitched_imgs.shape[0]):
        stitched_pil = transforms.ToPILImage()(stitched_imgs[i])
        basemap_pil = transforms.ToPILImage()(basemap_imgs[i])
        
        draw = ImageDraw.Draw(basemap_pil)
        
        # Convert NumPy array to a tuple of tuples
        # print(true_poses[i])
        # print(pred_poses[i])
        # print(stitched_pil.size)
        # print(basemap_pil.size)

        # true_poses[i] = (int(true_poses[i][0] / 6), int(true_poses[i][1] / 6))
        # pred_poses[i] = (int(pred_poses[i][0] / 6), int(pred_poses[i][1] / 6))

        # print(stitched_imgs[i].shape)
        # print(basemap_imgs[i].shape)

        ellipse_size=3
        draw.ellipse((true_poses[i][0]-ellipse_size, true_poses[i][1]-ellipse_size, true_poses[i][0]+ellipse_size, true_poses[i][1]+ellipse_size), fill="red")
        # draw.ellipse((pred_poses[i][0]-3, pred_poses[i][1]-3, pred_poses[i][0]+3, pred_poses[i][1]+3), fill="blue")
        
        # Save images
        stitched_pil.save(os.path.join(output_dir, f'stitched_{start_index + i}.png'))
        basemap_pil.save(os.path.join(output_dir, f'basemap_with_bbox_{start_index + i}.png'))


# Set up data transformations (same as in training)
transform_base = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
])
transform_gen = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
])
# transform=transforms.Compose([transforms.ToTensor()])

base_folder="/data1/"

# Create dataset and dataloader
print("Creating dataset ...")
dataset = MapDataset(base_folder+'all_val_metas_v2', base_folder+'all_val_basemaps_v2', base_folder+'all_val_maps_gt_v2/map/', transform_base=transform_base, transform_gen=transform_gen)
print("dataset created")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# # Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LocationModel().to(device)
# model.load_state_dict(torch.load('best_map_location_model_train.pth'))
# model.eval()

# Create output directory
output_dir = 'evaluation_results'
os.makedirs(output_dir, exist_ok=True)

# Evaluation loop
with torch.no_grad():
    for idx, (stitched_imgs, basemap_imgs, true_poses) in enumerate(dataloader):
        stitched_imgs = stitched_imgs.to(device)
        basemap_imgs = basemap_imgs.to(device)
        true_poses = true_poses.to(device)

        print(stitched_imgs.shape)
        print(basemap_imgs.shape)

        stitched_imgs = stitched_imgs.cpu()
        basemap_imgs = basemap_imgs.cpu()

        # Convert tensors to numpy for saving
        true_poses = true_poses.cpu().numpy()

        # Save results
        save_results(stitched_imgs, basemap_imgs, true_poses, output_dir, idx * dataloader.batch_size)

        print(f"Processed and saved image batch {idx+1}")

print("Evaluation complete. Results saved in the 'evaluation_results' directory.")

