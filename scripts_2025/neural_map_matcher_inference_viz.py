import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

# Import the MapDataset and LocationModel from your training script
from scripts_2025.neural_map_matcher_train import MapDataset, LocationModel


def save_results(stitched_imgs, basemap_imgs, true_bboxes, pred_bboxes, output_dir, start_index):
    for i in range(stitched_imgs.shape[0]):
        stitched_pil = transforms.ToPILImage()(stitched_imgs[i])
        basemap_pil = transforms.ToPILImage()(basemap_imgs[i])
        
        draw = ImageDraw.Draw(basemap_pil)
        
        # Convert NumPy array to a tuple of tuples
        true_bbox_tuple = tuple(map(tuple, true_bboxes[i].reshape(2, 2)))
        pred_bbox_tuple = tuple(map(tuple, pred_bboxes[i].reshape(2, 2)))

        # print(f"True bbox: {true_bbox_tuple}")
        # print(f"Pred bbox: {pred_bbox_tuple}")
        
        # Draw true bounding box (green)
        draw.rectangle(true_bbox_tuple, outline='green', width=2)
        
        # Draw predicted bounding box (red)
        draw.rectangle(pred_bbox_tuple, outline='red', width=2)
        
        # Save images
        stitched_pil.save(os.path.join(output_dir, f'stitched_{start_index + i}.png'))
        basemap_pil.save(os.path.join(output_dir, f'basemap_with_bbox_{start_index + i}.png'))


# Set up data transformations (same as in training)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_folder="/data1/"

# Create dataset and dataloader
print("Creating dataset ...")
dataset = MapDataset(base_folder+'all_val_basemaps_1km', base_folder+'all_val_maps_gt_stitched', transform=transform)
print("dataset created")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LocationModel().to(device)
model.load_state_dict(torch.load('map_location_model.pth'))
model.eval()

# Create output directory
output_dir = 'evaluation_results'
os.makedirs(output_dir, exist_ok=True)

# Evaluation loop
with torch.no_grad():
    for idx, (stitched_imgs, basemap_imgs, true_bboxes) in enumerate(dataloader):
        stitched_imgs = stitched_imgs.to(device)
        basemap_imgs = basemap_imgs.to(device)
        true_bboxes = true_bboxes.to(device)

        pred_bboxes = model(stitched_imgs, basemap_imgs)

        # Denormalize images for saving
        stitched_imgs = stitched_imgs.cpu()
        basemap_imgs = basemap_imgs.cpu()
        stitched_imgs = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                             std=[1/0.229, 1/0.224, 1/0.225])(stitched_imgs)
        basemap_imgs = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                            std=[1/0.229, 1/0.224, 1/0.225])(basemap_imgs)

        # Convert tensors to numpy for saving
        true_bboxes = true_bboxes.cpu().numpy()
        pred_bboxes = pred_bboxes.cpu().numpy()

        # Save results
        save_results(stitched_imgs, basemap_imgs, true_bboxes, pred_bboxes, output_dir, idx * dataloader.batch_size)

        print(f"Processed and saved image batch {idx+1}")

print("Evaluation complete. Results saved in the 'evaluation_results' directory.")

