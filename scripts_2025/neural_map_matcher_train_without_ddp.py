import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

class MapDataset(Dataset):
    def __init__(self, metas_folder, basemap_folder, stitched_folder, transform=None):
        self.basemap_folder = basemap_folder
        self.stitched_folder = stitched_folder
        self.metas_folder = metas_folder
        self.transform = transform
        
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
        stitched_file, basemap_file, metas_file = self.file_triplets[idx]
        
        stitched_img_path = os.path.join(self.stitched_folder, stitched_file)
        basemap_img_path = os.path.join(self.basemap_folder, basemap_file)
        metas_path = os.path.join(self.metas_folder, metas_file)

        stitched_img = Image.open(stitched_img_path).convert('RGB')
        basemap_img = Image.open(basemap_img_path).convert('RGB')

        # print(stitched_img.size)

        if self.transform:
            stitched_img = self.transform(stitched_img)
            basemap_img = self.transform(basemap_img)

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

        x_val=center_x - metas['perturbation'][0]/2 # Size mult by 2 and then divided by 4
        y_val=center_y - metas['perturbation'][1]/2

        location = torch.tensor([x_val, y_val], dtype=torch.float32)

        return stitched_img, basemap_img, location

class LocationModel(nn.Module):
    def __init__(self):
        super(LocationModel, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(984064, 1024)
        self.fc2 = nn.Linear(1024, 2)  # 4 outputs for bbox coordinates
        self.relu = nn.ReLU()

    def forward(self, stitched, basemap):
        x = torch.cat((stitched, basemap), dim=1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten while preserving batch size
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(num_epochs, model, criterion, optimizer, dataloader, val_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        # Validation step
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
                for stitched_imgs, basemap_imgs, locations in pbar:
                    stitched_imgs = stitched_imgs.to(device)
                    basemap_imgs = basemap_imgs.to(device)
                    locations = locations.to(device)

                    predicted_locations = model(stitched_imgs, basemap_imgs)
                    loss = criterion(predicted_locations, locations)
                    val_loss += loss.item()
                    pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})
            val_loss /= len(val_dataloader)
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                torch.save(best_model_state, 'best_map_location_model.pth')


        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}")


if __name__ == '__main__':
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_folder="/data1/"

    # Create dataset and dataloader
    dataset = MapDataset(base_folder+'all_train_metas_v2', base_folder+'all_train_basemaps_v2', base_folder+'all_train_maps_gt_v2/map/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    val_dataset = MapDataset(base_folder+'all_val_metas_v2', base_folder+'all_val_basemaps_v2', base_folder+'all_val_maps_gt_v2/map/', transform=transform)
    val_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    print('Init model ...')
    # Initialize the model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LocationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 300

    train_model(num_epochs, model, criterion, optimizer, dataloader, val_dataloader)

    # Save the trained model
    torch.save(model.state_dict(), 'map_location_model.pth')
