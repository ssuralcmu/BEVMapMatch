import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class MapDataset(Dataset):
    def __init__(self, basemap_folder, stitched_folder, transform=None):
        self.basemap_folder = basemap_folder
        self.stitched_folder = stitched_folder
        self.transform = transform
        
        # Get all stitched image files
        stitched_files = os.listdir(stitched_folder)
        
        # Create a list of matching file pairs
        self.file_pairs = []
        for stitched_file in stitched_files:
            if stitched_file.endswith("_stitched_map.png"):
                prefix = stitched_file.split("_stitched_map.png")[0]
                basemap_file = f"{prefix}_generated_map_image.png"
                if os.path.exists(os.path.join(basemap_folder, basemap_file)):
                    self.file_pairs.append((stitched_file, basemap_file))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        stitched_file, basemap_file = self.file_pairs[idx]
        
        stitched_img_path = os.path.join(self.stitched_folder, stitched_file)
        basemap_img_path = os.path.join(self.basemap_folder, basemap_file)

        stitched_img = Image.open(stitched_img_path).convert('RGB')
        basemap_img = Image.open(basemap_img_path).convert('RGB')

        if self.transform:
            stitched_img = self.transform(stitched_img)
            basemap_img = self.transform(basemap_img)

        # Calculate the center of the basemap image
        center_x, center_y = basemap_img.shape[1] // 2, basemap_img.shape[2] // 2

        # print(basemap_img.shape)
        
        # Calculate the bounding box coordinates (500x500 pixels around the center)
        x1 = center_x - 80
        y1 = center_y - 80
        x2 = center_x + 80
        y2 = center_y + 80
        
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        return stitched_img, basemap_img, bbox

class LocationModel(nn.Module):
    def __init__(self):
        super(LocationModel, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 4)  # 4 outputs for bbox coordinates
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


def train_model(num_epochs, model, criterion, optimizer, dataloader):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for stitched_imgs, basemap_imgs, bboxes in pbar:
            stitched_imgs = stitched_imgs.to(device)
            basemap_imgs = basemap_imgs.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()

            predicted_bboxes = model(stitched_imgs, basemap_imgs)
            loss = criterion(predicted_bboxes, bboxes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")


if __name__ == '__main__':
    # Set up data transformations
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

    print('Init model ...')
    # Initialize the model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LocationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 3

    train_model(num_epochs, model, criterion, optimizer, dataloader)

    # Save the trained model
    torch.save(model.state_dict(), 'map_location_model.pth')
