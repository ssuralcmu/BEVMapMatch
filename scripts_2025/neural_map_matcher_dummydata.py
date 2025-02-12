import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Define the neural network
class MapMatchingNet(nn.Module):
    def __init__(self):
        super(MapMatchingNet, self).__init__()
        # Feature extractor for stitched BEV map
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Feature extractor for global map
        self.global_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Matching layers
        self.fc = nn.Sequential(
            nn.Linear(2129920, 512),  # Adjust dimensions based on input size
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, bev_map, global_map):
        # Extract features
        bev_features = self.bev_encoder(bev_map)
        global_features = self.global_encoder(global_map)
        
        # Flatten features
        bev_features = bev_features.view(bev_features.size(0), -1)
        global_features = global_features.view(global_features.size(0), -1)
        
        # print("BEV shape:",bev_features.shape)
        # print("Global shape:",global_features.shape)

        # Concatenate features
        combined_features = torch.cat((bev_features, global_features), dim=1)

        # print("Combined shape:",combined_features.shape)
        
        # Pass through fully connected layers
        output = self.fc(combined_features)
        
        return output

# Dataset class (example implementation)
class MapMatchingDataset(Dataset):
    def __init__(self, bev_images, global_images, labels):
        self.bev_images = bev_images
        self.global_images = global_images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        bev_image = self.bev_images[idx]
        global_image = self.global_images[idx]
        label = self.labels[idx]
        
        return bev_image, global_image, label

# Training loop (example implementation)
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for epoch in range(100):  # Number of epochs
        epoch_loss = 0.0
        for bev_map, global_map, label in dataloader:
            bev_map = bev_map.to(device)
            global_map = global_map.to(device)
            label = label.to(device).float()
            
            optimizer.zero_grad()
            
            output = model(bev_map, global_map).squeeze()
            
            loss = criterion(output, label)
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}")

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy data (replace with actual data loading logic)
    num_samples = 100
    bev_images = torch.randn(num_samples, 3, 64, 64)  # Example dimensions (C x H x W)
    global_images = torch.randn(num_samples, 3, 512, 512)  # Example dimensions (C x H x W)
    labels = torch.randint(0, 2, (num_samples,))
    
    dataset = MapMatchingDataset(bev_images, global_images, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = MapMatchingNet().to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, device)
