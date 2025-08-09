import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml_utils.road_dataset import LaneDataset
from ml_utils.deeplab.deeplab_model import get_deeplab_model
from ml_utils.unet.unet_model import UNet

device = torch.device("cpu")

# Load dataset & dataloader
dataset = LaneDataset("data/images", "data/masks")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Initialize Unet model
# model = UNet().to(device)

# Initialize DeepLab model
weights_path = "ml_utils/weights/berkeley_deeplab.pth" # Get Berkeley weights
model = get_deeplab_model(num_classes=1, weights_path=weights_path, device=device)

# Set loss and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(50):
    model.train()
    total_loss = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)['out'] # DeepLab returns dict; use ['out']. UNet returns tensor directly.
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "road_model.pth")
print("Model saved to road_model.pth")