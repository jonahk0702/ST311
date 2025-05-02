# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
print("All imports successful.")

# Create Haversine Loss Function
class HaversineLoss(torch.nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()
        # Earth's radius in km (use 3956 for miles)
        self.R = torch.tensor(6371.0, requires_grad=False)

    def forward(self, preds, targets):
        # Ensure R is on the same device as inputs
        self.R = self.R.to(preds.device)

        # Convert degrees to radians
        preds_rad = torch.deg2rad(preds)
        targets_rad = torch.deg2rad(targets)

        # Split into latitude and longitude
        pred_lat, pred_lon = preds_rad[:, 0], preds_rad[:, 1]
        target_lat, target_lon = targets_rad[:, 0], targets_rad[:, 1]

        # Calculate differences
        dlat = target_lat - pred_lat
        dlon = target_lon - pred_lon

        # Haversine formula
        a = (torch.sin(dlat/2)**2 +
             torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon/2)**2)
        c = 2 * torch.asin(torch.sqrt(a))

        # Calculate distance
        distances = c * self.R

        # Return mean distance as loss
        return torch.mean(distances)

print("Haversine Loss Function created successfully.")

# Load the dataset
imgs = load_from_disk('processed_streetview')['train']
print("Dataset loaded successfully.")
print("Dataset length:", len(imgs))
print("Dataset keys:", imgs[0].keys())
print("Img dtype:", type(imgs[0]['image']))
#print(imgs[0]['image'][0][0])

# Create model specification
# Use mobilenet v3 as the model, without pretrained weights
model = torch.hub.load('pytorch/vision', 'mobilenet_v3_small', weights = None)
# replace the head to output 2 numbers for regression
model.classifier[3] = nn.Linear(in_features=1024, out_features=2, bias=True)
print("Model created successfully.")

# Test with one image
#example_image = imgs[0]['image']
#output = model(example_image.float().unsqueeze(0))
#print("Test Output Latitude and Longitude:", output)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('Model moved to device.')

# Define the loss function and optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = HaversineLoss()

# Create DataLoader
dataloader = DataLoader(imgs, batch_size=64, shuffle=True)
print('Dataloader created.')

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        images = torch.from_numpy(np.asarray(batch['image']))
        images = torch.permute(images, (3, 0, 1, 2)).float()
        images = images.to(device)
        targets = torch.stack((torch.tensor(batch['latitude']), torch.tensor(batch['longitude'])), dim=1)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")
    # save the model every epoch
    torch.save(model.state_dict(), f'cnn_scratch_epoch_{epoch+1}.pth')
    print(f"Model saved for epoch {epoch+1}")

print('Training Complete.')

# save the model
torch.save(model.state_dict(), 'cnn_scratch_trained.pth')
print("Model saved successfully.")
