
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
imgs = load_from_disk('processed_streetview_384_split')['train']
print("Dataset length:", len(imgs))
print("Dataset keys:", imgs[0].keys())
print("Img dtype:", type(imgs[0]['image']))
#print(imgs[0]['image'][0][0])

#imgs = imgs[:100] # for testing


from transformers import ViTModel, ViTFeatureExtractor

class ViTRegressor(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-384"):
        super(ViTRegressor, self).__init__()
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(model_name)
        # Add a dropout layer for regularization
        self.dropout = nn.Dropout(0.1)
        # Regression head: outputs 2 values (latitude and longitude)
        self.regressor = nn.Linear(self.vit.config.hidden_size, 2)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # Use the [CLS] token representation (pooler output)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        regression_output = self.regressor(pooled_output)
        return regression_output

# Initialize model and feature extractor
model = ViTRegressor()

# Freeze the ViT backbone so that only the regression head is trainable
for param in model.vit.parameters():
    param.requires_grad = False

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-384", size=384, do_rescale=False)
# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('Model moved to device.')

# Define the loss function and optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = HaversineLoss()


#Convert pixels into tensors

# Collate function for batching
def collate_fn(batch):
    pixel_values = []
    for item in batch:
        pixel_tensor = torch.as_tensor(item["image"])
        pixel_values.append(pixel_tensor)

    pixel_values = torch.stack(pixel_values)
    targets = torch.tensor([[item["latitude"], item["longitude"]] for item in batch], dtype=torch.float32)
    return {"pixel_values": pixel_values, "targets": targets}


# Create DataLoader
train_loader = DataLoader(imgs, batch_size=64, shuffle=True, collate_fn=collate_fn)
print('Dataloader created.')


# Inside your training loop, apply the feature extractor to resize images
num_epochs = 20
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Extract the images and resize them to 224x224 using the feature extractor
        pixel_values = feature_extractor(batch["pixel_values"], return_tensors="pt").pixel_values.to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(pixel_values)

        # Calculate the loss
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        #print('One mini-batch complete')

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")
    losses.append(epoch_loss/len(train_loader))
    torch.save(model.state_dict(), f'finetune_v2_epoch_{epoch+1}.pth')
    print(f"Model saved for epoch {epoch+1}")



print('Training Complete.')

# save the model
torch.save(model.state_dict(), 'finetune_v2_trained.pth')
print("Model saved successfully.")

# save the losses
epoch_nos = range(1, 21)
train_log = pd.DataFrame({'epoch': epoch_nos, 'loss': losses})
train_log.to_csv('train_log_finetune.csv')
