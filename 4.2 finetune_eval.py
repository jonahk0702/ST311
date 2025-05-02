# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import ViTModel, ViTFeatureExtractor
print("All imports successful.")

# Load the dataset
imgs = load_from_disk('processed_streetview_384_split')['test']
print("Dataset loaded successfully.")
print("Dataset length:", len(imgs))
print("Dataset keys:", imgs[0].keys())
print("Img dtype:", type(imgs[0]['image']))

imgs = imgs[6000:] # For testing

# Create model specification
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
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-384", size=384, do_rescale=False)
print("Model created successfully.")

# Load the model weights
model.load_state_dict(torch.load('finetune_trained.pth'))
model.eval()
print("Model weights loaded successfully.")

# Get predictions for the test set
with torch.no_grad():
    images = torch.from_numpy(np.asarray(imgs['image']))
    images = feature_extractor(images, return_tensors="pt").pixel_values
    #images = images.float()
    predictions = model(images)
print("Predictions made successfully.")

# Save predictions along with original lat/long values
pred_df = pd.DataFrame(predictions.numpy(), columns=['pred_latitude', 'pred_longitude'])
pred_df['true_latitude'] = imgs['latitude']
pred_df['true_longitude'] = imgs['longitude']
pred_df['address'] = imgs['address']
pred_df['country_iso_alpha2'] = imgs['country_iso_alpha2']
pred_df.to_csv('finetune_predictions7.csv', index=False)
print("Predictions saved.")
