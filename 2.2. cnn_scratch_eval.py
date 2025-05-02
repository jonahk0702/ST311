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
print("All imports successful.")

# Load the dataset
imgs = load_from_disk('processed_streetview_384_split')['test']
print("Dataset loaded successfully.")
print("Dataset length:", len(imgs))
print("Dataset keys:", imgs[0].keys())
print("Img dtype:", type(imgs[0]['image']))

imgs = imgs[5000:] # for testing

# Create model specification
# Use mobilenet v3 as the model, without pretrained weights
model = torch.hub.load('pytorch/vision', 'mobilenet_v3_small', weights = None)
# replace the head to output 2 numbers for regression
model.classifier[3] = nn.Linear(in_features=1024, out_features=2, bias=True)
print("Model created successfully.")

# Load the model weights
model.load_state_dict(torch.load('cnn_scratch_trained.pth'))
model.eval()
print("Model weights loaded successfully.")

# Get predictions for the test set
with torch.no_grad():
    images = torch.from_numpy(np.asarray(imgs['image']))
    #images = torch.permute(images, (3, 0, 1, 2)).float()
    images = images.float()
    predictions = model(images)
print("Predictions made successfully.")

# Save predictions along with original lat/long values
pred_df = pd.DataFrame(predictions.numpy(), columns=['pred_latitude', 'pred_longitude'])
pred_df['true_latitude'] = imgs['latitude']
pred_df['true_longitude'] = imgs['longitude']
pred_df['address'] = imgs['address']
pred_df['country_iso_alpha2'] = imgs['country_iso_alpha2']
pred_df.to_csv('cnn_scratch_predictions4.csv', index=False)
print('Predictions saved.')
