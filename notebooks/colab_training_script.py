# COPY THIS CODE INTO A GOOGLE COLAB CELL
# ==========================================

# 1. Install Ultralytics YOLO
%pip install ultralytics

import os
from ultralytics import YOLO

# 2. Download Dataset (Using a robust Pothole dataset from Roboflow/Kaggle)
# We will use a direct download link to a pre-packaged YOLO format dataset for ease of use.
# This dataset is publicly available for pothole detection.
!mkdir -p datasets
%cd datasets
!curl -L "https://universe.roboflow.com/ds/6WdK75F8bA?key=YOUR_KEY" > roboflow.zip; unzip -o roboflow.zip; rm roboflow.zip
# NOTE: If the above link expires, we can use the Kaggle API.
# For now, let's use a standard sample command or ask user to upload 'data.yaml' if they have one.
# BETTER APPROACH FOR AUTOMATION:
# We will use a demo dataset or simple download if the user doesn't have a kaggle token set up.
# Let's assume we want to download the "Pothole Detection" dataset from Kaggle via API or direct URL.
# Since we can't guarantee API keys, we will use the official YOLOv8 demo first to verify, 
# but for potholes specifically, here is a working snippet for a public dataset:

%cd /content
!git clone https://github.com/miki998/Pothole-Detection-YOLOv8.git
# This repo contains a dataset or links to it. 
# ACTUALLY, simpler path: Use the Roboflow public dataset which is very common.
# Url: https://universe.roboflow.com/v1-dec-2021/pothole-detection-1
# We will simply instruct the user to use the Roboflow snippet in the notebook output.

# LET'S WRITE A GENERIC TRAINING SCRIPT THAT WORKS ONCE DATA IS PRESENT
print("Please upload your 'data.yaml' and dataset folder if you have one, or use the command below to download a sample.")

# 3. Train the Model
# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs usage is not available in free tier but standard usage is fine
# We train for 20-50 epochs
print("Starting Training...")
results = model.train(data='pothole.yaml', epochs=20, imgsz=640)

# 4. Validate
print("Validating...")
metrics = model.val()

# 5. Export for Deployment
print("Exporting...")
path = model.export(format='onnx')  # export the model to ONNX format or keep as .pt
print(f"Model exported to {path}")
print("Please download the 'best.pt' file from runs/detect/train/weights/best.pt")
