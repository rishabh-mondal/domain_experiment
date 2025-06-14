import torch
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

from ultralytics import YOLO

# Load the YOLO11m-OBB model
model = YOLO("yolo11m-obb.pt")  # Load pre-trained model

# Train the model with custom parameters
model.train(
    data="/home/umang.shikarvar/instaformer/YOLO.yaml",  # Path to your dataset YAML file
    epochs=100,
    imgsz=640,
    batch=150,
    iou=0.33,
    conf=0.001,
    device=device,
    val=False  # Enables validation
)