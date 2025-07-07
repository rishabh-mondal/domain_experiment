import torch
import sys
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torchvision
# from model import VisionTransformer# or wherever it's defined
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from PIL import Image

# Paths
image_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/processed_data/delhi_airshed/images"
label_dir_yolo = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/processed_data/delhi_airshed/labels"
label_dir_voc = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/processed_data/delhi_airshed/label_aa_pascal_voc"

# -----------------------------
checkpoint = torch.load("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/satclip/satclip/satclip_logs/satclip/satclip/checkpoints/last-v1.ckpt", map_location="cpu")
state_dict = checkpoint["state_dict"] 
vit_state_dict = {
    k.replace("model.visual.", ""): v
    for k, v in state_dict.items()
    if k.startswith("model.visual.")
}
sys.path.append('/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/satclip/satclip')
from model import VisionTransformer# or wherever it's defined



vit_model = VisionTransformer(
    input_resolution=640,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    output_dim=768,
    in_channels=3,

)
vit_model.load_state_dict(vit_state_dict, strict=True)

class ViTBackboneForDetection(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.out_channels = 768  # Needed by torchvision

    def forward(self, x):
        _, patch_tokens = self.vit(x)
        B, N, C = patch_tokens.shape  # [B, 1600, 768]
        # print('patch_tokens shape', patch_tokens.shape)
        H = W = int(N ** 0.5)         # H = W = 40 for 640x640
        features = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)  # [B, 768, 40, 40]
        return {"0": features}
    


# -----------------------------
# Albumentations Transform Setup
# -----------------------------
def get_train_transform(resize_crop_size=640):
    return A.Compose([
        A.Resize(height=resize_crop_size, width=resize_crop_size),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.2),
        A.ToFloat(max_value=255.0, p=1.0),  # Convert to float for normalization
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# ---------------------
# Dataset for TIF images
# ---------------------
class TIFRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        image = np.array(Image.open(image_path).convert("RGB"))
        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        cls, x_min, y_min, x_max, y_max = parts
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(int(cls))  # background = 0

        # If no boxes, use dummy box to keep training stable (optional)
        if not boxes:
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        transformed = self.transforms(
            image=image,
            bboxes=boxes,
            class_labels=labels
        )

        transformed_image = transformed['image']
        transformed_boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        transformed_labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)

        target = {
            "boxes": transformed_boxes,
            "labels": transformed_labels
        }

        return transformed_image, target

# ---------------------
# Collate function
# ---------------------
def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------
# Backbone Wrapper
# ---------------------
class ViTBackboneForDetection(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        _, tokens = self.vit(x)
        B, N, C = tokens.shape
        H = W = int(N**0.5)
        tokens = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return {"0": tokens}
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")



# ---------------------
# Main Training
# ---------------------
Dataset = TIFRCNNDataset(
    image_dir=image_dir,
    label_dir=label_dir_voc,
    transforms=get_train_transform(640)  # Resize to 640x640
)
dataloader = DataLoader(
    Dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn
)


backbone = ViTBackboneForDetection(vit_model)
backbone.out_channels = 768
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),             # Only 1 feature map → 1 tuple of sizes
    aspect_ratios=((0.5, 1.0, 2.0),)              # Only 1 feature map → 1 tuple of ratios
)


roi_pooler = MultiScaleRoIAlign(
    featmap_names=["0"],
    output_size=7,
    sampling_ratio=2
)

model = FasterRCNN(
    backbone=backbone,
    num_classes=3,  # Adjust for your task
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler)

model_transform = GeneralizedRCNNTransform(
    min_size=640,
    max_size=640,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)
epoch_losses = []  # To store losses for each epoch
model.transform = model_transform  # Set the transform for the model
# model
model.to(device)
model.train()  # Set the model to training mode
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
for epoch in range(num_epochs):
    start_time = time.time()
    for images, targets in dataloader:
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
    epoch_losses.append(losses.item())
    end_time = time.time()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}, Time: {end_time - start_time:.2f}s")
    
model_path = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/satclip/satclip/satclip_logs/satclip/satclip/checkpoints/fasterrcnn_vit.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
# -----------------------------
    