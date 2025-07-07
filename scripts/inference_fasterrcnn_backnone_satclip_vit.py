import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from albumentations.pytorch import ToTensorV2
import albumentations as A

# Path to ViT model class
sys.path.append('/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/satclip/satclip')
from model import VisionTransformer

# ---------------------
# Vision Transformer Setup
# ---------------------
vit_model = VisionTransformer(
    input_resolution=640,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    output_dim=768,
    in_channels=3,
)

# ---------------------
# ViT Backbone Wrapper
# ---------------------
class ViTBackboneForDetection(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.out_channels = 768

    def forward(self, x):
        _, tokens = self.vit(x)
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        return {"0": tokens.permute(0, 2, 1).reshape(B, C, H, W)}

# ---------------------
# Load Model
# ---------------------
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

backbone = ViTBackboneForDetection(vit_model)
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)
roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

model = FasterRCNN(
    backbone=backbone,
    num_classes=3,  # 2 classes + background
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)
model.transform = GeneralizedRCNNTransform(
    min_size=640,
    max_size=640,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)
model.load_state_dict(torch.load(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/satclip/satclip/satclip_logs/satclip/satclip/checkpoints/fasterrcnn_vit.pth",
    map_location=device
))
model.to(device).eval()

# ---------------------
# Dataset for Inference
# ---------------------
class TIFRCNNDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        transformed = self.transforms(image=image_np)
        return transformed["image"], img_name

# ---------------------
# Albumentations Transform
# ---------------------
test_transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # A.ToFloat(max_value=255.0, p=1.0),  
    ToTensorV2()
])

# ---------------------
# DataLoader
# ---------------------
image_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/processed_data/delhi_airshed/images"
output_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/processed_data/delhi_airshed/predictions"
os.makedirs(output_dir, exist_ok=True)

test_dataset = TIFRCNNDataset(image_dir=image_dir, transforms=test_transform)

def collate_fn(batch):
    return tuple(zip(*batch))

test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# ---------------------
# Inference
# ---------------------
CONF_THRESHOLD = 0.33

with torch.no_grad():
    for i, (images, filenames) in enumerate(tqdm(test_loader, desc="Evaluating")):
        image_tensor = images[0].to(device)
        outputs = model([image_tensor])[0]

        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs["labels"]

        keep = scores > CONF_THRESHOLD
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            continue

        pred_lines = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.cpu().numpy()
            print(f"Image: {filenames[0]}, Box: {box}, Score: {score}, Label: {label}")
            pred_lines.append(f"{label.item()-1} {score.item():.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")

        out_file = os.path.join(output_dir, os.path.splitext(filenames[0])[0] + ".txt")
        with open(out_file, "w") as f:
            f.write("\n".join(pred_lines))
