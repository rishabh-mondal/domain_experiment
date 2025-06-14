import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm

# Set device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Load model
backbone = resnet_fpn_backbone(
    backbone_name='resnet50',
    weights=ResNet50_Weights.IMAGENET1K_V1  # Or ResNet50_Weights.DEFAULT
)
# Your number of classes (3 foreground + 1 background)
model = FasterRCNN(backbone, num_classes=4)
model.load_state_dict(torch.load("/home/umang.shikarvar/Style_GAN/detector/detector_epoch_90.pth"))
model.roi_heads.nms_thresh = 0.33
model.to(device)
model.eval()

# Paths
image_dir = "/home/umang.shikarvar/Style_GAN/lucknow/images"
output_dir = "/home/umang.shikarvar/Style_GAN/lucknow/predictions"
os.makedirs(output_dir, exist_ok=True)

# Image transform
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Inference loop
with torch.no_grad():
    for filename in tqdm(os.listdir(image_dir)):
        if not filename.endswith((".jpg", ".png", ".tif")):
            continue

        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        outputs = model(image_tensor)[0]

        # Format output
        pred_lines = []
        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        if len(boxes) == 0:
            continue

        print(f"{filename} → {len(outputs['boxes'])} boxes")

        for label, score, box in zip(labels, scores, boxes):
            x1, y1, x2, y2 = box.tolist()
            line = f"{label.item()-1} {score.item():.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
            pred_lines.append(line)

        # Write predictions to .txt
        out_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
        with open(out_file, "w") as f:
            f.write("\n".join(pred_lines))