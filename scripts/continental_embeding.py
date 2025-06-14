import os
import glob
import torch
import random
import logging
import numpy as np
from PIL import Image
from datetime import datetime
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import timm

# Logging setup
logging.basicConfig(filename='training_log.log', level=logging.INFO)

# Reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

# DatasetHHHHH                                                                   
class RotationDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.rotations = [0, 90, 180, 270]

    def __len__(self):
        return len(self.image_paths) * 4

    def __getitem__(self, idx):
        img_idx = idx // 4
        rot_idx = idx % 4
        angle = self.rotations[rot_idx]
        image = Image.open(self.image_paths[img_idx]).convert('RGB')
        image = image.rotate(angle)
        if self.transform:
            image = self.transform(image)
        return image, rot_idx

# Model
class RotationNet(nn.Module):
    def __init__(self, base_model, is_vit=False):
        super().__init__()
        if is_vit:
            self.backbone = base_model
            self.head = nn.Linear(base_model.num_features, 4)
        else:
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            self.head = nn.Linear(base_model.fc.in_features, 4)

    def forward(self, x):
        if isinstance(self.backbone, nn.Sequential):
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        else:
            features = self.backbone.forward_features(x)
        out = self.head(features)
        return out, features

# Training
def train_model(state, gpu_id, model_name):
    device = torch.device(f"cuda:{gpu_id}")
    image_paths = glob.glob(os.path.join(BASE_DIR, state, '*.tif'))
    if len(image_paths) == 0:
        logging.error(f"No images in {state}")
        return

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    dataset = RotationDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True, worker_init_fn=worker_init_fn)

    is_vit = False
    if model_name == 'resnet18':
        base_model = models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        base_model = models.resnet50(pretrained=True)
    elif model_name == 'vit_base_patch16_224':
        base_model = timm.create_model(model_name, img_size=640, pretrained=True)
        is_vit = True
    else:
        raise ValueError(f"Unsupported model {model_name}")

    model = RotationNet(base_model, is_vit=is_vit).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs("checkpoints", exist_ok=True)
    log_dir = f'logs/{state}_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=log_dir)

    model.train()
    for epoch in range(25):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            writer.add_scalar('Loss/Batch', loss.item(), epoch * len(dataloader) + i)

        epoch_loss = running_loss / len(dataloader.dataset)
        writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
        print(f"{state}-{model_name} on GPU {gpu_id} | Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/{state}_{model_name}_epoch_{epoch+1}.pth")

    writer.close()

# Main
if __name__ == '__main__':
    BASE_DIR = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/data/continent_classification"

    job_list = [
        # ("uttar_pradesh", 1, "resnet18"),
        # ("uttar_pradesh", 1, "resnet50"),
        # ("uttar_pradesh", 1, "vit_base_patch16_224"),
        # ("pak_punjab", 2, "resnet18"),
        # ("pak_punjab", 2, "resnet50"),
        # ("pak_punjab", 2, "vit_base_patch16_224"),
        # ("dhaka", 2, "resnet18"),
        # ("dhaka", 2, "resnet50"),
        ("dhaka", 2, "vit_base_patch16_224"),

    ]

    for state, gpu_id, model_name in job_list:
        train_model(state, gpu_id, model_name)
        logging.info(f"Started training for {state} on GPU {gpu_id} with model {model_name}")
        print(f"Started training for {state} on GPU {gpu_id} with model {model_name}")
        torch.cuda.empty_cache()
        logging.info(f"Finished training for {state} on GPU {gpu_id} with model {model_name}")
        print(f"Finished training for {state} on GPU {gpu_id} with model {model_name}")