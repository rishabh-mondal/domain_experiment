import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm

def get_transform(model_name):
    if 'vit' in model_name.lower():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        config = resolve_data_config({}, model=timm.create_model(model_name, pretrained=True))
        return create_transform(**config)

def train_model(state, gpu_id, model_name):
    if torch.cuda.is_available():
        assert gpu_id < torch.cuda.device_count(), f"Invalid GPU ID: {gpu_id}"
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    data_dir = f"/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/data/continent_crop_classification/{state}"
    num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)
    transform = get_transform(model_name)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints_for_crops", exist_ok=True)
    log_dir = f'logs/{state}_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=log_dir)

    model.train()
    for epoch in range(50):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            writer.add_scalar('Loss/Batch', loss.item(), epoch * len(dataloader) + i)

        epoch_loss = running_loss / len(dataloader.dataset)
        writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
        print(f"{state}-{model_name} on GPU {gpu_id} | Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints_for_crops/{state}_{model_name}_epoch_{epoch+1}.pth")

    writer.close()

if __name__ == '__main__':
    job_list = [
        ("uttar_pradesh", 0, "resnet18"),
        # ("uttar_pradesh", 0, "resnet50"),
        # ("uttar_pradesh", 0, "vit_base_patch16_224"),
        ("pak_punjab", 1, "resnet18"),
        # ("pak_punjab", 1, "resnet50"),
        # ("pak_punjab", 1, "vit_base_patch16_224"),
        # ("dhaka", 2, "resnet18"),
        # ("dhaka", 2, "resnet50"),
        # ("dhaka", 2, "vit_base_patch16_224"),
    ]

    for state, gpu_id, model_name in job_list:
        try:
            train_model(state, gpu_id, model_name)
            print(f"Finished training for {state} on GPU {gpu_id} with model {model_name}")
        except AssertionError as e:
            print(f"Skipping job: {e}")
        except RuntimeError as e:
            print(f"Runtime error for {state} on GPU {gpu_id}: {e}")
        torch.cuda.empty_cache()










