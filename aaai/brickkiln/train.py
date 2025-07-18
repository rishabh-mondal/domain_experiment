# brickkiln/train.py
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from .dataset import collate_fn  # <-- import your collate_fn!

def train_model(train_dataset, device, num_classes=4, epochs=50, batch_size=128, output_root=None):
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.transform = GeneralizedRCNNTransform(
        min_size=640, max_size=640,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    losses_per_epoch = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, targets in dataloader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        losses_per_epoch.append(total_loss)
        # if output_root:
        #     torch.save(model.state_dict(), f"{output_root}/model_epoch_{epoch+1}.pth")
    return model, losses_per_epoch
