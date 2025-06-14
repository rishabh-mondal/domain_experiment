import torch
from itertools import zip_longest
from dataset import SourceDataset, TargetDataset
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from torchvision.transforms import ToTensor
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 200
BATCH_SIZE = 10
λ_det = 1.0
λ_adv = 0.5
λ_id  = 5.0

print(f"Using device: {device}")

def collate_fn(batch):
    return tuple(zip(*batch))

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Load dataset
source = SourceDataset(
    image_dir='/home/umang.shikarvar/Style_GAN/delhi/images',
    label_dir='/home/umang.shikarvar/Style_GAN/delhi/labels',
    transforms=transform
)
source_loader = DataLoader(source, batch_size=BATCH_SIZE,pin_memory=True, num_workers=8, shuffle=True, collate_fn=collate_fn)

target= TargetDataset(
    image_dir='/home/umang.shikarvar/Style_GAN/lucknow/images',
    transforms=transform
)
target_loader = DataLoader(target, batch_size=BATCH_SIZE, pin_memory=True, num_workers=8, shuffle=True)

# Load backbone with pretrained weights
backbone = resnet_fpn_backbone(
    backbone_name='resnet50',
    weights=ResNet50_Weights.IMAGENET1K_V1  # Or ResNet50_Weights.DEFAULT
)
# Your number of classes (3 foreground + 1 background)
detector = FasterRCNN(backbone, num_classes=4)
detector.to(device)
detector.train()

# Load generator and discriminator
G = Generator(img_channels=3).to(device)
D = Discriminator().to(device)

opt_G = Adam(G.parameters(), lr=1e-4)
opt_D = Adam(D.parameters(), lr=1e-4)
params = [p for p in detector.parameters() if p.requires_grad]
opt_detector = torch.optim.Adam(params, lr=1e-4)

scheduler_G = CosineAnnealingLR(opt_G, T_max=100)
scheduler_D = CosineAnnealingLR(opt_D, T_max=100)
scheduler_detector = CosineAnnealingLR(opt_detector, T_max=100)

L1_loss = nn.L1Loss()

warmup_epochs = 10  # Number of epochs to warm up G and D

for epoch in range(num_epochs):
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    epoch_loss_det = 0.0
    epoch_loss_adv = 0.0
    epoch_loss_id  = 0.0

    for (batch_s, batch_t) in zip_longest(source_loader, target_loader):
        if batch_s is None or batch_t is None:
            continue

        x_s, y_s = batch_s
        x_t = batch_t

        x_s = torch.stack(x_s).to(device)
        y_s = [{k: v.to(device) for k, v in tgt.items()} for tgt in y_s]
        x_t = x_t.to(device)

        # --- 1. Source → Target-style translation ---
        x_s2t = G(x_s)

        # --- 2. Identity loss: G(x_t) ≈ x_t ---
        x_t_id = G(x_t)
        loss_id = L1_loss(x_t_id, x_t)

        # --- 3. Discriminator loss ---
        pred_real = D(x_t)
        pred_fake = D(x_s2t.detach())
        loss_D = mse_loss(pred_real, torch.ones_like(pred_real)) + \
                 mse_loss(pred_fake, torch.zeros_like(pred_fake))

        # --- 4. Generator adversarial loss ---
        pred_fake_G = D(x_s2t)
        loss_adv = mse_loss(pred_fake_G, torch.ones_like(pred_fake_G))

        if epoch >= warmup_epochs:
            # --- 5. Detection loss on translated source images ---
            loss_dict = detector(x_s2t, y_s)
            loss_det = sum(loss for loss in loss_dict.values())
        else:
            loss_det = torch.tensor(0.0, device=device)

        # --- 6. Total generator loss ---
        loss_G = λ_det * loss_det + λ_adv * loss_adv + λ_id * loss_id

        # --- 7. Backprop ---
        opt_G.zero_grad()
        if epoch >= warmup_epochs:
            opt_detector.zero_grad()
        loss_G.backward()
        opt_G.step()
        if epoch >= warmup_epochs:
            opt_detector.step()

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()


        # Step schedulers
        scheduler_G.step()
        scheduler_D.step()
        if epoch >= warmup_epochs:
            scheduler_detector.step()

        # Accumulate losses
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        epoch_loss_det += loss_det.item()
        epoch_loss_adv += loss_adv.item()
        epoch_loss_id += loss_id.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss G_total: {epoch_loss_G:.4f}, "
          f"Loss G_adv: {epoch_loss_adv:.4f}, "
          f"Loss G_detect: {epoch_loss_det:.4f}, "
          f"Loss G_identity: {epoch_loss_id:.4f}, "
          f"Loss D: {epoch_loss_D:.4f}, "
          f"{'WARMUP' if epoch < warmup_epochs else 'FULL'}")

    if (epoch + 1) % 10 == 0:
        torch.save(G.state_dict(), f'/home/umang.shikarvar/Style_GAN/generator/G_epoch_{epoch+1}.pth')
        torch.save(D.state_dict(), f'/home/umang.shikarvar/Style_GAN/discriminator/D_epoch_{epoch+1}.pth')
        if epoch >= warmup_epochs:
            torch.save(detector.state_dict(), f'/home/umang.shikarvar/Style_GAN/detector/detector_epoch_{epoch+1}.pth')
        print(f"Saved checkpoints for epoch {epoch + 1}")