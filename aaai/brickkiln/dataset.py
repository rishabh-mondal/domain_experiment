# brickkiln/dataset.py
import os
import torch
import torchvision
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image

class VOCTextDataset(Dataset):
    """
    PyTorch Dataset for VOC-style text labels and .tif images.
    """
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(self.labels_dir, img_filename.replace('.tif', '.txt'))
        img = Image.open(img_path).convert("RGB")
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        boxes.append(bbox)
                        labels.append(cls_id)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
        if self.transforms:
            img = self.transforms(img)
        return img, target

def get_transform():
    """
    Return standard transform: ToTensor()
    """
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def build_concat_dataset(image_dirs, label_dirs, transforms=None):
    """
    Build a concatenated dataset from (possibly multiple) image/label folders.
    """
    if isinstance(image_dirs, str) and isinstance(label_dirs, str):
        return VOCTextDataset(image_dirs, label_dirs, transforms)
    elif isinstance(image_dirs, list) and isinstance(label_dirs, list):
        assert len(image_dirs) == len(label_dirs), "image_dirs and label_dirs list must be same length"
        datasets = [VOCTextDataset(i, l, transforms) for i, l in zip(image_dirs, label_dirs)]
        return ConcatDataset(datasets)
    else:
        raise ValueError("Train images/labels must both be str or both be list of same length")

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Returns:
        images: list of image tensors
        targets: list of dicts (each dict: 'boxes', 'labels', 'image_id')
    """
    return tuple(zip(*batch))
