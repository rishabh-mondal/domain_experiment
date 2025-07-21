import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
from PIL import Image
import os
from tqdm import tqdm
import torchvision
##############################
# 1. Random Sampling
##############################
def random_acquisition(pool_loader, acquired_list, n_acquire, seed, **kwargs):
    """Randomly sample n_acquire images from pool."""
    pool_left = [fname for _, fname in pool_loader.dataset if fname not in acquired_list]
    if not pool_left: return []
    random.seed(seed)
    return random.sample(pool_left, min(n_acquire, len(pool_left)))

##############################
# 2. Least Confidence
##############################
def least_confidence_acquisition(
    pool_loader, acquired_list, n_acquire, seed,
    model=None, device=None, **kwargs
):
    """Select images where the highest box score is lowest."""
    image_leastconf = []
    model.eval()
    with torch.no_grad():
        for imgs, fnames in tqdm(pool_loader, desc="Least-confidence"):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for fname, out in zip(fnames, outputs):
                scores = out["scores"].detach().cpu().numpy()
                least_conf = 1 - scores.max() if len(scores) else 1.0
                image_leastconf.append((fname, least_conf))
    sorted_by_leastconf = sorted(image_leastconf, key=lambda x: x[1], reverse=True)
    return [fname for fname, _ in sorted_by_leastconf[:n_acquire]]

##############################
# 3. Margin Sampling
##############################
def get_per_box_class_probs(model, images, device):
    model.eval()
    images = [img.to(device) for img in images]
    with torch.no_grad():
        transformed = model.transform(images)
        features = model.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = [features]
        proposals, _ = model.rpn(transformed, features)
        box_features = model.roi_heads.box_roi_pool(features, proposals, transformed.image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, _ = model.roi_heads.box_predictor(box_features)
        class_probs = F.softmax(class_logits, dim=-1)
        probs_per_img = []
        idx = 0
        for p in proposals:
            count = p.shape[0]
            probs_per_img.append(class_probs[idx:idx+count])
            idx += count
    return probs_per_img, proposals

def margin_sampling_acquisition(
    pool_loader, acquired_list, n_acquire, seed,
    model=None, device=None, **kwargs
):
    """Pick images with smallest margin between top two class probs per box (most confusion)."""
    image_margins = []
    model.eval()
    with torch.no_grad():
        for imgs, fnames in tqdm(pool_loader, desc="Margin sampling"):
            imgs = [img.to(device) for img in imgs]
            batch_probs_per_img, _ = get_per_box_class_probs(model, imgs, device)
            for fname, probs in zip(fnames, batch_probs_per_img):
                if probs.shape[0] == 0:
                    mean_margin = 1.0
                else:
                    sorted_probs = torch.sort(probs, dim=1, descending=True).values
                    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
                    mean_margin = margins.mean().item()
                image_margins.append((fname, mean_margin))
    sorted_by_margin = sorted(image_margins, key=lambda x: x[1])
    return [fname for fname, _ in sorted_by_margin[:n_acquire]]
def compute_entropy(class_logits):
    """Compute entropy for each box's softmaxed class probability vector."""
    probs = F.softmax(class_logits, dim=1)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
    return entropy

def multiclass_entropy_acquisition(
    pool_loader, acquired_list, n_acquire, seed,
    model=None, device=None, transform=None, **kwargs
):
    """
    For each image in the pool:
        - Run full model (backbone, RPN, ROI, head) to get class logits for all classes for all proposals
        - Compute mean entropy of those logits per image
    """
    if transform is None:
        transform = lambda x: torchvision.transforms.ToTensor()(x)

    model.eval()
    image_uncertainties = []
    with torch.no_grad():
        for imgs, fnames in tqdm(pool_loader, desc="Multiclass entropy (per-class logits)"):
            for img, filename in zip(imgs, fnames):
                # If img is not tensor yet, convert
                if not torch.is_tensor(img):
                    img = transform(img)
                if img.ndim == 3:
                    img = img.unsqueeze(0)  # [1, 3, H, W]
                img = img.to(device)
                image_size = [tuple(img.shape[-2:])]

                # Backbone
                features = model.backbone(img)
                images = ImageList(img, image_size)
                # RPN
                proposals, _ = model.rpn(images, features)
                if len(proposals[0]) == 0:
                    image_uncertainties.append((filename, 0.0))
                    continue
                # ROI heads
                box_features = model.roi_heads.box_roi_pool(features, proposals, image_size)
                box_features = model.roi_heads.box_head(box_features)
                class_logits = model.roi_heads.box_predictor.cls_score(box_features)
                entropy = compute_entropy(class_logits)
                avg_entropy = entropy.mean().item() if entropy.numel() > 0 else 0.0
                image_uncertainties.append((filename, avg_entropy))

    # Rank and select the highest-entropy images
    sorted_by_entropy = sorted(image_uncertainties, key=lambda x: x[1], reverse=True)
    return [fname for fname, _ in sorted_by_entropy[:n_acquire]]
# ##############################
# # 4. Binary Entropy
# ##############################
# def binary_entropy_acquisition(
#     pool_loader, acquired_list, n_acquire, seed,
#     model=None, device=None, **kwargs
# ):
#     """Entropy using top box confidence only."""
#     image_entropies = []
#     model.eval()
#     with torch.no_grad():
#         for imgs, fnames in tqdm(pool_loader, desc="Binary entropy"):
#             imgs = [img.to(device) for img in imgs]
#             outputs = model(imgs)
#             for fname, out in zip(fnames, outputs):
#                 scores = out["scores"].detach().cpu().numpy()
#                 entropies = -scores * np.log(scores + 1e-12) - (1 - scores) * np.log(1 - scores + 1e-12)
#                 mean_entropy = entropies.mean() if len(entropies) else 0.0
#                 image_entropies.append((fname, mean_entropy))
#     sorted_by_entropy = sorted(image_entropies, key=lambda x: x[1], reverse=True)
#     return [fname for fname, _ in sorted_by_entropy[:n_acquire]]

# ##############################
# # 5. Multiclass Entropy
# ##############################
# def compute_multiclass_entropy(probs):
#     ent = -(probs * (probs + 1e-8).log()).sum(dim=1)
#     return ent

# def multiclass_entropy_acquisition(
#     pool_loader, acquired_list, n_acquire, seed,
#     model=None, device=None, **kwargs
# ):
#     """Entropy using the full class softmax per box."""
#     image_entropies = []
#     model.eval()
#     with torch.no_grad():
#         for imgs, fnames in tqdm(pool_loader, desc="Multiclass entropy"):
#             imgs = [img.to(device) for img in imgs]
#             batch_probs_per_img, _ = get_per_box_class_probs(model, imgs, device)
#             for fname, probs in zip(fnames, batch_probs_per_img):
#                 if probs.shape[0] == 0:
#                     mean_entropy = 0.0
#                 else:
#                     ent = compute_multiclass_entropy(probs)
#                     mean_entropy = ent.mean().item()
#                 image_entropies.append((fname, mean_entropy))
#     sorted_by_entropy = sorted(image_entropies, key=lambda x: x[1], reverse=True)
#     return [fname for fname, _ in sorted_by_entropy[:n_acquire]]

##############################
# 6. BALD (stub, MC-Dropout not in vanilla FasterRCNN)
##############################
def bald_acquisition(*args, **kwargs):
    """MC Dropout-based uncertainty (stub)"""
    print("BALD acquisition is not implemented (requires MC dropout).")
    return random_acquisition(*args, **kwargs)

##############################
# 7. Core-set (Farthest-First, requires feature extraction)
##############################
def core_set_acquisition(
    pool_loader, acquired_list, n_acquire, seed,
    feature_extractor=None, model=None, device=None, **kwargs
):
    """
    Selects n_acquire samples from pool that are farthest from the current labeled set in feature space.
    feature_extractor: callable that returns a vector for a given image (pretrained model, etc).
    """
    pool_left = [fname for _, fname in pool_loader.dataset if fname not in acquired_list]
    if not pool_left: return []
    # TODO: implement true feature extraction
    print("Core-set (farthest-first) is a stub. Needs feature extractor and distances.")
    random.seed(seed)
    return random.sample(pool_left, min(n_acquire, len(pool_left)))
