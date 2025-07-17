import os
import random
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision



# ---- 1. Utility: Symlink random subsets ----
def make_random_symlinks(
    src_img_dir, src_lbl_dir,
    dst_img_dir, dst_lbl_dir,
    n=100, seed=1008, filetype=".tif", log_file=None
):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    img_files = [f for f in os.listdir(src_img_dir) if f.endswith(filetype)]
    random.seed(seed)
    chosen = random.sample(img_files, n)
    if log_file:
        with open(log_file, "w") as f:
            for fname in chosen:
                f.write(fname + "\n")
    for fname in chosen:
        src_img = os.path.join(src_img_dir, fname)
        dst_img = os.path.join(dst_img_dir, fname)
        src_lbl = os.path.join(src_lbl_dir, fname.replace(filetype, ".txt"))
        dst_lbl = os.path.join(dst_lbl_dir, fname.replace(filetype, ".txt"))
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl) and os.path.exists(src_lbl):
            os.symlink(src_lbl, dst_lbl)
    print(f"✅ Symlinked {n} images+labels from {src_img_dir} to {dst_img_dir}")

# --------- 1. Split & Symlink Utility ---------
def make_split_symlinks(img_dir, lbl_dir, out_train_img, out_train_lbl, out_test_img, out_test_lbl, ratio=0.6, seed=2024):
    os.makedirs(out_train_img, exist_ok=True)
    os.makedirs(out_train_lbl, exist_ok=True)
    os.makedirs(out_test_img, exist_ok=True)
    os.makedirs(out_test_lbl, exist_ok=True)
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
    random.seed(seed)
    random.shuffle(all_imgs)
    n_train = int(len(all_imgs) * ratio)
    train_imgs, test_imgs = all_imgs[:n_train], all_imgs[n_train:]
    def link(img_list, dest_img_dir, dest_lbl_dir):
        for fname in img_list:
            img_src = os.path.join(img_dir, fname)
            img_dst = os.path.join(dest_img_dir, fname)
            lbl_src = os.path.join(lbl_dir, fname.replace(".tif", ".txt"))
            lbl_dst = os.path.join(dest_lbl_dir, fname.replace(".tif", ".txt"))
            if not os.path.exists(img_dst):
                os.symlink(img_src, img_dst)
            if os.path.exists(lbl_src) and not os.path.exists(lbl_dst):
                os.symlink(lbl_src, lbl_dst)
    link(train_imgs, out_train_img, out_train_lbl)
    link(test_imgs, out_test_img, out_test_lbl)
    print(f"✅ Split: {len(train_imgs)} train, {len(test_imgs)} test")

# --------- 2. Dataset Class ---------
class VOCTextDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(self.labels_dir, img_filename.replace('.tif', '.txt'))
        img = Image.open(img_path).convert("RGB")
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        boxes.append(bbox)
                        labels.append(cls_id)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
        if self.transforms: img = self.transforms(img)
        return img, target
def collate_fn(batch): return tuple(zip(*batch))
def get_transform(): return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
def build_concat_dataset(image_dirs, label_dirs, transforms=None):
    if isinstance(image_dirs, str) and isinstance(label_dirs, str):
        return VOCTextDataset(image_dirs, label_dirs, transforms)
    elif isinstance(image_dirs, list) and isinstance(label_dirs, list):
        datasets = [VOCTextDataset(i, l, transforms) for i, l in zip(image_dirs, label_dirs)]
        return ConcatDataset(datasets)
    else:
        raise ValueError("Train images/labels must both be str or both be list of same length")

# --------- 3. Prediction/Evaluation Utilities ---------
def predict(model, test_image_dir, transform, prediction_output_dir, device="cuda", batch_size=32, conf_thresh=None):
    os.makedirs(prediction_output_dir, exist_ok=True)
    model.eval()
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.lower().endswith(".tif")])
    for i in tqdm(range(0, len(image_files), batch_size), desc="Inference"):
        batch_files = image_files[i:i + batch_size]
        images = []
        original_filenames = []
        for img_file in batch_files:
            img_path = os.path.join(test_image_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image)
            images.append(image_tensor)
            original_filenames.append(img_file)
        images = [img.to(device) for img in images]
        with torch.no_grad():
            batch_outputs = model(images)
        for img_file, outputs in zip(original_filenames, batch_outputs):
            base_name = os.path.splitext(img_file)[0]
            pred_txt_path = os.path.join(prediction_output_dir, base_name + ".txt")
            with open(pred_txt_path, "w") as f:
                for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
                    if conf_thresh is not None and score.item() < conf_thresh:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    f.write(f"{label.item()} {score.item():.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
def load_txt_as_ground_truth(txt_path):
    boxes, labels = [], []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                cls, x1, y1, x2, y2 = map(float, parts)
                labels.append(int(cls))  
                boxes.append([x1, y1, x2, y2])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,))}
def load_txt_as_prediction(txt_path):
    boxes, labels, scores = [], [], []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6: continue
                cls, conf, x1, y1, x2, y2 = map(float, parts)
                labels.append(int(cls)) 
                scores.append(conf)
                boxes.append([x1, y1, x2, y2])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,)),
            "scores": torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros((0,))}
def make_class_agnostic(detection_list):
    for d in detection_list: d["labels"] = torch.zeros_like(d["labels"])
    return detection_list
def compute_map(test_img_dir, test_label_dir, pred_dir):
    image_filenames = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(".tif")])
    gt_targets, predictions = [], []
    for fname in image_filenames:
        base = os.path.splitext(fname)[0]
        gt = load_txt_as_ground_truth(os.path.join(test_label_dir, base + ".txt"))
        pred = load_txt_as_prediction(os.path.join(pred_dir, base + ".txt"))
        gt_targets.append(gt)
        predictions.append(pred)
    gt_targets = make_class_agnostic(gt_targets)
    predictions = make_class_agnostic(predictions)
    metric = MeanAveragePrecision()
    metric.update(predictions, gt_targets)
    results = metric.compute()
    return results["map_50"].item()

# --------- 4. Plotting ---------
def plot_map_csv(csv_path, title=None):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,5))
    if "map50_in" in df.columns and "map50_out" in df.columns:
        plt.plot(df["epoch"], df["map50_in"], marker="o", label="Delhi Test (in)")
        plt.plot(df["epoch"], df["map50_out"], marker="o", label="Lucknow Test (out)")
        plt.legend()
    else:
        plt.plot(df["epoch"], df["map50"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.50")
    plt.title(title or "mAP@0.50 over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.splitext(csv_path)[0] + "_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def plot_loss_curve(losses_per_epoch, output_root, exp_name):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(losses_per_epoch)+1), losses_per_epoch, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title(f"{exp_name} Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "loss_curve.png"))
    plt.close()
    print(f"Loss curve saved to {os.path.join(output_root, 'loss_curve.png')}")

# --------- 5. Config: Split and Paths ---------
base_path = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/data"
delhi = f"{base_path}/delhi_airshed"

# --- Create 60:40 symlinks (run once, will not overwrite) ---
make_split_symlinks(
    img_dir=f"{delhi}/images",
    lbl_dir=f"{delhi}/labels_voc",
    out_train_img=f"{delhi}/train60/images",
    out_train_lbl=f"{delhi}/train60/labels_voc",
    out_test_img=f"{delhi}/test40/images",
    out_test_lbl=f"{delhi}/test40/labels_voc",
    ratio=0.6, seed=2024
)


# === MAKE SYMLINKS for random splits ===
make_random_symlinks(
    src_img_dir=f"{base_path}/test_delhi_airshed/images",
    src_lbl_dir=f"{base_path}/test_delhi_airshed/labels_voc",
    dst_img_dir=f"{base_path}/delhi_airshed/images_random200",
    dst_lbl_dir=f"{base_path}/delhi_airshed/labels_voc_random200",
    n=200, seed=2024, log_file="random200_testdelhi.txt"
)
make_random_symlinks(
    src_img_dir=f"{base_path}/uttar_pradesh_pool_data/images",
    src_lbl_dir=f"{base_path}/uttar_pradesh_pool_data/labels_voc",
    dst_img_dir=f"{base_path}/uttar_pradesh_pool_data/images_random200",
    dst_lbl_dir=f"{base_path}/uttar_pradesh_pool_data/labels_voc_random200",
    n=200, seed=2024, log_file="random200_uppool.txt"
)
make_random_symlinks(
    src_img_dir=f"{base_path}/uttar_pradesh_pool_data/images",
    src_lbl_dir=f"{base_path}/uttar_pradesh_pool_data/labels_voc",
    dst_img_dir=f"{base_path}/uttar_pradesh_pool_data/images_random100",
    dst_lbl_dir=f"{base_path}/uttar_pradesh_pool_data/labels_voc_random100",
    n=100, seed=2024, log_file="random100_uppool.txt"
)
make_random_symlinks(
    src_img_dir=f"{base_path}/test_delhi_airshed/images",
    src_lbl_dir=f"{base_path}/test_delhi_airshed/labels_voc",
    dst_img_dir=f"{base_path}/delhi_airshed/images_random100",
    dst_lbl_dir=f"{base_path}/delhi_airshed/labels_voc_random100",
    n=100, seed=2024, log_file="random100_testdelhi.txt"
)






# --------- 6. Experiment Configs ---------


exp1_60_train_delhi_test_lucknow = {
    "experiment_name": "exp1_60_train_delhi_test_lucknow",
    "train_images": f"{delhi}/train60/images",
    "train_labels": f"{delhi}/train60/labels_voc",
    "test_images": f"{base_path}/lucknow_airshed_100/images",
    "test_labels": f"{base_path}/lucknow_airshed_100/labels_voc"
}

exp2_60_train_delhi_test_delhi40 = {
    "experiment_name": "exp2_60_train_delhi_test_delhi40",
    "train_images": f"{delhi}/train60/images",
    "train_labels": f"{delhi}/train60/labels_voc",
    "test_images": f"{delhi}/test40/images",
    "test_labels": f"{delhi}/test40/labels_voc"
}

exp3_train_60_plus_200_delhi_ncr_train_lucknow_test = {
    "experiment_name": "exp3_train_60_plus_200_delhi_ncr_train_lucknow_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{delhi}/images_random200"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{delhi}/labels_voc_random200"
    ],
    "test_images": f"{base_path}/lucknow_airshed_100/images",
    "test_labels": f"{base_path}/lucknow_airshed_100/labels_voc"
}

exp4_train_60_plus_200_uppool_train_lucknow_test = {
    "experiment_name": "exp4_train_60_plus_200_uppool_train_lucknow_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random200"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random200"
    ],
    "test_images": f"{base_path}/lucknow_airshed_100/images",
    "test_labels": f"{base_path}/lucknow_airshed_100/labels_voc"
}

exp5_delhi_plus_100_uppool_plus_100_delhi_ncr_train_lucknow_test = {
    "experiment_name": "exp5_delhi_plus_100_uppool_plus_100_delhi_ncr_train_lucknow_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random100",
        f"{base_path}/delhi_airshed/images_random100"
    ],
    "train_labels": [
        f"{base_path}/delhi_airshed/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random100",
        f"{base_path}/delhi_airshed/labels_voc_random100"
    ],
    "test_images": f"{base_path}/lucknow_airshed_100/images",
    "test_labels": f"{base_path}/lucknow_airshed_100/labels_voc"
}

exp6_delhi_60_plus_100_up_pool_plus_100_delhi_ncr_train_delhi_40_test = {
    "experiment_name": "exp6_delhi_60_plus_100_up_pool_plus_100_delhi_ncr_train_delhi_40_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random100",
        f"{base_path}/delhi_airshed/images_random100"
    ],
    "train_labels": [
        f"{base_path}/delhi_airshed/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random100",
        f"{base_path}/delhi_airshed/labels_voc_random100"
    ],
    "test_images": f"{delhi}/test40/images",
    "test_labels": f"{delhi}/test40/labels_voc"
}

exp7_train_delhi_60_plus_200_up_pool_train_delhi_40_test = {
    "experiment_name": "exp7_train_delhi_60_plus_200_up_pool_train_delhi_40_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random200"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random200"
    ],
    "test_images": f"{delhi}/test40/images",
    "test_labels": f"{delhi}/test40/labels_voc"
}

exp8_train_delhi_60_plus_200_delhi_ncr_train_delhi_40_test = {
    "experiment_name": "exp8_train_delhi_60_plus_200_delhi_ncr_train_delhi_40_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{delhi}/images_random200"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{delhi}/labels_voc_random200"
    ],
    "test_images": f"{delhi}/test40/images",
    "test_labels": f"{delhi}/test40/labels_voc"
}

experiments = [
    exp1_60_train_delhi_test_lucknow,
    exp2_60_train_delhi_test_delhi40,
    exp3_train_60_plus_200_delhi_ncr_train_lucknow_test,
    exp4_train_60_plus_200_uppool_train_lucknow_test,
    exp5_delhi_plus_100_uppool_plus_100_delhi_ncr_train_lucknow_test,
    exp6_delhi_60_plus_100_up_pool_plus_100_delhi_ncr_train_delhi_40_test,
    exp7_train_delhi_60_plus_200_up_pool_train_delhi_40_test,
    exp8_train_delhi_60_plus_200_delhi_ncr_train_delhi_40_test
]


# --------- 7. Main Training/Eval Loop ---------
def run_experiment(config):
    print(f"\n--- Experiment: {config['experiment_name']} ---")
    start_time = datetime.now()
    print(f"Started at: {start_time}")

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    num_classes = 4
    output_root = f"./outputs_{config['experiment_name']}"
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    train_dataset = build_concat_dataset(config["train_images"], config["train_labels"], transforms=get_transform())
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_images = config["test_images"]
    test_labels = config["test_labels"]

    transform = GeneralizedRCNNTransform(
        min_size=640, max_size=640,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.transform = transform
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    csv_path = os.path.join(output_root, "map_logs_baseline.csv")
    log_df = pd.DataFrame(columns=["epoch", "map50"])
    losses_per_epoch = []
    best_map50 = 0.0
    best_epoch = -1
    patience = 3
    epochs_no_improve = 0

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}")
        losses_per_epoch.append(total_loss)

        if epoch % 10 == 0:
            pred_dir = os.path.join(output_root, f"predictions_epoch_{epoch}")
            predict(model, test_images, get_transform(), pred_dir, device=device)
            map50 = compute_map(test_images, test_labels, pred_dir)
            print(f"[Epoch {epoch}] mAP@0.50: {map50:.4f}")
            log_df.loc[len(log_df)] = [epoch, map50]
            log_df.to_csv(csv_path, index=False)
            if map50 > best_map50:
                best_map50 = map50
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(output_root, "best_faster_rcnn.pth"))
                print(f"New best model saved at epoch {epoch} with mAP@0.50 = {map50:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} - no improvement for {patience} evals")
                    break

    plot_map_csv(csv_path, title=f"{config['experiment_name']} mAP@0.50")
    plot_loss_curve(losses_per_epoch, output_root, config['experiment_name'])

    end_time = datetime.now()
    print(f"Finished at: {end_time}")
    print(f"Best epoch: {best_epoch} | Best mAP@0.50: {best_map50:.4f}")
    print(f"Best model: {os.path.join(output_root, 'best_faster_rcnn.pth')}")

# --------- 8. Aggregate Results ---------
def aggregate_results(experiments):
    all_results = []
    for exp in experiments:
        exp_name = exp if isinstance(exp, str) else exp['experiment_name']
        base = f"./outputs_{exp_name}"
        csv_path = os.path.join(base, "map_logs_baseline.csv")
        if os.path.exists(csv_path):
            logs = pd.read_csv(csv_path)
            best_map = logs['map50'].max()
            all_results.append({
                "experiment": exp_name,
                "best_map50": best_map
            })
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("all_experiments_summary.csv", index=False)
    print(results_df)
    plt.figure(figsize=(8,4))
    plt.bar(results_df["experiment"], results_df["best_map50"])
    plt.ylabel("Best mAP@0.50")
    plt.title("Experiment Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("all_experiments_summary.png")
    plt.close()
    print("Aggregated results plot saved to all_experiments_summary.png")

# --------- 9. Main: Run experiments ---------
if __name__ == "__main__":
    for exp in experiments:
        run_experiment(exp)

    aggregate_results([
        exp1_60_train_delhi_test_lucknow,
        exp2_60_train_delhi_test_delhi40,
        exp3_train_60_plus_200_delhi_ncr_train_lucknow_test,
        exp4_train_60_plus_200_uppool_train_lucknow_test,
        exp5_delhi_plus_100_uppool_plus_100_delhi_ncr_train_lucknow_test,
        exp6_delhi_60_plus_100_up_pool_plus_100_delhi_ncr_train_delhi_40_test,
        exp7_train_delhi_60_plus_200_up_pool_train_delhi_40_test,
        exp8_train_delhi_60_plus_200_delhi_ncr_train_delhi_40_test
    ])
