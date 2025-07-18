# brickkiln/eval.py
import os, torch, pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

def predict(model, test_image_dir, transform, prediction_output_dir, device="cuda", batch_size=128, conf_thresh=None):
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

def plot_map_csv(csv_path, title=None):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,5))
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
    import matplotlib.pyplot as plt
    import os
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

