import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from active_learning.acquisition_functions import (
    multiclass_entropy_acquisition, least_confidence_acquisition
    # Add more as needed
)

acquisition_fn_list = [
    # ("multiclass_entropy", multiclass_entropy_acquisition),
    # ("least_confidence", least_confidence_acquisition),
    ("least_confidence", least_confidence_acquisition),
]

output_dir = "./output_activelearning"
os.makedirs(output_dir, exist_ok=True)

# --------- PATHS ---------
base_path = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/data"
uttar_pradesh_pool = f"{base_path}/uttar_pradesh_pool_data"
pool_images_dir = f"{uttar_pradesh_pool}/images"
pool_labels_dir = f"{uttar_pradesh_pool}/labels_voc"
delhi_images_dir = f"{base_path}/delhi_airshed/train60/images"
delhi_labels_dir = f"{base_path}/delhi_airshed/train60/labels_voc"
lucknow_images_dir = f"{base_path}/lucknow_airshed_100/train60/images"
lucknow_labels_dir = f"{base_path}/lucknow_airshed_100/train60/labels_voc"
test_lucknow_images_dir = f"{base_path}/lucknow_airshed_100/images"
test_lucknow_labels_dir = f"{base_path}/lucknow_airshed_100/labels_voc"
test_delhi_images_dir = f"{base_path}/delhi_airshed/images"
test_delhi_labels_dir = f"{base_path}/delhi_airshed/labels_voc"

delhi_init_ckpt = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/aaai/outputs_exp2_60_train_delhi_test_delhi40/best_faster_rcnn.pth"
lucknow_init_ckpt = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/aaai/outputs_exp2_60_train_lucknow_test_lucknow_40/best_faster_rcnn.pth"

num_classes = 4
acq_batch_size = 2
max_add_from_pool = 100
num_runs = 1
model_epochs_per_iter = 5
first_train_epochs = 50
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
dataloader_num_workers = 8
train_batch_size = 16
pool_batch_size = 200

# --------- DATASET DEFINITIONS ---------
class VOCTextListDataset(Dataset):
    # Expects a list of (img_fname, img_dir, label_dir)
    def __init__(self, image_label_triples, transforms=None):
        self.image_label_triples = image_label_triples
        self.transforms = transforms
    def __len__(self):
        return len(self.image_label_triples)
    def __getitem__(self, idx):
        img_filename, img_dir, label_dir = self.image_label_triples[idx]
        img_path = os.path.join(img_dir, img_filename)
        label_path = os.path.join(label_dir, img_filename.replace('.tif', '.txt'))
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

class PoolDataset(Dataset):
    def __init__(self, image_list, images_dir, transform=None):
        self.image_list = image_list
        self.images_dir = images_dir
        self.transform = transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        fname = self.image_list[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname

def get_transform():
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def collate_fn(batch): return tuple(zip(*batch))

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

def train_model(train_dataset, device, prev_model=None, epochs=20, num_classes=4, batch_size=32, init_ckpt_path=None):
    if prev_model is not None:
        model = prev_model
    else:
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
        if init_ckpt_path is not None:
            print(f"Loading initial weights from {init_ckpt_path}")
            state = torch.load(init_ckpt_path, map_location=device)
            model.load_state_dict(state)
    if not hasattr(model, 'transform') or not isinstance(model.transform, GeneralizedRCNNTransform):
        model.transform = GeneralizedRCNNTransform(
            min_size=640, max_size=640,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=dataloader_num_workers, pin_memory=True
    )
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0
        for imgs, targets in dataloader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
    return model

def predict(model, test_image_dir, transform, prediction_output_dir, device="cuda:1", batch_size=128, conf_thresh=None):
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

def run_active_learning(
    acquisition_fn,
    acquisition_fn_name,
    init_images_dir, init_labels_dir,
    pool_images_dir, pool_labels_dir,
    test_images_dir, test_labels_dir,
    num_classes=4,
    acq_batch_size=8,
    max_add_from_pool=100,
    num_runs=1,
    model_epochs_per_iter=10,
    device=None,
    results_csv_prefix="activelearning_results",
    first_train_epochs=20,
    delhi_init_ckpt=None,
    lucknow_init_ckpt=None
):
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    max_iterations = max_add_from_pool // acq_batch_size
    all_pool_imgs = sorted([f for f in os.listdir(pool_images_dir) if f.endswith(".tif")])
    init_train_imgs = sorted([f for f in os.listdir(init_images_dir) if f.endswith(".tif")])
    for run in range(1, num_runs+1):
        print(f"== {acquisition_fn_name}: Run {run} ==")
        # NEW: Train set is list of (img_fname, img_dir, label_dir)
        train_imgs = [(fname, init_images_dir, init_labels_dir) for fname in init_train_imgs]
        acquired_imgs = []
        prev_model = None
        records = []

        acq_logfile = os.path.join(output_dir, f"acquired_{acquisition_fn_name}_run{run}.txt")
        with open(acq_logfile, "w") as logf:
            logf.write("iteration,acquired_image\n")

        if "delhi" in init_images_dir:
            init_ckpt = delhi_init_ckpt
        elif "lucknow" in init_images_dir:
            init_ckpt = lucknow_init_ckpt
        else:
            init_ckpt = None

        for iteration in range(max_iterations+1):
            # Debug print: Current train set info
            print(f"\n[DEBUG] Train set at iteration {iteration} (showing up to 5):")
            for idx, (img_fname, img_dir, label_dir) in enumerate(train_imgs[:5]):
                print(f"  {img_fname} | img_dir: .../{os.path.basename(img_dir)} | label_dir: .../{os.path.basename(label_dir)}")
            print(f"  (Total training images: {len(train_imgs)})\n")

            train_dataset = VOCTextListDataset(train_imgs, transforms=get_transform())
            train_epochs = first_train_epochs if iteration == 0 else model_epochs_per_iter
            prev_model = train_model(
                train_dataset, device, prev_model=prev_model,
                epochs=train_epochs, num_classes=num_classes, batch_size=train_batch_size,
                init_ckpt_path=init_ckpt if iteration == 0 and prev_model is None else None
            )

            pred_dir = os.path.join(output_dir, f"preds_{acquisition_fn_name}_run{run}_iter{iteration}")
            predict(prev_model, test_images_dir, get_transform(), pred_dir, device=device, batch_size=pool_batch_size)
            map50 = compute_map(test_images_dir, test_labels_dir, pred_dir)
            print(f"MAP@50 for {acquisition_fn_name} | Run {run} | Iter {iteration}: {map50:.4f}")
            records.append({'acquisition': acquisition_fn_name, 'run': run, 'iteration': iteration, 'train_size': len(train_imgs), 'map50': map50})

            if iteration < max_iterations:
                seed = run * 1000 + iteration
                pool_left = sorted(list(set(all_pool_imgs) - set([x[0] for x in acquired_imgs])))
                pool_dataset = PoolDataset(pool_left, pool_images_dir, transform=get_transform())
                pool_loader = DataLoader(
                    pool_dataset, batch_size=pool_batch_size, shuffle=False,
                    num_workers=dataloader_num_workers, pin_memory=True
                )
                new_samples = acquisition_fn(
                    pool_loader=pool_loader,
                    acquired_list=[x[0] for x in acquired_imgs],
                    n_acquire=acq_batch_size,
                    seed=seed,
                    model=prev_model,
                    device=device,
                )
                # Debug print: What got acquired this iteration
                print("[DEBUG] Acquired in this iteration:")
                for fname in new_samples:
                    print(f"  {fname} | img_dir: .../{os.path.basename(pool_images_dir)} | label_dir: .../{os.path.basename(pool_labels_dir)}")
                if not new_samples:
                    print("Pool exhausted.")
                    break
                for fname in new_samples:
                    if fname not in [x[0] for x in acquired_imgs]:
                        acquired_imgs.append((fname, pool_images_dir, pool_labels_dir))
                        train_imgs.append((fname, pool_images_dir, pool_labels_dir))
                        with open(acq_logfile, "a") as logf:
                            logf.write(f"{iteration+1},{fname}\n")
        results_csv = os.path.join(output_dir, f"{results_csv_prefix}_{acquisition_fn_name}.csv")
        df = pd.DataFrame(records)
        if not os.path.exists(results_csv):
            df.to_csv(results_csv, index=False)
        else:
            df.to_csv(results_csv, mode='a', header=False, index=False)

if __name__ == "__main__":
    for acq_name, acq_fn in acquisition_fn_list:
        print(f"\n===== Running {acq_name} acquisition =====\n")
        run_active_learning(
            acquisition_fn=acq_fn,
            acquisition_fn_name=acq_name,
            init_images_dir=delhi_images_dir,
            init_labels_dir=delhi_labels_dir,
            pool_images_dir=pool_images_dir,
            pool_labels_dir=pool_labels_dir,
            test_images_dir=test_lucknow_images_dir,
            test_labels_dir=test_lucknow_labels_dir,
            num_classes=num_classes,
            acq_batch_size=acq_batch_size,
            max_add_from_pool=max_add_from_pool,
            num_runs=num_runs,
            model_epochs_per_iter=model_epochs_per_iter,
            device=device,
            results_csv_prefix=f"delhi_train_activelearning",
            first_train_epochs=first_train_epochs,
            delhi_init_ckpt=delhi_init_ckpt,
            lucknow_init_ckpt=lucknow_init_ckpt
        )
        run_active_learning(
            acquisition_fn=acq_fn,
            acquisition_fn_name=acq_name,
            init_images_dir=lucknow_images_dir,
            init_labels_dir=lucknow_labels_dir,
            pool_images_dir=pool_images_dir,
            pool_labels_dir=pool_labels_dir,
            test_images_dir=test_delhi_images_dir,
            test_labels_dir=test_delhi_labels_dir,
            num_classes=num_classes,
            acq_batch_size=acq_batch_size,
            max_add_from_pool=max_add_from_pool,
            num_runs=num_runs,
            model_epochs_per_iter=model_epochs_per_iter,
            device=device,
            results_csv_prefix=f"lucknow_train_activelearning",
            first_train_epochs=first_train_epochs,
            delhi_init_ckpt=delhi_init_ckpt,
            lucknow_init_ckpt=lucknow_init_ckpt
        )
    print("All acquisition strategies complete. Check CSVs for results.")
