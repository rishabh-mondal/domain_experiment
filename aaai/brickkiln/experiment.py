# brickkiln/experiment.py

import os
import json
import torch
import pandas as pd
from .dataset import build_concat_dataset, get_transform
from .train import train_model
from .eval import predict, compute_map, plot_map_csv, plot_loss_curve

def run_experiment(config, device="cuda:0", epochs=50, batch_size=32, patience=3, eval_every=25):
    print(f"\n--- Experiment: {config['experiment_name']} ---")
    output_root = f"./outputs_{config['experiment_name']}"
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    train_dataset = build_concat_dataset(config["train_images"], config["train_labels"], transforms=get_transform())
    test_images = config["test_images"]
    test_labels = config["test_labels"]
    csv_path = os.path.join(output_root, "map_logs_baseline.csv")
    log_df = pd.DataFrame(columns=["epoch", "map50"])
    best_map50 = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    losses_per_epoch = []

    # Build model
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    model = fasterrcnn_resnet50_fpn(num_classes=4)
    model.transform = GeneralizedRCNNTransform(
        min_size=640, max_size=640,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    from torch.utils.data import DataLoader
    from .dataset import collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Training + evaluation loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        losses_per_epoch.append(total_loss)

        # EVAL EVERY eval_every
        if epoch % eval_every == 0 or epoch == epochs:
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
                model_path = os.path.join(output_root, "best_faster_rcnn.pth")
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved at epoch {epoch} with mAP@0.50 = {map50:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} - no improvement for {patience} evals")
                    break

    plot_map_csv(csv_path, title=f"{config['experiment_name']} mAP@0.50")
    plot_loss_curve(losses_per_epoch, output_root, config['experiment_name'])
    print(f"Finished {config['experiment_name']}. Best epoch: {best_epoch} | Best mAP@0.50: {best_map50:.4f}")
    print(f"Best model: {os.path.join(output_root, 'best_faster_rcnn.pth')}")
