# Domain Adaptation for Brick Kiln Detection

This project investigates **unsupervised domain adaptation** techniques for detecting **brick kilns** in satellite imagery. It utilizes **image-to-image translation**, including **CycleGAN** and **CUT**, along with **object relabeling** and a **YOLO**-based detection pipeline. The primary objective is to improve **cross-region generalization** by enabling detection models to perform effectively in areas lacking **labeled data**.

---

## Motivation

- India has over **150,000 brick kilns**, contributing significantly to **air pollution** and **topsoil loss**.
- Kilns are often associated with **forced labor** and **unsafe working conditions**.
- Object detectors like **YOLO** perform well within a region but drop **40–50% in accuracy** across regions due to **domain shift**.
- This project addresses domain shift using **style transfer** and **label refinement**.

---

## Approach

1. **Translate source images** (e.g., from Delhi) to mimic the target region's style (e.g., West Bengal) using CycleGAN or CUT.
2. **Transfer brick kiln annotations** from source to translated images using original bounding boxes.
3. **Use a YOLO model trained on source images** to generate refined labels on the synthetic images.
4. **Train a new YOLO model** on these new images and labels **to evaluate on target region images**.

📄 [View Full Poster (PDF)](./Poster.pdf)

---

## Repository Structure

| File | Description |
|------|-------------|
| `Poster.pdf` | Research poster summarizing the approach and results |
| `CG.ipynb` | Code for training **CycleGAN** |
| `CUT.ipynb` | Code for training **CUT** |
| `YOLO.py` | Script for training the **YOLO** object detection model |
| `image_gen.ipynb` | Generates **target-style images** from source |
| `image_modify.ipynb` | Pastes **brick kilns** onto translated images |
| `label_formating.ipynb` | Extracts rectangular coordinates from OBB labels |
| `predicted_labels.ipynb` | Uses YOLO to generate **refined labels** |
| `map.ipynb` | Calculates **mAP (mean Average Precision)** for evaluation |

---

## Results

| Train Region | Test Region | mAP |
|--------------|-------------|-----|
| Delhi        | West Bengal | 0.506 |
| Delhi (CycleGAN) | West Bengal | **0.578** |
| Delhi (CUT)  | West Bengal | 0.535 |
| West Bengal  | Delhi       | 0.605 |
| WB (CycleGAN) | Delhi      | 0.638 |
| WB (CUT)     | Delhi       | **0.654** |


---

## Authors

- **Umang Shikarvar** — Undergraduate Researcher, BTech student, IIT Gandhinagar  
- **Prof. Nipun Batra** — Faculty Supervisor, Associate Professor, IIT Gandhinagar
- **Rishabh Mondal** — Project Mentor, Phd student, IIT Gandhinagar

---

## References

- [CycleGAN (ICCV 2017)](https://arxiv.org/abs/1703.10593)  
- [CUT (ECCV 2020)](https://arxiv.org/abs/2007.15651)