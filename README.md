# GA-MTNet

GA-MTNet (Gestational-Age-Conditioned Multi-Task Network) is a deep learning framework for abdominal landmark detection and sonographic pattern classification of fetal growth conditions on obstetric ultrasound.

## Overview

The model conditions a Swin Transformer backbone on gestational age using a FiLM-based Gestational Age Embedding Module (GAEM), and jointly optimizes three tasks:

- Detection of four abdominal landmarks: stomach, abdominal aorta, umbilical vein, liver
- Five-class sonographic pattern classification: normal fetal growth, IUGR, GDM, pre-eclampsia, AFV abnormality
- Biometric developmental trajectory regression: abdominal circumference, estimated fetal weight, deepest vertical pocket

A post-deployment continual learning loop combines Monte Carlo Dropout uncertainty estimation, Elastic Weight Consolidation, and Low-Rank Adaptation for safe incremental model updates.

## Repository Structure

```
ga_mtnet/
├── model.py               # GAEM, Swin backbone, detection/classification/trajectory heads, GA-MTNet
├── losses.py               # Focal loss, DIoU loss, Huber loss, combined multi-task loss
├── dataset.py               # Preprocessing (CLAHE, z-score normalization), augmentation, dataset class
├── continual_learning.py    # MC Dropout uncertainty flagging, EWC, LoRA adapters
├── train.py                 # 5-fold patient-stratified cross-validation training loop
├── evaluate.py               # AUC, F1, mAP, ECE, ICC, t-SNE, SHAP, Grad-CAM++
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

Training expects a JSON annotation file where each record has the following structure:

```json
{
  "image_path": "center_a/patient_0001/frame_003.png",
  "patient_id": "0001",
  "center": "A",
  "ga_days": 182,
  "class_label": 1,
  "boxes": [[0.42, 0.55, 0.10, 0.08]],
  "landmark_labels": [0],
  "traj_targets": [185.2, 720.5, 45.1]
}
```

- `boxes` are in YOLO format `(center_x, center_y, width, height)`, normalized to `[0, 1]`.
- `landmark_labels` index into `["stomach", "abdominal_aorta", "umbilical_vein", "liver"]`.
- `class_label` indexes into `["normal", "iugr", "gdm", "preeclampsia", "afv_abnormal"]`.
- `traj_targets` is optional and only present for patients with a longitudinal follow-up scan.

## Training

```bash
python train.py \
  --annotations /path/to/annotations.json \
  --image_root /path/to/images \
  --output_dir ./checkpoints \
  --batch_size 16 \
  --lr 1e-4 \
  --max_epochs 100
```

Ablation variants can be run by toggling `--use_ga` and `--use_multitask`.

## Continual Learning

`continual_learning.py` provides `ContinualLearningLoop`, which:

1. Calibrates an uncertainty threshold from Monte Carlo Dropout entropy on a calibration set.
2. Flags high-uncertainty inference-time cases for expert review.
3. Injects LoRA adapters into the backbone attention projections and unfreezes the task heads.
4. Fine-tunes on flagged, annotated cases while an EWC penalty discourages forgetting of previously learned representations.

## Evaluation

`evaluate.py` includes utilities for:

- Per-class and macro AUC-ROC, F1, precision, recall, confusion matrices
- Expected Calibration Error (ECE) with configurable bin count
- mAP@0.5 and mAP@[0.5:0.95] for landmark detection
- Intraclass correlation coefficient (ICC) for trajectory regression targets
- t-SNE projection and silhouette score for embedding space analysis
- SHAP value computation and Grad-CAM++ saliency maps for interpretability

## Hardware Notes

Training was designed for multi-GPU setups with mixed-precision (AMP) support. Inference is lightweight enough to run on a single consumer-grade GPU.
