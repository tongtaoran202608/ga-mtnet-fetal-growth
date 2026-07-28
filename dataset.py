import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def zscore_normalize(image, mean, std, clamp_range=(-3, 3)):
    normalized = (image.astype(np.float32) - mean) / std
    normalized = np.clip(normalized, clamp_range[0], clamp_range[1])
    return normalized


def build_train_augmentations():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=34, sigma=4, p=0.3),
            A.GaussNoise(var_limit=(0.0, 0.05), p=0.3),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1),
    )


def build_eval_transform():
    return A.Compose(
        [ToTensorV2()],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1),
    )


CLASS_NAMES = ["normal", "iugr", "gdm", "preeclampsia", "afv_abnormal"]
LANDMARK_NAMES = ["stomach", "abdominal_aorta", "umbilical_vein", "liver"]


class FetalAbdominalDataset(Dataset):
    def __init__(
        self,
        records,
        image_root,
        norm_mean=0.487,
        norm_std=0.221,
        image_size=224,
        clahe_clip=2.0,
        clahe_tile=(8, 8),
        train=True,
        max_traj_len=6,
    ):
        self.records = records
        self.image_root = image_root
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.image_size = image_size
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.train = train
        self.max_traj_len = max_traj_len
        self.transform = build_train_augmentations() if train else build_eval_transform()

    def __len__(self):
        return len(self.records)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = apply_clahe(img, clip_limit=self.clahe_clip, tile_grid_size=self.clahe_tile)
        img = zscore_normalize(img, self.norm_mean, self.norm_std)
        return img

    def __getitem__(self, idx):
        record = self.records[idx]
        img = self._load_image(os.path.join(self.image_root, record["image_path"]))
        boxes = record.get("boxes", [])
        labels = record.get("landmark_labels", [])

        transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
        image_tensor = transformed["image"].float()
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.repeat(3, 1, 1) if image_tensor.shape[0] == 1 else image_tensor

        det_boxes = torch.zeros(len(LANDMARK_NAMES), 4)
        det_obj = torch.zeros(len(LANDMARK_NAMES))
        for box, label in zip(transformed["bboxes"], transformed["class_labels"]):
            det_boxes[label] = torch.tensor(box, dtype=torch.float32)
            det_obj[label] = 1.0

        ga_days = torch.tensor(record["ga_days"], dtype=torch.float32)
        cls_target = torch.tensor(record.get("class_label", -1), dtype=torch.long)

        traj_sequences = torch.zeros(self.max_traj_len, 768)
        traj_mask = torch.zeros(self.max_traj_len)
        traj_targets = torch.zeros(3)
        has_traj = "traj_targets" in record and record["traj_targets"] is not None
        if has_traj:
            traj_targets = torch.tensor(record["traj_targets"], dtype=torch.float32)

        sample = {
            "image": image_tensor,
            "ga_days": ga_days,
            "cls_target": cls_target,
            "det_boxes": det_boxes,
            "det_obj": det_obj,
            "traj_sequences": traj_sequences,
            "traj_mask": traj_mask,
            "traj_targets": traj_targets,
            "has_traj": torch.tensor(1.0 if has_traj else 0.0),
            "patient_id": record.get("patient_id", -1),
            "center": record.get("center", "unknown"),
        }
        return sample


def compute_fold_statistics(image_paths):
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    count = 0
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        count += img.size
    mean = pixel_sum / count
    std = np.sqrt(pixel_sq_sum / count - mean ** 2)
    return mean, std
