import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

from model import GA_MTNet
from losses import MultiTaskLoss
from dataset import FetalAbdominalDataset, CLASS_NAMES
from continual_learning import ContinualLearningLoop


def build_class_weighted_sampler(labels, num_classes):
    class_counts = np.bincount(labels, minlength=num_classes)
    weights_per_class = 1.0 / np.sqrt(class_counts + 1e-6)
    sample_weights = weights_per_class[labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)
    return sampler


def smote_augment_features(features, labels, k_neighbors=5, synthetic_per_class=300):
    smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy="not majority")
    features_res, labels_res = smote.fit_resample(features, labels)
    return features_res, labels_res


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=True):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        ga_days = batch["ga_days"].to(device)
        cls_targets = batch["cls_target"].to(device)
        det_boxes = batch["det_boxes"].to(device)
        det_obj = batch["det_obj"].to(device)
        traj_sequences = batch["traj_sequences"].to(device)
        traj_mask = batch["traj_mask"].to(device)
        traj_targets = batch["traj_targets"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images, ga_days, traj_sequences=traj_sequences, traj_mask=traj_mask)
            targets = {
                "cls_targets": cls_targets,
                "det_boxes": det_boxes,
                "det_obj": det_obj,
                "det_valid_mask": det_obj.bool(),
                "traj_targets": traj_targets,
            }
            loss, components = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    for batch in loader:
        images = batch["image"].to(device)
        ga_days = batch["ga_days"].to(device)
        cls_targets = batch["cls_target"].to(device)
        det_boxes = batch["det_boxes"].to(device)
        det_obj = batch["det_obj"].to(device)
        traj_sequences = batch["traj_sequences"].to(device)
        traj_mask = batch["traj_mask"].to(device)
        traj_targets = batch["traj_targets"].to(device)

        outputs = model(images, ga_days, traj_sequences=traj_sequences, traj_mask=traj_mask)
        targets = {
            "cls_targets": cls_targets,
            "det_boxes": det_boxes,
            "det_obj": det_obj,
            "det_valid_mask": det_obj.bool(),
            "traj_targets": traj_targets,
        }
        loss, _ = criterion(outputs, targets)
        total_loss += loss.item()

        preds = torch.softmax(outputs["cls_logits"], dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(cls_targets.cpu().numpy())

    return total_loss / max(len(loader), 1), np.concatenate(all_preds), np.concatenate(all_targets)


def run_cross_validation(records, image_root, args, device):
    labels = np.array([r["class_label"] for r in records])
    patient_ids = np.array([r["patient_id"] for r in records])
    unique_patients = np.unique(patient_ids)
    patient_labels = np.array([labels[patient_ids == pid][0] for pid in unique_patients])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_pat_idx, val_pat_idx) in enumerate(skf.split(unique_patients, patient_labels)):
        train_patients = set(unique_patients[train_pat_idx])
        val_patients = set(unique_patients[val_pat_idx])

        train_records = [r for r in records if r["patient_id"] in train_patients]
        val_records = [r for r in records if r["patient_id"] in val_patients]

        train_dataset = FetalAbdominalDataset(train_records, image_root, train=True)
        val_dataset = FetalAbdominalDataset(val_records, image_root, train=False)

        train_labels = np.array([r["class_label"] for r in train_records])
        sampler = build_class_weighted_sampler(train_labels, num_classes=len(CLASS_NAMES))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = GA_MTNet(
            use_ga_conditioning=args.use_ga,
            use_multitask=args.use_multitask,
        ).to(device)

        criterion = MultiTaskLoss(
            lambda_cls=args.lambda_cls,
            lambda_det=args.lambda_det,
            lambda_traj=args.lambda_traj,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(args.max_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp=args.use_amp)
            val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= args.early_stopping_patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"ga_mtnet_fold{fold_idx}.pt"))

        fold_results.append({"fold": fold_idx, "best_val_loss": best_val_loss})

    return fold_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_det", type=float, default=0.5)
    parser.add_argument("--lambda_traj", type=float, default=0.3)
    parser.add_argument("--use_ga", action="store_true", default=True)
    parser.add_argument("--use_multitask", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.annotations, "r") as f:
        records = json.load(f)

    results = run_cross_validation(records, args.image_root, args, device)

    with open(os.path.join(args.output_dir, "cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
