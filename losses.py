import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, num_classes=5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def diou_loss(pred_boxes, target_boxes, eps=1e-7):
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)

    x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_pred = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=0)
    area_target = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=0) * (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp(min=0)
    union = area_pred + area_target - inter + eps
    iou = inter / union

    center_dist = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2 + (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2

    enc_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
    enc_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
    enc_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
    enc_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])
    diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    diou = iou - center_dist / diag
    return 1 - diou


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, pred_boxes, pred_obj, target_boxes, target_obj, valid_mask):
        obj_loss = self.bce(pred_obj, target_obj)
        pred_boxes_perm = pred_boxes.permute(0, 1, 3, 4, 2)
        mask = valid_mask.unsqueeze(-1).expand_as(pred_boxes_perm)
        if mask.sum() > 0:
            box_loss = diou_loss(pred_boxes_perm[mask.bool()].view(-1, 4), target_boxes[mask.bool()].view(-1, 4))
            box_loss = box_loss.mean()
        else:
            box_loss = torch.tensor(0.0, device=pred_boxes.device)
        return obj_loss + box_loss


class TrajectoryLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        return self.huber(pred, target)


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_det=0.5, lambda_traj=0.3, num_classes=5, focal_gamma=2.0, focal_alpha=0.25, huber_delta=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_det = lambda_det
        self.lambda_traj = lambda_traj
        self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, num_classes=num_classes)
        self.det_loss = DetectionLoss()
        self.traj_loss = TrajectoryLoss(delta=huber_delta)

    def forward(self, outputs, targets):
        cls_logits = outputs["cls_logits"]
        cls_targets = targets["cls_targets"]
        total = self.lambda_cls * self.cls_loss(cls_logits, cls_targets)
        components = {"cls": total.item()}

        if "det_boxes" in outputs and "det_boxes" in targets:
            det = self.det_loss(
                outputs["det_boxes"],
                outputs["det_obj"],
                targets["det_boxes"],
                targets["det_obj"],
                targets["det_valid_mask"],
            )
            total = total + self.lambda_det * det
            components["det"] = det.item()

        if "traj_pred" in outputs and "traj_targets" in targets:
            traj = self.traj_loss(outputs["traj_pred"], targets["traj_targets"])
            total = total + self.lambda_traj * traj
            components["traj"] = traj.item()

        return total, components
