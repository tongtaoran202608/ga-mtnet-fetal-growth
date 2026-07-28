import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pingouin as pg
import pandas as pd


def compute_classification_metrics(y_true, y_prob, num_classes):
    y_pred = np.argmax(y_prob, axis=1)
    per_class_auc = []
    for c in range(num_classes):
        binary_true = (y_true == c).astype(int)
        try:
            auc = roc_auc_score(binary_true, y_prob[:, c])
        except ValueError:
            auc = float("nan")
        per_class_auc.append(auc)
    macro_auc = np.nanmean(per_class_auc)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    accuracy = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "accuracy": accuracy,
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
    }


def expected_calibration_error(y_true, y_prob, num_bins=15):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return ece


def compute_iou(box1, box2):
    def to_xyxy(b):
        cx, cy, w, h = b
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    b1 = to_xyxy(box1)
    b2 = to_xyxy(box2)
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    order = np.argsort(-np.array(pred_scores))
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))

    for i, idx in enumerate(order):
        pred_box = pred_boxes[idx]
        best_iou = 0.0
        best_gt = -1
        for j, gt_box in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        if best_iou >= iou_threshold and best_gt >= 0:
            tp[i] = 1
            matched[best_gt] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(len(gt_boxes), 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def compute_map_range(pred_boxes, pred_scores, gt_boxes, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = [compute_average_precision(pred_boxes, pred_scores, gt_boxes, t) for t in iou_thresholds]
    return np.mean(aps)


def compute_icc(y_true, y_pred, subject_ids):
    df = pd.DataFrame(
        {
            "subject": np.tile(subject_ids, 2),
            "rater": ["true"] * len(subject_ids) + ["pred"] * len(subject_ids),
            "value": np.concatenate([y_true, y_pred]),
        }
    )
    icc_result = pg.intraclass_corr(data=df, targets="subject", raters="rater", ratings="value")
    icc_row = icc_result[icc_result["Type"] == "ICC2"]
    return icc_row["ICC"].values[0] if len(icc_row) > 0 else float("nan")


def embedding_tsne(embeddings, perplexity=30, learning_rate=200, n_iter=1000, seed=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=seed)
    projected = tsne.fit_transform(embeddings)
    return projected


def embedding_silhouette(embeddings, labels):
    return silhouette_score(embeddings, labels)


def shap_deep_explainer_values(model, background_images, background_ga, test_images, test_ga, device):
    import shap

    def predict_fn(inputs):
        images = torch.tensor(inputs[:, : 3 * 224 * 224].reshape(-1, 3, 224, 224), dtype=torch.float32).to(device)
        ga = torch.tensor(inputs[:, -1], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(images, ga)["cls_logits"]
        return F.softmax(logits, dim=-1).cpu().numpy()

    background_flat = np.concatenate(
        [background_images.reshape(background_images.shape[0], -1), background_ga.reshape(-1, 1)], axis=1
    )
    test_flat = np.concatenate([test_images.reshape(test_images.shape[0], -1), test_ga.reshape(-1, 1)], axis=1)

    explainer = shap.KernelExplainer(predict_fn, background_flat)
    shap_values = explainer.shap_values(test_flat, nsamples=100)
    return shap_values


def grad_cam_plus_plus(model, image_tensor, ga_tensor, target_class, target_layer):
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    outputs = model(image_tensor, ga_tensor)
    logits = outputs["cls_logits"]
    score = logits[:, target_class].sum()
    score.backward()

    handle_f.remove()
    handle_b.remove()

    acts = activations["value"]
    grads = gradients["value"]

    grads_power_2 = grads ** 2
    grads_power_3 = grads_power_2 * grads
    sum_acts = acts.sum(dim=(2, 3), keepdim=True)
    eps = 1e-8
    alpha = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + eps)
    weights = (alpha * F.relu(grads)).sum(dim=(2, 3))

    cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=1)
    cam = F.relu(cam)
    cam = cam / (cam.max() + eps)
    return cam
