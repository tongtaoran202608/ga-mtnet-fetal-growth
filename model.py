import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel


class SinusoidalGAEmbedding(nn.Module):
    def __init__(self, dim=64, min_freq=1.0, max_freq=10000.0):
        super().__init__()
        self.dim = dim
        freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq), dim // 2)
        self.register_buffer("freqs", freqs)

    def forward(self, ga_days):
        ga_days = ga_days.unsqueeze(-1).float()
        args = ga_days * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class GAEmbeddingModule(nn.Module):
    def __init__(self, sinusoidal_dim=64, hidden_dim=128, out_dim=256, feature_channels=768, dropout=0.1):
        super().__init__()
        self.sinusoidal = SinusoidalGAEmbedding(dim=sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gamma_proj = nn.Linear(out_dim, feature_channels)
        self.beta_proj = nn.Linear(out_dim, feature_channels)

    def forward(self, feature_map, ga_days):
        emb = self.sinusoidal(ga_days)
        ctx = self.mlp(emb)
        gamma = self.gamma_proj(ctx).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(ctx).unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * feature_map + beta
        return modulated


class SwinBackbone(nn.Module):
    def __init__(self, pretrained_name="microsoft/swin-tiny-patch4-window7-224"):
        super().__init__()
        self.swin = SwinModel.from_pretrained(pretrained_name, add_pooling_layer=False)
        self.out_channels = 768
        self.spatial_size = 7

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        last_hidden = outputs.last_hidden_state
        B, N, C = last_hidden.shape
        H = W = int(math.sqrt(N))
        fmap = last_hidden.permute(0, 2, 1).reshape(B, C, H, W)
        return fmap


class DetectionHead(nn.Module):
    def __init__(self, in_channels=768, num_landmarks=4):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        self.box_head = nn.Conv2d(256, num_landmarks * 4, kernel_size=1)
        self.obj_head = nn.Conv2d(256, num_landmarks, kernel_size=1)

    def forward(self, feature_map):
        x = self.shared(feature_map)
        boxes = self.box_head(x)
        obj = self.obj_head(x)
        B, _, H, W = boxes.shape
        boxes = boxes.view(B, self.num_landmarks, 4, H, W)
        obj = obj.view(B, self.num_landmarks, H, W)
        return boxes, obj


class ClassificationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=5, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, feature_map):
        x = self.pool(feature_map).flatten(1)
        logits = self.mlp(x)
        return logits, x


class TrajectoryHead(nn.Module):
    def __init__(self, in_channels=768, hidden_size=128, num_layers=2, num_targets=3):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, num_targets)

    def forward(self, sequence_embeddings, valid_mask):
        B, T, C = sequence_embeddings.shape
        mask_expand = self.mask_token.expand(B, T, C)
        valid_mask_exp = valid_mask.unsqueeze(-1).float()
        seq = sequence_embeddings * valid_mask_exp + mask_expand * (1 - valid_mask_exp)
        out, (h_n, c_n) = self.lstm(seq)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred


class GA_MTNet(nn.Module):
    def __init__(
        self,
        num_landmarks=4,
        num_classes=5,
        num_traj_targets=3,
        pretrained_name="microsoft/swin-tiny-patch4-window7-224",
        use_ga_conditioning=True,
        use_multitask=True,
    ):
        super().__init__()
        self.use_ga_conditioning = use_ga_conditioning
        self.use_multitask = use_multitask
        self.backbone = SwinBackbone(pretrained_name=pretrained_name)
        feature_channels = self.backbone.out_channels
        if use_ga_conditioning:
            self.gaem = GAEmbeddingModule(feature_channels=feature_channels)
        self.classification_head = ClassificationHead(in_channels=feature_channels, num_classes=num_classes)
        if use_multitask:
            self.detection_head = DetectionHead(in_channels=feature_channels, num_landmarks=num_landmarks)
            self.trajectory_head = TrajectoryHead(in_channels=feature_channels, num_targets=num_traj_targets)

    def encode(self, images, ga_days):
        fmap = self.backbone(images)
        if self.use_ga_conditioning:
            fmap = self.gaem(fmap, ga_days)
        return fmap

    def forward(self, images, ga_days, traj_sequences=None, traj_mask=None):
        fmap = self.encode(images, ga_days)
        cls_logits, embedding = self.classification_head(fmap)
        outputs = {"cls_logits": cls_logits, "embedding": embedding, "feature_map": fmap}
        if self.use_multitask:
            boxes, obj = self.detection_head(fmap)
            outputs["det_boxes"] = boxes
            outputs["det_obj"] = obj
            if traj_sequences is not None and traj_mask is not None:
                traj_pred = self.trajectory_head(traj_sequences, traj_mask)
                outputs["traj_pred"] = traj_pred
        return outputs

    def mc_dropout_predict(self, images, ga_days, num_passes=30):
        self.train()
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.eval()
        probs_list = []
        with torch.no_grad():
            for _ in range(num_passes):
                out = self.forward(images, ga_days)
                probs = F.softmax(out["cls_logits"], dim=-1)
                probs_list.append(probs.unsqueeze(0))
        self.eval()
        stacked = torch.cat(probs_list, dim=0)
        mean_probs = stacked.mean(dim=0)
        entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-8))).sum(dim=-1)
        return mean_probs, entropy
