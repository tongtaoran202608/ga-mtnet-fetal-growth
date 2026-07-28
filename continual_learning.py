import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=16, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora_into_attention(model, rank=16, alpha=16):
    target_keywords = ["query", "key", "value", "output.dense"]
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(k in f"{name}.{child_name}" for k in target_keywords):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, wrapped)
    return model


def unfreeze_task_heads(model):
    for param in model.classification_head.parameters():
        param.requires_grad = True
    if hasattr(model, "trajectory_head"):
        for param in model.trajectory_head.parameters():
            param.requires_grad = True


class ElasticWeightConsolidation:
    def __init__(self, model, lambda_ewc=400.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher(self, dataloader, criterion, device, num_batches=50):
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        count = 0
        for batch in dataloader:
            if count >= num_batches:
                break
            images = batch["image"].to(device)
            ga_days = batch["ga_days"].to(device)
            cls_targets = batch["cls_target"].to(device)
            self.model.zero_grad()
            outputs = self.model(images, ga_days)
            loss = criterion(outputs["cls_logits"], cls_targets)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
        for n in fisher:
            fisher[n] /= max(count, 1)
        self.fisher = fisher
        self.optimal_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return self.lambda_ewc * loss


class MCDropoutUncertaintyFlagger:
    def __init__(self, model, num_passes=30, percentile=90.0):
        self.model = model
        self.num_passes = num_passes
        self.percentile = percentile
        self.threshold = None

    def calibrate_threshold(self, dataloader, device):
        entropies = []
        for batch in dataloader:
            images = batch["image"].to(device)
            ga_days = batch["ga_days"].to(device)
            _, entropy = self.model.mc_dropout_predict(images, ga_days, num_passes=self.num_passes)
            entropies.append(entropy.detach().cpu())
        all_entropy = torch.cat(entropies)
        self.threshold = torch.quantile(all_entropy, self.percentile / 100.0).item()
        return self.threshold

    def flag_batch(self, images, ga_days):
        mean_probs, entropy = self.model.mc_dropout_predict(images, ga_days, num_passes=self.num_passes)
        flags = entropy > self.threshold
        return flags, entropy, mean_probs


class ContinualLearningLoop:
    def __init__(self, model, lora_rank=16, lora_alpha=16, ewc_lambda=400.0, mc_passes=30, uncertainty_percentile=90.0):
        self.model = model
        self.ewc = ElasticWeightConsolidation(model, lambda_ewc=ewc_lambda)
        self.flagger = MCDropoutUncertaintyFlagger(model, num_passes=mc_passes, percentile=uncertainty_percentile)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def prepare_adaptation(self):
        inject_lora_into_attention(self.model.backbone, rank=self.lora_rank, alpha=self.lora_alpha)
        unfreeze_task_heads(self.model)

    def calibrate(self, calibration_loader, device):
        self.flagger.calibrate_threshold(calibration_loader, device)

    def run_cycle(self, stream_loader, criterion, optimizer, device):
        flagged_batches = []
        for batch in stream_loader:
            images = batch["image"].to(device)
            ga_days = batch["ga_days"].to(device)
            flags, entropy, mean_probs = self.flagger.flag_batch(images, ga_days)
            if flags.any():
                flagged_batches.append(batch)

        for batch in flagged_batches:
            images = batch["image"].to(device)
            ga_days = batch["ga_days"].to(device)
            cls_targets = batch["cls_target"].to(device)
            optimizer.zero_grad()
            outputs = self.model(images, ga_days)
            task_loss = criterion(outputs["cls_logits"], cls_targets)
            ewc_penalty = self.ewc.penalty() if self.ewc.fisher else 0.0
            total_loss = task_loss + ewc_penalty
            total_loss.backward()
            optimizer.step()

        return len(flagged_batches)
