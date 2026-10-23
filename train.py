"""
Phase 1: fine-tune one backbone with the combined CORN+BSFL loss.
LLRD (manuscript Eq. 6) and a 5-epoch linear warm-up is applied when --backbone convnext_tiny.

Usage:
    python train.py --backbone densenet121 --data-root /path/to/oai
    python train.py --backbone convnext_tiny --data-root /path/to/oai
"""
import argparse

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import config
from src.data import build_dataloader
from src.losses import combined_loss, build_class_weights
from src.metrics import quadratic_weighted_kappa
from src.models import build_backbone, corn_predict


def build_llrd_param_groups(model, head_lr: float, decay: float, num_groups: int):
    """Layer-wise learning-rate decay (manuscript Section 3.4, Eq. 6): the CORN
    head plus `num_groups - 1` backbone groups. The backbone group nearest the
    head gets 0.1 x head_lr and each earlier group is scaled by `decay`, giving
    5e-5, 5e-6, 3.3e-6, 2.1e-6, 1.4e-6, 8.9e-7 for head_lr = 5e-5
    (see logs/01_phase1_training_and_clinical_log.txt)."""
    named_params = list(model.backbone.named_parameters())
    num_backbone = num_groups - 1
    chunk = -(-len(named_params) // num_backbone)
    top_lr = head_lr * config.BACKBONE_LR_RATIO
    groups = []
    for g in range(num_backbone):
        params = [p for _, p in named_params[g * chunk:(g + 1) * chunk]]
        if params:
            groups.append({"params": params, "lr": top_lr * decay ** (num_backbone - 1 - g)})
    groups.append({"params": model.head.parameters(), "lr": head_lr})
    return groups


def train_one_backbone(backbone_name: str, data_root: str, device: str = "cpu"):
    torch.manual_seed(config.RANDOM_SEED)
    model = build_backbone(backbone_name, num_classes=config.NUM_CLASSES).to(device)

    is_convnext = backbone_name == "convnext_tiny"
    head_lr = config.HEAD_LR_CONVNEXT if is_convnext else config.HEAD_LR_DEFAULT
    weight_decay = config.WEIGHT_DECAY_CONVNEXT if is_convnext else config.WEIGHT_DECAY_DEFAULT
    max_epochs = config.MAX_EPOCHS_CONVNEXT if is_convnext else config.MAX_EPOCHS_DEFAULT

    if is_convnext:
        param_groups = build_llrd_param_groups(model, head_lr, config.LLRD_DECAY, config.LLRD_GROUPS)
    else:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": head_lr * config.BACKBONE_LR_RATIO},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    if is_convnext:
        warmup = config.CONVNEXT_WARMUP_EPOCHS
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=1.0 / warmup, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=max_epochs - warmup),
        ], milestones=[warmup])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_loader = build_dataloader(f"{data_root}/train", config.IMAGE_SIZE, train=True,
                                     oversample_beta=config.OVERSAMPLE_BETA)
    val_loader = build_dataloader(f"{data_root}/val", config.IMAGE_SIZE, train=False)

    class_counts = torch.zeros(config.NUM_CLASSES)
    for _, y in train_loader.dataset.samples:
        class_counts[y] += 1
    class_weights = build_class_weights(class_counts, kl1_multiplier=config.KL1_WEIGHT_MULTIPLIER).to(device)

    best_qwk, patience_left = -1.0, config.EARLY_STOP_PATIENCE
    for epoch in range(max_epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, targets, class_weights, lambda_f=config.LAMBDA_F,
                                  gamma=config.BSFL_GAMMA)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images.to(device))
                y_pred.extend(corn_predict(logits).cpu().tolist())
                y_true.extend(targets.tolist())
        val_qwk = quadratic_weighted_kappa(y_true, y_pred)
        print(f"[{backbone_name}] epoch {epoch + 1}/{max_epochs}  val QWK={val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk, patience_left = val_qwk, config.EARLY_STOP_PATIENCE
            torch.save(model.state_dict(), f"{backbone_name}_best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{backbone_name}] early stopping at epoch {epoch + 1}")
                break

    return best_qwk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=config.BACKBONES)
    parser.add_argument("--data-root", default=config.DATA_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_one_backbone(args.backbone, args.data_root, args.device)
