"""
Heterogeneous backbone wrappers with a shared CORN ordinal head.

Four ImageNet-1K pretrained backbones (via timm), each producing K-1
binary "grade-exceedance" logits per the CORN framework (Shi et al.,
2023). See the manuscript, Section 3.2, Eq. (1)-(2).
"""
import timm
import torch
import torch.nn as nn

BACKBONE_FEATURE_DIMS = {
    "densenet121": 1024,
    "efficientnetv2_s": 1280,
    "resnet50": 2048,
    "convnext_tiny": 768,
}


class CORNModel(nn.Module):
    """Wraps a timm backbone with a dropout + linear CORN head.

    Produces K-1 conditional grade-exceedance logits (sigmoid applied
    at loss/inference time, not inside forward()).
    """

    def __init__(self, backbone_name: str, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim = BACKBONE_FEATURE_DIMS[backbone_name]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # global-average-pooled feature, (B, feat_dim)
        return self.head(self.dropout(feat))


def build_backbone(name: str, num_classes: int = 5) -> CORNModel:
    if name not in BACKBONE_FEATURE_DIMS:
        raise ValueError(f"Unknown backbone '{name}'. Choices: {list(BACKBONE_FEATURE_DIMS)}")
    return CORNModel(name, num_classes=num_classes)


def corn_decode(logits: torch.Tensor) -> torch.Tensor:
    """CORN conditional decoding rule (manuscript Eq. 2): binary logits ->
    per-class probabilities."""
    probs_exceed = torch.sigmoid(logits)  # (B, K-1)
    b = probs_exceed.shape[0]
    k = probs_exceed.shape[1] + 1
    class_probs = torch.zeros(b, k, device=logits.device)
    class_probs[:, 0] = 1 - probs_exceed[:, 0]
    for c in range(1, k - 1):
        class_probs[:, c] = probs_exceed[:, c - 1] - probs_exceed[:, c]
    class_probs[:, k - 1] = probs_exceed[:, k - 2]
    return class_probs.clamp(min=0)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Predicted grade k_hat = sum_j 1[sigmoid(logit_j) >= 0.5]."""
    return (torch.sigmoid(logits) >= 0.5).sum(dim=1)
