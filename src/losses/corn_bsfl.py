"""
Combined CORN + Boundary-Sensitive Focal Loss (BSFL).
Manuscript Section 3.3, Eq. (3)-(5).
"""
import torch
import torch.nn.functional as F

from src.models.backbones import corn_decode


def corn_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy over K-1 conditional classifiers (Eq. 4)."""
    k_minus_1 = logits.shape[1]
    levels = torch.stack(
        [(targets > j).float() for j in range(k_minus_1)], dim=1
    )  # r_j = 1[k > j]
    return F.binary_cross_entropy_with_logits(logits, levels, reduction="mean")


def boundary_sensitive_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss (Lin et al., 2018) applied to CORN-decoded class
    probabilities, with per-class weights w_c (Eq. 5). class_weights
    should already include the KL-1 multiplier (see build_class_weights)."""
    probs = corn_decode(logits)  # (B, K)
    p_true = probs.gather(1, targets.view(-1, 1)).clamp(min=1e-6).squeeze(1)
    w = class_weights[targets]
    loss = -w * (1 - p_true).pow(gamma) * torch.log(p_true)
    return loss.mean()


def build_class_weights(class_counts: torch.Tensor, kl1_index: int = 1,
                         kl1_multiplier: float = 4.0) -> torch.Tensor:
    """w_c = n_max / n_c, with an extra multiplier on the KL-1 (Doubtful) class."""
    n_max = class_counts.max()
    weights = n_max / class_counts.float()
    weights[kl1_index] *= kl1_multiplier
    return weights


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    lambda_f: float = 0.4,
    gamma: float = 2.0,
) -> torch.Tensor:
    """L = (1 - lambda_f) * L_CORN + lambda_f * L_BSFL."""
    l_corn = corn_loss(logits, targets)
    l_bsfl = boundary_sensitive_focal_loss(logits, targets, class_weights, gamma)
    return (1 - lambda_f) * l_corn + lambda_f * l_bsfl
