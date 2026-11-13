"""
Attention Rollout, heatmaps visualization, and IoU-regularized loss functions
for multi-scale Vision Transformers (CrossViT).
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_patch_plant_ratio(segmented_image: Union[torch.Tensor, np.ndarray], patch_size: int = 16) -> torch.Tensor:
    """
    Computes foreground plant pixel proportion per patch.

    Args:
        segmented_image: [C, H, W] or [H, W] image where black background = 0, plant > 0.
        patch_size: patch dimension in pixels.

    Returns:
        Tensor of shape [num_patches_h, num_patches_w] with values in [0, 1].
    """
    if isinstance(segmented_image, torch.Tensor):
        if segmented_image.dim() == 3:
            gray = segmented_image.mean(0).detach().cpu().numpy()
        else:
            gray = segmented_image.detach().cpu().numpy()
    else:
        gray = segmented_image.mean(0) if segmented_image.ndim == 3 else segmented_image

    H, W = gray.shape
    binary = (gray > 0.1).astype(np.float32)
    ph, pw = H // patch_size, W // patch_size

    ratios = np.zeros((ph, pw), dtype=np.float32)
    for i in range(ph):
        for j in range(pw):
            ratios[i, j] = binary[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size].mean()

    return torch.tensor(ratios, dtype=torch.float32)


def weighting_function(ratio: torch.Tensor, alpha: float = 2.0, beta: float = 0.5) -> torch.Tensor:
    """
    Concave patch weighting function: f(r) = beta + (1 - beta) * r^(1 / alpha).
    """
    return beta + (1.0 - beta) * torch.pow(ratio.clamp(min=0.0, max=1.0), 1.0 / alpha)


def compute_patch_weights_batch(
    segmented_images: torch.Tensor,
    patch_size: int = 16,
    alpha: float = 2.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """Computes patch weights for a full batch of segmented images."""
    weights = [
        weighting_function(compute_patch_plant_ratio(segmented_images[i], patch_size), alpha, beta).flatten()
        for i in range(segmented_images.shape[0])
    ]
    return torch.stack(weights)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy loss amplifying samples with higher relevant plant content.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 0.5, patch_size: int = 16, use_weighting: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.patch_size = patch_size
        self.use_weighting = use_weighting
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        segmented_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        losses = self.ce(outputs, labels)
        if not self.use_weighting or segmented_images is None:
            return losses.mean()

        weights = compute_patch_weights_batch(
            segmented_images,
            self.patch_size,
            self.alpha,
            self.beta,
        ).to(outputs.device)

        image_weights = weights.mean(dim=1)
        total_w = image_weights.sum()
        return (losses * image_weights).sum() / total_w if total_w > 0 else losses.mean()


def attention_rollout(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    branch_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Attention Rollout matrix: A_tilde = Prod_l [ (A^(l) + I) / rowsum ]
    Tracking token information propagation across all self-attention layers.

    Returns:
        attention_map: [H, W] normalized float array in [0, 1].
        attention_rollout_matrix: raw 2D rollout matrix.
    """
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    # Collect attention weights per layer
    attn_matrices = []

    def get_attn_hook(module, input_, output):
        if hasattr(module, "last_attn") and module.last_attn is not None:
            attn_matrices.append(module.last_attn.detach().cpu())

    hooks = []
    for blk in model.blocks:
        branch_blocks = blk.blocks[branch_idx]
        for b in branch_blocks:
            hooks.append(b.attn.register_forward_hook(get_attn_hook))

    with torch.no_grad():
        _ = model(image)

    for h in hooks:
        h.remove()

    if not attn_matrices:
        # Fallback uniform attention if no hooks captured
        return np.zeros((224, 224), dtype=np.float32), np.eye(197)

    # Rollout accumulation
    # Average over attention heads: shape [B=1, N, N]
    num_tokens = attn_matrices[0].shape[-1]
    rollout = torch.eye(num_tokens)

    for attn in attn_matrices:
        attn_avg = attn.mean(dim=1).squeeze(0)  # average over heads
        attn_with_identity = attn_avg + torch.eye(num_tokens)
        attn_normalized = attn_with_identity / attn_with_identity.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_normalized, rollout)

    # Extract CLS token attention to patches
    cls_to_patches = rollout[0, 1:].numpy()
    grid_size = int(np.sqrt(cls_to_patches.shape[0]))
    if grid_size * grid_size != cls_to_patches.shape[0]:
        grid_size = 14
        cls_to_patches = cls_to_patches[:196]

    heatmap_grid = cls_to_patches.reshape(grid_size, grid_size)

    # Interpolate to 224x224
    heatmap_tensor = torch.tensor(heatmap_grid).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(heatmap_tensor, size=(224, 224), mode="bilinear", align_corners=False).squeeze().numpy()

    # Min-max normalization
    norm_map = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
    return norm_map, rollout.numpy()


def compute_rollout_iou(
    attention_map: np.ndarray,
    mask: np.ndarray,
    quantile: float = 0.8,
) -> float:
    """
    Computes Intersection over Union between binarized attention rollout and plant mask.
    Binarization is quantile-based (e.g. top 20% active attention pixels).
    """
    if mask.ndim == 3:
        mask = mask.mean(axis=0)

    binary_mask = (mask > 0.05).astype(np.float32)
    threshold = np.quantile(attention_map, quantile)
    binary_attn = (attention_map >= threshold).astype(np.float32)

    intersection = np.logical_and(binary_attn, binary_mask).sum()
    union = np.logical_or(binary_attn, binary_mask).sum()

    return float(intersection) / float(union) if union > 0 else 0.0


class IoURegularizedLoss(nn.Module):
    """
    Combined Cross-Entropy and Attention IoU Regularization loss:
    L = L_CE + lambda * (1 - IoU(attention, segmentation_mask))
    """

    def __init__(self, lambda_iou: float = 0.1, attention_threshold: float = 0.5):
        super().__init__()
        self.lambda_iou = lambda_iou
        self.attention_threshold = attention_threshold
        self.ce = nn.CrossEntropyLoss()

    def _single_iou(self, attn: torch.Tensor, seg_mask: torch.Tensor) -> torch.Tensor:
        """Differentiable surrogate IoU calculation for a single sample."""
        if attn.shape[0] in (197, 226):
            attn = attn[1:]  # strip CLS token

        num_patches = attn.shape[0]
        g = int(num_patches ** 0.5)
        target_size = g * g
        if num_patches < target_size:
            attn = F.pad(attn, (0, target_size - num_patches), value=0)
        else:
            attn = attn[:target_size]

        attn_grid = attn.view(1, 1, g, g)
        H, W = seg_mask.shape[-2:]
        attn_upsampled = F.interpolate(attn_grid, size=(H, W), mode="bilinear", align_corners=False).squeeze()

        # Min-max normalization
        attn_norm = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)

        attn_bin = (attn_norm > self.attention_threshold).float()
        mask_bin = (seg_mask > 0.05).float()

        intersection = (attn_bin * mask_bin).sum()
        union = (attn_bin + mask_bin).clamp(0, 1).sum()
        return intersection / union if union > 0 else torch.tensor(0.0, device=attn.device)

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        segmentation_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ce_loss = self.ce(outputs, labels)
        if attention_maps is None or segmentation_masks is None:
            return ce_loss, ce_loss, None

        batch_size = segmentation_masks.shape[0]
        seg = segmentation_masks[:, 0] if segmentation_masks.dim() == 4 else segmentation_masks
        ious = [self._single_iou(attention_maps[i], seg[i]) for i in range(batch_size)]
        mean_iou = torch.stack(ious).mean()
        iou_loss = 1.0 - mean_iou

        total_loss = ce_loss + self.lambda_iou * iou_loss
        return total_loss, ce_loss, iou_loss

