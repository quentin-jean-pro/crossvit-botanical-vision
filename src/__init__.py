"""
CrossViT for Botanical Vision and Interpretability
"""

from .model import VisionTransformer, crossvit_tiny_224, crossvit_tiny_same_res
from .dataset import HerbariumDataset, create_synthetic_dataset, get_default_transforms
from .rollout import (
    attention_rollout,
    compute_rollout_iou,
    IoURegularizedLoss,
    WeightedCrossEntropyLoss,
)

compute_attention_rollout = attention_rollout

__all__ = [
    "crossvit_tiny_224",
    "crossvit_tiny_same_res",
    "VisionTransformer",
    "HerbariumDataset",
    "create_synthetic_dataset",
    "get_default_transforms",
    "attention_rollout",
    "compute_attention_rollout",
    "compute_rollout_iou",
    "IoURegularizedLoss",
    "WeightedCrossEntropyLoss",
]
