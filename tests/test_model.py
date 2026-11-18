"""
Unit tests for CrossViT architectures, dataset loaders, and rollout IoU losses.
"""

from pathlib import Path
import sys, os
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    import pytest
except ImportError:
    pass
import torch

from src.model import crossvit_tiny_224, crossvit_tiny_same_res
from src.dataset import HerbariumDataset, create_synthetic_dataset
from src.rollout import attention_rollout, compute_rollout_iou, IoURegularizedLoss


def test_crossvit_tiny_forward():
    model = crossvit_tiny_224(pretrained=False, num_classes=2)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)


def test_crossvit_same_res_dual_forward():
    model = crossvit_tiny_same_res(pretrained=False, num_classes=2)
    model.eval()
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model.dual_forward(x1, x2)
    assert out.shape == (2, 2)


def test_dataset_and_rollout(tmp_path):
    # Generate temporary mock data
    df = create_synthetic_dataset(tmp_path, num_samples=4, img_size=(224, 224))
    assert len(df) == 4

    dataset = HerbariumDataset(df=df, raw_dir=tmp_path / "raw", seg_dir=tmp_path / "segmented")
    assert len(dataset) == 4

    img_seg, img_raw, label = dataset[0]
    assert img_raw.shape == (3, 224, 224)
    assert img_seg.shape == (3, 224, 224)
    assert label in (0, 1)

    # Test attention rollout
    model = crossvit_tiny_same_res(pretrained=False, num_classes=2)
    device = torch.device("cpu")
    attn_map, rollout_mat = attention_rollout(model, img_raw, device=device, branch_idx=0)

    assert attn_map.shape == (224, 224)
    assert rollout_mat.ndim == 2

    iou = compute_rollout_iou(attn_map, img_seg.numpy())
    assert 0.0 <= iou <= 1.0


def test_iou_regularized_loss():
    criterion = IoURegularizedLoss(lambda_iou=0.1, attention_threshold=0.5)
    logits = torch.randn(2, 2, requires_grad=True)
    labels = torch.tensor([0, 1])

    # Case 1: without attention map
    total_loss, ce_loss, _ = criterion(logits, labels)
    assert total_loss.item() > 0

    # Case 2: with dummy attention tokens and segmentation mask
    attn_maps = torch.rand(2, 197)
    seg_masks = torch.zeros(2, 1, 224, 224)
    seg_masks[:, :, 50:150, 50:150] = 1.0

    total_loss, ce_loss, iou_loss = criterion(logits, labels, attn_maps, seg_masks)
    assert total_loss.item() > 0
    assert iou_loss is not None

