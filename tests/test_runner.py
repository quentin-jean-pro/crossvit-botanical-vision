"""
Standalone zero-dependency test runner for CrossViT Botanical Vision.
Runs without requiring pytest CLI.
"""
import sys
import tempfile
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from tests.test_model import (
    test_crossvit_tiny_forward,
    test_crossvit_same_res_dual_forward,
    test_dataset_and_rollout,
    test_iou_regularized_loss
)

def run_all():
    print("Testing CrossViT Tiny Forward...")
    test_crossvit_tiny_forward()
    print("  -> test_crossvit_tiny_forward: PASSED ✅")

    print("Testing CrossViT Same Res Dual Forward...")
    test_crossvit_same_res_dual_forward()
    print("  -> test_crossvit_same_res_dual_forward: PASSED ✅")

    print("Testing Herbarium Dataset & Attention Rollout IoU...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dataset_and_rollout(Path(tmp_dir))
    print("  -> test_dataset_and_rollout: PASSED ✅")

    print("Testing IoU Regularized Loss...")
    test_iou_regularized_loss()
    print("  -> test_iou_regularized_loss: PASSED ✅")

    print("\n🎉 ALL 4 CROSSVIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all()

