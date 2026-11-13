"""
Dataset loading and preprocessing utilities for botanical specimen images.
Supports paired raw and segmented image inputs, standard data augmentations,
and synthetic dataset generation for local testing and reproducibility.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

# Optional torchvision support with pure PyTorch / PIL fallbacks
try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    transforms = None
    HAS_TORCHVISION = False


class SimpleNormalize:
    """Fallback normalization if torchvision is unavailable."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class SimpleToTensor:
    """Fallback ToTensor if torchvision is unavailable."""
    def __call__(self, pic: Image.Image) -> torch.Tensor:
        arr = np.array(pic, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]
        # HWC to CHW
        tensor = torch.from_numpy(arr.transpose((2, 0, 1)))
        return tensor


class HerbariumDataset(Dataset):
    """
    Herbarium botanical dataset providing paired raw and segmented specimen images.
    Returns (img_seg, img_raw, label) to support dual-branch architectures.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        raw_dir: Optional[Union[str, Path]] = None,
        seg_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.seg_dir = Path(seg_dir) if seg_dir else None
        self.transform = transform
        self.target_size = target_size

        if df is not None:
            self.df = df.reset_index(drop=True)
        elif self.raw_dir and self.raw_dir.exists():
            files = sorted([f.stem for f in self.raw_dir.glob("*.jpg")])
            if not files:
                files = sorted([f.stem for f in self.raw_dir.glob("*.png")])
            self.df = pd.DataFrame({"code": files, "epines": [0] * len(files)})
        else:
            self.df = pd.DataFrame(columns=["code", "epines"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        code = str(row["code"]).strip()
        label = int(row.get("epines", row.get("label", 0)))

        # Resolve raw image
        raw_path = self._find_image_path(self.raw_dir, code)
        if raw_path and raw_path.exists():
            img_raw = Image.open(raw_path).convert("RGB")
        else:
            img_raw = Image.new("RGB", self.target_size, color=(240, 235, 220))

        # Resolve segmented image (or fall back to raw if missing)
        seg_path = self._find_image_path(self.seg_dir, code) if self.seg_dir else None
        if seg_path and seg_path.exists():
            img_seg = Image.open(seg_path).convert("RGB")
        else:
            img_seg = img_raw.copy()

        img_raw = img_raw.resize(self.target_size, Image.Resampling.BILINEAR)
        img_seg = img_seg.resize(self.target_size, Image.Resampling.BILINEAR)

        if self.transform:
            img_raw = self.transform(img_raw)
            img_seg = self.transform(img_seg)
        else:
            to_tensor = transforms.ToTensor() if HAS_TORCHVISION else SimpleToTensor()
            img_raw = to_tensor(img_raw)
            img_seg = to_tensor(img_seg)

        return img_seg, img_raw, label

    @staticmethod
    def _find_image_path(base_dir: Optional[Path], code: str) -> Optional[Path]:
        if not base_dir:
            return None
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            candidate = base_dir / f"{code}{ext}"
            if candidate.exists():
                return candidate
        return None


def get_default_transforms(img_size: int = 224, is_train: bool = True):
    """Returns standard transforms for training or evaluation."""
    if HAS_TORCHVISION:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        if is_train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Graceful fallback without torchvision
        normalize = SimpleNormalize()
        to_tensor = SimpleToTensor()
        return lambda img: normalize(to_tensor(img))


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 24,
    img_size: Tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """
    Generates a synthetic mock botanical dataset with raw specimen images,
    segmented plant masks, and specimen labels for testing without private data.
    """
    output_path = Path(output_dir)
    raw_dir = output_path / "raw"
    seg_dir = output_path / "segmented"

    raw_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    records = []
    np.random.seed(42)

    for i in range(num_samples):
        code = f"SPEC_{i:04d}"
        label = int(i % 2)

        # Background sheet
        bg_color = (
            int(np.random.randint(235, 248)),
            int(np.random.randint(230, 243)),
            int(np.random.randint(215, 230)),
        )
        raw_img = Image.new("RGB", img_size, color=bg_color)
        seg_img = Image.new("RGB", img_size, color=(0, 0, 0))

        raw_draw = ImageDraw.Draw(raw_img)
        seg_draw = ImageDraw.Draw(seg_img)

        # Stem
        stem_points = [
            (img_size[0] // 2 + int(np.random.randint(-15, 15)), img_size[1] - 20),
            (img_size[0] // 2 + int(np.random.randint(-25, 25)), img_size[1] // 2),
            (img_size[0] // 2 + int(np.random.randint(-20, 20)), 30),
        ]
        raw_draw.line(stem_points, fill=(60, 45, 25), width=4)
        seg_draw.line(stem_points, fill=(255, 255, 255), width=4)

        # Leaves
        num_leaves = int(np.random.randint(3, 7))
        for _ in range(num_leaves):
            lx = int(np.random.randint(40, img_size[0] - 40))
            ly = int(np.random.randint(40, img_size[1] - 50))
            leaf_box = [lx - 16, ly - 8, lx + 16, ly + 8]
            leaf_color = (
                int(np.random.randint(40, 80)),
                int(np.random.randint(90, 140)),
                int(np.random.randint(30, 70)),
            )
            raw_draw.ellipse(leaf_box, fill=leaf_color)
            seg_draw.ellipse(leaf_box, fill=(255, 255, 255))

            if label == 1:
                thorn_end = (lx + 20, ly + 5)
                raw_draw.line([(lx + 10, ly), thorn_end], fill=(130, 50, 20), width=2)
                seg_draw.line([(lx + 10, ly), thorn_end], fill=(255, 255, 255), width=2)

        # Label in corner
        raw_draw.rectangle([img_size[0] - 65, img_size[1] - 35, img_size[0] - 10, img_size[1] - 10], fill=(255, 255, 255), outline=(180, 180, 180))

        raw_img.save(raw_dir / f"{code}.jpg", quality=90)
        seg_img.save(seg_dir / f"{code}.jpg", quality=90)
        records.append({"code": code, "epines": label})

    df = pd.DataFrame(records)
    csv_path = output_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    return df
