"""
CLI utility to generate a synthetic botanical dataset with paired raw images,
segmentation masks, and classification metadata for offline testing.
"""

import argparse
from pathlib import Path
from src.dataset import create_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic herbarium dataset.")
    parser.add_argument("--output", type=str, default="data/sample", help="Output directory path")
    parser.add_argument("--samples", type=int, default=24, help="Number of synthetic samples")
    parser.add_argument("--size", type=int, default=224, help="Image width and height in pixels")
    args = parser.parse_args()

    out_path = Path(args.output)
    print(f"🌿 Generating {args.samples} synthetic herbarium specimens in '{out_path}'...")
    df = create_synthetic_dataset(out_path, num_samples=args.samples, img_size=(args.size, args.size))
    print(f"✅ Created {len(df)} specimens (classes: {df['epines'].value_counts().to_dict()})")
    print(f"   Metadata saved to: {out_path / 'metadata.csv'}")
    print(f"   Raw images in:     {out_path / 'raw'}")
    print(f"   Masks in:          {out_path / 'segmented'}")


if __name__ == "__main__":
    main()

