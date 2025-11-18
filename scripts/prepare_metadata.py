"""
Prepare metadata for training

This script:
1. Loads raw metadata
2. Splits into train/validation sets
3. Creates processed metadata CSVs
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_metadata(
    metadata_csv,
    output_dir,
    train_ratio=0.85,
    val_ratio=0.15,
    random_seed=42
):
    """
    Prepare train/val metadata splits

    Args:
        metadata_csv: Path to raw metadata CSV
        output_dir: Output directory for processed data
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        random_seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("Metadata Preparation")
    print("=" * 80)

    # Set random seed
    np.random.seed(random_seed)

    # Load metadata
    print(f"\nLoading metadata from: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)

    print(f"  - Total images: {len(metadata)}")
    print(f"  - Unique users: {metadata['user_id'].nunique()}")

    # Get unique users
    unique_users = metadata['user_id'].unique()
    print(f"\n  - Splitting by user (not by image)")

    # Split users into train/val
    train_users, val_users = train_test_split(
        unique_users,
        test_size=val_ratio,
        random_state=random_seed
    )

    print(f"  - Train users: {len(train_users)}")
    print(f"  - Val users: {len(val_users)}")

    # Filter metadata
    train_metadata = metadata[metadata['user_id'].isin(train_users)].copy()
    val_metadata = metadata[metadata['user_id'].isin(val_users)].copy()

    print(f"\n  - Train images: {len(train_metadata)}")
    print(f"  - Val images: {len(val_metadata)}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    train_csv = output_dir / 'train_metadata.csv'
    val_csv = output_dir / 'val_metadata.csv'

    train_metadata.to_csv(train_csv, index=False)
    val_metadata.to_csv(val_csv, index=False)

    print(f"\n✓ Saved train metadata to: {train_csv}")
    print(f"✓ Saved val metadata to: {val_csv}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    print("\nTrain set:")
    print(f"  - Users: {train_metadata['user_id'].nunique()}")
    print(f"  - Images: {len(train_metadata)}")
    print(f"  - Avg images per user: {len(train_metadata) / train_metadata['user_id'].nunique():.2f}")

    if 'is_synthetic' in train_metadata.columns:
        synthetic_count = train_metadata['is_synthetic'].sum()
        print(f"  - Synthetic images: {synthetic_count} ({synthetic_count/len(train_metadata)*100:.1f}%)")

    print("\nValidation set:")
    print(f"  - Users: {val_metadata['user_id'].nunique()}")
    print(f"  - Images: {len(val_metadata)}")
    print(f"  - Avg images per user: {len(val_metadata) / val_metadata['user_id'].nunique():.2f}")

    if 'is_synthetic' in val_metadata.columns:
        synthetic_count = val_metadata['is_synthetic'].sum()
        print(f"  - Synthetic images: {synthetic_count} ({synthetic_count/len(val_metadata)*100:.1f}%)")

    # Check for users with single image
    train_single_image_users = train_metadata.groupby('user_id').size()
    train_single_count = (train_single_image_users == 1).sum()

    val_single_image_users = val_metadata.groupby('user_id').size()
    val_single_count = (val_single_image_users == 1).sum()

    print("\n" + "=" * 80)
    print("Warnings")
    print("=" * 80)

    if train_single_count > 0:
        print(f"\n⚠ Warning: {train_single_count} users in train set have only 1 image")
        print("  This may affect triplet sampling.")
        print("  Consider using online triplet mining or augmentation.")

    if val_single_count > 0:
        print(f"\n⚠ Warning: {val_single_count} users in val set have only 1 image")

    # Create copy script for easy access
    create_copy_script(output_dir, train_metadata, val_metadata)

    return train_metadata, val_metadata


def create_copy_script(output_dir, train_metadata, val_metadata):
    """
    Create a shell script to copy images to processed directory

    Args:
        output_dir: Output directory
        train_metadata: Train metadata DataFrame
        val_metadata: Val metadata DataFrame
    """
    script_path = output_dir / 'copy_images.sh'

    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to copy/organize images into train/val directories\n\n")

        f.write("echo 'Copying images to processed directories...'\n\n")

        # Create directories
        f.write("mkdir -p processed/train\n")
        f.write("mkdir -p processed/val\n\n")

        # Copy train images
        f.write("# Copy train images\n")
        f.write("echo 'Copying train images...'\n")
        for _, row in train_metadata.iterrows():
            src = f"raw/{row['image_path']}"
            dst = f"processed/train/{row['filename']}"
            f.write(f"cp {src} {dst}\n")

        f.write("\n# Copy val images\n")
        f.write("echo 'Copying val images...'\n")
        for _, row in val_metadata.iterrows():
            src = f"raw/{row['image_path']}"
            dst = f"processed/val/{row['filename']}"
            f.write(f"cp {src} {dst}\n")

        f.write("\necho 'Done!'\n")

    # Make script executable
    os.chmod(script_path, 0o755)

    print(f"\n✓ Created copy script: {script_path}")
    print("  Run this script to organize images into train/val directories")


def validate_metadata(metadata_csv, data_root):
    """
    Validate metadata - check if all files exist

    Args:
        metadata_csv: Path to metadata CSV
        data_root: Root directory for images
    """
    print("\n" + "=" * 80)
    print("Validating Metadata")
    print("=" * 80)

    metadata = pd.read_csv(metadata_csv)
    data_root = Path(data_root)

    missing_files = []

    print(f"\nChecking {len(metadata)} image files...")

    for _, row in metadata.iterrows():
        filepath = data_root / row['image_path']
        if not filepath.exists():
            missing_files.append(str(filepath))

    if len(missing_files) == 0:
        print("✓ All files exist!")
    else:
        print(f"\n✗ {len(missing_files)} files are missing:")
        for filepath in missing_files[:10]:  # Show first 10
            print(f"  - {filepath}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")


def main():
    parser = argparse.ArgumentParser(description='Prepare metadata for training')
    parser.add_argument('--metadata_csv', type=str, required=True,
                       help='Path to raw metadata CSV')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.85,
                       help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--validate', action='store_true',
                       help='Validate metadata files')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Data root directory (for validation)')

    args = parser.parse_args()

    # Prepare metadata
    train_metadata, val_metadata = prepare_metadata(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed
    )

    # Validate if requested
    if args.validate:
        validate_metadata(args.metadata_csv, args.data_root)


if __name__ == "__main__":
    main()
