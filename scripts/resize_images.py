"""
Resize large images to target size for efficient training

This script:
1. Reads images from input directory
2. Resizes them to target size (default 224x224)
3. Saves resized images to output directory
4. Preserves directory structure
5. Optionally applies quality assessment

Use this for preprocessing very large images (e.g., 2048x2048)
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil


def resize_image(
    input_path,
    output_path,
    target_size=(224, 224),
    quality=95,
    maintain_aspect_ratio=False,
    check_quality=True
):
    """
    Resize a single image

    Args:
        input_path: Input image path
        output_path: Output image path
        target_size: Target size (width, height)
        quality: JPEG quality (1-100)
        maintain_aspect_ratio: If True, resize while maintaining aspect ratio
        check_quality: If True, check image quality before saving

    Returns:
        result: Dict with processing results
    """
    result = {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'success': False,
        'original_size': None,
        'resized_size': None,
        'size_reduction': None,
        'error': None
    }

    try:
        # Open image
        with Image.open(input_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            result['original_size'] = img.size

            # Resize
            if maintain_aspect_ratio:
                # Resize maintaining aspect ratio
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                resized_img = img
            else:
                # Resize to exact target size
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

            result['resized_size'] = resized_img.size

            # Check quality if requested
            if check_quality:
                # Simple quality check: image should not be too small
                if min(resized_img.size) < 50:
                    result['error'] = f"Image too small after resize: {resized_img.size}"
                    return result

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save
            resized_img.save(output_path, 'JPEG', quality=quality, optimize=True)

            # Calculate size reduction
            original_size_bytes = os.path.getsize(input_path)
            resized_size_bytes = os.path.getsize(output_path)
            result['size_reduction'] = (
                (original_size_bytes - resized_size_bytes) / original_size_bytes * 100
            )

            result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def resize_dataset(
    input_dir,
    output_dir,
    metadata_csv=None,
    target_size=(224, 224),
    quality=95,
    maintain_aspect_ratio=False,
    num_workers=4,
    overwrite=False
):
    """
    Resize entire dataset

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for resized images
        metadata_csv: Optional metadata CSV (if provided, only process these images)
        target_size: Target size
        quality: JPEG quality
        maintain_aspect_ratio: Maintain aspect ratio
        num_workers: Number of parallel workers
        overwrite: Overwrite existing files
    """
    print("=" * 80)
    print("Image Resizing")
    print("=" * 80)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {target_size}")
    print(f"Quality: {quality}")
    print(f"Maintain aspect ratio: {maintain_aspect_ratio}")
    print(f"Workers: {num_workers}")

    # Get list of images to process
    if metadata_csv:
        # Process images from metadata
        print(f"\nReading metadata from: {metadata_csv}")
        metadata = pd.read_csv(metadata_csv)
        image_paths = [
            (input_dir / row['image_path'], output_dir / row['image_path'])
            for _, row in metadata.iterrows()
        ]
        print(f"  - Images in metadata: {len(image_paths)}")
    else:
        # Process all images in directory
        print(f"\nScanning directory for images...")
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_paths = [
            (path, output_dir / path.relative_to(input_dir))
            for path in input_dir.rglob('*')
            if path.suffix in extensions
        ]
        print(f"  - Images found: {len(image_paths)}")

    if len(image_paths) == 0:
        print("\n✗ No images to process!")
        return

    # Filter out already processed images if not overwrite
    if not overwrite:
        image_paths = [
            (inp, out) for inp, out in image_paths
            if not out.exists()
        ]
        print(f"  - Images to process (skipping existing): {len(image_paths)}")

    if len(image_paths) == 0:
        print("\n✓ All images already processed!")
        return

    # Process images
    print(f"\nProcessing {len(image_paths)} images...")

    results = []

    if num_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    resize_image,
                    inp,
                    out,
                    target_size,
                    quality,
                    maintain_aspect_ratio,
                    check_quality=True
                ): (inp, out)
                for inp, out in image_paths
            }

            with tqdm(total=len(image_paths), desc="Resizing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

                    # Update progress bar with stats
                    if result['success'] and result['size_reduction']:
                        pbar.set_postfix({'size_reduction': f"{result['size_reduction']:.1f}%"})
    else:
        # Sequential processing
        for inp, out in tqdm(image_paths, desc="Resizing"):
            result = resize_image(
                inp, out, target_size, quality,
                maintain_aspect_ratio, check_quality=True
            )
            results.append(result)

    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    print(f"\nTotal processed: {len(results)}")
    print(f"  ✓ Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  ✗ Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

    if successful:
        avg_reduction = sum(r['size_reduction'] for r in successful) / len(successful)
        print(f"\nAverage size reduction: {avg_reduction:.1f}%")

        # Original sizes
        original_sizes = [r['original_size'] for r in successful]
        avg_original = (
            sum(w * h for w, h in original_sizes) / len(original_sizes)
        ) ** 0.5
        print(f"Average original size: {avg_original:.0f}x{avg_original:.0f}")

    if failed:
        print(f"\n✗ Failed images ({len(failed)}):")
        for r in failed[:10]:  # Show first 10
            print(f"  - {r['input_path']}: {r['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    # Save results log
    results_df = pd.DataFrame(results)
    log_path = output_dir / 'resize_log.csv'
    results_df.to_csv(log_path, index=False)
    print(f"\n✓ Saved processing log to: {log_path}")

    return results


def update_metadata(metadata_csv, output_dir, new_metadata_csv):
    """
    Update metadata CSV with new image paths

    Args:
        metadata_csv: Original metadata CSV
        output_dir: Directory with resized images
        new_metadata_csv: Path to save updated metadata
    """
    print("\n" + "=" * 80)
    print("Updating Metadata")
    print("=" * 80)

    metadata = pd.read_csv(metadata_csv)

    # Update paths to point to resized images
    # Assuming same directory structure
    metadata['original_image_path'] = metadata['image_path']
    # Paths remain the same, just in different root directory

    metadata.to_csv(new_metadata_csv, index=False)

    print(f"\n✓ Updated metadata saved to: {new_metadata_csv}")
    print(f"  - Updated paths for {len(metadata)} images")


def main():
    parser = argparse.ArgumentParser(
        description='Resize images for efficient training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize all images in directory
  python scripts/resize_images.py \\
      --input_dir data/raw/profiles \\
      --output_dir data/processed/profiles_224 \\
      --target_size 224

  # Resize images from metadata
  python scripts/resize_images.py \\
      --input_dir data/raw \\
      --output_dir data/processed \\
      --metadata data/raw/metadata.csv \\
      --target_size 224 \\
      --quality 95

  # Resize maintaining aspect ratio
  python scripts/resize_images.py \\
      --input_dir data/raw/profiles \\
      --output_dir data/processed/profiles_224 \\
      --target_size 256 \\
      --maintain_aspect_ratio \\
      --workers 8
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with original images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for resized images')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Metadata CSV (optional, process only these images)')
    parser.add_argument('--target_size', type=int, default=224,
                       help='Target size (square images)')
    parser.add_argument('--width', type=int, default=None,
                       help='Target width (if different from height)')
    parser.add_argument('--height', type=int, default=None,
                       help='Target height (if different from width)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality (1-100)')
    parser.add_argument('--maintain_aspect_ratio', action='store_true',
                       help='Maintain aspect ratio (may not be exact target size)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--update_metadata', type=str, default=None,
                       help='Path to save updated metadata CSV')

    args = parser.parse_args()

    # Determine target size
    if args.width and args.height:
        target_size = (args.width, args.height)
    else:
        target_size = (args.target_size, args.target_size)

    # Resize images
    results = resize_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata,
        target_size=target_size,
        quality=args.quality,
        maintain_aspect_ratio=args.maintain_aspect_ratio,
        num_workers=args.workers,
        overwrite=args.overwrite
    )

    # Update metadata if requested
    if args.update_metadata and args.metadata:
        update_metadata(
            metadata_csv=args.metadata,
            output_dir=args.output_dir,
            new_metadata_csv=args.update_metadata
        )


if __name__ == "__main__":
    main()
