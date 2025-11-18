"""
Generate dummy dataset for testing the pipeline

This script generates:
1. Dummy profile images (colored rectangles with user IDs)
2. Metadata CSV files
3. Interactions CSV (for triplet training)
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from tqdm import tqdm


def generate_dummy_image(user_id, image_idx, image_size=224, seed=None):
    """
    Generate a dummy profile image

    Args:
        user_id: User identifier (str)
        image_idx: Image index for this user
        image_size: Size of output image
        seed: Random seed for reproducibility

    Returns:
        image: PIL Image
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random color based on user_id
    user_seed = hash(user_id) % 10000
    np.random.seed(user_seed + image_idx)

    # Create base color for this user
    base_color = np.random.randint(50, 206, size=3)

    # Add variation for different images of same user
    variation = np.random.randint(-30, 30, size=3)
    color = np.clip(base_color + variation, 0, 255).astype(np.uint8)

    # Create image
    image = Image.new('RGB', (image_size, image_size), tuple(color))

    # Add some random shapes (to simulate facial features)
    draw = ImageDraw.Draw(image)

    # Add circles (eyes)
    eye1_pos = (image_size // 3, image_size // 3)
    eye2_pos = (2 * image_size // 3, image_size // 3)
    eye_radius = image_size // 20

    eye_color = tuple(np.clip(color - 50, 0, 255).astype(np.uint8))
    draw.ellipse(
        [eye1_pos[0] - eye_radius, eye1_pos[1] - eye_radius,
         eye1_pos[0] + eye_radius, eye1_pos[1] + eye_radius],
        fill=eye_color
    )
    draw.ellipse(
        [eye2_pos[0] - eye_radius, eye2_pos[1] - eye_radius,
         eye2_pos[0] + eye_radius, eye2_pos[1] + eye_radius],
        fill=eye_color
    )

    # Add arc (mouth)
    mouth_box = [image_size // 4, image_size // 2,
                 3 * image_size // 4, 3 * image_size // 4]
    draw.arc(mouth_box, 0, 180, fill=eye_color, width=3)

    # Add user ID text
    try:
        # Try to use a default font
        font_size = image_size // 10
        # Just use default font if custom font not available
        draw.text(
            (10, image_size - 30),
            f"{user_id}_{image_idx}",
            fill=(255, 255, 255)
        )
    except:
        pass

    return image


def generate_dataset(
    output_dir,
    num_users=50,
    images_per_user_range=(1, 4),
    image_size=224,
    generate_augmented=True,
    num_augmented=100
):
    """
    Generate complete dummy dataset

    Args:
        output_dir: Output directory
        num_users: Number of users
        images_per_user_range: Range of images per user
        image_size: Size of images
        generate_augmented: Whether to generate augmented data
        num_augmented: Number of augmented images
    """
    output_dir = Path(output_dir)

    # Create directories
    raw_dir = output_dir / 'raw' / 'profiles'
    augmented_dir = output_dir / 'augmented' / 'generated'

    raw_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating dummy dataset in: {output_dir}")
    print(f"  - Users: {num_users}")
    print(f"  - Images per user: {images_per_user_range}")

    # Generate real user images
    print("\nGenerating real user images...")
    real_images_data = []

    for user_idx in tqdm(range(num_users)):
        user_id = f"user_{user_idx:03d}"

        # Random number of images per user
        num_images = np.random.randint(
            images_per_user_range[0],
            images_per_user_range[1] + 1
        )

        for img_idx in range(1, num_images + 1):
            # Generate image
            image = generate_dummy_image(user_id, img_idx, image_size, seed=user_idx * 100 + img_idx)

            # Save image
            filename = f"{user_id}_{img_idx}.jpg"
            filepath = raw_dir / filename
            image.save(filepath, quality=95)

            # Add to metadata
            real_images_data.append({
                'user_id': user_id,
                'image_idx': img_idx,
                'filename': filename,
                'image_path': f'profiles/{filename}',
                'is_synthetic': False
            })

    print(f"  ✓ Generated {len(real_images_data)} real user images")

    # Generate augmented images
    augmented_images_data = []

    if generate_augmented and num_augmented > 0:
        print(f"\nGenerating {num_augmented} augmented images...")

        for aug_idx in tqdm(range(num_augmented)):
            # Use a synthetic user ID
            synthetic_user_id = f"gen_{aug_idx:04d}"

            # Generate image
            image = generate_dummy_image(
                synthetic_user_id,
                1,
                image_size,
                seed=10000 + aug_idx
            )

            # Save image
            filename = f"gen_{aug_idx:04d}.jpg"
            filepath = augmented_dir / filename
            image.save(filepath, quality=95)

            # Add to metadata
            augmented_images_data.append({
                'user_id': synthetic_user_id,
                'image_idx': 1,
                'filename': filename,
                'image_path': f'generated/{filename}',
                'is_synthetic': True
            })

        print(f"  ✓ Generated {len(augmented_images_data)} augmented images")

    # Combine metadata
    all_images_data = real_images_data + augmented_images_data
    metadata_df = pd.DataFrame(all_images_data)

    # Save metadata
    metadata_path = output_dir / 'raw' / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n✓ Saved metadata to: {metadata_path}")

    # Generate interactions data (for triplet training)
    print("\nGenerating interaction data...")
    interactions = generate_interactions(real_images_data, num_interactions=500)

    interactions_path = output_dir / 'raw' / 'interactions.csv'
    interactions.to_csv(interactions_path, index=False)
    print(f"  ✓ Saved interactions to: {interactions_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Real users: {num_users}")
    print(f"  - Real images: {len(real_images_data)}")
    print(f"  - Augmented images: {len(augmented_images_data)}")
    print(f"  - Total images: {len(all_images_data)}")
    print(f"  - Interactions: {len(interactions)}")

    print(f"\nFiles created:")
    print(f"  - {raw_dir}")
    print(f"  - {augmented_dir}")
    print(f"  - {metadata_path}")
    print(f"  - {interactions_path}")

    return metadata_df, interactions


def generate_interactions(images_data, num_interactions=500):
    """
    Generate synthetic interaction data (likes/passes)

    Args:
        images_data: List of image metadata dicts
        num_interactions: Number of interactions to generate

    Returns:
        interactions_df: DataFrame with interactions
    """
    # Get unique users
    users = list(set([img['user_id'] for img in images_data]))

    interactions = []

    for _ in range(num_interactions):
        # Random user
        user_id = np.random.choice(users)

        # Random target user (different from user)
        target_id = user_id
        while target_id == user_id:
            target_id = np.random.choice(users)

        # Random action (70% like, 30% pass)
        action = 'like' if np.random.random() < 0.7 else 'pass'

        # If like, 20% chance of mutual
        is_mutual = False
        if action == 'like' and np.random.random() < 0.2:
            is_mutual = True

        interactions.append({
            'user_id': user_id,
            'target_user_id': target_id,
            'action': action,
            'is_mutual': is_mutual,
            'timestamp': f"2024-01-{np.random.randint(1, 31):02d}"
        })

    return pd.DataFrame(interactions)


def main():
    parser = argparse.ArgumentParser(description='Generate dummy dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--num_users', type=int, default=50,
                       help='Number of users')
    parser.add_argument('--images_per_user_min', type=int, default=1,
                       help='Minimum images per user')
    parser.add_argument('--images_per_user_max', type=int, default=4,
                       help='Maximum images per user')
    parser.add_argument('--num_augmented', type=int, default=100,
                       help='Number of augmented images')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--no_augmented', action='store_true',
                       help='Do not generate augmented images')

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_users=args.num_users,
        images_per_user_range=(args.images_per_user_min, args.images_per_user_max),
        image_size=args.image_size,
        generate_augmented=not args.no_augmented,
        num_augmented=args.num_augmented
    )


if __name__ == "__main__":
    main()
