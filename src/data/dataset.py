"""
Dataset classes for profile matching
"""

import os
import random
from typing import Optional, Callable, Tuple, List

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ProfileImageDataset(Dataset):
    """
    Basic dataset for profile images
    """

    def __init__(
        self,
        data_root: str,
        metadata_csv: str,
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        """
        Args:
            data_root: Root directory containing images
            metadata_csv: Path to metadata CSV file
            transform: Optional transform to apply
            image_size: Target image size
        """
        self.data_root = data_root
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform
        self.image_size = image_size

        # Default transform if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get image info
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.data_root, row['image_path'])
        user_id = row['user_id']

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return {
            'image': image,
            'user_id': user_id,
            'idx': idx
        }

    def _get_default_transform(self):
        """Get default transform"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class TripletDataset(Dataset):
    """
    Dataset for triplet learning
    Returns (anchor, positive, negative) triplets
    """

    def __init__(
        self,
        data_root: str,
        metadata_csv: str,
        interactions_csv: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        samples_per_class: int = 2
    ):
        """
        Args:
            data_root: Root directory containing images
            metadata_csv: Path to metadata CSV
            interactions_csv: Path to interactions CSV
            transform: Transform to apply
            image_size: Target image size
            samples_per_class: Number of samples per user
        """
        self.data_root = data_root
        self.metadata = pd.read_csv(metadata_csv)
        self.interactions = pd.read_csv(interactions_csv)
        self.transform = transform
        self.image_size = image_size
        self.samples_per_class = samples_per_class

        # Build user to image mapping
        self.user_to_images = self.metadata.groupby('user_id')['image_path'].apply(list).to_dict()

        # Build positive and negative pairs from interactions
        self._build_pairs()

        # Default transform
        if self.transform is None:
            self.transform = self._get_default_transform()

    def _build_pairs(self):
        """Build positive and negative pairs from interaction data"""
        self.positive_pairs = {}  # {user_id: [positive_user_ids]}
        self.negative_pairs = {}  # {user_id: [negative_user_ids]}

        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            target_id = row['target_user_id']
            action = row['action']

            # Initialize if not exists
            if user_id not in self.positive_pairs:
                self.positive_pairs[user_id] = []
            if user_id not in self.negative_pairs:
                self.negative_pairs[user_id] = []

            # Add to positive or negative
            if action == 'like' and row.get('is_mutual', False):
                self.positive_pairs[user_id].append(target_id)
            elif action == 'pass':
                self.negative_pairs[user_id].append(target_id)

        # Get all user IDs that have both positive and negative samples
        self.valid_users = [
            uid for uid in self.positive_pairs.keys()
            if len(self.positive_pairs.get(uid, [])) > 0 and
               len(self.negative_pairs.get(uid, [])) > 0 and
               uid in self.user_to_images
        ]

        print(f"Valid users for triplet training: {len(self.valid_users)}")

    def __len__(self):
        return len(self.valid_users) * self.samples_per_class

    def __getitem__(self, idx):
        # Get anchor user
        user_idx = idx // self.samples_per_class
        anchor_user = self.valid_users[user_idx]

        # Sample anchor image
        anchor_images = self.user_to_images[anchor_user]
        anchor_img_path = random.choice(anchor_images)

        # Sample positive user and image
        positive_users = self.positive_pairs[anchor_user]
        positive_user = random.choice(positive_users)
        positive_images = self.user_to_images.get(positive_user, [])

        if len(positive_images) == 0:
            # Fallback: use another image from same anchor user
            positive_img_path = random.choice(anchor_images)
        else:
            positive_img_path = random.choice(positive_images)

        # Sample negative user and image
        negative_users = self.negative_pairs[anchor_user]
        negative_user = random.choice(negative_users)
        negative_images = self.user_to_images.get(negative_user, [])

        if len(negative_images) == 0:
            # Fallback: random user
            negative_user = random.choice(self.valid_users)
            while negative_user == anchor_user or negative_user in positive_users:
                negative_user = random.choice(self.valid_users)
            negative_images = self.user_to_images[negative_user]

        negative_img_path = random.choice(negative_images)

        # Load images
        anchor_img = self._load_image(anchor_img_path)
        positive_img = self._load_image(positive_img_path)
        negative_img = self._load_image(negative_img_path)

        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'negative': negative_img,
            'anchor_user': anchor_user,
            'positive_user': positive_user,
            'negative_user': negative_user
        }

    def _load_image(self, image_path):
        """Load and transform image"""
        full_path = os.path.join(self.data_root, image_path)
        image = Image.open(full_path).convert('RGB')
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image

    def _get_default_transform(self):
        """Get default transform with augmentation"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class OnlineTripletDataset(Dataset):
    """
    Dataset for online triplet mining
    Returns images with labels, triplets are mined in the loss function
    """

    def __init__(
        self,
        data_root: str,
        metadata_csv: str,
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        self.data_root = data_root
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform
        self.image_size = image_size

        # Create user_id to label mapping
        self.unique_users = self.metadata['user_id'].unique()
        self.user_to_label = {user: idx for idx, user in enumerate(self.unique_users)}

        # Default transform
        if self.transform is None:
            self.transform = self._get_default_transform()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.data_root, row['image_path'])
        user_id = row['user_id']
        label = self.user_to_label[user_id]

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return {
            'image': image,
            'label': label,
            'user_id': user_id
        }

    def _get_default_transform(self):
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def get_transforms(image_size=224, augment=True):
    """
    Get train and validation transforms

    Args:
        image_size: Target image size
        augment: Whether to apply augmentation

    Returns:
        train_transform, val_transform
    """
    if augment:
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


def create_dataloaders(
    data_root: str,
    metadata_csv: str,
    interactions_csv: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    dataset_type: str = 'online_triplet'
):
    """
    Create train and validation dataloaders

    Args:
        data_root: Root directory containing images
        metadata_csv: Path to metadata CSV
        interactions_csv: Path to interactions CSV (for triplet dataset)
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size
        dataset_type: Type of dataset ('basic', 'triplet', 'online_triplet')

    Returns:
        train_loader, val_loader
    """
    # Get transforms
    train_transform, val_transform = get_transforms(image_size, augment=True)

    # Create datasets based on type
    if dataset_type == 'triplet':
        if interactions_csv is None:
            raise ValueError("interactions_csv required for triplet dataset")

        train_dataset = TripletDataset(
            data_root=data_root,
            metadata_csv=metadata_csv,
            interactions_csv=interactions_csv,
            transform=train_transform,
            image_size=image_size
        )

        val_dataset = TripletDataset(
            data_root=data_root,
            metadata_csv=metadata_csv,
            interactions_csv=interactions_csv,
            transform=val_transform,
            image_size=image_size
        )

    elif dataset_type == 'online_triplet':
        train_dataset = OnlineTripletDataset(
            data_root=data_root,
            metadata_csv=metadata_csv,
            transform=train_transform,
            image_size=image_size
        )

        val_dataset = OnlineTripletDataset(
            data_root=data_root,
            metadata_csv=metadata_csv,
            transform=val_transform,
            image_size=image_size
        )

    else:  # basic
        train_dataset = ProfileImageDataset(
            data_root=data_root,
            metadata_csv=metadata_csv,
            transform=train_transform,
            image_size=image_size
        )

        val_dataset = ProfileImageDataset(
            data_root=data_root,
            metadata_csv=metadata_csv,
            transform=val_transform,
            image_size=image_size
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the datasets
    print("Testing dataset classes...")

    # Note: This requires actual data files to run
    # Uncomment and modify paths to test with real data

    # data_root = "data/raw"
    # metadata_csv = "data/raw/metadata.csv"
    # interactions_csv = "data/raw/interactions.csv"

    # dataset = OnlineTripletDataset(
    #     data_root=data_root,
    #     metadata_csv=metadata_csv,
    #     image_size=224
    # )

    # print(f"Dataset size: {len(dataset)}")
    # sample = dataset[0]
    # print(f"Sample keys: {sample.keys()}")
    # print(f"Image shape: {sample['image'].shape}")
