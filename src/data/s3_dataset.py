import io
import os
import boto3
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, List, Callable

class S3Dataset(Dataset):
    """
    Dataset for loading images directly from AWS S3.
    """
    def __init__(
        self, 
        bucket_name: str, 
        prefix: str, 
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """
        Args:
            bucket_name (str): Name of the S3 bucket
            prefix (str): Prefix (folder path) in the bucket where images are stored
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_dir (str, optional): Directory to cache downloaded images. If None, no caching.
            limit (int, optional): Limit the number of images to load (for testing).
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        self.cache_dir = cache_dir
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # List objects in the bucket with the given prefix
        print(f"Listing objects in s3://{bucket_name}/{prefix}...")
        self.image_keys = self._list_objects()
        
        if limit:
            self.image_keys = self.image_keys[:limit]
            
        print(f"Found {len(self.image_keys)} images.")
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Caching enabled. Directory: {self.cache_dir}")

    def _list_objects(self) -> List[str]:
        """List all image objects in the specified S3 prefix."""
        keys = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Filter for image files (you can add more extensions if needed)
                    if key.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        keys.append(key)
        return keys

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        key = self.image_keys[idx]
        
        image_data = None
        
        # Try to load from cache first
        if self.cache_dir:
            local_path = os.path.join(self.cache_dir, os.path.basename(key))
            if os.path.exists(local_path):
                try:
                    image = Image.open(local_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image
                except Exception as e:
                    print(f"Error loading from cache {local_path}: {e}. Re-downloading.")
        
        # Download from S3
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = obj['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Save to cache if enabled
            if self.cache_dir:
                local_path = os.path.join(self.cache_dir, os.path.basename(key))
                with open(local_path, 'wb') as f:
                    f.write(image_data)
            
            if self.transform:
                image = self.transform(image)
                
            return image
            
        except Exception as e:
            print(f"Error loading image {key} from S3: {e}")
            # Return a dummy image or raise error depending on requirement
            # For now, raising error to be explicit
            raise e
