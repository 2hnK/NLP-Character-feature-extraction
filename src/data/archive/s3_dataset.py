import io
import os
import json
import boto3
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, List, Callable, Dict
from sklearn.preprocessing import LabelEncoder

class S3Dataset(Dataset):
    """
    Dataset for loading images from AWS S3 using a JSONL metadata file.
    """
    def __init__(
        self, 
        bucket_name: str, 
        jsonl_path: str,
        prefix: str = "",
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
        label_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            bucket_name (str): Name of the S3 bucket
            jsonl_path (str): Path to the JSONL metadata file (local path)
            prefix (str): Optional prefix to prepend to filenames in S3
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_dir (str, optional): Directory to cache downloaded images.
            limit (int, optional): Limit the number of images to load.
            label_mapping (dict, optional): Dictionary mapping label names to integers.
        """
        self.bucket_name = bucket_name
        self.jsonl_path = jsonl_path
        self.prefix = prefix
        self.transform = transform
        self.cache_dir = cache_dir
        self.label_mapping = label_mapping
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Check if jsonl_path exists locally, if not try to download from S3
        if not os.path.exists(jsonl_path):
            print(f"Metadata file '{jsonl_path}' not found locally. Attempting to download from S3 bucket '{bucket_name}'...")
            try:
                # Ensure directory exists
                local_dir = os.path.dirname(jsonl_path)
                if local_dir:
                    os.makedirs(local_dir, exist_ok=True)
                
                self.s3_client.download_file(bucket_name, jsonl_path, jsonl_path)
                print(f"Successfully downloaded '{jsonl_path}' from S3.")
            except Exception as e:
                raise FileNotFoundError(f"Could not find '{jsonl_path}' locally or download from S3. Error: {e}")

        # Load metadata
        print(f"Loading metadata from {jsonl_path}...")
        self.data = []
        self.labels = []
        
        # Load label mapping if exists, otherwise build it (but user requested persistent mapping)
        # Ideally, the mapping should be passed or loaded from a file.
        
        if hasattr(self, 'label_mapping') and self.label_mapping:
             # Already set via argument (see below)
             pass
        else:
            mapping_path = "label_mapping.json"
            if os.path.exists(mapping_path):
                print(f"Loading label mapping from {mapping_path}")
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.label_mapping = json.load(f)
            else:
                print("Warning: label_mapping.json not found. Building mapping from data (indices might vary).")
                self.label_mapping = None

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                self.data.append(item)
                
                # Extract label
                label = "Unknown"
                
                # 1. Try 'image_metadata' -> 'fashion_style' (Validation & Standard)
                if 'image_metadata' in item and 'fashion_style' in item['image_metadata']:
                    style = item['image_metadata']['fashion_style']
                    if style: # Ensure it's not empty string
                        label = style
                
                # 2. Try parsing from Qwen conversation (if label is hidden in text)
                # Currently, the provided train sample doesn't show explicit style label.
                # Assuming 'Unknown' for now if not found.
                
                self.labels.append(label)
                    
        if limit:
            self.data = self.data[:limit]
            self.labels = self.labels[:limit]
            
        print(f"Loaded {len(self.data)} items.")
        
        # Encode labels
        if self.label_mapping:
            self.encoded_labels = np.array([self.label_mapping.get(label, -1) for label in self.labels])
            self.classes = list(self.label_mapping.keys())
        else:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels)
            self.classes = self.label_encoder.classes_
            
            # Save the generated mapping
            self.label_mapping = {label: int(idx) for idx, label in enumerate(self.classes)}
            mapping_path = "label_mapping.json"
            print(f"Saving generated label mapping to {mapping_path}")
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.label_mapping, f, indent=4, ensure_ascii=False)
            
        self.num_classes = len(self.classes)
        print(f"Found {self.num_classes} classes.")
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.encoded_labels[idx]
        
        # Construct S3 key
        filename = None
        
        # 1. Try 'filename' (Standard)
        if 'filename' in item:
            filename = item['filename']
            
        # 2. Try 'image_filename' (Validation format)
        elif 'image_filename' in item:
            filename = item['image_filename']
            
        # 3. Try 'conversations' (Qwen Train format)
        elif 'conversations' in item:
            # Extract from <img>tag</img>
            # Example: "Picture 1: <img>images/aug_00000.jpg</img>..."
            for turn in item['conversations']:
                if '<img>' in turn['value']:
                    import re
                    match = re.search(r'<img>(.*?)</img>', turn['value'])
                    if match:
                        full_path = match.group(1) # e.g., images/aug_00000.jpg
                        filename = os.path.basename(full_path) # aug_00000.jpg
                        break
        
        if not filename:
             raise ValueError(f"Could not extract filename from item at index {idx}: {item.keys()}")
             
        key = os.path.join(self.prefix, filename).replace("\\", "/")
        
        image = self._load_image(key)
        
        # Get text input if available
        text_input = item.get('text_input', "")
        
        if self.transform:
            image = self.transform(image)
            
        # Return (image, label) or (image, text, label)?
        # The current train loop expects (images, labels).
        # If we want to support text, we should return it, but train.py needs to handle it.
        # For now, let's stick to (image, label) to avoid breaking train.py immediately,
        # unless we update train.py too.
        # User asked for "Text embedding input formatting", implying they want to use it.
        # Let's return it as extra info, but we need to update the collate_fn or train loop if we do.
        # Standard DataLoader handles strings in a batch as a list of strings.
        # So returning (image, text, label) is safe if we unpack it in train loop.
        
        return image, text_input, torch.tensor(label, dtype=torch.long)
        
    def _load_image(self, key: str) -> Image.Image:
        # Try cache first
        if self.cache_dir:
            local_path = os.path.join(self.cache_dir, os.path.basename(key))
            if os.path.exists(local_path):
                try:
                    return Image.open(local_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading from cache {local_path}: {e}")
        
        # Download from S3
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = obj['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Save to cache
            if self.cache_dir:
                local_path = os.path.join(self.cache_dir, os.path.basename(key))
                with open(local_path, 'wb') as f:
                    f.write(image_data)
            
            return image
        except Exception as e:
            print(f"Error loading image {key} from S3: {e}")
            # Return a black image as fallback to prevent crashing
            return Image.new('RGB', (224, 224), color='black')

    def get_labels(self):
        """Returns the list of labels for the sampler."""
        return self.encoded_labels.tolist()
