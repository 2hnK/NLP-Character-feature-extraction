import os
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, List, Callable, Dict, Tuple

class MatchingDataset(Dataset):
    """
    Dataset for Pairwise Matching (Siamese Network).
    Loads (UserA, UserB, Label) triplets.
    Uses 'pairId' to locate local images in the specified root directory.
    Constructs text description from metadata.
    """
    def __init__(
        self, 
        jsonl_path: str,
        image_root: str,
        transform: Optional[Callable] = None,
        limit: Optional[int] = None,
        mode: str = 'train', # 'train' or 'valid'
        split_ratio: float = 0.8,
        seed: int = 42
    ):
        """
        Args:
            jsonl_path (str): Path to dataset.jsonl
            image_root (str): Root directory containing pairId folders (e.g., c:/.../data/images)
            transform (callable, optional): Transform for images.
            limit (int, optional): Limit data size for debugging.
            mode (str): 'train' or 'valid' to split data.
            split_ratio (float): Ratio for training data.
            seed (int): Random seed for splitting.
        """
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.transform = transform
        self.mode = mode
        
        self.data = []
        
        # 1. Load Data
        print(f"[{mode.upper()}] Loading metadata from {jsonl_path}...")
        all_data = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load {jsonl_path}: {e}")
            
        # 2. Split Data
        random.seed(seed)
        random.shuffle(all_data)
        
        split_idx = int(len(all_data) * split_ratio)
        if mode == 'train':
            self.data = all_data[:split_idx]
        else:
            self.data = all_data[split_idx:]
            
        if limit:
            self.data = self.data[:limit]
            
        print(f"[{mode.upper()}] Loaded {len(self.data)} pairs.")
        
    def __len__(self):
        return len(self.data)
    
    def _get_image_path(self, pair_id: str, gender: str) -> str:
        """
        Finds the image path given pairId and gender.
        Assumes folder structure: image_root/pairId/{gender}*.png
        """
        pair_dir = os.path.join(self.image_root, pair_id)
        if not os.path.exists(pair_dir):
            # Fallback: try searching without strict case or check if images are flat?
            # Based on analysis, they are in folders.
            raise FileNotFoundError(f"Pair directory not found: {pair_dir}")
            
        # List files to find match
        files = os.listdir(pair_dir)
        # Filter by gender (case-insensitive)
        # We expect 'male_001.png' or 'female_001.png'
        # Check for 'male' or 'female' in filename
        target_files = [f for f in files if gender.lower() in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not target_files:
             # Fallback if specific gender file is missing (unexpected based on analysis)
             # Just take any image? No, that mixes users.
             # Return None and let getitem handle error
             return None
             
        # Take the first match
        return os.path.join(pair_dir, target_files[0])

    def _make_text_description(self, user_data: dict) -> str:
        """
        Constructs a natural language description from metadata.
        """
        gender = user_data.get('gender', 'unknown')
        interests = user_data.get('interests', [])
        
        # Simple template
        # "여성 사용자, 관심사: 여행, 맛집"
        
        gender_map = {'male': '남성', 'female': '여성'}
        gender_str = gender_map.get(gender, gender)
        
        interest_str = ", ".join(interests) if interests else "없음"
        
        description = f"{gender_str} 사용자, 관심사: {interest_str}"
        return description

    def __getitem__(self, idx):
        item = self.data[idx]
        pair_id = item['pairId']
        pair_type = item['pairType']
        
        # Label: Positive=1, Negative=-1
        label = 1 if pair_type == 'positive' else -1
        
        user_a = item['userA']
        user_b = item['userB']
        
        # Get Image Paths
        path_a = self._get_image_path(pair_id, user_a.get('gender', 'male'))
        path_b = self._get_image_path(pair_id, user_b.get('gender', 'female'))
        
        if not path_a or not path_b:
            print(f"Warning: Missing image for {pair_id}. path_a={path_a}, path_b={path_b}")
            # Return dummy if failed (should not happen based on analysis)
            # Create black image
            img_a = Image.new('RGB', (224, 224), 'black')
            img_b = Image.new('RGB', (224, 224), 'black')
        else:
            try:
                img_a = Image.open(path_a).convert('RGB')
                img_b = Image.open(path_b).convert('RGB')
            except Exception as e:
                print(f"Error opening image for {pair_id}: {e}")
                img_a = Image.new('RGB', (224, 224), 'black')
                img_b = Image.new('RGB', (224, 224), 'black')
        
        # Transform
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            
        # Text
        text_a = self._make_text_description(user_a)
        text_b = self._make_text_description(user_b)
        
        return img_a, text_a, img_b, text_b, torch.tensor(label, dtype=torch.float)

if __name__ == "__main__":
    # Test block
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    dataset = MatchingDataset(
        jsonl_path=r"c:\Users\kimgh\Downloads\data\dataset.jsonl",
        image_root=r"c:\Users\kimgh\Downloads\data\images",
        transform=transform,
        limit=5,
        mode='train'
    )
    
    print("Testing __getitem__...")
    img_a, text_a, img_b, text_b, label = dataset[0]
    print(f"Label: {label}")
    print(f"Text A: {text_a}")
    print(f"Text B: {text_b}")
    print(f"Img A Shape: {img_a.shape}")
