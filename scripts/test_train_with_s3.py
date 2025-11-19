"""
Test script for training loop with S3 data loading
"""

import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLFeatureExtractor
from src.data.s3_dataset import S3Dataset

def test_s3_training_loop():
    print("=" * 80)
    print("TEST: S3 Data Integration & Training Loop")
    print("=" * 80)

    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # 2. Initialize Model
        print("\n1. Initializing Model...")
        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        model = Qwen3VLFeatureExtractor(
            model_name=model_name,
            embedding_dim=512,
            freeze_vision_encoder=True,
            device=device
        )
        print("✓ Model initialized")

        # 3. Initialize S3 Dataset
        print("\n2. Initializing S3 Dataset...")
        bucket_name = "sometimes-ki-datasets"
        prefix = "dataset/qwen-vl-train-v1/images/"
        
        # Define transforms (resize to model input size if needed, though Qwen handles variable sizes, 
        # it's good to have consistent batches for testing)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = S3Dataset(
            bucket_name=bucket_name,
            prefix=prefix,
            transform=None, # Qwen processor handles raw images usually, but let's see. 
                            # Actually Qwen3VLFeatureExtractor.forward expects PIL images or list of them.
                            # So we should NOT convert to Tensor here if we pass to model.forward directly 
                            # as implemented in test_qwen_model.py.
                            # Let's check test_qwen_model.py again. 
                            # It passes list of PIL images.
            limit=8, # Limit to 8 images for quick testing
            cache_dir="./s3_cache" # Cache for faster re-runs
        )
        print(f"✓ Dataset initialized with {len(dataset)} images")

        # 4. Create DataLoader
        print("\n3. Creating DataLoader...")
        # Custom collate function might be needed if we want to pass list of PIL images
        def collate_fn(batch):
            return batch # Returns list of PIL images

        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        print("✓ DataLoader created")

        # 5. Run Training Loop Step
        print("\n4. Running Training Step...")
        model.train()
        
        # Simple optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        for i, batch_images in enumerate(dataloader):
            print(f"  Batch {i+1}: {len(batch_images)} images")
            
            # Forward pass
            embeddings = model.forward(batch_images)
            print(f"    - Embeddings shape: {embeddings.shape}")
            
            # Dummy loss (e.g., maximize distance from origin or something simple)
            # In real training, we would have labels or pairs
            loss = embeddings.norm(dim=1).mean() 
            # Let's just try to minimize norm as a dummy objective to check backward pass
            
            print(f"    - Loss: {loss.item():.4f}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("    ✓ Backward pass & Step successful")
            
            if i >= 1: # Run only 2 batches
                break

        print("\n" + "=" * 80)
        print("TEST PASSED: S3 Data Loading & Training Loop Verified! ✓")
        print("=" * 80)
        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ✗")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_s3_training_loop())
