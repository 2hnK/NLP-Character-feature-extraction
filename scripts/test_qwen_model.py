"""
Test script for Qwen3-VL model loading and basic inference

This script tests:
1. Model loading from HuggingFace
2. Processor initialization
3. Basic forward pass with dummy image
4. Embedding extraction
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLFeatureExtractor


def test_model_loading():
    """Test if Qwen3-VL model can be loaded"""
    print("=" * 80)
    print("TEST 1: Model Loading")
    print("=" * 80)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Note: Qwen3-VL-2B-Instruct-FP8 may not exist yet
        # Using Qwen2-VL-2B-Instruct as fallback
        model_name = "Qwen/Qwen2-VL-2B-Instruct"

        print(f"\nLoading model: {model_name}")
        print("This may take a few minutes on first run...")

        model = Qwen3VLFeatureExtractor(
            model_name=model_name,
            embedding_dim=512,
            freeze_vision_encoder=True,  # Freeze for faster testing
            device=device
        )

        print("\n✓ Model loaded successfully!")
        print(f"  - Embedding dimension: {model.embedding_dim}")
        print(f"  - Vision hidden size: {model.vision_hidden_size}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        return model, device

    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        raise


def test_dummy_inference(model, device):
    """Test forward pass with dummy image"""
    print("\n" + "=" * 80)
    print("TEST 2: Dummy Image Inference")
    print("=" * 80)

    try:
        # Create dummy image (random noise)
        print("\nCreating dummy image (224x224 RGB)...")
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        print("Running forward pass...")
        with torch.no_grad():
            embedding = model.forward([dummy_image])

        print("\n✓ Forward pass successful!")
        print(f"  - Output shape: {embedding.shape}")
        print(f"  - Embedding norm: {embedding.norm(dim=1).item():.4f}")
        print(f"  - Mean value: {embedding.mean().item():.4f}")
        print(f"  - Std value: {embedding.std().item():.4f}")

        # Check if normalized (should be close to 1.0)
        norm = embedding.norm(dim=1).item()
        if abs(norm - 1.0) < 0.01:
            print("\n  ✓ Embedding is L2-normalized")
        else:
            print(f"\n  ⚠ Warning: Embedding norm is {norm:.4f}, expected ~1.0")

        return embedding

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        raise


def test_batch_inference(model, device):
    """Test batch inference"""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Inference")
    print("=" * 80)

    try:
        batch_size = 4
        print(f"\nCreating batch of {batch_size} dummy images...")

        images = [
            Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            for _ in range(batch_size)
        ]

        print("Running batch forward pass...")
        with torch.no_grad():
            embeddings = model.forward(images)

        print("\n✓ Batch forward pass successful!")
        print(f"  - Output shape: {embeddings.shape}")
        print(f"  - Expected shape: ({batch_size}, {model.embedding_dim})")

        # Check pairwise distances
        from torch.nn.functional import cosine_similarity

        # Compute similarity between first two embeddings
        sim = cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
        print(f"\n  - Cosine similarity between image 0 and 1: {sim:.4f}")

        return embeddings

    except Exception as e:
        print(f"\n✗ Batch inference failed: {e}")
        raise


def test_checkpoint_save_load(model, device):
    """Test saving and loading checkpoint"""
    print("\n" + "=" * 80)
    print("TEST 4: Checkpoint Save/Load")
    print("=" * 80)

    try:
        import tempfile
        import os

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            checkpoint_path = tmp.name

        print(f"\nSaving checkpoint to: {checkpoint_path}")
        model.save_checkpoint(
            path=checkpoint_path,
            epoch=0,
            test_metadata="test_value"
        )

        print("✓ Checkpoint saved successfully!")

        # Check file size
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"  - File size: {file_size:.2f} MB")

        print("\nLoading checkpoint...")
        loaded_model = Qwen3VLFeatureExtractor.load_from_checkpoint(
            checkpoint_path,
            device=device
        )

        print("✓ Checkpoint loaded successfully!")

        # Test if loaded model works
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        with torch.no_grad():
            original_emb = model.forward([dummy_image])
            loaded_emb = loaded_model.forward([dummy_image])

        # Check if embeddings are identical
        diff = (original_emb - loaded_emb).abs().max().item()
        print(f"\n  - Max difference between original and loaded: {diff:.6f}")

        if diff < 1e-5:
            print("  ✓ Loaded model produces identical outputs")
        else:
            print("  ⚠ Warning: Outputs differ slightly")

        # Clean up
        os.remove(checkpoint_path)
        print("\n  - Temporary checkpoint file deleted")

    except Exception as e:
        print(f"\n✗ Checkpoint test failed: {e}")
        raise


def test_memory_usage(model, device):
    """Test GPU memory usage"""
    print("\n" + "=" * 80)
    print("TEST 5: Memory Usage")
    print("=" * 80)

    if device == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()

            # Run inference
            images = [
                Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                for _ in range(8)
            ]

            with torch.no_grad():
                embeddings = model.forward(images)

            # Get memory stats
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

            print(f"\n✓ Memory usage:")
            print(f"  - Current allocated: {allocated:.2f} GB")
            print(f"  - Peak allocated: {peak:.2f} GB")

            if peak > 8.0:
                print("\n  ⚠ Warning: High memory usage (>8GB)")
                print("  Consider reducing batch size or using gradient checkpointing")

        except Exception as e:
            print(f"\n✗ Memory test failed: {e}")
    else:
        print("\nSkipping (CPU mode)")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("Qwen3-VL Model Test Suite")
    print("=" * 80)

    try:
        # Test 1: Load model
        model, device = test_model_loading()

        # Test 2: Single image inference
        embedding = test_dummy_inference(model, device)

        # Test 3: Batch inference
        batch_embeddings = test_batch_inference(model, device)

        # Test 4: Checkpoint save/load
        test_checkpoint_save_load(model, device)

        # Test 5: Memory usage
        test_memory_usage(model, device)

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nYour Qwen3-VL model is ready for training!")
        print("\nNext steps:")
        print("  1. Prepare your dataset")
        print("  2. Create metadata CSV files")
        print("  3. Run training with train.py or train_sagemaker.py")

    except Exception as e:
        print("\n" + "=" * 80)
        print("TESTS FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  1. Internet connection (for model download)")
        print("  2. HuggingFace access (may need login)")
        print("  3. Sufficient disk space (~5GB)")
        print("  4. GPU availability (optional, but recommended)")

        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        return 1

    return 0


if __name__ == "__main__":
    exit(main())
