import sys
import os
import json
from unittest.mock import MagicMock

"""
S3 데이터셋 로딩 검증 스크립트 (Mocking)

이 스크립트는 실제 AWS 연결 없이 `S3Dataset` 클래스의 로직을 검증합니다.
주요 기능:
1. `boto3`를 Mocking하여 S3 의존성 제거.
2. JSONL 파일에서 생성된 S3 키가 예상 패턴과 일치하는지 확인.
3. Train 및 Valid 데이터셋의 경로 생성 로직 테스트.

사용법:
    python verify_s3_dataset_loading.py
"""

# Mock boto3 before importing S3Dataset because it might not be installed in this env
sys.modules["boto3"] = MagicMock()

# Add project root to path
sys.path.append(os.getcwd())

from src.data.s3_dataset import S3Dataset

class MockS3Dataset(S3Dataset):
    def _load_image(self, key):
        # Return dummy string instead of image to verify key
        return f"MOCK_IMAGE_FROM_{key}"

def verify_dataset(name, jsonl_path, prefix, expected_pattern):
    print(f"Testing {name} with {jsonl_path}...")
    
    # Mock boto3 client to avoid AWS calls during init
    with MagicMock() as mock_boto:
        dataset = MockS3Dataset(
            bucket_name="mock-bucket",
            jsonl_path=jsonl_path,
            prefix=prefix
        )
        # Inject mock client
        dataset.s3_client = mock_boto
        
        print(f"Loaded {len(dataset)} items.")
        
        # Check first 5 items
        for i in range(5):
            try:
                image, text, label = dataset[i]
                # image is our mock string containing the key
                key = image.replace("MOCK_IMAGE_FROM_", "")
                
                expected = expected_pattern.format(i=i)
                if key != expected:
                    print(f"❌ Mismatch at index {i}: Expected '{expected}', got '{key}'")
                else:
                    print(f"✅ Index {i}: Key matches '{key}'")
            except Exception as e:
                print(f"❌ Error at index {i}: {e}")

if __name__ == "__main__":
    # Verify Train Data
    # Expected: dataset/qwen-vl-train-v1/images/aug_00000.jpg
    verify_dataset(
        "Train Data",
        "train_aug_final.jsonl",
        "dataset/qwen-vl-train-v1/images",
        "dataset/qwen-vl-train-v1/images/aug_{i:05d}.jpg"
    )
    
    print("-" * 50)
    
    # Verify Validation Data
    # Expected: dataset/validation/images/val_00000.jpg
    verify_dataset(
        "Validation Data",
        "train_valid_final.jsonl",
        "dataset/validation/images",
        "dataset/validation/images/val_{i:05d}.jpg"
    )
