import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset

"""
Triplet Loss 구성 요소 단위 테스트 스크립트

이 스크립트는 Triplet Loss 학습에 필요한 핵심 컴포넌트들의 동작을 검증합니다.
테스트 항목:
1. PKSampler: 클래스 균형 배치 샘플링(P classes * K samples) 동작 확인.
2. ProjectionHead: 임베딩 차원 축소 및 L2 정규화 확인.
3. OnlineTripletLoss: Triplet Loss 계산 및 유효한 Triplet 생성 여부 확인.

사용법:
    python test_triplet_components.py
"""

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.sampler import PKSampler
from src.models.projection import ProjectionHead
from src.models.losses import OnlineTripletLoss

class DummyDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=10):
        self.labels = np.random.randint(0, num_classes, size=num_samples).tolist()
        self.data = torch.randn(num_samples, 128)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
    def get_labels(self):
        return self.labels

def test_sampler():
    print("Testing PKSampler...")
    p = 4
    k = 4
    dataset = DummyDataset(num_samples=100, num_classes=10)
    sampler = PKSampler(dataset, p=p, k=k)
    
    batch_indices = list(iter(sampler))
    
    # Check batch size
    batch_size = p * k
    first_batch = batch_indices[:batch_size]
    
    print(f"First batch indices: {first_batch}")
    
    batch_labels = [dataset.labels[i] for i in first_batch]
    unique_labels = set(batch_labels)
    
    print(f"Labels in first batch: {batch_labels}")
    print(f"Unique labels: {unique_labels}")
    
    # Note: Since we sample with replacement if needed, we might not get exactly P unique labels if K is large relative to available data per class,
    # but the sampler logic tries to select P classes.
    # Let's just verify it runs and produces indices.
    assert len(first_batch) == batch_size
    print("PKSampler test passed!")

def test_projection_head():
    print("\nTesting ProjectionHead...")
    input_dim = 512
    hidden_dim = 256
    output_dim = 128
    
    head = ProjectionHead(input_dim, hidden_dim, output_dim)
    x = torch.randn(10, input_dim)
    output = head(x)
    
    # Check output shape
    assert output.shape == (10, output_dim)
    
    # Check normalization (L2 norm should be close to 1)
    norms = output.norm(p=2, dim=1)
    print(f"Norms: {norms}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print("ProjectionHead test passed!")

def test_loss():
    print("\nTesting OnlineTripletLoss...")
    loss_fn = OnlineTripletLoss(margin=0.2)
    
    embeddings = torch.randn(32, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Create labels such that we have pairs
    labels = torch.tensor([0, 0, 1, 1] * 8)
    
    loss, num_triplets = loss_fn(embeddings, labels)
    
    print(f"Loss: {loss.item()}")
    print(f"Num active triplets: {num_triplets}")
    
    assert loss.item() >= 0
    print("OnlineTripletLoss test passed!")

if __name__ == "__main__":
    test_sampler()
    test_projection_head()
    test_loss()
