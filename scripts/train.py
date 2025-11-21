"""
Qwen3-VL 모델 학습 스크립트 (Triplet Loss 적용)

이 스크립트는 AWS S3 데이터를 사용하여 모델을 학습시킵니다.
주요 기능:
1. S3 데이터셋 로드 (JSONL 메타데이터 기반)
2. PKSampler를 사용한 Balanced Batch 구성
3. Projection Head 및 Online Triplet Loss 적용
4. 학습 및 검증 루프 실행
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLFeatureExtractor
from src.data.s3_dataset import S3Dataset
from src.data.sampler import PKSampler
from src.models.projection import ProjectionHead
from src.models.losses import OnlineTripletLoss

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def train(args):
    # 1. 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 2. 데이터셋 준비
    logger.info("Initializing S3 Dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    full_dataset = S3Dataset(
        bucket_name=args.bucket_name,
        jsonl_path=args.jsonl_path,
        prefix=args.prefix,
        transform=transform,
        cache_dir=args.cache_dir
    )
    
    # Train/Val 분할 (단순 Random Split은 Sampler와 충돌할 수 있으므로 주의 필요)
    # Triplet Loss 학습에서는 보통 전체 데이터를 사용하거나, 클래스별로 분할해야 함.
    # 여기서는 간단히 전체 데이터를 학습에 사용하고, 검증은 별도로 구성하지 않거나(혹은 동일 데이터로 Loss 확인),
    # 데이터셋 클래스 내부에서 분할 로직을 구현하는 것이 좋음.
    # 이번 구현에서는 전체 데이터를 학습에 사용합니다. (검증 데이터셋 분리는 추후 고도화 필요)
    train_dataset = full_dataset
    
    logger.info(f"Training Dataset size: {len(train_dataset)}")

    # PKSampler 설정
    # Batch Size = P * K
    batch_size = args.p * args.k
    logger.info(f"Batch Size: {batch_size} (P={args.p}, K={args.k})")
    
    train_sampler = PKSampler(train_dataset, p=args.p, k=args.k)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # 3. 모델 초기화
    logger.info(f"Loading model: {args.model_name}")
    backbone = Qwen3VLFeatureExtractor(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        freeze_vision_encoder=args.freeze_vision,
        device=device
    )
    
    projection_head = ProjectionHead(
        input_dim=args.embedding_dim,
        hidden_dim=args.projection_hidden_dim,
        output_dim=args.projection_output_dim
    ).to(device)
    
    # 4. Optimizer 및 Loss 설정
    # Backbone과 Projection Head 모두 학습 (Backbone이 Freeze되지 않았다면)
    params = list(projection_head.parameters())
    if not args.freeze_vision:
        params += list(backbone.parameters())
        
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    
    criterion = OnlineTripletLoss(margin=args.margin, type_of_triplets=args.miner_type)
    
    logger.info("Starting training loop...")
    
    for epoch in range(args.epochs):
        # --- Training ---
        backbone.train() if not args.freeze_vision else backbone.eval()
        projection_head.train()
        
        train_loss = 0.0
        total_triplets = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # S3Dataset now returns (image, text, label)
            # DataLoader collates them into (batch_images, batch_texts, batch_labels)
            if len(batch) == 3:
                batch_images, batch_texts, batch_labels = batch
            else:
                batch_images, batch_labels = batch
                
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            features = backbone.forward(batch_images)
            embeddings = projection_head(features)
            
            # Loss Calculation
            loss, num_triplets = criterion(embeddings, batch_labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_triplets += num_triplets
            
            pbar.set_postfix({'loss': loss.item(), 'active_triplets': num_triplets})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_triplets = total_triplets / len(train_loader)
        
        logger.info(f"Epoch {epoch+1}: Avg Loss = {avg_train_loss:.4f}, Avg Active Triplets = {avg_triplets:.1f}")
        
        # Checkpoint 저장
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(), # QwenFeatureExtractor의 save 메서드 대신 직접 저장
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, save_path)
            logger.info(f"Checkpoint saved to {save_path}")

    logger.info("Training complete!")

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL Model with Triplet Loss")
    
    # Data params
    parser.add_argument("--bucket_name", type=str, default="sometimes-ki-datasets", help="S3 Bucket Name")
    parser.add_argument("--prefix", type=str, default="dataset/qwen-vl-train-v1/images/", help="S3 Prefix")
    parser.add_argument("--jsonl_path", type=str, default="train_processed.jsonl", help="Path to processed JSONL file")
    parser.add_argument("--cache_dir", type=str, default="./s3_cache", help="Local cache directory")
    
    # Sampler params
    parser.add_argument("--p", type=int, default=8, help="Number of classes per batch")
    parser.add_argument("--k", type=int, default=4, help="Number of samples per class")
    
    # Model params
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Model name")
    parser.add_argument("--embedding_dim", type=int, default=1536, help="Backbone output dimension (Qwen2-VL usually 1536 or similar, check specific model)")
    parser.add_argument("--projection_hidden_dim", type=int, default=1024, help="Projection head hidden dimension")
    parser.add_argument("--projection_output_dim", type=int, default=128, help="Final embedding dimension")
    parser.add_argument("--freeze_vision", action="store_true", help="Freeze vision encoder")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=224, help="Image input size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet loss margin")
    parser.add_argument("--miner_type", type=str, default="semihard", choices=["all", "hard", "semihard", "easy"], help="Miner type")
    
    # Output params
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=5, help="Checkpoint save interval")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()
