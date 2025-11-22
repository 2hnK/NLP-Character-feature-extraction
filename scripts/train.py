"""
Qwen3-VL 모델 학습 스크립트 (Triplet Loss 적용)

이 스크립트는 AWS S3 데이터를 사용하여 모델을 학습시킵니다.
주요 기능:
1. S3 데이터셋 로드 (JSONL 메타데이터 기반)
2. PKSampler를 사용한 Balanced Batch 구성
3. Projection Head 및 Online Triplet Loss 적용
4. 학습 및 검증 루프 실행 (Recall@K, t-SNE 포함)
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.manifold import TSNE

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

def validate(model, projection_head, val_loader, criterion, device, epoch, output_dir, classes):
    """
    검증 함수: Loss 계산, Recall@K 측정, t-SNE 시각화
    """
    model.eval()
    projection_head.eval()
    
    val_loss = 0.0
    total_triplets = 0
    
    all_embeddings = []
    all_labels = []
    
    logger.info("Running Validation...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if len(batch) == 3:
                batch_images, _, batch_labels = batch
            else:
                batch_images, batch_labels = batch
                
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward
            features = model.forward(batch_images)
            embeddings = projection_head(features)
            
            # Loss Calculation
            loss, num_triplets = criterion(embeddings, batch_labels)
            
            val_loss += loss.item()
            total_triplets += num_triplets
            
            # Collect for metrics
            all_embeddings.append(embeddings.cpu())
            all_labels.append(batch_labels.cpu())
            
    avg_val_loss = val_loss / len(val_loader)
    avg_triplets = total_triplets / len(val_loader)
    
    # Concatenate all
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    
    # 1. Calculate Recall@K using AccuracyCalculator
    calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r"), k=5)
    metrics = calculator.get_accuracy(
        all_embeddings, 
        all_labels,
        embeddings_and_labels_are_gpu=False
    )
    
    r_at_1 = metrics["precision_at_1"]
    map_at_r = metrics["mean_average_precision_at_r"]
    
    logger.info(f"Validation Results - Epoch {epoch+1}:")
    logger.info(f"  Avg Loss: {avg_val_loss:.4f}")
    logger.info(f"  Recall@1: {r_at_1:.4f}")
    logger.info(f"  MAP@R: {map_at_r:.4f}")
    
    # 2. t-SNE Visualization (Every 5 epochs or first epoch)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        logger.info("Generating t-SNE plot...")
        try:
            # Use a subset if too large
            if len(all_embeddings) > 2000:
                indices = np.random.choice(len(all_embeddings), 2000, replace=False)
                tsne_embeddings = all_embeddings[indices].numpy()
                tsne_labels = all_labels[indices].numpy()
            else:
                tsne_embeddings = all_embeddings.numpy()
                tsne_labels = all_labels.numpy()
                
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_embeddings)-1))
            embeddings_2d = tsne.fit_transform(tsne_embeddings)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1], 
                c=tsne_labels, 
                cmap='tab10', 
                alpha=0.6
            )
            plt.colorbar(scatter)
            plt.title(f"t-SNE Visualization (Epoch {epoch+1})")
            
            # Add legend if classes are available
            if classes:
                # Create a legend with class names
                handles, _ = scatter.legend_elements(prop="colors")
                # Ensure we don't exceed the number of handles or classes
                num_classes_to_show = min(len(handles), len(classes))
                plt.legend(handles[:num_classes_to_show], classes[:num_classes_to_show], title="Styles")

            tsne_path = os.path.join(output_dir, f"tsne_epoch_{epoch+1}.png")
            plt.savefig(tsne_path)
            plt.close()
            logger.info(f"t-SNE plot saved to {tsne_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate t-SNE: {e}")

    return avg_val_loss, r_at_1

def train(args):
    # 1. 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 2. 데이터셋 준비
    logger.info("Initializing Training Dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    # Train Dataset
    train_dataset = S3Dataset(
        bucket_name=args.bucket_name,
        jsonl_path=args.jsonl_path,
        prefix=args.prefix,
        transform=transform,
        cache_dir=args.cache_dir
    )
    logger.info(f"Training Dataset size: {len(train_dataset)}")
    
    # Validation Dataset
    if args.val_jsonl:
        logger.info("Initializing Validation Dataset...")
        val_dataset = S3Dataset(
            bucket_name=args.val_bucket if args.val_bucket else args.bucket_name,
            jsonl_path=args.val_jsonl,
            prefix=args.val_prefix if args.val_prefix else args.prefix,
            transform=transform,
            cache_dir=args.cache_dir,
            label_mapping=train_dataset.label_mapping # Share label mapping!
        )
        logger.info(f"Validation Dataset size: {len(val_dataset)}")
        
        # Validation DataLoader (No PKSampler, Shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        val_dataset = None
        val_loader = None
        logger.warning("No validation dataset provided. Skipping validation.")

    # PKSampler 설정 (Train only)
    batch_size = args.p * args.k
    logger.info(f"Train Batch Size: {batch_size} (P={args.p}, K={args.k})")
    
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
    params = list(projection_head.parameters())
    if not args.freeze_vision:
        params += list(backbone.parameters())
        
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    
    # Triplet Loss
    criterion = OnlineTripletLoss(margin=args.margin, type_of_triplets=args.miner_type)
    
    logger.info("Starting training loop...")
    
    best_recall = 0.0
    
    for epoch in range(args.epochs):
        # --- Training ---
        backbone.train() if not args.freeze_vision else backbone.eval()
        projection_head.train()
        
        train_loss = 0.0
        total_triplets = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            if len(batch) == 3:
                batch_images, _, batch_labels = batch
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
            
            pbar.set_postfix({'loss': loss.item(), 'triplets': num_triplets})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_triplets = total_triplets / len(train_loader)
        
        logger.info(f"Epoch {epoch+1} Train Summary: Loss = {avg_train_loss:.4f}, Avg Triplets = {avg_triplets:.1f}")
        
        # --- Validation ---
        if val_loader:
            val_loss, recall_1 = validate(
                backbone, 
                projection_head, 
                val_loader, 
                criterion, 
                device, 
                epoch, 
                args.output_dir,
                train_dataset.classes
            )
            
            # Save Best Model based on Recall@1
            if recall_1 > best_recall:
                best_recall = recall_1
                save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'backbone_state_dict': backbone.state_dict(),
                    'projection_head_state_dict': projection_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall_at_1': best_recall,
                }, save_path)
                logger.info(f"New Best Model Saved! (Recall@1: {best_recall:.4f})")

        # Regular Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, save_path)
            logger.info(f"Checkpoint saved to {save_path}")

    logger.info("Training complete!")

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL Model with Triplet Loss")
    
    # Train Data params
    parser.add_argument("--bucket_name", type=str, default="sometimes-ki-datasets", help="S3 Bucket Name")
    parser.add_argument("--prefix", type=str, default="dataset/qwen-vl-train-v1/images/", help="S3 Prefix for Train")
    parser.add_argument("--jsonl_path", type=str, default="train_processed.jsonl", help="Path to processed Train JSONL")
    
    # Validation Data params
    parser.add_argument("--val_bucket", type=str, default=None, help="S3 Bucket for Validation (default: same as train)")
    parser.add_argument("--val_prefix", type=str, default=None, help="S3 Prefix for Validation (default: same as train)")
    parser.add_argument("--val_jsonl", type=str, default=None, help="Path to Validation JSONL")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Validation batch size")
    
    parser.add_argument("--cache_dir", type=str, default="./s3_cache", help="Local cache directory")
    
    # Sampler params
    parser.add_argument("--p", type=int, default=8, help="Number of classes per batch")
    parser.add_argument("--k", type=int, default=4, help="Number of samples per class")
    
    # Model params
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Model name")
    parser.add_argument("--embedding_dim", type=int, default=1536, help="Backbone output dimension")
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
