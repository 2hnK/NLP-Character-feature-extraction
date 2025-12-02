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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
import wandb
from PIL import Image

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

class ResizeLongestEdge:
    """
    Resizes the longest edge of the image to 'max_size' while maintaining aspect ratio.
    """
    def __init__(self, max_size: int, interpolation=Image.BICUBIC):
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.max_size / max(w, h)
        if scale >= 1:
            return img
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), self.interpolation)

    def __repr__(self):
        return f"ResizeLongestEdge(max_size={self.max_size}, interpolation={self.interpolation})"

def validate(model, projection_head, val_loader, criterion, device, epoch, output_dir, classes, writer=None, use_wandb=False):
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
                
            # # batch_images = batch_images.to(device)
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
    # k="max_bin_count"로 설정하여 MAP@R 계산 시 모든 관련 샘플을 고려하도록 함 (Warning 해결)
    calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r"), k="max_bin_count")
    metrics = calculator.get_accuracy(
        all_embeddings, 
        all_labels,
    )
    
    r_at_1 = metrics["precision_at_1"]
    map_at_r = metrics["mean_average_precision_at_r"]
    
    logger.info(f"Validation Results - Epoch {epoch+1}:")
    logger.info(f"  Avg Loss: {avg_val_loss:.4f}")
    logger.info(f"  Recall@1: {r_at_1:.4f}")

    logger.info(f"  MAP@R: {map_at_r:.4f}")
    
    # Log Validation Metrics
    if writer:
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/Recall_at_1", r_at_1, epoch)
        writer.add_scalar("Val/MAP_at_R", map_at_r, epoch)
        
    if use_wandb:
        wandb.log({
            "val/loss": avg_val_loss,
            "val/recall_at_1": r_at_1,
            "val/map_at_r": map_at_r,
            "epoch": epoch
        })
    
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
            
            # Log t-SNE image
            if writer:
                # Convert plot to image for TensorBoard
                # Re-open the saved image to log it (simplest way without buffer manipulation)
                import PIL.Image
                image = PIL.Image.open(tsne_path)
                image = transforms.ToTensor()(image)
                writer.add_image("Val/t-SNE", image, epoch)
                
            if use_wandb:
                wandb.log({"val/t-sne": wandb.Image(tsne_path, caption=f"Epoch {epoch+1}")})
            
        except Exception as e:
            logger.error(f"Failed to generate t-SNE: {e}")

    return avg_val_loss, r_at_1

def train(args):
    # 1. 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"  - CUDA Version: {torch.version.cuda}")
        
        # Set precision
        if args.use_bfloat16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("  - Using bfloat16 precision")
        else:
            dtype = torch.float16
            logger.info("  - Using float16 precision")
    else:
        dtype = torch.float32
        logger.info("  - Using float32 precision (CPU)")

    # 2. 데이터셋 준비
    logger.info("Initializing Training Dataset...")
    
    # Resize Logic
    transform = ResizeLongestEdge(max_size=args.image_size)
    logger.info(f"Using Transform: {transform}")
    
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
        
        def collate_fn(batch):
            images = [item[0] for item in batch]
            # item[1] is text_input (ignored for now)
            labels = torch.stack([item[2] for item in batch])
            return images, labels

        # Validation DataLoader (No PKSampler, Shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    else:
        val_dataset = None
        val_loader = None
        logger.warning("No validation dataset provided. Skipping validation.")

    # PKSampler 설정 (Train only)
    batch_size = args.p * args.k
    logger.info(f"Train Batch Size: {batch_size} (P={args.p}, K={args.k})")
    
    train_sampler = PKSampler(train_dataset, p=args.p, k=args.k)
    
    def collate_fn(batch):
        images = [item[0] for item in batch]
        # item[1] is text_input (ignored for now)
        labels = torch.stack([item[2] for item in batch])
        return images, labels

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False
    )
    
    # 3. 모델 초기화
    logger.info(f"Loading model: {args.model_name}")
    backbone = Qwen3VLFeatureExtractor(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        freeze_vision_encoder=args.freeze_vision,
        use_projection_head=False,
        device=device
    )
    
    # Explicitly freeze the entire backbone if requested
    # (Qwen3VLFeatureExtractor only freezes vision tower by default)
    if args.freeze_vision:
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone (Vision + LLM) frozen.")
    
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
        # Load state dicts
        # Handle case where backbone might be wrapped or have different keys
        # strict=False allows loading checkpoints that might have extra keys (e.g. internal projection_head)
        backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    

    
    # --- Monitoring Setup ---
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))
    
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args.__dict__,
            name=f"{args.model_name.split('/')[-1]}_e{args.epochs}_b{batch_size}"
        )
    
    logger.info("Starting training loop...")
    
    best_recall = 0.0
    
    for epoch in range(start_epoch, args.epochs):
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
                
            # # batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward & Loss Calculation with Autocast
            with torch.amp.autocast('cuda', dtype=dtype, enabled=(device=="cuda")):
                with torch.no_grad():
                    features = backbone.forward(batch_images)
                embeddings = projection_head(features)
                loss, num_triplets = criterion(embeddings, batch_labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_triplets += num_triplets
            
            pbar.set_postfix({'loss': loss.item(), 'triplets': num_triplets})
            
            # Log Train Metrics (Step-wise)
            global_step = epoch * len(train_loader) + pbar.n
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/Triplets", num_triplets, global_step)
            
            if args.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/triplets": num_triplets,
                    "global_step": global_step
                })
            
        avg_train_loss = train_loss / len(train_loader)
        avg_triplets = total_triplets / len(train_loader)
        
        logger.info(f"Epoch {epoch+1} Train Summary: Loss = {avg_train_loss:.4f}, Avg Triplets = {avg_triplets:.1f}")
        
        # --- Safety Checkpoint (Save BEFORE Validation) ---
        latest_save_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'backbone_state_dict': backbone.state_dict(),
            'projection_head_state_dict': projection_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, latest_save_path)
        logger.info(f"Safety checkpoint saved to {latest_save_path}")
        
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

                train_dataset.classes,
                writer=writer,
                use_wandb=args.use_wandb
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
    writer.close()
    if args.use_wandb:
        wandb.finish()

# ==========================================
# 설정 (Configuration)
# ==========================================
@dataclass
class Config:
    # 1. Train Data params
    bucket_name: str = "sometimes-ki-datasets"
    prefix: str = "dataset/qwen-vl-train-v1/images/"
    jsonl_path: str = "dataset/qwen-vl-train-v1/train_aug_relabeled.jsonl"
    
    # 2. Validation Data params
    val_bucket: Optional[str] = None  # None이면 bucket_name과 동일하게 사용
    val_prefix: Optional[str] = "dataset/validation/images/"  # 검증 이미지 경로
    val_jsonl: Optional[str] = "dataset/validation/validation_relabeled.jsonl"   # 검증 메타데이터 경로
    val_batch_size: int = 32
    
    cache_dir: str = "./s3_cache"
    
    # 3. Sampler params (Batch Size = p * k)
    p: int = 5  # 클래스(스타일) 개수 (최대 5개)
    k: int = 4  # 클래스당 샘플 개수 (Batch Size = 20)
    
    # 4. Model params
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    embedding_dim: int = 1536
    projection_hidden_dim: int = 1024
    projection_output_dim: int = 256
    freeze_vision: bool = True
    
    # 5. Training params
    epochs: int = 30
    learning_rate: float = 1e-4
    image_size: int = 768  # Reduced from 1024 to save memory
    num_workers: int = 4
    margin: float = 0.3
    miner_type: str = "semihard"  # choices=["all", "hard", "semihard", "easy"]
    use_bfloat16: bool = True
    
    # 6. Output params
    output_dir: str = "./checkpoints"

    save_interval: int = 5
    
    # 7. Monitoring params
    use_wandb: bool = False
    wandb_project: str = "fashion-style-triplet"
    wandb_entity: Optional[str] = None
    
    # 8. Resume params
    resume_from_checkpoint: Optional[str] = "./checkpoints/latest_checkpoint.pth"

def main():
    # 설정값 인스턴스 생성
    # 위 Config 클래스의 값을 수정하여 설정을 변경하세요.
    args = Config()
    
    print(f"Training Configuration:")
    print(f"  - Bucket: {args.bucket_name}")
    print(f"  - Train Prefix: {args.prefix}")
    print(f"  - Output Dir: {args.output_dir}")
    print(f"  - Epochs: {args.epochs}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()
