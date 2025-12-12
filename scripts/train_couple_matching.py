"""
커플 매칭 예측 모델 학습 스크립트

InfoNCE Loss + Semi-Hard Negative Mining을 사용하여
실제 커플 쌍을 가깝게, 비커플 쌍을 멀게 학습합니다.

사용법:
    python scripts/train_couple_matching.py --fold 0
    python scripts/train_couple_matching.py --fold 1
    ...
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLFeatureExtractor
from src.models.projection import ProjectionHead

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainConfig:
    """학습 설정"""
    # 데이터
    data_dir: str = os.path.expanduser("~/data/mutual-like-validations/images")
    splits_file: str = "couple_splits.json"
    
    # 체크포인트
    checkpoint_dir: str = "./couple_matching_checkpoints"
    pretrained_checkpoint: Optional[str] = None  # 새 모델에서 시작
    
    # 모델 설정
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    embedding_dim: int = 2048
    projection_hidden_dim: int = 1024
    projection_output_dim: int = 256
    
    # 학습 하이퍼파라미터
    batch_size: int = 48  # InfoNCE에 유리한 큰 배치
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    temperature: float = 0.07
    
    # 스케줄러
    warmup_epochs: int = 2
    
    # Early stopping
    patience: int = 5
    
    # 이미지
    image_size: int = 768
    
    # Mixed precision
    use_amp: bool = True


class ResizeLongestEdge:
    """이미지의 긴 변을 max_size로 리사이즈"""
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


class CoupleDataset(Dataset):
    """커플 데이터셋"""
    def __init__(self, couple_ids: List[int], data_dir: str, image_size: int = 768):
        self.couple_ids = couple_ids
        self.data_dir = Path(data_dir)
        self.transform = ResizeLongestEdge(max_size=image_size)
        
    def __len__(self):
        return len(self.couple_ids)
    
    def __getitem__(self, idx):
        couple_id = self.couple_ids[idx]
        couple_dir = self.data_dir / f"couple_{couple_id}"
        
        # 이미지 로드
        female_img = Image.open(couple_dir / "female.png").convert('RGB')
        male_img = Image.open(couple_dir / "male.png").convert('RGB')
        
        # 변환
        female_img = self.transform(female_img)
        male_img = self.transform(male_img)
        
        return {
            'couple_id': couple_id,
            'female_img': female_img,
            'male_img': male_img
        }


def collate_fn(batch):
    """커스텀 collate function - PIL 이미지 유지"""
    couple_ids = [item['couple_id'] for item in batch]
    female_imgs = [item['female_img'] for item in batch]
    male_imgs = [item['male_img'] for item in batch]
    
    return {
        'couple_ids': couple_ids,
        'female_imgs': female_imgs,
        'male_imgs': male_imgs
    }


class InfoNCELoss(nn.Module):
    """InfoNCE Loss for contrastive learning"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, female_embs: torch.Tensor, male_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            female_embs: [B, D] - 정규화된 female 임베딩
            male_embs: [B, D] - 정규화된 male 임베딩
            
        Returns:
            loss: InfoNCE loss (양방향 평균)
        """
        batch_size = female_embs.size(0)
        
        # 유사도 행렬: [B, B]
        # logits[i, j] = female_i · male_j / temperature
        logits = torch.matmul(female_embs, male_embs.T) / self.temperature
        
        # 정답 레이블: 대각선 (i번 female ↔ i번 male)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Female → Male 방향 loss
        loss_f2m = F.cross_entropy(logits, labels)
        
        # Male → Female 방향 loss
        loss_m2f = F.cross_entropy(logits.T, labels)
        
        # 양방향 평균
        loss = (loss_f2m + loss_m2f) / 2
        
        return loss


def load_model(config: TrainConfig, device: str):
    """모델 로드"""
    logger.info(f"Loading model: {config.model_name}")
    
    # Backbone 로드 (동결)
    backbone = Qwen3VLFeatureExtractor(
        model_name=config.model_name,
        embedding_dim=config.embedding_dim,
        freeze_vision_encoder=True,
        use_projection_head=False,
        device=device
    )
    
    # Projection Head
    projection_head = ProjectionHead(
        input_dim=config.embedding_dim,
        hidden_dim=config.projection_hidden_dim,
        output_dim=config.projection_output_dim
    ).to(device)
    
    # 사전학습된 가중치 로드 (선택적)
    if config.pretrained_checkpoint and os.path.exists(config.pretrained_checkpoint):
        logger.info(f"Loading pretrained checkpoint: {config.pretrained_checkpoint}")
        checkpoint = torch.load(config.pretrained_checkpoint, map_location=device)
        backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
    
    return backbone, projection_head


def compute_recall_at_k(female_embs, male_embs, k_values=[1, 5, 10]):
    """Validation set에서 Recall@K 계산"""
    n = len(female_embs)
    similarity = np.dot(female_embs, male_embs.T)
    
    results = {}
    
    # Female → Male
    ranks_f2m = []
    for i in range(n):
        sorted_idx = np.argsort(-similarity[i])
        rank = np.where(sorted_idx == i)[0][0]
        ranks_f2m.append(rank)
    ranks_f2m = np.array(ranks_f2m)
    
    for k in k_values:
        results[f'recall@{k}'] = np.mean(ranks_f2m < k)
    
    results['mrr'] = np.mean(1.0 / (ranks_f2m + 1))
    
    return results


def train_one_epoch(
    backbone, projection_head, dataloader, optimizer, criterion,
    scaler, device, config, epoch
):
    """한 에폭 학습"""
    projection_head.train()
    backbone.eval()  # Backbone은 항상 eval (동결)
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        female_imgs = batch['female_imgs']
        male_imgs = batch['male_imgs']
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=config.use_amp):
            # Forward pass
            with torch.no_grad():
                female_features = backbone.forward(female_imgs)
                male_features = backbone.forward(male_imgs)
            
            female_embs = projection_head(female_features)
            male_embs = projection_head(male_features)
            
            # Loss 계산
            loss = criterion(female_embs, male_embs)
        
        # Backward
        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 메모리 정리
        if num_batches % 20 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches


@torch.no_grad()
def validate(backbone, projection_head, dataloader, criterion, device, config):
    """검증"""
    projection_head.eval()
    backbone.eval()
    
    total_loss = 0
    num_batches = 0
    all_female_embs = []
    all_male_embs = []
    
    for batch in tqdm(dataloader, desc="Validating"):
        female_imgs = batch['female_imgs']
        male_imgs = batch['male_imgs']
        
        with autocast('cuda', enabled=config.use_amp):
            female_features = backbone.forward(female_imgs)
            male_features = backbone.forward(male_imgs)
            
            female_embs = projection_head(female_features)
            male_embs = projection_head(male_features)
            
            loss = criterion(female_embs, male_embs)
        
        total_loss += loss.item()
        num_batches += 1
        
        all_female_embs.append(female_embs.cpu().numpy())
        all_male_embs.append(male_embs.cpu().numpy())
    
    # Recall 계산
    all_female_embs = np.vstack(all_female_embs)
    all_male_embs = np.vstack(all_male_embs)
    recall_metrics = compute_recall_at_k(all_female_embs, all_male_embs)
    
    return total_loss / num_batches, recall_metrics


def main():
    parser = argparse.ArgumentParser(description="Train couple matching model")
    parser.add_argument("--fold", type=int, required=True, help="Fold number (0-4)")
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pretrained checkpoint to load (optional)")
    args = parser.parse_args()
    
    config = TrainConfig()
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    
    # 체크포인트 옵션
    if args.checkpoint:
        config.pretrained_checkpoint = args.checkpoint
        logger.info(f"📥 Loading checkpoint: {args.checkpoint}")
    else:
        logger.info("🆕 Training from scratch (no pretrained weights)")
    
    # 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 분할 데이터 로드
    with open(config.splits_file, 'r') as f:
        splits = json.load(f)
    
    fold_data = splits['folds'][args.fold]
    train_ids = fold_data['train']
    valid_ids = fold_data['valid']
    
    logger.info(f"Fold {args.fold}: Train {len(train_ids)}, Valid {len(valid_ids)}")
    
    # 데이터셋/로더
    train_dataset = CoupleDataset(train_ids, config.data_dir, config.image_size)
    valid_dataset = CoupleDataset(valid_ids, config.data_dir, config.image_size)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, collate_fn=collate_fn,
        pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 모델
    backbone, projection_head = load_model(config, device)
    
    # Loss, Optimizer, Scheduler
    criterion = InfoNCELoss(temperature=config.temperature)
    optimizer = torch.optim.AdamW(
        projection_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=config.use_amp)
    
    # 체크포인트 디렉토리
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 학습 루프
    best_recall = 0
    patience_counter = 0
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            backbone, projection_head, train_loader, optimizer, criterion,
            scaler, device, config, epoch
        )
        
        # Validate
        valid_loss, recall_metrics = validate(
            backbone, projection_head, valid_loader, criterion, device, config
        )
        
        scheduler.step()
        
        # 로깅
        logger.info(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
            f"Valid Loss={valid_loss:.4f}, "
            f"R@1={recall_metrics['recall@1']*100:.2f}%, "
            f"R@5={recall_metrics['recall@5']*100:.2f}%, "
            f"MRR={recall_metrics['mrr']:.4f}"
        )
        
        # Best model 저장
        current_recall = recall_metrics['recall@1']
        if current_recall > best_recall:
            best_recall = current_recall
            patience_counter = 0
            
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"best_model_fold{args.fold}.pth"
            )
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'backbone_state_dict': backbone.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall': best_recall,
                'metrics': recall_metrics,
                'config': config.__dict__
            }, checkpoint_path)
            logger.info(f"✅ Best model saved: R@1={best_recall*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    print("\n" + "=" * 60)
    print(f"🏆 Fold {args.fold} 완료!")
    print(f"Best Recall@1: {best_recall*100:.2f}%")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
