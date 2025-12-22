"""
커플 매칭 모델 학습 스크립트

구조:
  - Backbone: Qwen3-VL (동결)
  - Projection: 성별별 독립 헤드 (Female/Male)
  - Loss: InfoNCE (양방향)

사용법:
    python scripts/prepare_splits.py  # 먼저 분할 파일 생성
    python scripts/train_couple_matching.py
    python scripts/train_couple_matching.py --splits couple_splits.json --epochs 50
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLFeatureExtractor
from src.models.projection import GenderSpecificProjection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    pretrained_checkpoint: Optional[str] = None
    
    # 모델
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    embedding_dim: int = 2048
    projection_hidden_dim: int = 1024
    projection_output_dim: int = 256
    
    # 학습 파라미터
    batch_size: int = 48
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    epochs: int = 30
    temperature: float = 0.1
    warmup_epochs: int = 2
    patience: int = 10
    
    # 이미지
    image_size: int = 768
    
    # AMP
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
    """커플 데이터셋 (손상된 이미지 자동 필터링)"""
    def __init__(self, couple_ids: List[int], data_dir: str, image_size: int = 768):
        self.data_dir = Path(data_dir)
        self.transform = ResizeLongestEdge(max_size=image_size)
        
        # 유효한 커플만 필터링
        self.valid_couples = []
        skipped = []
        for cid in couple_ids:
            couple_dir = self.data_dir / f"couple_{cid}"
            female_path = couple_dir / "female.png"
            male_path = couple_dir / "male.png"
            
            # 파일 존재 및 열기 가능 확인
            try:
                if female_path.exists() and male_path.exists():
                    Image.open(female_path).verify()
                    Image.open(male_path).verify()
                    self.valid_couples.append(cid)
                else:
                    skipped.append(cid)
            except Exception:
                skipped.append(cid)
        
        if skipped:
            logger.warning(f"Skipped {len(skipped)} corrupted couples: {skipped[:10]}...")
        logger.info(f"Valid couples: {len(self.valid_couples)}/{len(couple_ids)}")
        
    def __len__(self):
        return len(self.valid_couples)
    
    def __getitem__(self, idx):
        couple_id = self.valid_couples[idx]
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
    """모델 로드
    
    구조:
    - Backbone: Qwen3-VL (동결)
    - Projection: 성별별 독립 헤드 (Female/Male)
    """
    logger.info(f"Loading model: {config.model_name}")
    
    # Backbone 로드 (동결)
    backbone = Qwen3VLFeatureExtractor(
        model_name=config.model_name,
        embedding_dim=config.embedding_dim,
        freeze_vision_encoder=True,
        use_projection_head=False,
        device=device
    )
    
    # 성별별 Projection Head
    projection = GenderSpecificProjection(
        input_dim=config.embedding_dim,
        hidden_dim=config.projection_hidden_dim,
        output_dim=config.projection_output_dim
    ).to(device)
    
    logger.info("✅ Gender-specific projection heads initialized (Female/Male)")
    
    # 사전학습된 가중치 로드 (선택적)
    if config.pretrained_checkpoint and os.path.exists(config.pretrained_checkpoint):
        logger.info(f"Loading pretrained checkpoint: {config.pretrained_checkpoint}")
        checkpoint = torch.load(config.pretrained_checkpoint, map_location=device)
        backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
        projection.load_state_dict(checkpoint['projection_state_dict'])
    
    return backbone, projection


def compute_metrics(female_embs, male_embs, k_values=[5, 10, 20, 50]):
    """
    Validation set에서 평가 지표 계산
    
    측정 지표:
    - Hit@K (K=5,10,20,50): 정답이 Top-K에 포함되면 1, 아니면 0
    - Recall@K (K=5,10,20,50): Hit@K와 동일 (정답이 1개일 때)
    - MRR (Mean Reciprocal Rank): 정답 순위의 역수 평균
    - Accuracy: 정답이 1위인 비율 (Hit@1)
    
    평가 방법:
    - 유사도 행렬 (N x N) 계산: similarity[i, j] = female_i · male_j
    - Female→Male: 각 여성에 대해 모든 남성 중 정답 파트너의 순위 계산
    - Male→Female: 각 남성에 대해 모든 여성 중 정답 파트너의 순위 계산
    
    Args:
        female_embs: (N, D) 정규화된 female 임베딩
        male_embs: (N, D) 정규화된 male 임베딩
        k_values: 계산할 K 값 리스트 (기본: [5, 10, 20, 50])
        
    Returns:
        results: 양방향 평균 지표
    """
    n = len(female_embs)
    
    # 유사도 행렬: (N, N)
    # similarity[i, j] = i번째 female과 j번째 male 간의 코사인 유사도
    similarity = np.dot(female_embs, male_embs.T)
    
    results = {}
    
    # === Female → Male 방향 ===
    # 각 female i에 대해, 모든 male 중 정답 male i가 몇 위인지 계산
    ranks_f2m = []
    for i in range(n):
        scores = similarity[i]  # i번째 female의 모든 male에 대한 유사도
        sorted_idx = np.argsort(-scores)  # 유사도 높은 순 정렬
        rank = np.where(sorted_idx == i)[0][0]  # 정답(i번째 male)의 순위
        ranks_f2m.append(rank)
    ranks_f2m = np.array(ranks_f2m)
    
    # === Male → Female 방향 ===
    # 각 male i에 대해, 모든 female 중 정답 female i가 몇 위인지 계산
    ranks_m2f = []
    for i in range(n):
        scores = similarity[:, i]  # i번째 male에 대한 모든 female의 유사도
        sorted_idx = np.argsort(-scores)  # 유사도 높은 순 정렬
        rank = np.where(sorted_idx == i)[0][0]  # 정답(i번째 female)의 순위
        ranks_m2f.append(rank)
    ranks_m2f = np.array(ranks_m2f)
    
    # === Accuracy (Hit@1) ===
    # 정답이 1위인 비율
    acc_f2m = np.mean(ranks_f2m == 0)
    acc_m2f = np.mean(ranks_m2f == 0)
    results['accuracy'] = (acc_f2m + acc_m2f) / 2
    results['f2m_accuracy'] = acc_f2m
    results['m2f_accuracy'] = acc_m2f
    
    # === Hit@K / Recall@K (양방향 평균) ===
    # 정답이 1개일 때 Hit@K = Recall@K
    for k in k_values:
        hit_f2m = np.mean(ranks_f2m < k)
        hit_m2f = np.mean(ranks_m2f < k)
        
        # Hit@K
        results[f'hit@{k}'] = (hit_f2m + hit_m2f) / 2
        results[f'f2m_hit@{k}'] = hit_f2m
        results[f'm2f_hit@{k}'] = hit_m2f
        
        # Recall@K (Hit@K와 동일, 호환성 위해 둘 다 저장)
        results[f'recall@{k}'] = results[f'hit@{k}']
        results[f'f2m_recall@{k}'] = hit_f2m
        results[f'm2f_recall@{k}'] = hit_m2f
    
    # === MRR (Mean Reciprocal Rank) ===
    mrr_f2m = np.mean(1.0 / (ranks_f2m + 1))
    mrr_m2f = np.mean(1.0 / (ranks_m2f + 1))
    results['mrr'] = (mrr_f2m + mrr_m2f) / 2
    results['f2m_mrr'] = mrr_f2m
    results['m2f_mrr'] = mrr_m2f
    
    # === 추가 통계 ===
    results['mean_rank'] = (np.mean(ranks_f2m) + np.mean(ranks_m2f)) / 2
    results['median_rank'] = (np.median(ranks_f2m) + np.median(ranks_m2f)) / 2
    
    return results


def train_one_epoch(
    backbone, projection, dataloader, optimizer, criterion,
    scaler, device, config, epoch
):
    """
    한 에폭 학습
    
    구조:
    - 여성 이미지 → 여성 프롬프트 → Backbone → Female Projection Head
    - 남성 이미지 → 남성 프롬프트 → Backbone → Male Projection Head
    """
    projection.train()
    backbone.eval()  # Backbone은 항상 eval (동결)
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        female_imgs = batch['female_imgs']
        male_imgs = batch['male_imgs']
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=config.use_amp):
            # Forward pass (Backbone은 gradient 계산 안 함)
            with torch.no_grad():
                female_features = backbone.forward(female_imgs)
                male_features = backbone.forward(male_imgs)
            
            # 성별별 Projection Head 사용
            female_embs, male_embs = projection(female_features, male_features)
            
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
def validate(backbone, projection, dataloader, criterion, device, config):
    """검증"""
    projection.eval()
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
            
            # 성별별 Projection Head 사용
            female_embs, male_embs = projection(female_features, male_features)
            
            loss = criterion(female_embs, male_embs)
        
        total_loss += loss.item()
        num_batches += 1
        
        all_female_embs.append(female_embs.cpu().numpy())
        all_male_embs.append(male_embs.cpu().numpy())
    
    # 평가 지표 계산
    all_female_embs = np.vstack(all_female_embs)
    all_male_embs = np.vstack(all_male_embs)
    metrics = compute_metrics(all_female_embs, all_male_embs)
    
    return total_loss / num_batches, metrics


def main():
    parser = argparse.ArgumentParser(description="커플 매칭 모델 학습")
    parser.add_argument("--data-dir", type=str, 
                        default=os.path.expanduser("~/data/mutual-like-validations/images"))
    parser.add_argument("--splits", type=str, default="couple_splits.json",
                        help="분할 JSON 파일 (prepare_splits.py로 생성)")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="사전학습 체크포인트 경로")
    args = parser.parse_args()
    
    config = TrainConfig()
    config.data_dir = args.data_dir
    config.splits_file = args.splits
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    
    if args.checkpoint:
        config.pretrained_checkpoint = args.checkpoint
        logger.info(f"📥 체크포인트 로드: {args.checkpoint}")
    else:
        logger.info("🆕 새로운 학습 시작")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"디바이스: {device}")
    
    # 분할 파일 로드
    if not os.path.exists(config.splits_file):
        logger.error(f"분할 파일 없음: {config.splits_file}")
        logger.error("python scripts/prepare_splits.py 실행 필요")
        return 1
    
    with open(config.splits_file, 'r') as f:
        splits = json.load(f)
    
    train_ids = splits['train']
    valid_ids = splits['valid']
    
    logger.info(f"Train: {len(train_ids)}, Valid: {len(valid_ids)}")
    
    # 데이터셋
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
    backbone, projection = load_model(config, device)
    
    # 학습 설정
    criterion = InfoNCELoss(temperature=config.temperature)
    optimizer = torch.optim.AdamW(
        projection.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler('cuda', enabled=config.use_amp)
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 학습 루프
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            backbone, projection, train_loader, optimizer, criterion,
            scaler, device, config, epoch
        )
        
        valid_loss, metrics = validate(
            backbone, projection, valid_loader, criterion, device, config
        )
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch}: Train={train_loss:.4f}, Valid={valid_loss:.4f}, "
            f"Acc={metrics['accuracy']*100:.2f}%, H@10={metrics['hit@10']*100:.2f}%, MRR={metrics['mrr']:.4f}"
        )
        
        # Best 저장
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            patience_counter = 0
            
            checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'projection_state_dict': projection.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_acc,
                'metrics': metrics,
                'config': config.__dict__
            }, checkpoint_path)
            logger.info(f"✅ Best 저장: Acc={best_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping (epoch {epoch})")
                break
    
    print(f"\n{'='*50}")
    print(f"🏆 학습 완료! Best Accuracy: {best_acc*100:.2f}%")
    print(f"{'='*50}")
    
    return 0


if __name__ == "__main__":
    exit(main())
