"""
Qwen3-VL 베이스라인 모델 평가 스크립트

사전 학습된 Qwen3-VL 모델(프로젝션 헤드 없이)의 성능을 측정합니다.
학습된 모델과 비교하기 위한 베이스라인 성능을 제공합니다.

측정 지표:
- Hit@K (K=5, 10, 20, 50)
- MRR (Mean Reciprocal Rank)
- Accuracy

사용법:
    python scripts/evaluate_baseline.py
    python scripts/evaluate_baseline.py --data-dir /path/to/data --pooling-mode eos
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLFeatureExtractor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class EvalConfig:
    """베이스라인 평가 설정"""
    # 데이터
    data_dir: str = os.path.expanduser("~/data/mutual-like-validations/images")
    
    # 모델 설정
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    embedding_dim: int = 2048  # Qwen3-VL hidden dimension
    pooling_mode: str = "mean"  # 'mean' or 'eos'
    
    # 평가 설정
    batch_size: int = 16
    image_size: int = 768
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # 출력
    output_dir: str = "./baseline_evaluation_results"


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


class CoupleDataset:
    """커플 데이터셋"""
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
            logger.warning(f"Skipped {len(skipped)} corrupted/missing couples: {skipped[:10]}...")
        logger.info(f"Valid couples: {len(self.valid_couples)}/{len(couple_ids)}")

    def __len__(self):
        return len(self.valid_couples)
    
    def get_couple(self, idx):
        """특정 인덱스의 커플 이미지 로드"""
        couple_id = self.valid_couples[idx]
        couple_dir = self.data_dir / f"couple_{couple_id}"
        
        female_img = Image.open(couple_dir / "female.png").convert('RGB')
        male_img = Image.open(couple_dir / "male.png").convert('RGB')
        
        female_img = self.transform(female_img)
        male_img = self.transform(male_img)
        
        return couple_id, female_img, male_img


def compute_recall_and_mrr(female_embs: np.ndarray, male_embs: np.ndarray, 
                            k_values: List[int] = [1, 5, 10, 20, 50]):
    """
    양방향 Recall@K 및 MRR 계산
    
    Args:
        female_embs: (N, D) - 정규화된 female 임베딩
        male_embs: (N, D) - 정규화된 male 임베딩
        k_values: 계산할 K 값 리스트
        
    Returns:
        metrics: 딕셔너리 (f2m, m2f, avg 각각의 recall@k, mrr)
    """
    n = len(female_embs)
    
    # 유사도 행렬: (N, N)
    similarity = np.dot(female_embs, male_embs.T)
    
    metrics = {
        'f2m': {},  # Female → Male
        'm2f': {},  # Male → Female
        'avg': {}   # 양방향 평균
    }
    
    # === Female → Male ===
    ranks_f2m = []
    for i in range(n):
        sorted_idx = np.argsort(-similarity[i])
        rank = np.where(sorted_idx == i)[0][0]
        ranks_f2m.append(rank)
    ranks_f2m = np.array(ranks_f2m)
    
    for k in k_values:
        metrics['f2m'][f'recall@{k}'] = np.mean(ranks_f2m < k)
    metrics['f2m']['mrr'] = np.mean(1.0 / (ranks_f2m + 1))
    
    # === Male → Female ===
    ranks_m2f = []
    for i in range(n):
        sorted_idx = np.argsort(-similarity[:, i])
        rank = np.where(sorted_idx == i)[0][0]
        ranks_m2f.append(rank)
    ranks_m2f = np.array(ranks_m2f)
    
    for k in k_values:
        metrics['m2f'][f'recall@{k}'] = np.mean(ranks_m2f < k)
    metrics['m2f']['mrr'] = np.mean(1.0 / (ranks_m2f + 1))
    
    # === 양방향 평균 ===
    for k in k_values:
        metrics['avg'][f'recall@{k}'] = (
            metrics['f2m'][f'recall@{k}'] + metrics['m2f'][f'recall@{k}']
        ) / 2
    metrics['avg']['mrr'] = (metrics['f2m']['mrr'] + metrics['m2f']['mrr']) / 2
    
    return metrics


@torch.no_grad()
def evaluate_baseline(backbone, dataset: CoupleDataset, device: str, 
                       batch_size: int = 16):
    """
    베이스라인 모델 평가
    
    Args:
        backbone: Qwen3VLFeatureExtractor 모델
        dataset: CoupleDataset 인스턴스
        device: 디바이스
        batch_size: 배치 크기
        
    Returns:
        metrics: 평가 결과
    """
    backbone.eval()
    
    all_female_embs = []
    all_male_embs = []
    
    n_couples = len(dataset)
    n_batches = (n_couples + batch_size - 1) // batch_size
    
    logger.info(f"Extracting embeddings for {n_couples} couples...")
    
    for batch_idx in tqdm(range(n_batches), desc="Extracting"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_couples)
        
        female_imgs = []
        male_imgs = []
        
        for idx in range(start_idx, end_idx):
            _, female_img, male_img = dataset.get_couple(idx)
            female_imgs.append(female_img)
            male_imgs.append(male_img)
        
        # Forward pass
        female_features = backbone.forward(female_imgs)
        male_features = backbone.forward(male_imgs)
        
        # L2 정규화
        female_features = F.normalize(female_features, p=2, dim=1)
        male_features = F.normalize(male_features, p=2, dim=1)
        
        all_female_embs.append(female_features.cpu().numpy())
        all_male_embs.append(male_features.cpu().numpy())
        
        # 메모리 정리
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # 결과 합치기
    all_female_embs = np.vstack(all_female_embs)
    all_male_embs = np.vstack(all_male_embs)
    
    logger.info(f"Embeddings shape: {all_female_embs.shape}")
    
    # Recall@K, MRR 계산
    k_values = [1, 5, 10, 20, 50]
    metrics = compute_recall_and_mrr(all_female_embs, all_male_embs, k_values)
    
    return metrics


def print_results(metrics: dict, n_couples: int):
    """결과 출력"""
    print("\n" + "=" * 50)
    print("   Qwen3-VL 베이스라인 평가 결과")
    print("=" * 50)
    
    print(f"데이터: {n_couples} couples")
    print()
    
    k_values = [5, 10, 20, 50]
    
    # Female → Male
    print("📊 Female → Male 검색:")
    print(f"    Accuracy:  {metrics['f2m'].get('recall@1', 0) * 100:6.2f}%")
    for k in k_values:
        recall = metrics['f2m'][f'recall@{k}'] * 100
        print(f"    Hit@{k:2d}:   {recall:6.2f}%")
    print(f"    MRR:       {metrics['f2m']['mrr']:.4f}")
    print()
    
    # Male → Female
    print("📊 Male → Female 검색:")
    print(f"    Accuracy:  {metrics['m2f'].get('recall@1', 0) * 100:6.2f}%")
    for k in k_values:
        recall = metrics['m2f'][f'recall@{k}'] * 100
        print(f"    Hit@{k:2d}:   {recall:6.2f}%")
    print(f"    MRR:       {metrics['m2f']['mrr']:.4f}")
    print()
    
    # 양방향 평균
    print("📊 평균 (양방향):")
    print(f"    Accuracy:  {metrics['avg'].get('recall@1', 0) * 100:6.2f}%")
    for k in k_values:
        recall = metrics['avg'][f'recall@{k}'] * 100
        print(f"    Hit@{k:2d}:   {recall:6.2f}%")
    print(f"    MRR:       {metrics['avg']['mrr']:.4f}")
    
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="베이스라인 모델 평가")
    parser.add_argument("--data-dir", type=str, 
                        default=os.path.expanduser("~/data/mutual-like-validations/images"))
    parser.add_argument("--splits", type=str, default="couple_splits.json",
                        help="분할 JSON 파일")
    parser.add_argument("--pooling-mode", type=str, default="mean", choices=["mean", "eos"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="./baseline_evaluation_results")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"디바이스: {device}")
    
    # 분할 파일에서 test set 로드
    if not os.path.exists(args.splits):
        logger.error(f"분할 파일 없음: {args.splits}")
        logger.error("python scripts/prepare_splits.py 실행 필요")
        return 1
    
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    
    test_ids = splits['test']
    logger.info(f"Test set: {len(test_ids)}개")
    
    # 데이터셋
    dataset = CoupleDataset(test_ids, args.data_dir, 768)
    
    if len(dataset) == 0:
        logger.error("유효한 데이터 없음")
        return 1
    
    # 모델 로드
    logger.info(f"모델 로드: Qwen/Qwen3-VL-2B-Instruct (pooling: {args.pooling_mode})")
    backbone = Qwen3VLFeatureExtractor(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        embedding_dim=2048,
        freeze_vision_encoder=True,
        use_projection_head=False,
        pooling_mode=args.pooling_mode,
        device=device
    )
    
    # 평가
    metrics = evaluate_baseline(backbone, dataset, device, args.batch_size)
    
    print_results(metrics, len(dataset))
    
    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, "baseline_results.json")
    with open(result_file, 'w') as f:
        json.dump({
            'n_couples': len(dataset),
            'pooling_mode': args.pooling_mode,
            'metrics': {
                'f2m': {k: float(v) for k, v in metrics['f2m'].items()},
                'm2f': {k: float(v) for k, v in metrics['m2f'].items()},
                'avg': {k: float(v) for k, v in metrics['avg'].items()}
            }
        }, f, indent=2)
    logger.info(f"저장됨: {result_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
