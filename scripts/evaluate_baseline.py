"""
Qwen3-VL 베이스라인 모델 평가 스크립트

사전 학습된 Qwen3-VL 모델(프로젝션 헤드 없이)의 성능을 측정합니다.
학습된 모델과 비교하기 위한 베이스라인 성능을 제공합니다.

측정 지표:
- Recall@1, @5, @10, @20, @50
- MRR (Mean Reciprocal Rank)

사용법:
    python scripts/evaluate_baseline.py --fold 0
    python scripts/evaluate_baseline.py --test-ids 1,2,3,4,5

작성: 2024-12-21
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    """평가 설정"""
    # 데이터
    data_dir: str = os.path.expanduser("~/data/mutual-like-validations/images")
    splits_file: str = "couple_splits.json"
    
    # 모델 설정
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    embedding_dim: int = 2048  # Qwen3-VL hidden dimension
    
    # 평가 하이퍼파라미터
    batch_size: int = 48
    image_size: int = 768


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


def print_results(metrics: dict, n_couples: int, fold: Optional[int] = None):
    """결과 출력"""
    print("\n" + "=" * 50)
    print("   Qwen3-VL 베이스라인 평가 결과")
    print("=" * 50)
    
    dataset_info = f"Fold {fold} test set" if fold is not None else "Custom test set"
    print(f"데이터: {n_couples} couples ({dataset_info})")
    print()
    
    k_values = [1, 5, 10, 20, 50]
    
    # Female → Male
    print("📊 Female → Male 검색:")
    for k in k_values:
        recall = metrics['f2m'][f'recall@{k}'] * 100
        print(f"    Recall@{k:2d}: {recall:6.2f}%")
    print(f"    MRR:       {metrics['f2m']['mrr']:.4f}")
    print()
    
    # Male → Female
    print("📊 Male → Female 검색:")
    for k in k_values:
        recall = metrics['m2f'][f'recall@{k}'] * 100
        print(f"    Recall@{k:2d}: {recall:6.2f}%")
    print(f"    MRR:       {metrics['m2f']['mrr']:.4f}")
    print()
    
    # 양방향 평균
    print("📊 평균 (양방향):")
    for k in k_values:
        recall = metrics['avg'][f'recall@{k}'] * 100
        print(f"    Recall@{k:2d}: {recall:6.2f}%")
    print(f"    MRR:       {metrics['avg']['mrr']:.4f}")
    
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL baseline model")
    
    # 데이터 관련
    parser.add_argument("--data-dir", type=str, 
                        default=os.path.expanduser("~/data/mutual-like-validations/images"),
                        help="커플 이미지 데이터 디렉토리")
    parser.add_argument("--splits-file", type=str, default="couple_splits.json",
                        help="커플 분할 JSON 파일")
    parser.add_argument("--fold", type=int, default=None,
                        help="평가할 Fold 번호 (0-4). splits-file에서 test set 로드")
    parser.add_argument("--test-ids", type=str, default=None,
                        help="직접 지정할 테스트 커플 ID (쉼표 구분)")
    
    # 모델 관련
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Qwen3-VL 모델 이름")
    parser.add_argument("--embedding-dim", type=int, default=2048,
                        help="임베딩 차원")
    
    # 평가 관련
    parser.add_argument("--batch-size", type=int, default=16,
                        help="배치 크기")
    parser.add_argument("--image-size", type=int, default=768,
                        help="이미지 리사이즈 크기 (긴 변)")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 테스트 커플 ID 결정
    if args.test_ids:
        test_ids = [int(x.strip()) for x in args.test_ids.split(",")]
        fold = None
    elif args.fold is not None:
        with open(args.splits_file, 'r') as f:
            splits = json.load(f)
        
        fold_data = splits['folds'][args.fold]
        test_ids = fold_data.get('test', fold_data.get('valid', []))
        fold = args.fold
        logger.info(f"Loaded Fold {args.fold}: {len(test_ids)} test couples")
    else:
        # 기본값: splits 파일에서 전체 valid/test set 사용
        if os.path.exists(args.splits_file):
            with open(args.splits_file, 'r') as f:
                splits = json.load(f)
            # 첫 번째 fold의 valid 사용
            test_ids = splits['folds'][0].get('test', splits['folds'][0].get('valid', []))
            fold = 0
        else:
            logger.error("--fold 또는 --test-ids를 지정해주세요.")
            return 1
    
    logger.info(f"Evaluating on {len(test_ids)} couples")
    
    # 데이터셋 생성
    dataset = CoupleDataset(test_ids, args.data_dir, args.image_size)
    
    if len(dataset) == 0:
        logger.error("유효한 커플 데이터가 없습니다.")
        return 1
    
    # 모델 로드 (베이스라인: 프로젝션 헤드 없음)
    logger.info(f"Loading baseline model: {args.model_name}")
    backbone = Qwen3VLFeatureExtractor(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        freeze_vision_encoder=True,
        use_projection_head=False,
        device=device
    )
    
    # 평가 수행
    metrics = evaluate_baseline(backbone, dataset, device, args.batch_size)
    
    # 결과 출력
    print_results(metrics, len(dataset), fold)
    
    # 결과를 JSON으로도 저장 (선택적)
    result_file = f"baseline_results_fold{fold}.json" if fold is not None else "baseline_results.json"
    with open(result_file, 'w') as f:
        json.dump({
            'fold': fold,
            'n_couples': len(dataset),
            'metrics': {
                'f2m': {k: float(v) for k, v in metrics['f2m'].items()},
                'm2f': {k: float(v) for k, v in metrics['m2f'].items()},
                'avg': {k: float(v) for k, v in metrics['avg'].items()}
            },
            'config': {
                'model_name': args.model_name,
                'embedding_dim': args.embedding_dim,
                'image_size': args.image_size
            }
        }, f, indent=2)
    logger.info(f"Results saved to {result_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
