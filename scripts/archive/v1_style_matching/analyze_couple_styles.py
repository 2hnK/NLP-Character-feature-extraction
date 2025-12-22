"""
커플 데이터 스타일 라벨링 및 Recall 측정 스크립트

기존 스타일 모델(Epoch 3)을 사용하여:
1. 775쌍 커플 이미지에 스타일 라벨 부여
2. 커플 간 스타일 일치율 분석
3. 이미지+라벨 기반 Recall@K 측정

목적: "같은 스타일 = 커플" 가설 검증
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import torch
import torch.nn.functional as F
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

# 스타일 라벨 정의
STYLE_LABELS = [
    "Casual_Basic",
    "Street_Hip", 
    "Sporty_Athleisure",
    "Chic_Modern",
    "Classy_Elegant"
]


class ResizeLongestEdge:
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


def load_model(checkpoint_path: str, device: str):
    """기존 스타일 모델 로드"""
    logger.info("Loading style model...")
    
    backbone = Qwen3VLFeatureExtractor(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        embedding_dim=2048,
        freeze_vision_encoder=True,
        use_projection_head=False,
        device=device
    )
    
    projection_head = ProjectionHead(
        input_dim=2048,
        hidden_dim=1024,
        output_dim=256
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
    projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
    
    backbone.eval()
    projection_head.eval()
    
    logger.info(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'N/A')}")
    
    return backbone, projection_head


def predict_style(backbone, projection_head, image: Image.Image, device: str) -> Tuple[str, np.ndarray]:
    """이미지의 스타일 예측 및 임베딩 반환"""
    with torch.no_grad():
        features = backbone.forward([image])
        embedding = projection_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding.cpu().numpy()[0]


def load_and_process_couples(
    data_dir: str, 
    backbone, 
    projection_head,
    device: str,
    start: int = 5, 
    end: int = 778
) -> Tuple[List[Dict], List[int]]:
    """커플 데이터 로드 및 임베딩 추출"""
    transform = ResizeLongestEdge(max_size=768)
    data_dir = Path(data_dir)
    
    couples = []
    skipped = []
    
    for couple_num in tqdm(range(start, end + 1), desc="Processing couples"):
        try:
            couple_dir = data_dir / f"couple_{couple_num}"
            female_path = couple_dir / "female.png"
            male_path = couple_dir / "male.png"
            
            if not female_path.exists() or not male_path.exists():
                skipped.append(couple_num)
                continue
            
            # 이미지 로드
            female_img = Image.open(female_path).convert('RGB')
            male_img = Image.open(male_path).convert('RGB')
            
            female_img = transform(female_img)
            male_img = transform(male_img)
            
            # 임베딩 추출
            female_emb = predict_style(backbone, projection_head, female_img, device)
            male_emb = predict_style(backbone, projection_head, male_img, device)
            
            couples.append({
                'couple_id': couple_num,
                'female_emb': female_emb,
                'male_emb': male_emb
            })
            
            if couple_num % 50 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Skipping couple_{couple_num}: {e}")
            skipped.append(couple_num)
    
    logger.info(f"Processed {len(couples)} couples, skipped {len(skipped)}")
    return couples, skipped


def assign_style_labels(couples: List[Dict], reference_embeddings: Dict[str, np.ndarray] = None):
    """각 이미지에 가장 가까운 스타일 라벨 할당 (클러스터링 기반)"""
    # 모든 임베딩 수집
    all_female_embs = np.array([c['female_emb'] for c in couples])
    all_male_embs = np.array([c['male_emb'] for c in couples])
    
    # 간단한 K-means 클러스터링으로 5개 스타일 분류
    from sklearn.cluster import KMeans
    
    all_embs = np.vstack([all_female_embs, all_male_embs])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    all_labels = kmeans.fit_predict(all_embs)
    
    n_couples = len(couples)
    female_labels = all_labels[:n_couples]
    male_labels = all_labels[n_couples:]
    
    for i, couple in enumerate(couples):
        couple['female_style'] = int(female_labels[i])
        couple['male_style'] = int(male_labels[i])
        couple['style_match'] = female_labels[i] == male_labels[i]
    
    return couples


def analyze_style_matching(couples: List[Dict]):
    """스타일 일치율 분석"""
    n_total = len(couples)
    n_match = sum(1 for c in couples if c['style_match'])
    match_rate = n_match / n_total
    
    # 랜덤 기대값 (5개 클래스 → 20%)
    random_rate = 1 / 5
    
    print("\n" + "=" * 60)
    print("📊 커플 스타일 일치율 분석")
    print("=" * 60)
    print(f"\n총 커플 수: {n_total}")
    print(f"같은 스타일 커플: {n_match} ({match_rate*100:.1f}%)")
    print(f"랜덤 기대값: {random_rate*100:.1f}%")
    print(f"랜덤 대비: {match_rate/random_rate:.2f}x")
    
    if match_rate > random_rate * 1.5:
        print("\n✅ 스타일 일치율이 랜덤보다 유의미하게 높음!")
        print("   → '비슷한 스타일 = 커플' 가설이 어느 정도 유효")
    else:
        print("\n⚠️ 스타일 일치율이 랜덤과 비슷함")
        print("   → 스타일만으로는 커플 예측 어려움")
    
    print("=" * 60)
    
    return {'match_rate': match_rate, 'random_rate': random_rate}


def compute_style_based_recall(couples: List[Dict], k_values: List[int] = [1, 5, 10, 20, 50]):
    """스타일 기반 Recall@K 측정
    
    Female 기준: 같은 스타일의 male 중 파트너가 있는지
    """
    n_couples = len(couples)
    
    # 임베딩 추출
    female_embs = np.array([c['female_emb'] for c in couples])
    male_embs = np.array([c['male_emb'] for c in couples])
    female_styles = np.array([c['female_style'] for c in couples])
    male_styles = np.array([c['male_style'] for c in couples])
    
    # 유사도 계산
    similarity = np.dot(female_embs, male_embs.T)
    
    results = {
        'all_males': {},  # 전체 male에서 검색
        'same_style_males': {}  # 같은 스타일 male에서만 검색
    }
    
    # 1. 전체 male에서 검색 (기존 방식)
    ranks_all = []
    for i in range(n_couples):
        sorted_idx = np.argsort(-similarity[i])
        rank = np.where(sorted_idx == i)[0][0]
        ranks_all.append(rank)
    ranks_all = np.array(ranks_all)
    
    for k in k_values:
        results['all_males'][f'recall@{k}'] = np.mean(ranks_all < k)
    results['all_males']['mrr'] = np.mean(1.0 / (ranks_all + 1))
    
    # 2. 같은 스타일 male에서만 검색
    ranks_same_style = []
    valid_count = 0
    
    for i in range(n_couples):
        my_style = female_styles[i]
        # 같은 스타일의 male 인덱스
        same_style_idx = np.where(male_styles == my_style)[0]
        
        if len(same_style_idx) == 0:
            continue
        
        # 같은 스타일 male들과의 유사도
        same_style_sim = similarity[i, same_style_idx]
        sorted_local_idx = np.argsort(-same_style_sim)
        sorted_global_idx = same_style_idx[sorted_local_idx]
        
        # 파트너(i)가 몇 번째인지
        if i in sorted_global_idx:
            rank = np.where(sorted_global_idx == i)[0][0]
            ranks_same_style.append(rank)
            valid_count += 1
    
    if ranks_same_style:
        ranks_same_style = np.array(ranks_same_style)
        for k in k_values:
            results['same_style_males'][f'recall@{k}'] = np.mean(ranks_same_style < k)
        results['same_style_males']['mrr'] = np.mean(1.0 / (ranks_same_style + 1))
        results['same_style_males']['valid_count'] = valid_count
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 Recall@K 비교")
    print("=" * 60)
    
    print("\n📌 전체 Male에서 검색 (기준선)")
    for k, v in results['all_males'].items():
        if k.startswith('recall'):
            print(f"  {k}: {v*100:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")
    
    print(f"\n📌 같은 스타일 Male에서만 검색 ({valid_count}쌍)")
    for k, v in results['same_style_males'].items():
        if k == 'valid_count':
            continue
        if k.startswith('recall'):
            print(f"  {k}: {v*100:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")
    
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, 
                        default=os.path.expanduser("~/data/mutual-like-validations/images"))
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.expanduser("~/checkpoints/best_model_epoch3.pth"))
    parser.add_argument("--output", type=str, default="style_analysis_results.json")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 모델 로드
    backbone, projection_head = load_model(args.checkpoint, device)
    
    # 커플 데이터 처리
    couples, skipped = load_and_process_couples(
        args.data_dir, backbone, projection_head, device
    )
    
    # 스타일 라벨 할당 (클러스터링)
    couples = assign_style_labels(couples)
    
    # 스타일 일치율 분석
    style_stats = analyze_style_matching(couples)
    
    # Recall@K 측정
    recall_results = compute_style_based_recall(couples)
    
    # 결과 저장
    results = {
        'n_couples': len(couples),
        'skipped': skipped,
        'style_stats': style_stats,
        'recall_results': {
            k: {k2: float(v2) if isinstance(v2, (np.floating, float)) else v2 
                for k2, v2 in v.items()}
            for k, v in recall_results.items()
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
