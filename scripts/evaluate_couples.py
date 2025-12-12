"""
커플 데이터 기반 Recall@K 평가 스크립트 (로컬 버전)

실제 매칭된 커플 데이터(774쌍)를 사용하여:
1. 각 사용자의 임베딩 추출
2. Female → Male 검색 시 파트너가 Top-K에 있는지 평가
3. Male → Female 검색 시 파트너가 Top-K에 있는지 평가
4. Recall@K, MRR, Hit Rate 등 지표 계산

SageMaker JupyterLab 로컬 경로:
- 체크포인트: ~/checkpoints/best_model_epoch3.pth
- 커플 데이터: ~/data/mutual-like-validations/images/couple_{5-778}/
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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


@dataclass
class CoupleEvalConfig:
    """커플 평가 설정"""
    # 로컬 경로 (SageMaker JupyterLab)
    couple_data_dir: str = os.path.expanduser("~/data/mutual-like-validations/images")
    checkpoint_path: str = os.path.expanduser("~/checkpoints/best_model_epoch3.pth")
    splits_file: str = "couple_splits.json"  # 분할 파일 경로
    
    # 커플 범위 (splits 미사용 시)
    start_couple: int = 5
    end_couple: int = 778
    
    # 모델 설정 (학습 시와 동일)
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    embedding_dim: int = 2048
    projection_hidden_dim: int = 1024
    projection_output_dim: int = 256
    
    # 평가 설정
    batch_size: int = 1  # 메모리 안전
    image_size: int = 768
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    
    # 출력
    output_dir: str = "./couple_evaluation_results"


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


def load_model(checkpoint_path: str, config: CoupleEvalConfig, device: str):
    """모델 및 체크포인트 로드"""
    logger.info(f"Loading model: {config.model_name}")
    
    # Backbone 로드
    backbone = Qwen3VLFeatureExtractor(
        model_name=config.model_name,
        embedding_dim=config.embedding_dim,
        freeze_vision_encoder=True,
        use_projection_head=False,
        device=device
    )
    
    # Projection Head 로드
    projection_head = ProjectionHead(
        input_dim=config.embedding_dim,
        hidden_dim=config.projection_hidden_dim,
        output_dim=config.projection_output_dim
    ).to(device)
    
    # 체크포인트 로드
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # State dict 로드
    backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
    projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
    
    backbone.eval()
    projection_head.eval()
    
    logger.info(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    return backbone, projection_head


def load_couple_images(
    config: CoupleEvalConfig
) -> Tuple[List[Tuple[int, Image.Image, Image.Image]], List[int]]:
    """로컬에서 커플 이미지 로드"""
    transform = ResizeLongestEdge(max_size=config.image_size)
    couples = []
    skipped = []
    
    data_dir = Path(config.couple_data_dir)
    logger.info(f"Loading couple images from: {data_dir}")
    
    for couple_num in tqdm(
        range(config.start_couple, config.end_couple + 1), 
        desc="Loading couples"
    ):
        try:
            couple_dir = data_dir / f"couple_{couple_num}"
            female_path = couple_dir / "female.png"
            male_path = couple_dir / "male.png"
            
            # 파일 존재 확인
            if not female_path.exists() or not male_path.exists():
                logger.warning(f"Skipping couple_{couple_num}: files not found")
                skipped.append(couple_num)
                continue
            
            # Female 이미지 로드
            female_img = Image.open(female_path).convert('RGB')
            female_img = transform(female_img)
            
            # Male 이미지 로드
            male_img = Image.open(male_path).convert('RGB')
            male_img = transform(male_img)
            
            couples.append((couple_num, female_img, male_img))
            
        except Exception as e:
            logger.warning(f"Skipping couple_{couple_num}: {e}")
            skipped.append(couple_num)
    
    logger.info(f"Loaded {len(couples)} couples, skipped {len(skipped)}")
    
    return couples, skipped


def extract_embeddings(
    backbone: Qwen3VLFeatureExtractor,
    projection_head: ProjectionHead,
    couples: List[Tuple[int, Image.Image, Image.Image]],
    device: str
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """커플 이미지에서 임베딩 추출"""
    female_embeddings = []
    male_embeddings = []
    couple_ids = []
    
    logger.info("Extracting embeddings...")
    
    with torch.no_grad():
        for couple_num, female_img, male_img in tqdm(couples, desc="Extracting"):
            try:
                # Female 임베딩
                female_feat = backbone.forward([female_img])
                female_emb = projection_head(female_feat)
                female_emb = F.normalize(female_emb, p=2, dim=1)
                
                # Male 임베딩
                male_feat = backbone.forward([male_img])
                male_emb = projection_head(male_feat)
                male_emb = F.normalize(male_emb, p=2, dim=1)
                
                female_embeddings.append(female_emb.cpu().numpy())
                male_embeddings.append(male_emb.cpu().numpy())
                couple_ids.append(couple_num)
                
                # 메모리 정리
                if couple_num % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing couple_{couple_num}: {e}")
                continue
    
    female_embeddings = np.vstack(female_embeddings)
    male_embeddings = np.vstack(male_embeddings)
    
    logger.info(f"Extracted embeddings: {len(couple_ids)} couples")
    logger.info(f"Female embeddings shape: {female_embeddings.shape}")
    logger.info(f"Male embeddings shape: {male_embeddings.shape}")
    
    return female_embeddings, male_embeddings, couple_ids


def compute_couple_metrics(
    female_embeddings: np.ndarray,
    male_embeddings: np.ndarray,
    k_values: List[int]
) -> Dict:
    """커플 기반 Recall@K 및 기타 지표 계산"""
    n_couples = len(female_embeddings)
    
    # 코사인 유사도 행렬 계산
    similarity_f2m = np.dot(female_embeddings, male_embeddings.T)
    similarity_m2f = similarity_f2m.T
    
    metrics = {
        "n_couples": n_couples,
        "female_to_male": {},
        "male_to_female": {},
        "bidirectional": {}
    }
    
    # --- Female → Male 방향 ---
    ranks_f2m = []
    for i in range(n_couples):
        scores = similarity_f2m[i]
        sorted_indices = np.argsort(-scores)
        rank = np.where(sorted_indices == i)[0][0]
        ranks_f2m.append(rank)
    
    ranks_f2m = np.array(ranks_f2m)
    
    for k in k_values:
        hits = np.sum(ranks_f2m < k)
        metrics["female_to_male"][f"recall@{k}"] = hits / n_couples
    
    mrr_f2m = np.mean(1.0 / (ranks_f2m + 1))
    metrics["female_to_male"]["mrr"] = float(mrr_f2m)
    metrics["female_to_male"]["mean_rank"] = float(np.mean(ranks_f2m))
    metrics["female_to_male"]["median_rank"] = float(np.median(ranks_f2m))
    
    # --- Male → Female 방향 ---
    ranks_m2f = []
    for i in range(n_couples):
        scores = similarity_m2f[i]
        sorted_indices = np.argsort(-scores)
        rank = np.where(sorted_indices == i)[0][0]
        ranks_m2f.append(rank)
    
    ranks_m2f = np.array(ranks_m2f)
    
    for k in k_values:
        hits = np.sum(ranks_m2f < k)
        metrics["male_to_female"][f"recall@{k}"] = hits / n_couples
    
    mrr_m2f = np.mean(1.0 / (ranks_m2f + 1))
    metrics["male_to_female"]["mrr"] = float(mrr_m2f)
    metrics["male_to_female"]["mean_rank"] = float(np.mean(ranks_m2f))
    metrics["male_to_female"]["median_rank"] = float(np.median(ranks_m2f))
    
    # --- 양방향 ---
    for k in k_values:
        bidirectional_hits = np.sum((ranks_f2m < k) & (ranks_m2f < k))
        metrics["bidirectional"][f"recall@{k}"] = bidirectional_hits / n_couples
    
    # 평균 지표
    metrics["average"] = {}
    for k in k_values:
        avg = (metrics["female_to_male"][f"recall@{k}"] + 
               metrics["male_to_female"][f"recall@{k}"]) / 2
        metrics["average"][f"recall@{k}"] = avg
    
    metrics["average"]["mrr"] = (mrr_f2m + mrr_m2f) / 2
    
    return metrics


def print_results(metrics: Dict):
    """결과 출력"""
    print("\n" + "=" * 70)
    print("📊 커플 데이터 기반 Recall@K 평가 결과")
    print("=" * 70)
    
    print(f"\n총 커플 수: {metrics['n_couples']}")
    
    print("\n📌 Female → Male (여성 기준 남성 파트너 검색)")
    print("-" * 50)
    for key, value in metrics["female_to_male"].items():
        if key.startswith("recall"):
            print(f"  {key}: {value*100:.2f}%")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\n📌 Male → Female (남성 기준 여성 파트너 검색)")
    print("-" * 50)
    for key, value in metrics["male_to_female"].items():
        if key.startswith("recall"):
            print(f"  {key}: {value*100:.2f}%")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\n📌 평균 (양방향)")
    print("-" * 50)
    for key, value in metrics["average"].items():
        if key.startswith("recall"):
            print(f"  {key}: {value*100:.2f}%")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\n📌 양방향 동시 Hit (둘 다 Top-K 내 파트너 찾음)")
    print("-" * 50)
    for key, value in metrics["bidirectional"].items():
        print(f"  {key}: {value*100:.2f}%")
    
    print("\n" + "=" * 70)


def save_results(metrics: Dict, output_dir: str):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 저장
    json_path = os.path.join(output_dir, "couple_recall_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to: {json_path}")
    
    # Markdown 리포트 생성
    md_path = os.path.join(output_dir, "couple_recall_report.md")
    md_content = f"""# 커플 데이터 기반 Recall@K 평가 결과

**평가 일시**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**총 커플 수**: {metrics['n_couples']}  
**체크포인트**: best_model_epoch3.pth

---

## 📊 주요 지표

| 방향 | Recall@1 | Recall@5 | Recall@10 | MRR |
|------|----------|----------|-----------|-----|
| Female→Male | {metrics['female_to_male'].get('recall@1', 0)*100:.2f}% | {metrics['female_to_male'].get('recall@5', 0)*100:.2f}% | {metrics['female_to_male'].get('recall@10', 0)*100:.2f}% | {metrics['female_to_male'].get('mrr', 0):.4f} |
| Male→Female | {metrics['male_to_female'].get('recall@1', 0)*100:.2f}% | {metrics['male_to_female'].get('recall@5', 0)*100:.2f}% | {metrics['male_to_female'].get('recall@10', 0)*100:.2f}% | {metrics['male_to_female'].get('mrr', 0):.4f} |
| **평균** | **{metrics['average'].get('recall@1', 0)*100:.2f}%** | **{metrics['average'].get('recall@5', 0)*100:.2f}%** | **{metrics['average'].get('recall@10', 0)*100:.2f}%** | **{metrics['average'].get('mrr', 0):.4f}** |

---

## 📈 상세 결과

### Female → Male
"""
    for key, value in metrics["female_to_male"].items():
        if key.startswith("recall"):
            md_content += f"- **{key}**: {value*100:.2f}%\n"
        else:
            md_content += f"- **{key}**: {value:.4f}\n"
    
    md_content += "\n### Male → Female\n"
    for key, value in metrics["male_to_female"].items():
        if key.startswith("recall"):
            md_content += f"- **{key}**: {value*100:.2f}%\n"
        else:
            md_content += f"- **{key}**: {value:.4f}\n"
    
    md_content += "\n### 양방향 동시 Hit\n"
    for key, value in metrics["bidirectional"].items():
        md_content += f"- **{key}**: {value*100:.2f}%\n"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    logger.info(f"Report saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on couple data (local)")
    parser.add_argument("--data-dir", type=str, default=None, 
                        help="Couple data directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path")
    parser.add_argument("--splits-file", type=str, default=None,
                        help="Splits JSON file (from prepare_couple_splits.py)")
    parser.add_argument("--split", type=str, default=None, choices=['test', 'all'],
                        help="Which split to evaluate: 'test' or 'all'")
    parser.add_argument("--start", type=int, default=5, 
                        help="Start couple number (ignored if --splits-file used)")
    parser.add_argument("--end", type=int, default=778, 
                        help="End couple number (ignored if --splits-file used)")
    args = parser.parse_args()
    
    config = CoupleEvalConfig()
    
    # 명령줄 인자로 덮어쓰기
    if args.data_dir:
        config.couple_data_dir = args.data_dir
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.splits_file:
        config.splits_file = args.splits_file
    
    # 커플 ID 결정
    couple_ids_to_eval = None
    if args.split and os.path.exists(config.splits_file):
        with open(config.splits_file, 'r') as f:
            splits = json.load(f)
        if args.split == 'test':
            couple_ids_to_eval = splits['test']
            logger.info(f"Evaluating TEST set: {len(couple_ids_to_eval)} couples")
        elif args.split == 'all':
            couple_ids_to_eval = None  # 전체 평가
    
    config.start_couple = args.start
    config.end_couple = args.end
    
    # 경로 확인
    if not Path(config.checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {config.checkpoint_path}")
        return 1
    if not Path(config.couple_data_dir).exists():
        logger.error(f"Data directory not found: {config.couple_data_dir}")
        return 1
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. 모델 로드
    backbone, projection_head = load_model(config.checkpoint_path, config, device)
    
    # 2. 커플 이미지 로드 (로컬)
    couples, skipped = load_couple_images(config)
    
    if not couples:
        logger.error("No couples loaded. Exiting.")
        return 1
    
    # 3. 임베딩 추출
    female_embs, male_embs, couple_ids = extract_embeddings(
        backbone, projection_head, couples, device
    )
    
    # 4. 지표 계산
    metrics = compute_couple_metrics(female_embs, male_embs, config.k_values)
    metrics["skipped_couples"] = skipped
    metrics["couple_ids"] = couple_ids
    
    # 5. 결과 출력 및 저장
    print_results(metrics)
    save_results(metrics, config.output_dir)
    
    print(f"\n✅ 결과가 저장되었습니다: {config.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
