"""
라벨 기반 커플 Recall@K 측정 스크립트

label_couple_images.py로 생성된 라벨 데이터를 사용하여:
1. 스타일 일치율 분석
2. 같은 스타일 내에서 Recall@K 측정
3. 전체 검색 대비 개선 효과 분석
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.qwen_backbone import Qwen3VLFeatureExtractor
from src.models.projection import ProjectionHead

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ResizeLongestEdge:
    def __init__(self, max_size: int):
        self.max_size = max_size
    def __call__(self, img):
        w, h = img.size
        scale = self.max_size / max(w, h)
        if scale >= 1: return img
        return img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)


def load_model(checkpoint_path: str, device: str):
    """모델 로드"""
    backbone = Qwen3VLFeatureExtractor(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        embedding_dim=2048, freeze_vision_encoder=True,
        use_projection_head=False, device=device
    )
    projection_head = ProjectionHead(input_dim=2048, hidden_dim=1024, output_dim=256).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
    projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
    backbone.eval()
    projection_head.eval()
    
    return backbone, projection_head


def load_labels(label_file: str) -> Dict[int, Dict]:
    """라벨 파일 로드"""
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            labels[data['couple_id']] = {
                'female_style': data['female']['fashion_style'],
                'male_style': data['male']['fashion_style'],
                'style_match': data['style_match']
            }
    return labels


def extract_embeddings(backbone, projection_head, data_dir: str, labels: Dict, device: str):
    """임베딩 추출"""
    transform = ResizeLongestEdge(768)
    data_dir = Path(data_dir)
    
    couples = []
    
    for couple_id, label_info in tqdm(labels.items(), desc="Extracting embeddings"):
        couple_dir = data_dir / f"couple_{couple_id}"
        female_path = couple_dir / "female.png"
        male_path = couple_dir / "male.png"
        
        if not female_path.exists() or not male_path.exists():
            continue
        
        try:
            female_img = transform(Image.open(female_path).convert('RGB'))
            male_img = transform(Image.open(male_path).convert('RGB'))
            
            with torch.no_grad():
                f_feat = backbone.forward([female_img])
                f_emb = F.normalize(projection_head(f_feat), p=2, dim=1)
                
                m_feat = backbone.forward([male_img])
                m_emb = F.normalize(projection_head(m_feat), p=2, dim=1)
            
            couples.append({
                'couple_id': couple_id,
                'female_emb': f_emb.cpu().numpy()[0],
                'male_emb': m_emb.cpu().numpy()[0],
                'female_style': label_info['female_style'],
                'male_style': label_info['male_style']
            })
            
            if len(couples) % 50 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Error couple_{couple_id}: {e}")
    
    return couples


def compute_recall(couples: List[Dict], k_values=[1, 5, 10, 20, 50]):
    """Recall@K 계산"""
    n = len(couples)
    female_embs = np.array([c['female_emb'] for c in couples])
    male_embs = np.array([c['male_emb'] for c in couples])
    female_styles = [c['female_style'] for c in couples]
    male_styles = [c['male_style'] for c in couples]
    
    similarity = np.dot(female_embs, male_embs.T)
    
    results = {'all': {}, 'same_style': {}}
    
    # 전체 검색
    ranks_all = []
    for i in range(n):
        sorted_idx = np.argsort(-similarity[i])
        rank = np.where(sorted_idx == i)[0][0]
        ranks_all.append(rank)
    ranks_all = np.array(ranks_all)
    
    for k in k_values:
        results['all'][f'recall@{k}'] = float(np.mean(ranks_all < k))
    results['all']['mrr'] = float(np.mean(1.0 / (ranks_all + 1)))
    results['all']['mean_rank'] = float(np.mean(ranks_all))
    
    # 같은 스타일 내 검색
    ranks_style = []
    valid = 0
    
    for i in range(n):
        my_style = female_styles[i]
        same_idx = [j for j in range(n) if male_styles[j] == my_style]
        
        if i not in same_idx or len(same_idx) < 2:
            continue
        
        same_sim = [(j, similarity[i, j]) for j in same_idx]
        same_sim.sort(key=lambda x: -x[1])
        
        rank = next(idx for idx, (j, _) in enumerate(same_sim) if j == i)
        ranks_style.append(rank)
        valid += 1
    
    ranks_style = np.array(ranks_style)
    
    for k in k_values:
        results['same_style'][f'recall@{k}'] = float(np.mean(ranks_style < k))
    results['same_style']['mrr'] = float(np.mean(1.0 / (ranks_style + 1)))
    results['same_style']['valid_couples'] = valid
    results['same_style']['avg_pool_size'] = n * (valid / n) if valid else 0
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="couple_labels.jsonl")
    parser.add_argument("--data-dir", type=str, 
                        default=os.path.expanduser("~/data/mutual-like-validations/images"))
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.expanduser("~/checkpoints/best_model_epoch3.pth"))
    parser.add_argument("--output", type=str, default="style_recall_results.json")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 라벨 로드
    labels = load_labels(args.labels)
    logger.info(f"Loaded {len(labels)} couple labels")
    
    # 스타일 일치율
    match_count = sum(1 for l in labels.values() if l['style_match'])
    match_rate = match_count / len(labels)
    
    print(f"\n{'='*60}")
    print(f"📊 스타일 일치율 분석")
    print(f"{'='*60}")
    print(f"총 커플: {len(labels)}")
    print(f"같은 스타일: {match_count} ({match_rate*100:.1f}%)")
    print(f"랜덤 기대값: 20%")
    print(f"랜덤 대비: {match_rate/0.2:.2f}x")
    
    # 스타일 분포
    all_styles = [l['female_style'] for l in labels.values()] + [l['male_style'] for l in labels.values()]
    style_dist = Counter(all_styles)
    print(f"\n스타일 분포:")
    for style, count in style_dist.most_common():
        print(f"  {style}: {count} ({count/len(all_styles)*100:.1f}%)")
    
    # 모델 로드 및 임베딩 추출
    backbone, projection_head = load_model(args.checkpoint, device)
    couples = extract_embeddings(backbone, projection_head, args.data_dir, labels, device)
    
    # Recall 계산
    results = compute_recall(couples)
    
    print(f"\n{'='*60}")
    print(f"📊 Recall@K 비교")
    print(f"{'='*60}")
    
    print(f"\n📌 전체 Male 검색 (N={len(couples)})")
    for k, v in results['all'].items():
        if k.startswith('recall'):
            print(f"  {k}: {v*100:.2f}%")
        elif k == 'mrr':
            print(f"  MRR: {v:.4f}")
    
    print(f"\n📌 같은 스타일 내 검색 (N={results['same_style']['valid_couples']})")
    for k, v in results['same_style'].items():
        if k.startswith('recall'):
            print(f"  {k}: {v*100:.2f}%")
        elif k == 'mrr':
            print(f"  MRR: {v:.4f}")
    
    print(f"\n{'='*60}")
    
    # 결과 저장
    output_data = {
        'n_couples': len(couples),
        'style_match_rate': match_rate,
        'style_distribution': dict(style_dist),
        'recall_results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
