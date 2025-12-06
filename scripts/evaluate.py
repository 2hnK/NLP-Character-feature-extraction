"""
모델 평가 스크립트 (t-SNE 시각화 + 정량적 지표)

학습된 체크포인트를 로드하고 테스트 이미지에 대해:
1. t-SNE 시각화 (클래스별 임베딩 분포)
2. Recall@K (K=1, 5)
3. Silhouette Score
4. 클래스별 Precision/Recall

논문용 Figure 및 평가 지표를 생성합니다.
"""

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

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


@dataclass
class EvalConfig:
    """평가 설정"""
    # 체크포인트 경로 (JupyterLab 환경)
    # best_model.pth = Epoch 3 기준 (R@1: 87.76%, MAP@R: 0.7852)
    checkpoint_path: str = "scripts/checkpoints/best_model.pth"
    
    # 테스트 데이터 (JupyterLab 경로)
    test_jsonl: str = "vaildation/validation_labeled.jsonl"
    test_image_dir: str = "vaildation/images_renamed"
    
    # 모델 설정 (학습 시 설정과 동일해야 함)
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    embedding_dim: int = 2048
    projection_hidden_dim: int = 1024
    projection_output_dim: int = 256
    
    # 평가 설정
    batch_size: int = 4
    image_size: int = 768
    
    # 출력
    output_dir: str = "./evaluation_results"


# 레이블 매핑 (학습과 동일)
LABEL_MAPPING = {
    "Casual_Basic": 0,
    "Street_Hip": 1,
    "Sporty_Athleisure": 2,
    "Chic_Modern": 3,
    "Classy_Elegant": 4
}

CLASS_NAMES = list(LABEL_MAPPING.keys())


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


def load_test_data(config: EvalConfig) -> List[Dict]:
    """테스트 데이터 로드"""
    # 프로젝트 루트 기준 절대 경로로 변환
    test_jsonl_path = project_root / config.test_jsonl
    test_image_dir = project_root / config.test_image_dir
    
    data = []
    with open(test_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            # 이미지 경로 구성
            filename = item.get('filename')
            image_path = test_image_dir / filename
            
            # 라벨 추출
            fashion_style = item.get('image_metadata', {}).get('fashion_style')
            
            if fashion_style not in LABEL_MAPPING:
                logger.warning(f"Unknown fashion_style: {fashion_style}, skipping {filename}")
                continue
                
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            data.append({
                'id': item.get('id'),
                'image_path': str(image_path),
                'label': LABEL_MAPPING[fashion_style],
                'fashion_style': fashion_style
            })
    
    logger.info(f"Loaded {len(data)} test samples")
    return data


def load_model(config: EvalConfig, device: str):
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
    checkpoint_path = project_root / config.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # State dict 로드
    backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
    projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
    
    backbone.eval()
    projection_head.eval()
    
    logger.info(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    return backbone, projection_head


def extract_embeddings(
    backbone: Qwen3VLFeatureExtractor,
    projection_head: ProjectionHead,
    test_data: List[Dict],
    config: EvalConfig,
    device: str
) -> tuple:
    """테스트 이미지에서 임베딩 추출"""
    transform = ResizeLongestEdge(max_size=config.image_size)
    
    all_embeddings = []
    all_labels = []
    all_styles = []
    
    logger.info("Extracting embeddings...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), config.batch_size), desc="Extracting"):
            batch = test_data[i:i + config.batch_size]
            
            # 이미지 로드 및 전처리
            images = []
            labels = []
            styles = []
            
            for item in batch:
                try:
                    img = Image.open(item['image_path']).convert('RGB')
                    img = transform(img)
                    images.append(img)
                    labels.append(item['label'])
                    styles.append(item['fashion_style'])
                except Exception as e:
                    logger.error(f"Error loading {item['image_path']}: {e}")
                    continue
            
            if not images:
                continue
            
            # Forward pass
            features = backbone.forward(images)
            embeddings = projection_head(features)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels)
            all_styles.extend(styles)
    
    # 결합
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels)
    
    logger.info(f"Extracted {len(all_embeddings)} embeddings")
    
    return all_embeddings.numpy(), all_labels.numpy(), all_styles


def compute_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """평가 지표 계산"""
    metrics = {}
    
    # 1. Recall@K
    logger.info("Computing Recall@K...")
    try:
        calculator = AccuracyCalculator(
            include=("precision_at_1", "r_precision", "mean_average_precision_at_r"),
            k="max_bin_count"
        )
        accuracy = calculator.get_accuracy(
            torch.tensor(embeddings),
            torch.tensor(labels)
        )
        metrics['recall_at_1'] = accuracy['precision_at_1']
        metrics['r_precision'] = accuracy['r_precision']
        metrics['map_at_r'] = accuracy['mean_average_precision_at_r']
    except Exception as e:
        logger.error(f"Error computing Recall@K: {e}")
        metrics['recall_at_1'] = None
    
    # 2. Silhouette Score
    logger.info("Computing Silhouette Score...")
    try:
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = float(silhouette_score(embeddings, labels))
        else:
            metrics['silhouette_score'] = None
    except Exception as e:
        logger.error(f"Error computing Silhouette Score: {e}")
        metrics['silhouette_score'] = None
    
    # 3. 클래스별 분포
    unique, counts = np.unique(labels, return_counts=True)
    metrics['class_distribution'] = {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}
    
    return metrics


def visualize_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    styles: List[str],
    output_path: str,
    metrics: Dict
):
    """t-SNE 시각화 (논문용)"""
    logger.info("Generating t-SNE visualization...")
    
    # t-SNE 계산
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 색상 팔레트 (5개 클래스용)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 플롯 생성
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, class_name in enumerate(CLASS_NAMES):
        mask = labels == i
        if np.sum(mask) > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors[i],
                label=f'{class_name} (n={np.sum(mask)})',
                alpha=0.7,
                s=100,
                edgecolors='white',
                linewidth=0.5
            )
    
    # 스타일링
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Fashion Style Embeddings', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 지표 텍스트 추가
    if metrics.get('recall_at_1') is not None:
        text = f"Recall@1: {metrics['recall_at_1']:.3f}"
        if metrics.get('silhouette_score') is not None:
            text += f"\nSilhouette: {metrics['silhouette_score']:.3f}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"t-SNE plot saved to: {output_path}")


def main():
    config = EvalConfig()
    
    # 출력 디렉토리 생성 (프로젝트 루트 기준)
    output_dir = project_root / config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. 테스트 데이터 로드
    test_data = load_test_data(config)
    if not test_data:
        logger.error("No test data loaded. Exiting.")
        return
    
    # 2. 모델 로드
    backbone, projection_head = load_model(config, device)
    
    # 3. 임베딩 추출
    embeddings, labels, styles = extract_embeddings(
        backbone, projection_head, test_data, config, device
    )
    
    # 4. 지표 계산
    metrics = compute_metrics(embeddings, labels)
    
    # 5. t-SNE 시각화
    tsne_path = output_dir / "tsne_visualization.png"
    visualize_tsne(embeddings, labels, styles, str(tsne_path), metrics)
    
    # 6. 결과 저장
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # 7. 결과 출력
    print("\n" + "="*60)
    print("📊 Evaluation Results")
    print("="*60)
    print(f"Total samples: {len(embeddings)}")
    print(f"Recall@1: {metrics.get('recall_at_1', 'N/A')}")
    print(f"R-Precision: {metrics.get('r_precision', 'N/A')}")
    print(f"MAP@R: {metrics.get('map_at_r', 'N/A')}")
    print(f"Silhouette Score: {metrics.get('silhouette_score', 'N/A')}")
    print("\nClass Distribution:")
    for cls, count in metrics.get('class_distribution', {}).items():
        print(f"  - {cls}: {count}")
    print("="*60)
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
