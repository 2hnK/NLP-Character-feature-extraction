# Getting Started Guide

AWS SageMaker AI Studio에서 Qwen3-VL-2B 모델로 데이팅 프로필 매칭 시스템을 빠르게 시작하는 가이드입니다.

## 목차

1. [환경 설정](#환경-설정)
2. [데이터 준비](#데이터-준비)
3. [모델 학습](#모델-학습)
4. [모델 평가](#모델-평가)
5. [추론 및 매칭](#추론-및-매칭)

## 환경 설정

### 1. SageMaker Studio 접속

```
1. AWS 콘솔 → SageMaker → Studio
2. "Open Studio" 클릭
3. Domain/User 선택
```

### 2. 프로젝트 클론 및 설정

```bash
# SageMaker Studio Terminal
cd ~/SageMaker
git clone <your-repo-url> dating-profile-matcher
cd dating-profile-matcher

# 의존성 설치
pip install -r requirements.txt

# 디렉토리 생성
mkdir -p data/{raw/profiles,raw/augmented,processed/{train,val}}
mkdir -p models/{checkpoints,saved_models}
mkdir -p logs
```

### 3. Notebook 커널 설정

- **Kernel**: Python 3 (PyTorch 2.0 GPU Optimized)
- **Instance**: 
  - 개발/테스트: `ml.t3.medium` (CPU)
  - 모델 학습: `ml.g5.xlarge` (1 GPU)

**예상 비용:**
- ml.t3.medium: ~$0.05/시간
- ml.g5.xlarge: ~$1.41/시간 (Spot 사용 시 ~$0.42/시간)

## 데이터 준비

### 1. 데이터 디렉토리 구조

```bash
data/
├── raw/
│   ├── profiles/          # 실제 사용자 이미지 (100개)
│   └── augmented/         # 증강 이미지 (3,200개)
└── processed/
    ├── train/             # 전처리된 학습 데이터
    ├── val/               # 전처리된 검증 데이터
    ├── train_metadata.csv
    └── val_metadata.csv
```

### 2. 원본 데이터 업로드

**SageMaker Studio 파일 브라우저 사용:**

1. 좌측 메뉴에서 파일 브라우저 열기
2. `data/raw/profiles/`로 이동
3. 실제 사용자 이미지 (100개) 업로드
4. `data/raw/augmented/`로 이동
5. 증강 이미지 (3,200개) 업로드

**또는 AWS CLI 사용:**

```bash
# 로컬에서 S3로 업로드
aws s3 sync local_data/profiles/ s3://your-bucket/dating-matcher/profiles/
aws s3 sync local_data/augmented/ s3://your-bucket/dating-matcher/augmented/

# SageMaker에서 S3에서 다운로드
aws s3 sync s3://your-bucket/dating-matcher/ ~/SageMaker/dating-profile-matcher/data/raw/
```

### 3. 메타데이터 생성

```python
# create_metadata.py
import pandas as pd
from pathlib import Path

# 실제 이미지 메타데이터
real_images = []
for img_path in Path('data/raw/profiles').glob('*.jpg'):
    filename = img_path.name
    # 파일명 형식: user_001_1.jpg
    parts = filename.replace('.jpg', '').split('_')
    user_id = f"{parts[0]}_{parts[1]}"
    image_idx = int(parts[2])
    
    real_images.append({
        'filename': filename,
        'user_id': user_id,
        'image_idx': image_idx,
        'filepath': f'data/raw/profiles/{filename}',
        'is_synthetic': False
    })

# 증강 이미지 메타데이터
aug_images = []
for img_path in Path('data/raw/augmented').glob('*.jpg'):
    filename = img_path.name
    # 파일명 형식: gen_0001.jpg
    user_id = filename.replace('.jpg', '')
    
    aug_images.append({
        'filename': filename,
        'user_id': user_id,
        'image_idx': 1,
        'filepath': f'data/raw/augmented/{filename}',
        'is_synthetic': True
    })

# 데이터프레임 생성
df = pd.DataFrame(real_images + aug_images)

# Train/Val 분할 (85:15)
from sklearn.model_selection import train_test_split
user_ids = df['user_id'].unique()
train_users, val_users = train_test_split(user_ids, test_size=0.15, random_state=42)

train_df = df[df['user_id'].isin(train_users)]
val_df = df[df['user_id'].isin(val_users)]

# 저장
train_df.to_csv('data/processed/train_metadata.csv', index=False)
val_df.to_csv('data/processed/val_metadata.csv', index=False)

print(f"Train: {len(train_df)} images, {len(train_users)} users")
print(f"Val: {len(val_df)} images, {len(val_users)} users")
```

실행:

```bash
python create_metadata.py
```

## 모델 학습

### 1. Qwen 모델 로드 테스트

먼저 Jupyter Notebook에서 모델이 정상적으로 로드되는지 확인합니다.

```python
# test_model_loading.ipynb
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 모델 로드
model_name = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

print(f"Model loaded: {model.__class__.__name__}")
print(f"Device: {next(model.parameters()).device}")
```

### 2. 학습 설정 확인

`configs/config.yaml` 파일에서 주요 설정을 확인/수정합니다:

```yaml
# 데이터 경로
paths:
  train_metadata: "data/processed/train_metadata.csv"
  val_metadata: "data/processed/val_metadata.csv"
  data_root: "data/raw"

# 모델 설정
model:
  name: "Qwen/Qwen2-VL-2B-Instruct"
  embedding_dim: 512
  freeze_vision_encoder: true  # Stage 1에서는 freeze

# 학습 파라미터
training:
  batch_size: 16
  num_epochs: 15
  learning_rate: 1e-4
  margin: 1.0  # Triplet Loss margin
  
# SageMaker 설정
sagemaker:
  instance_type: "ml.g5.xlarge"
  use_spot_instances: true
```

### 3. 학습 시작 (SageMaker Training Job)

**노트북에서 실행:**

```python
# notebooks/sagemaker_training.ipynb
import sagemaker
from sagemaker.pytorch import PyTorch

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Training Job 설정
estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='../src/training',
    role=role,
    instance_type='ml.g5.xlarge',
    instance_count=1,
    framework_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'config': '../configs/config.yaml',
        'num-epochs': 15,
        'batch-size': 16
    },
    use_spot_instances=True,
    max_run=3*60*60,  # 3시간
    max_wait=3*60*60
)

# 학습 시작
estimator.fit()
```

### 4. 학습 모니터링

**CloudWatch Logs:**

```python
# 로그 실시간 확인
estimator.logs()
```

**주요 확인 사항:**

- Triplet Loss 감소 추세
- Training/Validation Loss 차이
- GPU 메모리 사용량
- Epoch당 소요 시간

**예상 학습 시간:**

- 3,300개 이미지, batch_size=16: 약 2-3시간 (15 epochs)

## 모델 평가

### 1. 학습된 모델 로드

```python
# evaluate_model.ipynb
import torch
from src.models.qwen_backbone import Qwen3VLFeatureExtractor

# 모델 로드
model = Qwen3VLFeatureExtractor.load_from_checkpoint(
    'models/saved_models/best_model.pth',
    device='cuda'
)
model.eval()
```

### 2. 임베딩 품질 평가

```python
from src.evaluation.metrics import evaluate_embeddings

# 검증 데이터로 임베딩 생성
val_metadata = pd.read_csv('data/processed/val_metadata.csv')
embeddings, user_ids = [], []

for _, row in val_metadata.iterrows():
    img_path = row['filepath']
    embedding = model.extract_from_path(img_path)
    embeddings.append(embedding)
    user_ids.append(row['user_id'])

embeddings = torch.stack(embeddings)

# 평가 지표 계산
metrics = evaluate_embeddings(embeddings, user_ids)

print("Evaluation Results:")
print(f"  Intra-class distance: {metrics['intra_distance']:.4f}")
print(f"  Inter-class distance: {metrics['inter_distance']:.4f}")
print(f"  Separation: {metrics['separation']:.4f}")
```

**목표 지표:**

- Intra-class distance: < 0.3 (같은 사용자 이미지끼리 가까움)
- Inter-class distance: > 0.7 (다른 사용자끼리 멀리)
- Separation: > 0.4

### 3. 시각화

```python
from src.evaluation.visualize import plot_tsne, plot_similarity_heatmap

# t-SNE 시각화
plot_tsne(embeddings, user_ids, save_path='logs/tsne_visualization.png')

# 유사도 히트맵 (샘플 사용자)
sample_users = user_ids[:20]
plot_similarity_heatmap(
    embeddings[:20], 
    sample_users,
    save_path='logs/similarity_heatmap.png'
)
```

## 추론 및 매칭

### 1. 매칭 엔진 구축

```python
# matching_demo.ipynb
from src.inference.matcher import MatchingEngine
import pandas as pd

# 매칭 엔진 초기화
engine = MatchingEngine(
    model_path='models/saved_models/best_model.pth',
    device='cuda'
)

# 모든 검증 데이터로 인덱스 구축
val_metadata = pd.read_csv('data/processed/val_metadata.csv')
for _, row in val_metadata.iterrows():
    engine.add_user(row['user_id'], row['filepath'])

# 인덱스 구축
engine.build_index()
print(f"Index built with {len(engine.user_embeddings)} users")
```

### 2. 매칭 테스트

```python
# 특정 사용자에 대한 Top-K 매칭
query_user = 'user_001'
matches = engine.find_matches(query_user, top_k=10)

print(f"Top 10 matches for {query_user}:")
for i, (user_id, similarity) in enumerate(matches, 1):
    print(f"  {i}. {user_id}: {similarity:.4f}")
```

**예상 출력:**

```
Top 10 matches for user_001:
  1. user_042: 0.8912
  2. user_137: 0.8654
  3. user_089: 0.8432
  ...
```

### 3. 시각적 확인

```python
from PIL import Image
import matplotlib.pyplot as plt

# 쿼리 사용자와 매칭 결과 시각화
fig, axes = plt.subplots(1, 6, figsize=(18, 3))

# 쿼리 이미지
query_img = Image.open(f'data/raw/profiles/{query_user}_1.jpg')
axes[0].imshow(query_img)
axes[0].set_title(f'Query: {query_user}')
axes[0].axis('off')

# Top-5 매칭
for i, (match_id, sim) in enumerate(matches[:5], 1):
    match_img = Image.open(f'data/raw/profiles/{match_id}_1.jpg')
    axes[i].imshow(match_img)
    axes[i].set_title(f'{match_id}\n{sim:.3f}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('logs/matching_results.png')
plt.show()
```

### 4. 인덱스 저장 및 로드

```python
# 인덱스 저장
engine.save_index('models/matching_index.pkl')

# 나중에 로드
engine_new = MatchingEngine.load_index(
    'models/matching_index.pkl',
    'models/saved_models/best_model.pth'
)
```

## 문제 해결

### GPU 메모리 부족

```yaml
# config.yaml에서 batch size 줄이기
training:
  batch_size: 8  # 16에서 8로 줄임
```

또는 Gradient Accumulation 사용:

```python
# train_sagemaker.py에서
gradient_accumulation_steps = 4  # Effective batch = 8 × 4 = 32
```

### 모델 다운로드 실패

HuggingFace 토큰 설정:

```bash
# SageMaker Terminal에서
export HF_TOKEN=your_huggingface_token

# 또는 Python에서
from huggingface_hub import login
login(token="your_token")
```

### 학습이 너무 느림

1. **Spot 인스턴스 사용**: 비용 70% 절감
2. **Mixed Precision Training**: FP16 사용으로 2배 속도 향상
3. **Vision Encoder Freeze**: Stage 1에서 freeze로 빠른 학습

## 다음 단계

### 프로젝트 완료 후

1. **성능 분석**: 베이스라인 대비 개선 확인
2. **보고서 작성**: 실험 결과 및 인사이트 정리
3. **코드 정리**: 주석 추가 및 문서화

### 향후 확장 (선택사항)

1. **실제 사용자 피드백 수집**: 좋아요/패스 데이터로 재학습
2. **멀티모달 확장**: 텍스트 프로필 정보 추가
3. **API 서버 구축**: FastAPI로 REST API 제공
4. **프로덕션 배포**: Docker + Kubernetes

더 자세한 내용은 다음 문서를 참고하세요:

- [아키텍처 설계](ARCHITECTURE.md)
- [데이터 명세](DATA_SPEC.md)
- [SageMaker 가이드](SAGEMAKER_GUIDE.md)
- [프로젝트 컨텍스트](PROJECT_CONTEXT.md)
