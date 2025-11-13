# Getting Started Guide

빠르게 시작하기 위한 가이드입니다.

## 목차

1. [환경 설정](#환경-설정)
2. [데이터 준비](#데이터-준비)
3. [모델 학습](#모델-학습)
4. [모델 평가](#모델-평가)
5. [API 서버 실행](#api-서버-실행)

## 환경 설정

### 1. 저장소 클론 및 이동

```bash
cd dating-profile-matcher
```

### 2. 가상환경 생성 및 활성화

```bash
# Python 가상환경 생성
python -m venv venv

# 활성화 (Linux/Mac)
source venv/bin/activate

# 활성화 (Windows)
venv\Scripts\activate
```

### 3. 의존성 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt

# GPU 사용 시 Faiss GPU 버전 설치 (선택)
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 4. 설치 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 데이터 준비

### 1. 데이터 디렉토리 구조 생성

```bash
mkdir -p data/raw/profiles
mkdir -p data/processed
```

### 2. 원본 이미지 및 메타데이터 준비

원본 프로필 이미지를 `data/raw/profiles/`에 배치하고, 메타데이터 CSV를 생성합니다.

**metadata.csv 예시:**

```csv
user_id,image_path,gender,age,face_detected,face_confidence,image_quality,is_synthetic,source
user_001,user_001.jpg,F,28,True,0.95,0.87,False,real
user_002,user_002.jpg,M,32,True,0.89,0.82,False,real
user_003,user_003.jpg,F,25,True,0.91,0.79,False,real
```

저장 위치: `data/raw/metadata.csv`

**interactions.csv 예시 (선택):**

```csv
interaction_id,user_id,target_user_id,action,timestamp,is_mutual,conversation_started
int_001,user_001,user_002,like,2024-01-15 15:45:30,True,True
int_002,user_001,user_003,pass,2024-01-15 15:46:10,False,False
```

저장 위치: `data/raw/interactions.csv`

### 3. 이미지 전처리 (얼굴 검출 및 크롭)

```bash
python src/data/preprocessing.py \
    --input_dir data/raw/profiles \
    --output_dir data/processed \
    --metadata_csv data/raw/metadata.csv \
    --output_metadata_csv data/processed/metadata.csv \
    --image_size 224 \
    --detector mtcnn
```

**처리 결과:**
- 전처리된 이미지: `data/processed/`
- 성공한 이미지의 메타데이터: `data/processed/metadata.csv`
- 디버그 정보: `data/processed/metadata_debug.csv`

### 4. 데이터 분할 (Train/Val/Test)

```python
# 간단한 분할 스크립트 (Python 인터프리터에서 실행)
import pandas as pd
from sklearn.model_selection import train_test_split

# 메타데이터 로드
df = pd.read_csv('data/processed/metadata.csv')

# 사용자 ID 기준 분할
user_ids = df['user_id'].unique()

train_users, temp_users = train_test_split(user_ids, test_size=0.3, random_state=42)
val_users, test_users = train_test_split(temp_users, test_size=0.5, random_state=42)

# 분할된 데이터 저장
train_df = df[df['user_id'].isin(train_users)]
val_df = df[df['user_id'].isin(val_users)]
test_df = df[df['user_id'].isin(test_users)]

train_df.to_csv('data/processed/train_metadata.csv', index=False)
val_df.to_csv('data/processed/val_metadata.csv', index=False)
test_df.to_csv('data/processed/test_metadata.csv', index=False)

print(f"Train: {len(train_df)} images, {len(train_users)} users")
print(f"Val: {len(val_df)} images, {len(val_users)} users")
print(f"Test: {len(test_df)} images, {len(test_users)} users")
```

## 모델 학습

### 1. 설정 파일 확인 및 수정

`configs/config.yaml` 파일을 열어서 경로와 하이퍼파라미터를 확인합니다.

```yaml
# 주요 설정 항목
paths:
  processed_data: "data/processed"

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 1e-4

model:
  backbone: "efficientnet_b0"
  embedding_dim: 512
```

### 2. 학습 시작

```bash
# 기본 학습
python src/training/train.py --config configs/config.yaml

# Weights & Biases 로깅 비활성화
# config.yaml에서 logging.wandb.enabled: false로 설정
```

### 3. 학습 중단 후 재개

```bash
python src/training/train.py \
    --config configs/config.yaml \
    --resume models/checkpoints/last.pth
```

### 4. 학습 모니터링

**TensorBoard:**
```bash
tensorboard --logdir logs/tensorboard
# 브라우저에서 http://localhost:6006 접속
```

**Weights & Biases:**
- W&B 대시보드에서 실시간 모니터링
- wandb.ai에서 확인

### 5. 학습 결과 확인

```bash
# 체크포인트 확인
ls -lh models/checkpoints/

# 최종 모델 확인
ls -lh models/saved_models/
```

## 모델 평가

### 1. 평가 스크립트 작성

```python
# evaluate.py (간단한 평가 예시)
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.backbone import ProfileFeatureExtractor
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import evaluate_model

# 설정
model_path = 'models/saved_models/best_model.pth'
device = 'cuda'

# 모델 로드
model = ProfileFeatureExtractor.load_from_checkpoint(model_path, device=device)

# 데이터 로더 생성
_, val_loader = create_dataloaders(
    data_root='data/processed',
    metadata_csv='data/processed/val_metadata.csv',
    batch_size=64,
    num_workers=4,
    image_size=224,
    dataset_type='online_triplet'
)

# 평가
print("Evaluating model...")
metrics = evaluate_model(model, val_loader, device=device)

print("\nEvaluation Results:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")
```

```bash
python evaluate.py
```

### 2. 예상 출력

```
Evaluation Results:
  intra_class_mean: 0.2543
  inter_class_mean: 0.8721
  separation_ratio: 3.4301
  top_1_accuracy: 0.7234
  top_5_accuracy: 0.8912
  top_10_accuracy: 0.9401
  mAP: 0.8123
  silhouette_score: 0.6543
```

## API 서버 실행

### 1. 환경 변수 설정

```bash
export MODEL_PATH=models/saved_models/best_model.pth
export DEVICE=cuda
export INDEX_TYPE=Flat
```

### 2. 서버 시작

```bash
# 개발 모드 (자동 재시작)
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

# 프로덕션 모드
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. API 문서 확인

브라우저에서 접속:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. API 사용 예시

**Health Check:**
```bash
curl http://localhost:8000/
```

**특징 추출:**
```bash
curl -X POST "http://localhost:8000/extract_features" \
  -F "image=@path/to/profile.jpg" \
  -F "user_id=user_123"
```

**사용자 추가:**
```bash
curl -X POST "http://localhost:8000/add_user?user_id=user_123" \
  -F "file=@path/to/profile.jpg"
```

**매칭 조회:**
```bash
curl "http://localhost:8000/matches/user_123?top_k=10"
```

**선호도 업데이트:**
```bash
curl -X POST "http://localhost:8000/update_preferences" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "liked_users": ["user_456", "user_789"],
    "passed_users": ["user_111"]
  }'
```

## Python 스크립트에서 사용

```python
from src.models.backbone import ProfileFeatureExtractor
from src.inference.matcher import MatchingEngine
import torch

# 1. 모델 로드
model = ProfileFeatureExtractor.load_from_checkpoint(
    'models/saved_models/best_model.pth',
    device='cuda'
)

# 2. 매칭 엔진 초기화
matcher = MatchingEngine(
    model_path='models/saved_models/best_model.pth',
    device='cuda',
    index_type='Flat'
)

# 3. 사용자 추가
matcher.add_user('user_001', 'data/processed/user_001.jpg')
matcher.add_user('user_002', 'data/processed/user_002.jpg')
matcher.add_user('user_003', 'data/processed/user_003.jpg')

# 4. 인덱스 구축
matcher.build_index_from_users()

# 5. 매칭 조회
matches = matcher.find_matches('user_001', top_k=5)
print("Matches for user_001:")
for user_id, similarity in matches:
    print(f"  {user_id}: {similarity:.4f}")

# 6. 선호도 업데이트
matcher.update_preferences(
    user_id='user_001',
    liked_users=['user_002']
)

# 7. 개인화된 매칭 조회
matches = matcher.find_matches('user_001', top_k=5)
print("\nPersonalized matches for user_001:")
for user_id, similarity in matches:
    print(f"  {user_id}: {similarity:.4f}")

# 8. 인덱스 저장
matcher.save_index('models/faiss_index.bin', 'models/index_metadata.npy')
```

## 문제 해결

### GPU 메모리 부족

```yaml
# config.yaml에서 batch size 줄이기
training:
  batch_size: 16  # 32에서 16으로
```

### 얼굴 검출 실패

```bash
# 더 관대한 설정으로 재실행
python src/data/preprocessing.py \
    --input_dir data/raw/profiles \
    --output_dir data/processed \
    --metadata_csv data/raw/metadata.csv \
    --output_metadata_csv data/processed/metadata.csv \
    --min_quality 0.2  # 기본값 0.3에서 낮춤
```

### 학습이 너무 느림

```yaml
# config.yaml에서 workers 수 조정
data:
  num_workers: 8  # CPU 코어 수에 맞게 조정
```

## 다음 단계

1. **데이터 증강**: 생성형 AI로 추가 데이터 생성
2. **멀티태스크 학습**: 나이, 스타일 등 보조 태스크 추가
3. **하이퍼파라미터 튜닝**: Optuna로 최적 파라미터 탐색
4. **A/B 테스트**: 실제 환경에서 성능 검증
5. **프로덕션 배포**: Docker, Kubernetes 활용

더 자세한 내용은 다음 문서를 참고하세요:
- [아키텍처 설계](ARCHITECTURE.md)
- [데이터 명세](DATA_SPEC.md)
- [API 문서](http://localhost:8000/docs)
