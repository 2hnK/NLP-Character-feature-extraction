# Dating Profile Matcher

> **외모 유사도 기반 데이팅 앱 매칭률 향상 시스템**  
> AWS SageMaker, Qwen3-VL-2B 기반 프로필 사진 특징 추출 및 매칭 프로젝트

---

## 목차

- [프로젝트 개요](#-프로젝트-개요)
- [핵심 가설](#-핵심-가설)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [빠른 시작](#-빠른-시작)
- [개발 환경 설정](#-개발-환경-설정)
- [데이터 구조](#-데이터-구조)
- [주요 컴포넌트](#-주요-컴포넌트)
- [개발 가이드](#-개발-가이드)
- [성능 지표](#-성능-지표)
- [프로젝트 로드맵](#-프로젝트-로드맵)
- [문서](#-문서)
- [참고 자료](#-참고-자료)

---

## 프로젝트 개요

데이팅 앱에서 **프로필 사진의 시각적 특징**을 학습하여, 서로 선호할 만한 외모 스타일을 가진 사용자끼리 매칭하는 딥러닝 시스템입니다.

### 핵심 기능

- **이미지 임베딩 추출**: Qwen3-VL-2B 모델로 프로필 사진 → 512차원 벡터 변환
- **유사도 기반 매칭**: Cosine Similarity를 활용한 Top-K 사용
- **Metric Learning**: Triplet Loss로 같은 스타일끼리 가깝게 학습
- **매칭 시뮬레이션**: 임베딩 기반 추천 시스템 프로토타입

---

## 핵심 가설

> **"외적으로 뛰어난 사람은 상대의 외모가 뛰어나길 바라는 경향이 많다"**

→  비슷한 매력도/스타일의 사용자끼리 매칭하면  
→ 좋아요 비율이 증가할 것

---

## 기술 스택

### 핵심 기술
- **프레임워크**: PyTorch 2.0+
- **모델**: Qwen3-VL-2B-Instruct-FP8 (Vision-Language Model)
- **학습 방법**: Triplet Loss / Metric Learning
- **플랫폼**: AWS SageMaker AI Studio (GPU 학습)

### 라이브러리
```python
# 핵심
torch>=2.0.0
transformers>=4.37.2
pillow>=10.2.0
numpy>=1.24.0

# SageMaker
sagemaker>=2.190.0
boto3>=1.28.0

# 실험 관리 (선택)
wandb>=0.15.0

# 시각화
matplotlib>=3.8.2
seaborn>=0.13.1
scikit-learn>=1.4.0
```

### 데이터
- **학습 데이터**: 증강된 이미지 3,200개 (생성형 AI로 생성)
- **검증 데이터**: 실제 사용자 이미지 100개
- **메타데이터**: user_id 기반 그룹화

---

## 📁 프로젝트 구조

```
dating-profile-matcher/
├── data/                           # 데이터 디렉토리
│   ├── raw/
│   │   └── profiles/              # 실제 사용자 이미지 (100개)
│   ├── augmented/
│   │   └── generated/             # 증강 이미지 (3,200개)
│   └── processed/
│       ├── train/                 # 전처리된 학습 데이터
│       ├── val/                   # 전처리된 검증 데이터
│       ├── train_metadata.csv     # 학습 메타데이터
│       └── val_metadata.csv       # 검증 메타데이터
│
├── models/                         # 모델 저장
│   ├── checkpoints/               # 학습 중 체크포인트
│   │   ├── epoch_2.pt
│   │   ├── epoch_4.pt
│   │   └── ...
│   └── saved_models/              # 최종 모델
│       ├── best_model.pt          # Best validation loss 모델
│       ├── baseline_embeddings.pt  # 베이스라인 임베딩
│       └── matching_index.pt      # 매칭 인덱스
│
├── src/                           # 소스 코드
│   ├── data/
│   │   ├── dataset.py            # TripletDataset 클래스
│   │   └── preprocessing.py      # 이미지 전처리 파이프라인
│   ├── models/
│   │   ├── feature_extractor.py  # Qwen 기반 Feature Extractor
│   │   └── losses.py             # TripletLoss 구현
│   ├── training/
│   │   ├── trainer.py            # Training Loop 클래스
│   │   └── utils.py              # 학습 유틸리티
│   ├── evaluation/
│   │   ├── metrics.py            # 평가 지표 (Intra/Inter distance)
│   │   └── visualize.py          # t-SNE, 유사도 히트맵
│   └── inference/
│       └── matcher.py            # MatchingEngine 클래스
│
├── notebooks/                     # Jupyter 노트북 (SageMaker용)
│   ├── 01_data_exploration.ipynb  # 데이터 탐색
│   ├── 02_model_loading.ipynb     # Qwen 모델 로드 테스트
│   ├── 03_training.ipynb          # 학습 실행
│   └── 04_evaluation.ipynb        # 평가 및 시각화
│
├── configs/
│   └── config.yaml               # 하이퍼파라미터 설정
│
├── logs/                          # 로그 및 시각화 결과
│   ├── training_losses.png
│   ├── baseline_embeddings.png
│   ├── similarity_heatmap.png
│   └── evaluation_results.json
│
├── docs/                          # 프로젝트 문서
│   ├── PROJECT_CONTEXT.md        # 프로젝트 전체 맥락
│   ├── TECHNICAL_GUIDE.md        # 기술 가이드 (상세 설명)
│   ├── WORKFLOW.md               # 단계별 작업 가이드
│   └── GLOSSARY.md               # 용어 사전
│
├── requirements.txt               # Python 의존성
├── .gitignore                     # Git 제외 파일
└── README.md                      # 이 파일
```

---

## 개발 환경 설정

### 옵션 A: AWS SageMaker Studio (GPU 학습)

#### 1. SageMaker Studio 접속
```
1. AWS 콘솔 → SageMaker → Studio
2. "Open Studio" 클릭
3. Domain/User 선택
```

#### 2. 프로젝트 클론 & 설정
```bash
# SageMaker Studio Terminal
cd ~/SageMaker
git clone <your-repo-url> dating-profile-matcher
cd dating-profile-matcher

# 환경 설정
pip install -r requirements.txt

# 디렉토리 생성
mkdir -p data/{raw/profiles,augmented/generated,processed/{train,val}}
mkdir -p models/{checkpoints,saved_models}
mkdir -p logs
```

#### 3. Notebook 환경
- **Kernel**: Python 3 (PyTorch 2.0 GPU Optimized)
- **Instance**: 
  - 개발: `ml.t3.medium` (CPU, 저렴)
  - 학습: `ml.g5.xlarge` (1 GPU, ~$1/hour)

### 옵션 B: 로컬 환경 (VSCode + Claude Code)

#### 1. 저장소 클론
```bash
git clone <your-repo-url>
cd dating-profile-matcher
```

#### 2. 가상환경 생성
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

---

## 데이터 구조

### 파일명 규칙
```
# 실제 사용자 이미지
user_{user_id}_{image_idx}.jpg

예시:
- user_001_1.jpg  (user_001의 첫 번째 사진)
- user_001_2.jpg  (user_001의 두 번째 사진)
- user_042_1.jpg  (user_042의 첫 번째 사진)

# 증강 이미지
gen_{idx}.jpg

예시:
- gen_0001.jpg
- gen_0002.jpg
```

### 메타데이터 (CSV)
```csv
filename,user_id,image_idx,filepath
user_001_1.jpg,user_001,1,data/processed/train/user_001_1.jpg
user_001_2.jpg,user_001,2,data/processed/train/user_001_2.jpg
user_042_1.jpg,user_042,1,data/processed/train/user_042_1.jpg
```

### 데이터 통계
```
총 이미지: 3,300개
├── 학습용 (증강): 3,200개
│   └── 예상 사용자 수: 약 1,000명 (평균 3장/사용자)
└── 검증용 (실제): 100개
    └── 사용자 수: 약 30-50명 (1-3장/사용자)
```

---

## 주요 컴포넌트

### 1. TripletDataset (`src/data/dataset.py`)
```python
# (Anchor, Positive, Negative) 조합 생성
dataset = TripletDataset(
    metadata_csv="data/processed/train_metadata.csv",
    image_dir="data/processed/train",
    transform=train_transform
)

# 샘플 출력
anchor, positive, negative = dataset[0]
# anchor: user_001_1.jpg
# positive: user_001_2.jpg (같은 사용자)
# negative: user_042_1.jpg (다른 사용자)
```

### 2. FeatureExtractor (`src/models/feature_extractor.py`)
```python
# Qwen Vision Encoder 기반 임베딩 추출
extractor = FeatureExtractor(vision_model, processor)

# 단일 이미지
embedding = extractor.extract_from_path("user_001.jpg")
# embedding.shape: [512]

# 배치 처리
embeddings = extractor.extract_batch_from_paths(
    image_paths, 
    batch_size=32
)
# embeddings.shape: [N, 512]
```

### 3. Trainer (`src/training/trainer.py`)
```python
# 학습 설정
config = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'margin': 1.0,
    'epochs': 10
}

trainer = Trainer(model, train_loader, val_loader, config)
train_losses, val_losses = trainer.train(num_epochs=10)
```

### 4. MatchingEngine (`src/inference/matcher.py`)
```python
# 매칭 엔진
engine = MatchingEngine(model, processor)
engine.build_index("data/processed/val")

# Top-K 추천
matches = engine.search("user_001", top_k=5)
# [('user_042', 0.89), ('user_137', 0.85), ...]
```

---

## 워크플로우

```
Phase 0: 환경 설정 (1-2일)
  └─→ SageMaker Studio 또는 로컬 환경 구축
  └─→ 데이터 업로드
  └─→ 기본 테스트

Phase 1: 데이터 준비 (2-3일)
  └─→ 데이터 탐색 (EDA)
  └─→ 전처리 파이프라인 구현
  └─→ TripletDataset 구현
  └─→ DataLoader 테스트

Phase 2: 베이스라인 구축 (3-4일)
  └─→ Qwen 모델 로드
  └─→ Feature Extraction 구현
  └─→ 유사도 계산 검증
  └─→ 베이스라인 성능 측정

Phase 3: Fine-tuning (5-7일)
  └─→ Triplet Loss 구현
  └─→ Trainer 클래스 구현
  └─→ 모델 학습 실행
  └─→ 하이퍼파라미터 실험

Phase 4: 평가 및 검증 (2-3일)
  └─→ Fine-tuned 모델 평가
  └─→ 베이스라인 대비 성능 비교
  └─→ 시각화 (t-SNE, 히트맵)
  └─→ 매칭 시뮬레이션

Phase 5: 문서화 (2-3일)
  └─→ 코드 정리 및 주석
  └─→ 최종 보고서 작성
  └─→ 발표 자료 준비
```

---

## 성능 지표

### 평가 메트릭

#### 1. 임베딩 품질
```python
# 같은 사용자 내 거리 (작을수록 좋음)
Intra-class distance = 평균(같은 사용자 사진 간 거리)
목표: < 0.3

# 다른 사용자 간 거리 (클수록 좋음)
Inter-class distance = 평균(다른 사용자 사진 간 거리)
목표: > 0.7

# 분리도 (클수록 좋음)
Separation = Inter-class - Intra-class
목표: > 0.4
```

#### 2. 비즈니스 KPI
```
좋아요 비율 = (좋아요 수) / (매칭 추천 수) × 100%
```

---

## 문서

### 필수 문서 (먼저 읽기)

1. 

### 참고 자료

- **[Qwen-VL GitHub](https://github.com/QwenLM/Qwen-VL)**: 공식 문서
- **[SageMaker 가이드](https://docs.aws.amazon.com/sagemaker/)**: AWS 공식 문서
- **[PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)**: 라이브러리
- **[PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)**: 프로젝트 목표, 데이터, 모델 아키텍처

---

## 참고 논문

### Metric Learning
- **[FaceNet](https://arxiv.org/abs/1503.03832)**: Triplet Loss의 원조
- **[Deep Metric Learning Survey](https://arxiv.org/abs/2002.08473)**: 전체 개요

### Vision Models
- **[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)**: Transformer for Images
- **[CLIP](https://arxiv.org/abs/2103.00020)**: Vision-Language Learning