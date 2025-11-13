# Dating Profile Matcher

데이팅 앱 매칭률 향상을 위한 프로필 사진 특징 추출 시스템

## 프로젝트 개요

사용자의 프로필 사진에서 시각적 특징을 추출하여 개인화된 매칭 추천을 제공하는 딥러닝 시스템입니다.

### 핵심 기능

- **특징 추출**: 프로필 사진에서 512차원 임베딩 벡터 추출
- **유사도 기반 매칭**: Cosine similarity를 활용한 매칭 점수 계산
- **개인화 추천**: 사용자 행동 피드백 기반 선호도 학습
- **실시간 추론**: FastAPI 기반 REST API 제공

## 기술 스택

- **프레임워크**: PyTorch 2.0+
- **모델 아키텍처**:
  - **Qwen3-VL-2B-Instruct-FP8**: 비전-언어 모델 (SageMaker)
  - EfficientNet-B0 + Metric Learning (로컬 학습)
- **학습 방법**: Triplet Loss / Contrastive Learning / Online Triplet Mining
- **벡터 검색**: Faiss
- **클라우드 플랫폼**: AWS SageMaker AI Studio
- **API 서버**: FastAPI
- **실험 관리**: Weights & Biases / SageMaker Experiments

## 프로젝트 구조

```
dating-profile-matcher/
├── data/                      # 데이터 디렉토리
│   ├── raw/                   # 원본 이미지
│   ├── processed/             # 전처리된 데이터
│   └── augmented/             # 증강 데이터
├── models/                    # 모델 저장
│   ├── checkpoints/           # 학습 체크포인트
│   └── saved_models/          # 최종 모델
├── src/                       # 소스 코드
│   ├── data/                  # 데이터 처리
│   │   ├── dataset.py         # Dataset 클래스
│   │   ├── preprocessing.py   # 전처리 파이프라인
│   │   └── augmentation.py    # 데이터 증강
│   ├── models/                # 모델 정의
│   │   ├── backbone.py        # Backbone 네트워크
│   │   ├── embedding.py       # Embedding 레이어
│   │   └── losses.py          # Loss 함수
│   ├── training/              # 학습 관련
│   │   ├── trainer.py         # Training loop
│   │   └── utils.py           # 유틸리티
│   ├── evaluation/            # 평가
│   │   ├── metrics.py         # 평가 지표
│   │   └── visualize.py       # 시각화
│   └── inference/             # 추론 API
│       ├── api.py             # FastAPI 서버
│       └── matcher.py         # 매칭 엔진
├── configs/                   # 설정 파일
│   └── config.yaml            # 하이퍼파라미터
├── notebooks/                 # Jupyter 노트북
│   └── exploration.ipynb      # 데이터 탐색
├── docs/                      # 문서
│   ├── ARCHITECTURE.md        # 아키텍처 설계
│   ├── DATA_SPEC.md           # 데이터 명세
│   └── API_SPEC.md            # API 명세
├── tests/                     # 테스트 코드
├── logs/                      # 로그 파일
├── requirements.txt           # 의존성
└── README.md                  # 프로젝트 설명
```

## 빠른 시작

### 옵션 A: AWS SageMaker Studio (권장)

```bash
# 1. SageMaker Studio에서 터미널 열기
cd ~/SageMaker/dating-profile-matcher

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 데이터 전처리
python src/data/preprocessing.py \
    --input_dir ~/SageMaker/dating-matcher/data/raw/profiles \
    --output_dir ~/SageMaker/dating-matcher/data/processed \
    --metadata_csv ~/SageMaker/dating-matcher/data/raw/metadata.csv \
    --output_metadata_csv ~/SageMaker/dating-matcher/data/processed/metadata.csv

# 4. Jupyter Notebook으로 학습 시작
# notebooks/sagemaker_training.ipynb 열기
```

**자세한 내용**: [SageMaker 사용 가이드](docs/SAGEMAKER_GUIDE.md)

### 옵션 B: 로컬 환경

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 데이터 전처리
python src/data/preprocessing.py --input data/raw --output data/processed

# 4. 모델 학습
python src/training/train.py --config configs/config.yaml

# 학습 재개
python src/training/train.py --config configs/config.yaml --resume models/checkpoints/last.pth
```

### 4. 모델 평가

```bash
# 평가 실행
python src/evaluation/evaluate.py --model models/saved_models/best_model.pth --data data/processed/test

# 시각화
python src/evaluation/visualize.py --model models/saved_models/best_model.pth
```

### 5. API 서버 실행

```bash
# FastAPI 서버 시작
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

# API 문서: http://localhost:8000/docs
```

## 사용 예제

### 특징 추출

```python
from src.models.embedding import ProfileFeatureExtractor
import torch
from PIL import Image

# 모델 로드
model = ProfileFeatureExtractor.load_from_checkpoint('models/saved_models/best_model.pth')
model.eval()

# 이미지 로드 및 전처리
image = Image.open('profile.jpg')
tensor = preprocess(image).unsqueeze(0)

# 특징 추출
with torch.no_grad():
    embedding = model(tensor)

print(f"Feature vector shape: {embedding.shape}")  # [1, 512]
```

### 매칭 추천

```python
from src.inference.matcher import MatchingEngine

# 매칭 엔진 초기화
matcher = MatchingEngine(model_path='models/saved_models/best_model.pth')

# 사용자별 특징 벡터 등록
matcher.add_user(user_id='user_001', image_path='user_001.jpg')
matcher.add_user(user_id='user_002', image_path='user_002.jpg')

# 매칭 추천
matches = matcher.recommend_matches(user_id='user_001', top_k=10)
print(matches)
# [('user_042', 0.89), ('user_137', 0.85), ...]
```

### REST API 호출

```bash
# 특징 추출
curl -X POST "http://localhost:8000/extract_features" \
  -F "image=@profile.jpg"

# 매칭 추천
curl -X GET "http://localhost:8000/matches/user_001?top_k=10"
```

## 성능 지표

### 현재 베이스라인 성능

- **임베딩 품질**: Intra-class distance < 0.3, Inter-class distance > 0.7
- **Retrieval Accuracy**: Top-1: 72%, Top-5: 89%, Top-10: 94%
- **추론 속도**: ~15ms/image (GPU), ~50ms/image (CPU)

### 목표 지표

- **비즈니스 KPI**: 매칭 성사율 20% 향상
- **사용자 참여도**: 좋아요 비율 15% 증가
- **시스템 성능**: P95 latency < 100ms

## 개발 로드맵

- [x] Phase 1: 프로젝트 세팅 및 베이스라인 구현
- [ ] Phase 2: 사용자 행동 데이터 통합
- [ ] Phase 3: 멀티태스크 학습 적용
- [ ] Phase 4: 모델 경량화 및 최적화
- [ ] Phase 5: 프로덕션 배포
- [ ] Phase 6: A/B 테스트 및 개선

## 기여 가이드

1. 이슈 생성 또는 기존 이슈 확인
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 연락처

프로젝트 관리자: [Your Name]
이메일: [your.email@example.com]

## 참고 자료

- [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Metric Learning Survey](https://arxiv.org/abs/2002.08473)
