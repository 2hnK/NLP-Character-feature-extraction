# Scripts Directory

이 디렉토리는 프로젝트의 데이터 전처리, 모델 학습, 그리고 각 컴포넌트와 파이프라인을 검증하기 위한 스크립트들을 포함하고 있습니다.

## 🧪 테스트 스크립트 (권장 실행 순서)

전체 학습 파이프라인을 돌리기 전에, 아래 순서대로 테스트를 진행하여 각 모듈이 정상 작동하는지 확인하는 것을 권장합니다.

### 1. `test_triplet_components.py`
**역할**: 학습에 필요한 핵심 컴포넌트들의 단위 테스트를 수행합니다.
- **검증 항목**:
  - `PKSampler`: 배치 내 클래스($P$)와 샘플($K$)이 균형 있게 샘플링되는지 확인.
  - `ProjectionHead`: 임베딩 차원 변환 및 L2 정규화(Normalization) 동작 확인.
  - `OnlineTripletLoss`: 마진(margin) 기반의 손실 계산 및 Triplet 마이닝 로직 확인.
- **실행**:
  ```bash
  python scripts/test_triplet_components.py
  ```

### 2. `test_qwen_model.py`
**역할**: Qwen3-VL 백본 모델의 로딩 및 기본 추론 기능을 검증합니다.
- **검증 항목**:
  - HuggingFace로부터 모델 및 프로세서 다운로드/로드.
  - 더미 이미지를 이용한 Forward Pass 성공 여부.
  - 배치 단위 추론 및 임베딩 출력 형태(Shape) 확인.
  - 체크포인트 저장 및 로드 테스트.
  - GPU 메모리 사용량 체크.
- **실행**:
  ```bash
  python scripts/test_qwen_model.py
  ```

### 3. `test_train_with_s3.py`
**역할**: S3 데이터셋과 연동하여 실제 학습 루프가 돌아가는지 통합 테스트를 수행합니다.
- **검증 항목**:
  - AWS S3 버킷 연결 및 이미지 데이터 로딩.
  - `S3Dataset` 및 `DataLoader` 정상 작동 확인.
  - 모델 Forward/Backward Pass 및 Optimizer 업데이트 단계(Step) 실행 확인.
- **실행**:
  ```bash
  python scripts/test_train_with_s3.py
  ```

---

## 🛠 데이터 처리 및 학습 스크립트

### `train.py`
**역할**: Triplet Loss를 이용한 메인 학습 스크립트입니다.
- **기능**:
  - S3 데이터셋 로드.
  - Qwen3-VL 모델 파인튜닝 (또는 어댑터 학습).
  - Metric Learning 학습 루프 실행.
  - 모델 체크포인트 저장.
- **사용법**:
  ```bash
  python scripts/train.py --jsonl_path <path> --bucket_name <bucket> ...
  ```
