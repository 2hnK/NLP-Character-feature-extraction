# NLP Character Feature Extraction

> **Qwen3-VL 기반 캐릭터 스타일 특징 추출 및 Metric Learning 프로젝트**  
> AWS S3 데이터와 Triplet Loss를 활용하여 패션 스타일/분위기 기반의 임베딩 모델을 학습합니다.

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [설치 및 환경 설정](#-설치-및-환경-설정)
- [데이터 준비](#-데이터-준비)
- [학습 실행](#-학습-실행)
- [프로젝트 구조](#-프로젝트-구조)

---

## 🚀 프로젝트 개요

이 프로젝트는 **Qwen3-VL** 멀티모달 모델을 백본으로 사용하여, 이미지와 텍스트 설명에서 캐릭터의 **패션 스타일(Fashion Style)** 및 **분위기(Vibe)** 특징을 추출하는 모델을 학습합니다.  
**Triplet Loss**와 **Online Mining** 기법을 적용하여, 동일한 스타일을 가진 이미지는 가깝게, 다른 스타일은 멀게 임베딩 공간에 배치하도록 학습합니다.

---

## ✨ 주요 기능

1.  **Qwen3-VL Backbone**: 강력한 Vision-Language 모델을 특징 추출기로 사용.
2.  **Triplet Loss & Online Mining**: `pytorch-metric-learning` 라이브러리를 활용한 안정적인 Metric Learning 구현.
3.  **S3 Data Pipeline**: AWS S3에서 이미지를 직접 로드하고, 로컬 JSONL 메타데이터와 연동.
4.  **Balanced Batch Sampling (PKSampler)**: 각 배치에 $P$개의 클래스와 $K$개의 샘플을 보장하여 학습 안정성 확보.
5.  **Label Encoding & Text Formatting**: 문자열 라벨 자동 인코딩 및 텍스트 입력 포맷팅 지원.

---

## 🛠 설치 및 환경 설정
# NLP Character Feature Extraction

> **Qwen3-VL 기반 캐릭터 스타일 특징 추출 및 Metric Learning 프로젝트**  
> AWS S3 데이터와 Triplet Loss를 활용하여 패션 스타일/분위기 기반의 임베딩 모델을 학습합니다.

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [설치 및 환경 설정](#-설치-및-환경-설정)
- [데이터 준비](#-데이터-준비)
- [학습 실행](#-학습-실행)
- [프로젝트 구조](#-프로젝트-구조)

---

## 🚀 프로젝트 개요

이 프로젝트는 **Qwen3-VL** 멀티모달 모델을 백본으로 사용하여, 이미지와 텍스트 설명에서 캐릭터의 **패션 스타일(Fashion Style)** 및 **분위기(Vibe)** 특징을 추출하는 모델을 학습합니다.  
**Triplet Loss**와 **Online Mining** 기법을 적용하여, 동일한 스타일을 가진 이미지는 가깝게, 다른 스타일은 멀게 임베딩 공간에 배치하도록 학습합니다.

---

## ✨ 주요 기능

1.  **Qwen3-VL Backbone**: 강력한 Vision-Language 모델을 특징 추출기로 사용.
2.  **Triplet Loss & Online Mining**: `pytorch-metric-learning` 라이브러리를 활용한 안정적인 Metric Learning 구현.
3.  **S3 Data Pipeline**: AWS S3에서 이미지를 직접 로드하고, 로컬 JSONL 메타데이터와 연동.
4.  **Balanced Batch Sampling (PKSampler)**: 각 배치에 $P$개의 클래스와 $K$개의 샘플을 보장하여 학습 안정성 확보.
5.  **Label Encoding & Text Formatting**: 문자열 라벨 자동 인코딩 및 텍스트 입력 포맷팅 지원.

---

## 🛠 설치 및 환경 설정

### 필수 요구 사항
- Python 3.8+
- PyTorch 2.0+
- CUDA (권장)

### 설치
```bash
# 저장소 클론
git clone <repository-url>
cd NLP-Character-feature-extraction

# Conda 가상환경 활성화
conda activate drl

# 의존성 설치
pip install -r requirements.txt
```

---

## 📊 데이터 준비

학습을 위해서는 S3에 저장된 이미지와 이에 대응하는 JSONL 메타데이터 파일이 필요합니다.

### 1. 메타데이터 전처리
원본 JSONL 파일에는 이미지 파일명이나 정수형 라벨이 없을 수 있습니다. 전처리 스크립트를 통해 이를 생성합니다.

```bash
python scripts/preprocess_jsonl.py \
    --input enhanced_train_batch_44_94.jsonl \
    --output train_processed.jsonl \
    --mapping_output label_mapping.json
```

- **`--input`**: 원본 JSONL 파일 경로
- **`--output`**: 전처리된 JSONL 파일 저장 경로 (파일명 `filename` 및 포맷팅된 `text_input` 추가됨)
- **`--mapping_output`**: 라벨(스타일)과 정수 인덱스 매핑 정보 저장 경로

### 2. 생성된 파일 예시
- **`train_processed.jsonl`**:
  ```json
  {
    "filename": "aug_00000.jpg",
    "image_metadata": { "fashion_style": "Dandy_Minimal", ... },
    "text_input": "Style: Dandy_Minimal. Features: ... Vibe: Warm_Friendly."
  }
  ```
- **`label_mapping.json`**:
  ```json
  {
    "Casual_Basic": 0,
    "Dandy_Minimal": 1,
    ...
  }
  ```

---

## 🏃‍♂️ 학습 실행

전처리된 데이터와 S3 버킷 정보를 사용하여 학습을 시작합니다.

```bash
python scripts/train.py \
    --jsonl_path train_processed.jsonl \
    --bucket_name sometimes-ki-datasets \
    --prefix "characters/augmented/generated/" \
    --p 8 \
    --k 4 \
    --epochs 10 \
    --lr 1e-5 \
    --margin 0.2
```

### 주요 인자 설명
- **`--jsonl_path`**: 전처리된 메타데이터 파일 경로
- **`--bucket_name`**: AWS S3 버킷 이름
- **`--prefix`**: 이미지가 저장된 S3 경로 접두사
- **`--p`**: 배치 당 클래스(스타일) 개수 (기본값: 8)
- **`--k`**: 클래스 당 샘플 이미지 개수 (기본값: 4)
  - *Batch Size = P × K*
- **`--margin`**: Triplet Loss 마진 (기본값: 0.2)

---

## 📂 프로젝트 구조

```
NLP-Character-feature-extraction/
├── scripts/
│   ├── preprocess_jsonl.py      # 데이터 전처리 및 라벨 매핑 생성
│   ├── train.py                 # Triplet Loss 학습 스크립트
│   ├── test_triplet_components.py # 컴포넌트 단위 테스트
│   ├── test_qwen_model.py       # Qwen 모델 로드 및 추론 테스트
│   ├── test_train_with_s3.py    # S3 연동 학습 루프 통합 테스트
│   └── README.md                # 스크립트 사용 가이드
├── src/
│   ├── data/
│   │   ├── s3_dataset.py        # S3 데이터셋 및 이미지 로더
│   │   └── sampler.py           # PKSampler (Balanced Batch)
│   ├── models/
│   │   ├── losses.py            # OnlineTripletLoss 래퍼
│   │   ├── projection.py        # Projection Head 모듈
│   │   └── qwen_backbone.py     # Qwen3-VL 백본 모델
├── requirements.txt             # 프로젝트 의존성
├── label_mapping.json           # (생성됨) 라벨 매핑 파일
├── train_processed.jsonl        # (생성됨) 전처리된 메타데이터
└── README.md                    # 메인 문서
```

---

## 🧪 검증 및 테스트

구현된 컴포넌트와 파이프라인이 정상 작동하는지 확인하려면 다음 스크립트들을 순서대로 실행해보세요.

### 1. 컴포넌트 단위 테스트
Sampler, Loss, ProjectionHead 등 핵심 모듈의 동작을 검증합니다.
```bash
python scripts/test_triplet_components.py
```

### 2. 모델 로드 및 추론 테스트
Qwen3-VL 모델이 정상적으로 로드되고 추론되는지 확인합니다.
```bash
python scripts/test_qwen_model.py
```

### 3. S3 연동 학습 통합 테스트
S3에서 데이터를 가져와 학습 루프가 정상적으로 도는지 확인합니다.
```bash
python scripts/test_train_with_s3.py
```