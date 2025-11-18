# 베이스라인 코드 테스팅 가이드

데이터 준비 전에 베이스라인 코드가 정상 작동하는지 확인하는 단계별 가이드입니다.

---

## 📋 목차

1. [환경 확인](#1-환경-확인)
2. [이미지 전처리 (리사이징)](#2-이미지-전처리-리사이징)
3. [Qwen 모델 테스트](#3-qwen-모델-테스트)
4. [전체 파이프라인 테스트](#4-전체-파이프라인-테스트)
5. [문제 해결](#5-문제-해결)

---

## 1. 환경 확인

### 1.1 Python 환경
```bash
# Python 버전 확인 (3.8 이상 필요)
python --version

# 가상환경 활성화 (있는 경우)
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 1.2 필수 패키지 설치
```bash
# requirements.txt로 일괄 설치
pip install -r requirements.txt

# 또는 핵심 패키지만 설치
pip install torch torchvision transformers qwen-vl-utils pillow pandas numpy tqdm
```

### 1.3 GPU 확인 (선택사항, 권장)
```bash
# CUDA 사용 가능 여부 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# GPU 정보 확인
nvidia-smi

# 예상 출력:
# CUDA available: True
# CUDA version: 11.8
```

**GPU 메모리 권장사항:**
- **최소:** 8GB VRAM (RTX 3060 12GB, RTX 4060 Ti 등)
- **권장:** 16GB+ VRAM (RTX 4080, A100 등)
- **CPU만 사용 가능:** 테스트는 가능하지만 학습이 매우 느림

---

## 2. 이미지 전처리 (리사이징)

### 2.1 왜 필요한가?

**문제:**
- 원본 이미지가 2048x2048 같이 매우 큰 경우
- 매번 학습 시 로드해서 224x224로 리사이징하는 것은 비효율적
- 메모리 사용량 증가, 학습 속도 저하

**해결:**
- 사전에 이미지를 224x224로 리사이징해서 저장
- 학습 시 빠른 로딩 가능

### 2.2 실제 이미지 리사이징 (2048→224)

```bash
# 단일 디렉토리의 모든 이미지 리사이징
python scripts/resize_images.py \
    --input_dir data/raw/profiles \
    --output_dir data/processed/profiles_224 \
    --target_size 224 \
    --quality 95 \
    --workers 4
```

**파라미터 설명:**
- `--input_dir`: 원본 이미지 디렉토리 (2048x2048)
- `--output_dir`: 리사이징된 이미지 저장 위치
- `--target_size`: 목표 크기 (224x224)
- `--quality`: JPEG 품질 (95 권장)
- `--workers`: 병렬 처리 워커 수

**예상 출력:**
```
================================================================================
Image Resizing
================================================================================

Input directory: data/raw/profiles
Output directory: data/processed/profiles_224
Target size: (224, 224)
Quality: 95
Workers: 4

Scanning directory for images...
  - Images found: 150
  - Images to process: 150

Processing 150 images...
Resizing: 100%|███████████████████| 150/150 [00:12<00:00, 12.3it/s]

================================================================================
Results
================================================================================

Total processed: 150
  ✓ Successful: 150 (100.0%)
  ✗ Failed: 0 (0.0%)

Average size reduction: 95.2%
Average original size: 2048x2048

✓ Saved processing log to: data/processed/profiles_224/resize_log.csv
```

### 2.3 메타데이터와 함께 리사이징

```bash
# 메타데이터 CSV에 있는 이미지만 리사이징
python scripts/resize_images.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --metadata data/raw/metadata.csv \
    --target_size 224 \
    --quality 95 \
    --workers 8
```

### 2.4 리사이징 결과 확인

```bash
# 원본 이미지 크기 확인
ls -lh data/raw/profiles/user_001_1.jpg
# 출력 예: -rw-r--r-- 1 user user 2.3M Nov 18 10:00 user_001_1.jpg

# 리사이징된 이미지 크기 확인
ls -lh data/processed/profiles_224/user_001_1.jpg
# 출력 예: -rw-r--r-- 1 user user 15K Nov 18 10:05 user_001_1.jpg

# 약 95% 크기 감소!
```

---

## 3. Qwen 모델 테스트

### 3.1 기본 모델 로드 테스트

```bash
python scripts/test_qwen_model.py
```

**예상 소요 시간:**
- **첫 실행:** 5-10분 (모델 다운로드 ~5GB)
- **이후 실행:** 1-2분

**예상 출력:**
```
================================================================================
Qwen3-VL Model Test Suite
================================================================================

================================================================================
TEST 1: Model Loading
================================================================================
Using device: cuda

Loading model: Qwen/Qwen2-VL-2B-Instruct
This may take a few minutes on first run...

✓ Model loaded successfully!
  - Embedding dimension: 512
  - Vision hidden size: 1536
  - Total parameters: 2,134,567,890
  - Trainable parameters: 524,544

================================================================================
TEST 2: Dummy Image Inference
================================================================================

Creating dummy image (224x224 RGB)...
Running forward pass...

✓ Forward pass successful!
  - Output shape: torch.Size([1, 512])
  - Embedding norm: 1.0000
  - Mean value: 0.0023
  - Std value: 0.1234

  ✓ Embedding is L2-normalized

================================================================================
TEST 3: Batch Inference
================================================================================

Creating batch of 4 dummy images...
Running batch forward pass...

✓ Batch forward pass successful!
  - Output shape: torch.Size([4, 512])
  - Expected shape: (4, 512)

  - Cosine similarity between image 0 and 1: 0.3456

================================================================================
TEST 4: Checkpoint Save/Load
================================================================================

Saving checkpoint to: /tmp/tmpXXXX.pth

✓ Checkpoint saved successfully!
  - File size: 2.05 MB

Loading checkpoint...
✓ Checkpoint loaded successfully!

  - Max difference between original and loaded: 0.000001
  ✓ Loaded model produces identical outputs

  - Temporary checkpoint file deleted

================================================================================
TEST 5: Memory Usage
================================================================================

✓ Memory usage:
  - Current allocated: 2.34 GB
  - Peak allocated: 2.87 GB

================================================================================
ALL TESTS PASSED! ✓
================================================================================

Your Qwen3-VL model is ready for training!

Next steps:
  1. Prepare your dataset
  2. Create metadata CSV files
  3. Run training with train.py or train_sagemaker.py
```

### 3.2 모델 테스트 실패 시

#### HuggingFace 로그인 필요
```bash
# HuggingFace CLI 설치
pip install huggingface-hub

# 로그인
huggingface-cli login

# 토큰 입력 (https://huggingface.co/settings/tokens 에서 발급)
```

#### CUDA Out of Memory
```python
# test_qwen_model.py 수정 (batch size 줄이기)
# Line 140 근처
batch_size = 2  # 4에서 2로 줄임
```

---

## 4. 전체 파이프라인 테스트

### 4.1 End-to-End 테스트 실행

```bash
python scripts/test_pipeline.py
```

**이 스크립트가 자동으로 수행:**
1. ✅ 테스트 환경 설정
2. ✅ 더미 데이터 생성 (30명, ~100개 이미지)
3. ✅ 메타데이터 준비 (train/val 분할)
4. ✅ 데이터 로더 생성
5. ✅ Qwen 모델 로드
6. ✅ 2 epoch 학습 실행
7. ✅ Checkpoint 저장/로드 테스트

**예상 소요 시간:** 10-15분

### 4.2 상세 출력 예시

```
================================================================================
DATING PROFILE MATCHER - PIPELINE TEST
================================================================================

This script tests the complete pipeline from data generation to training.
It may take 10-15 minutes depending on your hardware.

================================================================================
STEP 1: Setup Test Environment
================================================================================

Cleaning up previous test data: test_data
Cleaning up previous test output: test_output
✓ Environment setup complete

================================================================================
STEP 2: Generate Dummy Data
================================================================================

Generating dummy dataset in: test_data
  - Users: 30
  - Images per user: (2, 4)

Generating real user images...
100%|███████████████████████████████| 30/30 [00:02<00:00, 12.5it/s]
  ✓ Generated 87 real user images

Generating 50 augmented images...
100%|███████████████████████████████| 50/50 [00:03<00:00, 15.2it/s]
  ✓ Generated 50 augmented images

✓ Saved metadata to: test_data/raw/metadata.csv

Generating interaction data...
  ✓ Saved interactions to: test_data/raw/interactions.csv

================================================================================
Dataset Generation Complete!
================================================================================

Summary:
  - Real users: 30
  - Real images: 87
  - Augmented images: 50
  - Total images: 137
  - Interactions: 500

================================================================================
STEP 3: Prepare Metadata
================================================================================

Loading metadata from: test_data/raw/metadata.csv
  - Total images: 137
  - Unique users: 80

  - Splitting by user (not by image)
  - Train users: 64
  - Val users: 16

  - Train images: 110
  - Val images: 27

✓ Saved train metadata to: test_data/processed/train_metadata.csv
✓ Saved val metadata to: test_data/processed/val_metadata.csv

================================================================================
STEP 4: Create Data Loaders
================================================================================

Creating data loaders...
  - Train batches: 27
  - Val batches: 6

Testing batch loading...
  - Batch keys: dict_keys(['image', 'label', 'user_id'])
  - Image shape: torch.Size([4, 3, 224, 224])
  - Label shape: torch.Size([4])

✓ Data loader creation complete

================================================================================
STEP 5: Load Model
================================================================================
Using device: cuda

Loading Qwen3-VL model: Qwen/Qwen2-VL-2B-Instruct

✓ Model loaded successfully
  - Embedding dim: 512
  - Vision hidden size: 1536
  - Total params: 2,134,567,890
  - Trainable params: 524,544

================================================================================
STEP 6: Run Training Loop
================================================================================

Training for 2 epochs...

Epoch 1/2
----------------------------------------
Training: 100%|████████████| 3/3 [00:08<00:00, 2.67s/it, loss=0.8734]
  Train Loss: 0.8734
Validation: 100%|██████████| 2/2 [00:02<00:00, 1.23s/it]
  Val Loss: 0.9123

Epoch 2/2
----------------------------------------
Training: 100%|████████████| 3/3 [00:07<00:00, 2.34s/it, loss=0.7891]
  Train Loss: 0.7891
Validation: 100%|██████████| 2/2 [00:02<00:00, 1.12s/it]
  Val Loss: 0.8456

✓ Training loop complete

================================================================================
STEP 7: Test Model Save/Load
================================================================================

Saving checkpoint to: test_output/test_model.pth
✓ Checkpoint saved

Loading checkpoint...
✓ Checkpoint loaded

================================================================================
ALL TESTS PASSED! ✓
================================================================================

Your pipeline is working correctly!

Next steps:
  1. Prepare your real dataset
  2. Update config.yaml with your settings
  3. Run full training with: python src/training/train.py

================================================================================
Cleanup
================================================================================

Cleaning up test data...
  ✓ Removed test_data
  ✓ Removed test_output
```

### 4.3 테스트 출력 유지하기

```bash
# 테스트 결과 파일 유지
python scripts/test_pipeline.py --keep_outputs

# 생성된 더미 데이터도 유지
python scripts/test_pipeline.py --skip_cleanup
```

---

## 5. 문제 해결

### 5.1 HuggingFace 모델 다운로드 실패

**증상:**
```
✗ Failed to load model: HTTP Error 403: Forbidden
```

**해결:**
```bash
# 1. HuggingFace 토큰 발급
# https://huggingface.co/settings/tokens

# 2. 로그인
huggingface-cli login
# 토큰 입력

# 3. 재시도
python scripts/test_qwen_model.py
```

### 5.2 GPU 메모리 부족 (OOM)

**증상:**
```
RuntimeError: CUDA out of memory. Tried to allocate XX GB
```

**해결 방법:**

**Option 1: Batch size 줄이기**
```python
# test_pipeline.py 수정
class PipelineTestConfig:
    batch_size = 2  # 4에서 2로 줄임
```

**Option 2: Vision encoder freeze**
```python
# qwen_backbone.py에서 이미 freeze=True로 설정됨
freeze_vision_encoder=True
```

**Option 3: CPU로 테스트**
```python
# test_qwen_model.py 수정
device = "cpu"  # "cuda" 대신
```

### 5.3 이미지 파일 없음 에러

**증상:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/profiles/user_001_1.jpg'
```

**해결:**
```bash
# 1. 더미 데이터 먼저 생성
python scripts/generate_dummy_data.py

# 2. 메타데이터 검증
python scripts/prepare_metadata.py \
    --metadata_csv data/raw/metadata.csv \
    --validate \
    --data_root data

# 3. 경로 확인
ls data/raw/profiles/
```

### 5.4 transformers 버전 호환 문제

**증상:**
```
ImportError: cannot import name 'Qwen2VLForConditionalGeneration'
```

**해결:**
```bash
# transformers 업데이트
pip install --upgrade transformers>=4.45.0

# qwen-vl-utils 설치
pip install qwen-vl-utils>=0.0.8
```

### 5.5 Pillow 이미지 로드 에러

**증상:**
```
PIL.UnidentifiedImageError: cannot identify image file
```

**해결:**
```bash
# Pillow 업데이트
pip install --upgrade Pillow>=10.0.0

# 또는 이미지 파일 확인
file data/raw/profiles/user_001_1.jpg
# JPEG image data 출력되어야 함
```

---

## 6. 테스트 체크리스트

베이스라인 코드 완성 후 확인사항:

### 환경 확인
- [ ] Python 3.8+ 설치 확인
- [ ] 필수 패키지 설치 완료
- [ ] GPU 사용 가능 (선택사항)
- [ ] CUDA 버전 확인 (GPU 사용 시)

### 이미지 전처리
- [ ] 원본 이미지 크기 확인 (2048x2048?)
- [ ] `resize_images.py` 실행 성공
- [ ] 리사이징된 이미지 확인 (224x224)
- [ ] 파일 크기 감소 확인 (~95% 감소)

### 모델 테스트
- [ ] `test_qwen_model.py` 모든 테스트 통과
- [ ] 모델 로드 성공
- [ ] Forward pass 정상 작동
- [ ] Checkpoint save/load 정상
- [ ] GPU 메모리 사용량 확인 (<8GB)

### 파이프라인 테스트
- [ ] `test_pipeline.py` 전체 통과
- [ ] 더미 데이터 생성 확인
- [ ] 데이터 로더 정상 작동
- [ ] 학습 루프 실행 성공
- [ ] Loss 값 정상 감소

### 다음 단계 준비
- [ ] 실제 데이터 준비 계획 수립
- [ ] SageMaker vs 로컬 환경 결정
- [ ] 하이퍼파라미터 설정 검토
- [ ] `config.yaml` 수정 계획

**모두 체크되면 → 실제 데이터로 학습 준비 완료!** ✅

---

## 7. 다음 단계

테스트가 모두 성공했다면:

### 7.1 실제 데이터 준비

```bash
# 1. 이미지를 data/raw/profiles/에 복사
# 파일명 규칙: user_XXX_Y.jpg

# 2. 이미지 리사이징
python scripts/resize_images.py \
    --input_dir data/raw/profiles \
    --output_dir data/processed/profiles_224 \
    --target_size 224 \
    --quality 95 \
    --workers 8

# 3. 메타데이터 생성 (수동 또는 스크립트)
# CSV 형식:
# user_id,image_idx,filename,image_path,is_synthetic
# user_001,1,user_001_1.jpg,profiles/user_001_1.jpg,False

# 4. 메타데이터 분할
python scripts/prepare_metadata.py \
    --metadata_csv data/raw/metadata.csv \
    --output_dir data/processed
```

### 7.2 설정 파일 업데이트

`configs/config.yaml` 수정:
```yaml
paths:
  raw_data: "data/processed/profiles_224"  # 리사이징된 이미지
  processed_data: "data/processed"

training:
  batch_size: 16  # GPU 메모리에 따라 조정
  num_epochs: 30
  learning_rate: 1e-4
```

### 7.3 학습 실행

```bash
# 로컬 환경
python src/training/train.py --config configs/config.yaml

# SageMaker 환경
# notebooks/sagemaker_training.ipynb 실행
```

---

**문서 작성일:** 2025-11-18
**버전:** 1.0
**마지막 업데이트:** 이미지 리사이징 전처리 추가
