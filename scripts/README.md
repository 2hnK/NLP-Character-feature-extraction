# Scripts Directory

베이스라인 코드 테스트 및 데이터 준비를 위한 유틸리티 스크립트 모음입니다.

## 📋 스크립트 목록

### 1. `test_qwen_model.py`
Qwen3-VL 모델 로드 및 기본 추론 테스트

**사용법:**
```bash
python scripts/test_qwen_model.py
```

**테스트 항목:**
- ✓ 모델 로드 (HuggingFace)
- ✓ Processor 초기화
- ✓ Forward pass (단일 이미지)
- ✓ Batch inference
- ✓ Checkpoint save/load
- ✓ GPU 메모리 사용량

**예상 소요 시간:** 5-10분 (첫 실행 시 모델 다운로드)

---

### 2. `generate_dummy_data.py`
테스트용 더미 데이터 생성

**사용법:**
```bash
# 기본 설정 (50명, 100개 증강 이미지)
python scripts/generate_dummy_data.py

# 커스텀 설정
python scripts/generate_dummy_data.py \
    --output_dir data \
    --num_users 100 \
    --images_per_user_min 2 \
    --images_per_user_max 5 \
    --num_augmented 200
```

**생성 파일:**
```
data/
├── raw/
│   ├── profiles/          # 실제 사용자 이미지
│   │   ├── user_000_1.jpg
│   │   ├── user_000_2.jpg
│   │   └── ...
│   ├── metadata.csv       # 전체 메타데이터
│   └── interactions.csv   # 사용자 상호작용 (좋아요/패스)
└── augmented/
    └── generated/         # 증강 이미지
        ├── gen_0000.jpg
        └── ...
```

**옵션:**
- `--num_users`: 사용자 수 (기본: 50)
- `--images_per_user_min`: 최소 이미지/사용자 (기본: 1)
- `--images_per_user_max`: 최대 이미지/사용자 (기본: 4)
- `--num_augmented`: 증강 이미지 수 (기본: 100)
- `--no_augmented`: 증강 이미지 생성 안 함

---

### 3. `prepare_metadata.py`
메타데이터를 train/val로 분할

**사용법:**
```bash
python scripts/prepare_metadata.py \
    --metadata_csv data/raw/metadata.csv \
    --output_dir data/processed
```

**생성 파일:**
```
data/processed/
├── train_metadata.csv     # 학습 데이터 메타데이터
├── val_metadata.csv       # 검증 데이터 메타데이터
└── copy_images.sh         # 이미지 복사 스크립트
```

**옵션:**
- `--train_ratio`: 학습 비율 (기본: 0.85)
- `--val_ratio`: 검증 비율 (기본: 0.15)
- `--seed`: Random seed (기본: 42)
- `--validate`: 메타데이터 검증 (파일 존재 확인)

**중요:**
- 사용자 단위로 분할 (같은 사용자의 이미지는 항상 같은 set에)
- 단일 이미지 사용자는 경고 메시지 출력

---

### 4. `test_pipeline.py`
전체 파이프라인 end-to-end 테스트

**사용법:**
```bash
# 기본 테스트 (완료 후 자동 정리)
python scripts/test_pipeline.py

# 출력 파일 유지
python scripts/test_pipeline.py --keep_outputs

# 정리 건너뛰기 (디버깅용)
python scripts/test_pipeline.py --skip_cleanup
```

**테스트 단계:**
1. ✓ 환경 설정
2. ✓ 더미 데이터 생성
3. ✓ 메타데이터 준비
4. ✓ 데이터 로더 생성
5. ✓ 모델 로드
6. ✓ 학습 루프 (2 epoch)
7. ✓ Checkpoint save/load

**예상 소요 시간:** 10-15분

### 5. `preprocess_jsonl.py`
JSONL 메타데이터 전처리 (파일명 추가, 텍스트 포맷팅, 라벨 매핑 생성)

**사용법:**
```bash
python scripts/preprocess_jsonl.py \
    --input enhanced_train_batch_44_94.jsonl \
    --output train_processed.jsonl \
    --mapping_output label_mapping.json
```

**기능:**
- `filename` 필드 추가 (`aug_00000.jpg` 형식)
- `text_input` 필드 생성 (Style, Features, Vibe 결합)
- `label_mapping.json` 생성 (Style 문자열 -> 정수 인덱스 매핑)

---

## 🚀 빠른 시작 가이드

### Step 1: 모델 테스트
먼저 Qwen3-VL 모델이 정상적으로 로드되는지 확인합니다.

```bash
python scripts/test_qwen_model.py
```

**예상 출력:**
```
================================================================================
TEST 1: Model Loading
================================================================================
Using device: cuda
Loading model: Qwen/Qwen2-VL-2B-Instruct
This may take a few minutes on first run...

✓ Model loaded successfully!
  - Embedding dimension: 512
  - Vision hidden size: 1536
  ...
ALL TESTS PASSED! ✓
```

### Step 2: 더미 데이터 생성
테스트용 데이터를 생성합니다.

```bash
python scripts/generate_dummy_data.py \
    --num_users 50 \
    --num_augmented 100
```

### Step 3: 메타데이터 준비
Train/val 분할을 수행합니다.

```bash
python scripts/prepare_metadata.py \
    --metadata_csv data/raw/metadata.csv \
    --output_dir data/processed
```

### Step 4: 전체 파이프라인 테스트
모든 것이 정상 작동하는지 확인합니다.

```bash
python scripts/test_pipeline.py
```

**성공 시:**
```
================================================================================
ALL TESTS PASSED! ✓
================================================================================

Your pipeline is working correctly!

Next steps:
  1. Prepare your real dataset
  2. Update config.yaml with your settings
  3. Run full training with: python src/training/train.py
```

---

## 🐛 문제 해결

### HuggingFace 모델 다운로드 실패

**증상:**
```
✗ Failed to load model: HTTP Error 403: Forbidden
```

**해결:**
1. HuggingFace 로그인
```bash
pip install huggingface-hub
huggingface-cli login
```

2. 토큰 입력 후 재시도

---

### GPU 메모리 부족

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
1. Batch size 줄이기 (config.yaml)
```yaml
training:
  batch_size: 8  # 16에서 8로 줄임
```

2. Vision encoder freeze
```yaml
model:
  freeze_vision_encoder: true
```

---

### 데이터 로더 에러

**증상:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**해결:**
1. 메타데이터 검증
```bash
python scripts/prepare_metadata.py \
    --metadata_csv data/raw/metadata.csv \
    --validate \
    --data_root data
```

2. 경로 확인
- `metadata.csv`의 `image_path` 컬럼이 올바른지 확인
- 상대 경로가 `data_root` 기준인지 확인

---

## 📌 다음 단계

베이스라인 코드 테스트가 완료되면:

1. **실제 데이터 준비**
   - 프로필 이미지를 `data/raw/profiles/` 에 업로드
   - 파일명 규칙: `user_XXX_Y.jpg` (XXX: user ID, Y: image index)

2. **메타데이터 생성**
   - 실제 데이터로 `metadata.csv` 생성
   - `interactions.csv` 준비 (좋아요/패스 데이터)

3. **설정 파일 업데이트**
   - `configs/config.yaml` 수정
   - 하이퍼파라미터, 경로 등 설정

4. **학습 실행**
   ```bash
   # 로컬 환경
   python src/training/train.py --config configs/config.yaml

   # SageMaker 환경
   # notebooks/sagemaker_training.ipynb 실행
   ```

5. **평가 및 시각화**
   ```bash
   python src/evaluation/evaluate.py --checkpoint models/saved_models/best_model.pth
   ```

---

## 📚 참고 문서

- [프로젝트 컨텍스트](../docs/PROJECT_CONTEXT.md)
- [시작 가이드](../docs/GETTING_STARTED.md)
- [아키텍처](../docs/ARCHITECTURE.md)
- [SageMaker 가이드](../docs/SAGEMAKER_GUIDE.md)

---

**마지막 업데이트:** 2025-11-18
