# 커플 매칭 모델 기술 문서

> **버전**: v3.0 (2024-12-22)  
> **목적**: 이미지 기반 커플 호환성 예측을 위한 학습 및 평가 파이프라인

---

## 1. 개요

본 프로젝트는 Vision-Language Model(VLM)을 활용하여 커플 이미지 간의 호환성을 예측하는 모델을 학습합니다.

### 핵심 특징

- **Backbone**: Qwen3-VL-2B (사전학습된 Vision-Language Model)
- **학습 방식**: Contrastive Learning (InfoNCE Loss)
- **성별별 독립 Projection Head**: 남/녀 이미지를 각각 다른 임베딩 공간으로 매핑

---

## 2. 모델 구조

### 2.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Female Image                        Male Image              │
│       │                                   │                  │
│       ▼                                   ▼                  │
│  ┌─────────────┐                   ┌─────────────┐          │
│  │  Female     │                   │   Male      │          │
│  │  Prompt     │                   │   Prompt    │          │
│  └─────────────┘                   └─────────────┘          │
│       │                                   │                  │
│       ▼                                   ▼                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Qwen3-VL Backbone (동결)                │    │
│  │              - Vision Encoder                        │    │
│  │              - LLM (Hidden States 추출)              │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                   │                  │
│       ▼                                   ▼                  │
│  [2048-dim]                          [2048-dim]             │
│       │                                   │                  │
│       ▼                                   ▼                  │
│  ┌───────────────┐                ┌───────────────┐         │
│  │ Female Head   │                │  Male Head    │         │
│  │ Projection    │                │  Projection   │         │
│  └───────────────┘                └───────────────┘         │
│       │                                   │                  │
│       ▼                                   ▼                  │
│  [256-dim]                           [256-dim]              │
│       │                                   │                  │
│       └────────────┬──────────────────────┘                  │
│                    ▼                                         │
│             ┌─────────────┐                                  │
│             │ InfoNCE Loss│                                  │
│             └─────────────┘                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 구성 요소

#### Backbone (Qwen3-VL-2B)

| 항목             | 값                              |
| ---------------- | ------------------------------- |
| 모델명           | `Qwen/Qwen3-VL-2B-Instruct`     |
| Hidden Dimension | 2048                            |
| Pooling 방식     | Mean Pooling (기본) / EOS Token |
| 학습 상태        | **동결 (Frozen)**               |

#### Gender-Specific Projection Head

```python
class GenderSpecificProjection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=256):
        self.female_head = ProjectionHead(input_dim, hidden_dim, output_dim)
        self.male_head = ProjectionHead(input_dim, hidden_dim, output_dim)
```

| 레이어       | 입력 → 출력 |
| ------------ | ----------- |
| Linear       | 2048 → 1024 |
| BatchNorm    | 1024        |
| ReLU         | -           |
| Linear       | 1024 → 256  |
| L2 Normalize | -           |

---

## 3. 프롬프트 전략

### 3.1 시스템 프롬프트

```
Female: "You are analyzing a female profile for dating compatibility..."
Male:   "You are analyzing a male profile for dating compatibility..."
```

### 3.2 사용자 프롬프트

```
Female: "Analyze this woman's dating profile photo. Extract visual features..."
Male:   "Analyze this man's dating profile photo. Extract visual features..."
```

### 3.3 특징 추출 과정

1. 이미지 + 성별별 프롬프트 → Qwen3-VL 입력
2. LLM의 마지막 레이어 Hidden States 추출
3. Mean Pooling 또는 EOS Token 선택
4. 성별별 Projection Head 통과

---

## 4. 학습 방법

### 4.1 손실 함수: InfoNCE Loss

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def forward(self, female_embs, male_embs):
        # 유사도 행렬 계산
        logits = torch.matmul(female_embs, male_embs.T) / self.temperature

        # 대각선이 정답 (i번 female ↔ i번 male)
        labels = torch.arange(batch_size)

        # 양방향 Cross Entropy
        loss_f2m = F.cross_entropy(logits, labels)
        loss_m2f = F.cross_entropy(logits.T, labels)

        return (loss_f2m + loss_m2f) / 2
```

### 4.2 학습 하이퍼파라미터

| 파라미터      | 값   | 설명                     |
| ------------- | ---- | ------------------------ |
| Batch Size    | 48   | InfoNCE에 유리한 큰 배치 |
| Learning Rate | 5e-5 | AdamW                    |
| Weight Decay  | 1e-3 | 규제 강화                |
| Epochs        | 30   | 최대 에폭                |
| Temperature   | 0.1  | Softmax 스케일링         |
| Patience      | 10   | Early Stopping           |

### 4.3 학습 파이프라인

```bash
# 1. 데이터 분할 생성
python scripts/prepare_splits.py --train-ratio 0.7 --valid-ratio 0.15 --test-ratio 0.15

# 2. 학습 실행
python scripts/train_couple_matching.py --splits couple_splits.json
```

---

## 5. 평가 방법

### 5.1 평가 지표

| 지표                 | 설명                           |
| -------------------- | ------------------------------ |
| **Accuracy (Hit@1)** | 정확히 1위로 매칭된 비율       |
| **Hit@K**            | 상위 K개 내에 정답이 있는 비율 |
| **MRR**              | Mean Reciprocal Rank           |

### 5.2 평가 방향

- **Female → Male**: 여성 임베딩으로 남성 검색
- **Male → Female**: 남성 임베딩으로 여성 검색
- **Average**: 양방향 평균

### 5.3 베이스라인 vs 학습 모델

| 구분          | 베이스라인                     | 학습 모델                       |
| ------------- | ------------------------------ | ------------------------------- |
| Backbone      | Qwen3-VL                       | Qwen3-VL                        |
| Projection    | ❌ 없음                        | ✅ 성별별 Head                  |
| 임베딩 차원   | 2048                           | 256                             |
| 스크립트      | `evaluate_baseline.py`         | `evaluate_finetuned.py`         |
| 결과 디렉토리 | `baseline_evaluation_results/` | `finetuned_evaluation_results/` |

### 5.4 평가 실행

```bash
# 베이스라인 평가
python scripts/evaluate_baseline.py --splits couple_splits.json

# 학습 모델 평가
python scripts/evaluate_finetuned.py --splits couple_splits.json \
    --checkpoint ./couple_matching_checkpoints/best_model.pth
```

---

## 6. 데이터 분할

### 6.1 분할 비율

| Set   | 비율 | 용도                          |
| ----- | ---- | ----------------------------- |
| Train | 70%  | 모델 학습                     |
| Valid | 15%  | 학습 중 검증 (Early Stopping) |
| Test  | 15%  | 최종 성능 평가                |

### 6.2 분할 파일 구조 (`couple_splits.json`)

```json
{
  "train": [5, 12, 23, 45, ...],
  "valid": [78, 102, 156, ...],
  "test": [234, 389, 456, ...],
  "config": {
    "train_ratio": 0.7,
    "valid_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 42,
    "total_couples": 774
  }
}
```

---

## 7. 파일 구조

```
NLP-Character-feature-extraction/
├── scripts/
│   ├── prepare_splits.py           # 데이터 분할 생성
│   ├── train_couple_matching.py    # 모델 학습
│   ├── evaluate_baseline.py        # 베이스라인 평가
│   └── evaluate_finetuned.py       # 학습 모델 평가
├── src/
│   └── models/
│       ├── qwen_backbone.py        # Qwen3-VL Feature Extractor
│       └── projection.py           # Projection Head 정의
├── couple_splits.json              # 데이터 분할 정보
├── couple_matching_checkpoints/    # 학습된 가중치
│   └── best_model.pth
├── baseline_evaluation_results/    # 베이스라인 평가 결과
└── finetuned_evaluation_results/   # 학습 모델 평가 결과
```

---

## 8. 체크포인트 구조

```python
{
    'epoch': 15,
    'backbone_state_dict': {...},      # Qwen3-VL 가중치
    'projection_state_dict': {         # 성별별 Head 가중치
        'female_head.layer1.weight': ...,
        'female_head.layer2.weight': ...,
        'male_head.layer1.weight': ...,
        'male_head.layer2.weight': ...,
    },
    'optimizer_state_dict': {...},
    'best_accuracy': 0.15,
    'metrics': {...},
    'config': {...}
}
```

---

## 9. 실행 가이드

### 전체 파이프라인

```bash
# Step 1: 데이터 분할
python scripts/prepare_splits.py \
    --data-dir ~/data/mutual-like-validations/images \
    --train-ratio 0.7 \
    --valid-ratio 0.15 \
    --test-ratio 0.15

# Step 2: 모델 학습
python scripts/train_couple_matching.py \
    --splits couple_splits.json \
    --epochs 30 \
    --lr 5e-5

# Step 3: 베이스라인 평가
python scripts/evaluate_baseline.py \
    --splits couple_splits.json

# Step 4: 학습 모델 평가
python scripts/evaluate_finetuned.py \
    --splits couple_splits.json \
    --checkpoint ./couple_matching_checkpoints/best_model.pth
```

---

## 10. 주요 설계 결정

### Q: 왜 성별별 독립 Projection Head?

동일한 Projection Head를 사용할 경우, 남녀 이미지가 동일한 임베딩 공간에 매핑되어
성별 고유의 특징을 학습하기 어렵습니다. 성별별 Head를 분리함으로써:

- 여성 이미지의 "매력적인 특징" → Female Head → Female 임베딩 공간
- 남성 이미지의 "매력적인 특징" → Male Head → Male 임베딩 공간

각 성별에 최적화된 임베딩을 학습할 수 있습니다.

### Q: 왜 Backbone을 동결?

- Qwen3-VL은 대규모 사전학습으로 이미 강력한 시각적 이해 능력 보유
- 774쌍의 작은 데이터셋으로 미세조정 시 과적합 위험
- Projection Head만 학습하여 효율적인 전이 학습 수행

### Q: 왜 InfoNCE Loss?

- Contrastive Learning의 표준 손실 함수
- 배치 내 모든 negative pair를 효율적으로 활용
- Temperature 파라미터로 유사도 분포 조절 가능

---

## 11. 향후 개선 방안

1. **Hard Negative Mining**: 더 어려운 negative 샘플 선택
2. **Multi-Modal Features**: 프로필 텍스트 정보 추가
3. **Larger Backbone**: Qwen3-VL-7B 사용 (메모리 허용 시)
4. **Data Augmentation**: 이미지 증강 기법 적용
5. **Cross-Attention**: 남녀 임베딩 간 교차 어텐션 추가
