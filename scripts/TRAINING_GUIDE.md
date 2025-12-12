# 커플 매칭 예측 모델 학습 스크립트

> **Qwen3-VL 기반 커플 매칭 예측 시스템**  
> 실제 데이팅 앱에서 매칭된 커플 데이터(775쌍)를 활용하여 매칭 호환성을 학습합니다.

---

## 📋 스크립트 목록

| 스크립트                   | 설명                            |
| -------------------------- | ------------------------------- |
| `prepare_couple_splits.py` | 데이터 분할 (80/20 + 5-Fold CV) |
| `train_couple_matching.py` | InfoNCE Loss 기반 학습          |
| `evaluate_couples.py`      | Recall@K 평가                   |

---

## 🚀 실행 순서

### 1. 데이터 분할

```bash
python scripts/prepare_couple_splits.py \
    --data-dir ~/data/mutual-like-validations/images \
    --output couple_splits.json
```

**출력**: `couple_splits.json`

- Test Set: 155쌍 (20%)
- 5-Fold CV: 620쌍 (80%)

---

### 2. 모델 학습 (5-Fold)

```bash
# Fold 0~4 순차 실행
python scripts/train_couple_matching.py --fold 0
python scripts/train_couple_matching.py --fold 1
python scripts/train_couple_matching.py --fold 2
python scripts/train_couple_matching.py --fold 3
python scripts/train_couple_matching.py --fold 4
```

**주요 옵션**:

```
--fold        : Fold 번호 (0-4)
--batch-size  : 배치 크기 (기본: 48)
--epochs      : 에폭 수 (기본: 30)
--lr          : 학습률 (기본: 1e-4)
```

**출력**: `./couple_checkpoints/best_model_fold{N}.pth`

---

### 3. Test Set 평가

```bash
python scripts/evaluate_couples.py \
    --checkpoint ./couple_checkpoints/best_model_fold0.pth \
    --splits-file couple_splits.json \
    --split test
```

**출력**:

- `couple_evaluation_results/couple_recall_metrics.json`
- `couple_evaluation_results/couple_recall_report.md`

---

## 📊 학습 설정

| 파라미터       | 값              | 설명                       |
| -------------- | --------------- | -------------------------- |
| Backbone       | Qwen3-VL-2B     | 동결                       |
| Loss           | InfoNCE         | τ=0.07                     |
| Optimizer      | AdamW           | lr=1e-4, weight_decay=1e-4 |
| Scheduler      | CosineAnnealing |                            |
| Batch Size     | 48              | OOM 시 32로 감소           |
| Early Stopping | patience=5      |                            |

---

## � 디렉토리 구조

```
scripts/
├── prepare_couple_splits.py   # 데이터 분할
├── train_couple_matching.py   # 학습
├── evaluate_couples.py        # 평가
├── labeling/                  # 라벨링 유틸리티
└── archive/                   # 이전 버전 스크립트
```

---

## 🔧 환경 요구사항

- **GPU**: NVIDIA A10G 24GB (ml.g5.2xlarge) 권장
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **Transformers**: 4.57+
