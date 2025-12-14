# 커플 매칭 모델 실험 현황

> **최종 업데이트**: 2025-12-13

---

## 📌 프로젝트 개요

- **목표**: 데이팅 앱 커플 프로필 이미지 기반 매칭 예측
- **데이터**: 775쌍 실제 커플 이미지 (mutual-like)
- **모델**: Qwen3-VL-2B (Backbone 동결) + Projection Head
- **Loss**: InfoNCE (Contrastive Learning)

---

## ✅ 완료된 실험

### 실험 1: 기본 학습 (Dropout ❌)

| 항목           | 값                                           |
| -------------- | -------------------------------------------- |
| 파라미터       | LR=1e-4, Batch=48, τ=0.07                    |
| Validation R@1 | **8.06%** (Epoch 7)                          |
| **Test R@1**   | **1.04%**                                    |
| 체크포인트     | `couple_matching_checkpoints_v1_no_dropout/` |

**결론**: Validation 대비 Test 성능 하락 → 과적합 의심

---

### 실험 2: Dropout 0.3 추가

| 항목         | 값                                  |
| ------------ | ----------------------------------- |
| 변경         | Projection Head에 Dropout(0.3) 추가 |
| **Test R@1** | **0.97%** (하락)                    |
| 체크포인트   | `couple_matching_checkpoints/`      |

**결론**: Dropout이 오히려 성능 하락 → 데이터 부족으로 학습 방해

---

## 🔬 향후 실험 계획

### 실험 3: Temperature 조정 (권장 1순위)

```python
# train_couple_matching.py
temperature: float = 0.1  # 현재 0.07 → 0.1로 변경
```

**가설**: 0.07이 너무 sharp → 0.1로 완화하면 일반화 개선

---

### 실험 4: Learning Rate 감소

```bash
python scripts/train_couple_matching.py --fold 0 --lr 5e-5
```

**가설**: 더 천천히 학습하면 과적합 방지

---

### 실험 5: 배치 크기 증가

```bash
python scripts/train_couple_matching.py --fold 0 --batch-size 64
```

**가설**: 더 많은 Negative 샘플 = 더 강한 학습 신호

> ⚠️ OOM 발생 시: 이미지 크기 768 → 512로 축소

---

### 실험 6: 스타일 라벨 결합 (복잡)

- 입력: 이미지 + Gemini 스타일 라벨 텍스트
- 구현 필요: 멀티모달 임베딩 결합

---

## 📁 파일 구조

```
scripts/
├── train_couple_matching.py      # 학습 스크립트
├── evaluate_couples.py           # 평가 스크립트
├── prepare_couple_splits.py      # 데이터 분할
├── label_couple_images.py        # Gemini 라벨링
└── evaluate_style_recall.py      # 스타일 기반 평가

src/models/
├── qwen_backbone.py              # Qwen3-VL 백본
└── projection.py                 # Projection Head (Dropout 포함)

checkpoints/
├── couple_matching_checkpoints_v1_no_dropout/  # 실험 1
└── couple_matching_checkpoints/                 # 실험 2
```

---

## 🚀 다음 세션에서 할 일

1. **Dropout 제거** (projection.py 원복)
2. **Temperature 0.1로 변경** (train_couple_matching.py)
3. **Fold 0 학습** → Test 평가
4. 결과에 따라 다음 실험 진행

---

## 📊 기준 성능 (랜덤)

| 지표 | 랜덤 기대값 | 현재 Best        |
| ---- | ----------- | ---------------- |
| R@1  | 0.13%       | 1.04% (**8배**)  |
| R@10 | 1.30%       | 10.18% (**8배**) |
