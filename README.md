# Couple Matching Prediction System

> **Qwen3-VL 기반 데이팅 앱 커플 매칭 예측 시스템**  
> 실제 매칭된 커플 데이터를 활용하여 시각적 매칭 호환성을 학습합니다.

---

## 🎯 프로젝트 개요

온라인 데이팅 앱에서 실제 매칭이 성사된 커플 데이터(775쌍)를 활용하여, 프로필 이미지 기반으로 매칭 호환성을 예측하는 딥러닝 모델을 개발합니다.

### 핵심 기술

- **Backbone**: Qwen3-VL-2B (동결)
- **Loss**: InfoNCE (Contrastive Learning)
- **평가**: 5-Fold Cross Validation + Hold-out Test Set

---

## 📊 데이터셋

| 항목                 | 값                                         |
| -------------------- | ------------------------------------------ |
| 총 커플 수           | 775쌍                                      |
| Train+Valid (5-Fold) | 620쌍 (80%)                                |
| Test (Hold-out)      | 155쌍 (20%)                                |
| 이미지 형식          | `couple_N/female.png`, `couple_N/male.png` |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd NLP-Character-feature-extraction

# 의존성 설치
pip install -r requirements.txt
```

### 2. 학습 실행

```bash
# 데이터 분할
python scripts/prepare_couple_splits.py

# 5-Fold 학습 (Fold 0)
python scripts/train_couple_matching.py --fold 0

# Test set 평가
python scripts/evaluate_couples.py --split test
```

자세한 실행 방법은 [학습 가이드](scripts/TRAINING_GUIDE.md)를 참조하세요.

---

## 📁 프로젝트 구조

```
NLP-Character-feature-extraction/
├── scripts/
│   ├── prepare_couple_splits.py   # 데이터 분할
│   ├── train_couple_matching.py   # 모델 학습
│   ├── evaluate_couples.py        # 평가
│   └── TRAINING_GUIDE.md          # 학습 가이드
├── src/
│   ├── models/
│   │   ├── qwen_backbone.py       # Qwen3-VL 백본
│   │   └── projection.py          # Projection Head
│   └── data/
│       └── s3_dataset.py          # 데이터 로더
├── docs/
│   ├── research_v2.md             # 연구 논문
│   └── research.md                # (이전 버전)
├── paper/                         # LaTeX 논문
└── requirements.txt
```

---

## 🛠 학습 설정

| 파라미터   | 값          | 비고           |
| ---------- | ----------- | -------------- |
| Backbone   | Qwen3-VL-2B | 동결           |
| Projection | 2048 → 256  | 학습           |
| Loss       | InfoNCE     | τ=0.07         |
| Batch Size | 48          | A10G 24GB 기준 |
| Optimizer  | AdamW       | lr=1e-4        |
| Epochs     | 30          | Early Stopping |

---

## 📈 평가 지표

- **Recall@K**: 파트너가 상위 K명 내 검색될 확률
- **MRR**: Mean Reciprocal Rank
- **양방향 평가**: Female→Male, Male→Female

---

## 🔗 참고 자료

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [CLIP](https://openai.com/research/clip)
- [Matching Hypothesis](https://en.wikipedia.org/wiki/Matching_hypothesis)

---

## 📄 라이선스

MIT License
