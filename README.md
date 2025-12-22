# Couple Matching Prediction System

> **Qwen3-VL 기반 커플 매칭 예측 시스템**  
> 실제 매칭된 커플 데이터를 활용하여 시각적 호환성을 학습합니다.

---

## 🎯 프로젝트 개요

온라인 데이팅 앱에서 실제 매칭이 성사된 커플 데이터(774쌍)를 활용하여, 프로필 이미지 기반으로 매칭 호환성을 예측하는 딥러닝 모델을 개발합니다.

### 핵심 특징

- **Backbone**: Qwen3-VL-2B-Instruct (동결)
- **Projection Head**: 성별별 독립 헤드 (Female/Male)
- **Loss**: InfoNCE (Contrastive Learning)
- **평가**: Train/Valid/Test 분할 (70/15/15)

---

## 📊 데이터셋

| 항목        | 값                                         |
| ----------- | ------------------------------------------ |
| 총 커플 수  | ~774쌍                                     |
| Train       | 70%                                        |
| Valid       | 15%                                        |
| Test        | 15%                                        |
| 이미지 형식 | `couple_N/female.png`, `couple_N/male.png` |

---

## 🏗 모델 아키텍처

```
Female Image                          Male Image
     │                                     │
     ▼                                     ▼
┌─────────────┐                    ┌─────────────┐
│Female Prompt│                    │ Male Prompt │
└─────────────┘                    └─────────────┘
     │                                     │
     ▼                                     ▼
┌─────────────────────────────────────────────────┐
│            Qwen3-VL Backbone (동결)              │
│            Hidden Dim: 2048                      │
└─────────────────────────────────────────────────┘
     │                                     │
     ▼                                     ▼
┌─────────────┐                    ┌─────────────┐
│ Female Head │                    │  Male Head  │
│ Projection  │                    │ Projection  │
└─────────────┘                    └─────────────┘
     │                                     │
     ▼                                     ▼
  [256-dim]                            [256-dim]
     │                                     │
     └──── L2 Normalize ───────────────────┘
                   │
                   ▼
        ┌───────────────────┐
        │ Cosine Similarity │
        │     (τ = 0.1)     │
        └───────────────────┘
                   │
                   ▼
           InfoNCE Loss
```

자세한 모델 구조는 [기술 문서](docs/model_architecture.md)를 참조하세요.

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

### 2. 데이터 분할

```bash
python scripts/prepare_splits.py \
    --data-dir ~/data/mutual-like-validations/images \
    --train-ratio 0.7 \
    --valid-ratio 0.15 \
    --test-ratio 0.15
```

### 3. 모델 학습

```bash
python scripts/train_couple_matching.py \
    --splits couple_splits.json \
    --epochs 30 \
    --lr 5e-5
```

### 4. 평가

```bash
# 베이스라인 평가 (Projection Head 없이)
python scripts/evaluate_baseline.py --splits couple_splits.json

# 학습 모델 평가
python scripts/evaluate_finetuned.py \
    --splits couple_splits.json \
    --checkpoint ./couple_matching_checkpoints/best_model.pth
```

---

## 📁 프로젝트 구조

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
│       └── projection.py           # 성별별 Projection Head
├── docs/
│   ├── model_architecture.md       # 기술 문서
│   └── prompt_strategy.md          # 프롬프트 전략
├── couple_splits.json              # 데이터 분할 정보
├── couple_matching_checkpoints/    # 학습된 가중치
│   └── best_model.pth
├── baseline_evaluation_results/    # 베이스라인 평가 결과
├── finetuned_evaluation_results/   # 학습 모델 평가 결과
└── requirements.txt
```

---

## 🛠 학습 설정

| 파라미터   | 값                | 비고             |
| ---------- | ----------------- | ---------------- |
| Backbone   | Qwen3-VL-2B       | 동결             |
| Projection | 2048 → 1024 → 256 | 성별별 독립      |
| Loss       | InfoNCE           | τ=0.1            |
| Batch Size | 48                | A10G 24GB 기준   |
| Optimizer  | AdamW             | lr=5e-5, wd=1e-3 |
| Epochs     | 30                | Early Stopping   |
| Patience   | 10                | Accuracy 기준    |

---

## 📈 평가 지표

| 지표                 | 설명                                          |
| -------------------- | --------------------------------------------- |
| **Accuracy (Hit@1)** | 정확히 1위로 매칭된 비율                      |
| **Hit@K**            | 상위 K개 내에 정답이 있는 비율 (K=5,10,20,50) |
| **MRR**              | Mean Reciprocal Rank                          |

### 평가 방향

- **Female → Male**: 여성 임베딩으로 남성 검색
- **Male → Female**: 남성 임베딩으로 여성 검색
- **Average**: 양방향 평균

---

## 📚 문서

| 문서                                     | 설명                                 |
| ---------------------------------------- | ------------------------------------ |
| [기술 문서](docs/model_architecture.md)  | 모델 구조, 학습 방법, 평가 방법 상세 |
| [프롬프트 전략](docs/prompt_strategy.md) | 성별별 프롬프트 설계                 |

---

## 🔗 참고 자료

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [CLIP](https://openai.com/research/clip)
- [Matching Hypothesis](https://en.wikipedia.org/wiki/Matching_hypothesis)

---

## 📄 라이선스

MIT License
