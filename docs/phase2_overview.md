# Phase 2: Experimental Couple Matching (Frozen Backbone + InfoNCE)

본 문서는 Phase 3 이전 단계인 **Phase 2 (초기 실험 단계)**의 모델 구조, 학습 방법, 실험 결과를 요약합니다. 이 단계에서는 **사전 학습된 지식(Pre-trained Knowledge)**을 최대한 보존하면서 커플 매칭을 시도했으나, 데이터셋 특성에 따른 한계점을 확인했습니다.

---

## 1. 개요 (Overview)
Phase 2는 **"사전 학습된 Qwen3-VL이 데이팅 앱 매칭을 zero-shot 또는 few-shot으로 얼마나 잘 수행할까?"**라는 질문에서 출발했습니다. Backbone 모델을 고정(Freeze)하고 Projection Head만 학습시키는 전략을 취했습니다.

## 2. 모델 아키텍처 (Model Architecture)

### 2.1 Backbone: Qwen3-VL-2B (Frozen)
-   **파라미터 동결**: Vision Encoder 및 LLM 파라미터를 전혀 업데이트하지 않았습니다.
-   **의도**: 거대 모델의 시각적/언어적 이해력을 그대로 활용하여 과적합(Overfitting)을 방지하고자 함.

### 2.2 Projection Head (Trainable)
-   Backbone의 출력(Hidden States)을 256차원 임베딩 공간으로 매핑하는 선형 변환 층(Linear Layer)만 학습했습니다.
    -   `Linear -> LayerNorm -> GELU -> Linear` 구조

---

## 3. 학습 방법 (Training Methodology)

### 3.1 Loss Function: InfoNCE (Contrastive Loss)
배치(Batch) 내에서 Positive Pair를 찾고, 나머지를 모두 Negative로 간주하는 방식입니다.

$$
L = - \log \frac{\exp(\text{sim}(q, k_+)/\tau)}{\sum_{i=0}^{K} \exp(\text{sim}(q, k_i)/\tau)}
$$

-   **Temperature ($\tau$)**: 0.07 (Sharp한 분포 유도)
-   **Negative Sampling**: In-batch Negatives (배치 내의 다른 유저들을 오답으로 사용)

### 3.2 실험 설정
-   **Dataset**: 775쌍 (Phase 3와 동일)
-   **Batch Size**: 48 ~ 64 (메모리를 아껴서 배치 사이즈를 키움)
-   **Learning Rate**: 1e-4

---

## 4. 실험 결과 및 분석 (Results & Analysis)

| 실험 ID | 설정 | Validation R@1 | Test R@1 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | Baseline (No Dropout) | 8.06% | **1.04%** | 극심한 과적합 및 일반화 실패 |
| **Exp 2** | Dropout (0.3) 추가 | - | **0.97%** | 학습 방해로 성능 더 하락 |

### 🛑 실패 원인 분석 (Key Findings)
1.  **Frozen Backbone의 한계**: 사전 학습된 Qwen3-VL은 일반적인 객체는 잘 알지만, "한국인 데이팅 앱 유저의 매칭 포인트(관상, 스타일 조화 등)"라는 **도메인 특화 지식(Domain Knowledge)**은 전혀 없습니다. Backbone을 고정하니 이 미묘한 특징을 추출하지 못했습니다.
2.  **Explicit Negative 미사용**: 실제 데이터셋에는 "거절(Negative)" 데이터가 존재했으나, InfoNCE 방식은 이를 직접 활용하지 않고 배치 내의 임의의 쌍을 Negative로 썼습니다. 이는 '실제로 싫어해서 거절한' 강력한 신호를 무시하는 셈이 되었습니다.

---

## 5. Phase 3로의 발전 (Transition to Phase 3)
Phase 2의 실패를 교훈 삼아 Phase 3에서는 다음과 같이 전략을 수정했습니다.

1.  **Full Fine-tuning**: Backbone을 **Unfreeze**하여 도메인 특화 특징을 학습하도록 변경.
2.  **Cosine Embedding Loss**: InfoNCE 대신, **Explicit Negative(거절 데이터)**를 직접 학습에 반영하는 Loss로 변경.
3.  **Siamese Network**: 두 입력을 명시적으로 비교하는 구조 채택.

> Phase 3는 Phase 2의 **약 8배 이상**의 성능 향상(R@1 1% -> 8% 이상)을 목표로 설계되었습니다.
