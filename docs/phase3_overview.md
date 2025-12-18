# Phase 3: Multimodal Matching Model (Qwen3-VL Siamese Network)

본 문서는 프로젝트의 **Phase 3**에 해당하는 **'사용자 매칭 호환성 예측 모델'**의 구조, 학습 방법, 데이터 처리 파이프라인을 상세히 기술합니다.

---

## 1. 개요 (Overview)
Phase 3의 목표는 실제 데이팅 앱 데이터(User A, User B)를 기반으로 **두 사용자가 매칭될 확률(Compatibility)**을 예측하는 것입니다. 단순한 이미지 매칭을 넘어, 프로필 텍스트(성별, 관심사 등)와 이미지를 결합한 **멀티모달(Multimodal)** 학습을 수행합니다.

## 2. 모델 아키텍처 (Model Architecture)

### 2.1 Backbone: Qwen3-VL-2B
-   **기반 모델**: `Qwen/Qwen3-VL-2B-Instruct`
-   **특징**: Vision-Language Model (VLM)로, 이미지와 텍스트를 동시에 이해할 수 있는 강력한 성능을 보유.
-   **역할**: 각 사용자의 프로필 이미지와 텍스트 설명을 입력받아 고차원 특징 벡터(Embedding)를 추출합니다.

### 2.2 Siamese Network (샴 네트워크)
두 개의 입력을 동일한 가중치를 가진 하나의 모델(Backbone)에 통과시켜 비교하는 구조입니다.
-   **User A -> Backbone -> Embedding A**
-   **User B -> Backbone -> Embedding B**
-   두 임베딩 벡터 간의 거리를 계산하여 유사도를 측정합니다.

### 2.3 Projection Head
Backbone에서 나온 특징 벡터를 매칭 예측에 적합한 공간으로 변환합니다.
-   **Vision Features**: 이미지에서 추출된 시각적 특징.
-   **Text Features**: "여성 사용자, 관심사: 여행, 맛집" 등의 텍스트 정보.
-   **Fusion**: 이미지(70%) + 텍스트(30%) 가중치 합산 (코드상의 `0.7 * vision + 0.3 * text`).
-   **Output**: 256차원의 최종 임베딩 벡터.

---

## 3. 학습 방법 (Training Methodology)

### 3.1 Loss Function: Cosine Embedding Loss
Contrastive Learning(대조 학습)의 일종으로, 두 벡터의 코사인 유사도(Cosine Similarity)를 기반으로 학습합니다.

$$
L(x_1, x_2, y) = \begin{cases}
1 - \cos(x_1, x_2), & \text{if } y = 1 \text{ (Positive)} \\
\max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1 \text{ (Negative)}
\end{cases}
$$

*   **Positive Pair ($y=1$)**: 실제 매칭된 커플. 두 벡터의 각도를 **좁혀서(유사하게)** 만듭니다.
*   **Negative Pair ($y=-1$)**: 매칭되지 않은 쌍. 두 벡터의 각도를 **벌려서(다르게)** 만듭니다. (Margin 0.3 이상으로)

### 3.2 학습 프로세스 (Step-by-Step)
1.  **Batch Load**: 데이터셋에서 N개의 쌍(Pair)을 가져옵니다. (Batch Size 2라면 총 4명)
2.  **Forward Pass**: 
    *   User A의 이미지/텍스트를 모델에 입력 -> $E_A$ 추출
    *   User B의 이미지/텍스트를 모델에 입력 -> $E_B$ 추출
3.  **Compute Loss**: $E_A$와 $E_B$ 사이의 코사인 유사도를 구하고, 실제 라벨(Positive/Negative)과 비교하여 Loss 계산.
4.  **Backpropagation**: Loss를 줄이는 방향으로 Qwen3-VL 모델의 파라미터 업데이트.

---

## 4. 데이터셋 구조 (Dataset)
-   **Format**: JSONL (`dataset.jsonl`)
-   **구성**:
    -   `pairId`: 매칭 쌍 고유 ID (예: `positive_001`)
    -   `userA`, `userB`: 각각의 이미지 경로 및 메타데이터(성별, 관심사)
    -   `pairType`: `positive` (성공) / `negative` (실패)
-   **전처리**:
    -   이미지 리사이징 (448x448)
    -   텍스트 설명 생성 (예: "남성 사용자, 관심사: 운동")

---

## 5. 하이퍼파라미터 설정 (Hyperparameters)
`ml.g5.2xlarge (A10G 24GB)` 환경 기준 최적화된 값입니다.

| 파라미터 | 값 | 설명 |
| :--- | :--- | :--- |
| **Batch Size** | **2 ~ 4** | 2B 모델 Full Fine-tuning 및 Siamese 구조(2배 메모리)로 인한 제약. |
| **Learning Rate** | **1e-5 ~ 2e-5** | Catastrophic Forgetting 방지를 위해 매우 낮은 학습률 사용. |
| **Epochs** | **5 ~ 10** | 데이터 수가 적으므로(774쌍) 과적합 방지를 위해 짧게 학습. |
| **Margin** | **0.3** | Positive/Negative 구분을 위한 최소 거리 기준. |
| **Embedding Dim** | **256** | 벡터 공간의 효율성을 위해 512에서 압축. |

---

## 6. 기대 효과
이 모델은 단순 외모 유사성이 아닌, **"성공한 커플들의 시각적/취향적 패턴"**을 학습합니다. 이를 통해 새로운 사용자 쌍이 주어졌을 때, 이들이 얼마나 잘 어울리는지(매칭 확률)를 수치화하여 제공할 수 있습니다.
