## 1. 개요 (Overview)

본 프로젝트는 데이팅 앱 사용자의 프로필 이미지를 분석하여, **시각적 스타일(Fashion Style)과 분위기(Vibe)**가 유사한 사용자끼리 매칭해주는 추천 시스템을 구축하는 것을 목표로 한다. 단순한 객체 분류를 넘어, **Metric Learning** 기법을 통해 '스타일적 유사도'를 벡터 공간상의 거리로 정량화한다.

## 2. 핵심 가설 (Key Hypothesis)

- **매칭 가설 (Matching Hypothesis):** 사람들은 자신과 유사한 수준의 시각적 매력도나 스타일 분위기를 가진 상대에게 더 높은 호감을 느낀다.
- **기술적 접근:** 텍스트 기반의 필터링(나이, 거주지)으로는 포착할 수 없는 '취향'과 '분위기'를 이미지 임베딩 벡터의 코사인 유사도(Cosine Similarity)로 해결한다.

---

## 3. 데이터셋 구축 (Data Pipeline)

### 3.1. 데이터 수집 및 라벨링 (AI-Assisted Labeling)

- **도구: GPT-5.1** (Vision-Language Model)
- **방식:** 원본 프로필 이미지를 입력하여 구조화된 JSON 메타데이터를 자동 추출.
- **스키마 구조 (Hybrid Schema):**
    - **명목형 (Categorical - 학습 기준):** `fashion_style` (5 Class), `shot_type` (3 Class), `visual_quality` (3 Class).
    - **서술형 (Descriptive - 임베딩 보강):** `physical_features` (헤어, 액세서리 등 외형 특징), `caption` (텍스트-이미지 매칭용 자연어 설명).

### 3.2. 클래스 정의 (Class Definition)

Triplet Loss 학습을 위한 Positive/Negative 기준이 되는 핵심 스타일 클래스 5종 정의.

- **Fashion Style (5 Classes)**
    - `Casual_Basic`: 편안함, 티셔츠, 청바지, 후드.
    - `Street_Hip`: 오버핏, 레이어드, 스트릿 무드.
    - `Sporty_Athleisure`: 운동복, 레깅스, 저지.
    - `Chic_Modern`: 블랙, 가죽, 시크함, 도시적.
    - `Classy_Elegant`: 셔츠, 슬랙스, 수트, 블라우스, 원피스 (깔끔/격식).

- **Shot Type (3 Classes)**
    - `Selfie_CloseUp`: 얼굴 위주 셀카.
    - `Mirrored_Selfie`: 거울 셀카 (전신/반신).
    - `Others_Cam`: 남이 찍어준 사진 (전신/반신/스냅).

### 3.3. 전처리 (Preprocessing)

- **Label Encoding:** 문자열 클래스를 0~N의 정수 인덱스로 변환.
- **Class Balancing:** 특정 스타일(예: Casual)에 편중되지 않도록 데이터 증강(Augmentation) 및 언더샘플링 적용.
- **Quality Filtering:** `visual_quality`가 'Low'인 데이터는 학습에서 제외하여 노이즈 제거.

---

## 4. 모델 아키텍처 (Model Architecture)

### 4.1. Backbone Network (Feature Extractor)

- **모델:** **Qwen3-VL-2B** (Vision-Language Model)
- **역할:** 이미지를 입력받아 풍부한 시각적/의미적 특징이 담긴 고차원 임베딩 추출.
- **설정:** **Freeze (동결)**. 사전 학습된 일반화 능력을 유지하기 위해 가중치를 업데이트하지 않음.

### 4.2. Projection Head (Trainable)

- **구조:** `Linear` → `BatchNorm` → `ReLU` → `Linear` → `L2 Normalize`
- **역할:** Backbone의 범용 특징을 **'스타일 매칭 전용'** 저차원 벡터(예: 128~256 dim)로 압축 및 변환.
- **특이사항:** 마지막에 반드시 **L2 Normalization**을 적용하여 모든 벡터를 단위 구(Unit Hypersphere) 위에 배치 (Cosine Similarity 최적화).

---

## 5. 학습 방법론 (Training Strategy)

### 5.1. 학습 목표 (Objective)

- *Metric Learning (거리 학습)**을 통해 같은 스타일(`AnchorPositive`)은 가깝게, 다른 스타일(`AnchorNegative`)은 멀게 배치한다.

### 5.2. 손실 함수 (Loss Function)

**Triplet Margin Loss**를 사용한다.

![image.png](attachment:c84724d6-fcfa-4a6d-8733-0bf77ca589b1:image.png)

### 5.3. 마이닝 전략 (Mining Strategy) - 핵심 기술

- **Online Mining (Batch Hard):** 학습 배치(Batch) 내에서 실시간으로 가장 구분하기 어려운(Loss가 큰) 샘플들을 찾아 학습한다.
    - *Hard Positive:* 같은 스타일인데 다르게 생긴 사진 (예: 여름 댄디룩 vs 겨울 댄디룩)
    - *Hard Negative:* 다른 스타일인데 비슷하게 생긴 사진 (예: 깔끔한 캐주얼 vs 댄디)

### 5.4. 배치 구성 (Batch Sampling)

- **기법:** **PK Sampler (M-Per-Class Sampler)**
- **원리:** 매 배치마다 P개의 스타일 클래스를 선택하고, 각 클래스당 K장의 이미지를 강제로 포함시킴.

![image.png](attachment:78f42f26-85f8-4d9a-9b55-090a645692c3:image.png)

- **이유:** 배치 내에 반드시 Positive Pair가 존재하도록 보장하여 Online Mining이 가능하게 함.

---

## 6. 평가 및 검증 (Evaluation)

### 6.1. 정량적 지표 (Quantitative Metrics)

- **Validation Loss:** 검증 데이터셋에 대한 Triplet Loss 감소 추이 확인.
- **Recall@K (Top-K Accuracy):**
    - 테스트 이미지를 쿼리로 넣었을 때, 상위 K개 추천 결과 중 실제로 같은 스타일인 비율.
    - 추천 시스템의 실질적인 만족도를 대변하는 지표.

### 6.2. 정성적 지표 (Qualitative Metrics)

- **t-SNE 시각화:** 학습된 고차원 임베딩을 2차원으로 축소하여 시각화.
    - 성공 기준: 같은 스타일의 점들이 서로 뭉쳐 있고(Cluster), 다른 스타일과는 명확한 경계(Margin)가 형성되어야 함.

---

## 7. 기대 효과 (Expected Impact)

- **사용자 경험:** 텍스트 필터로는 찾을 수 없는 '취향 저격' 매칭 제공으로 매칭 성사율 및 앱 체류 시간 증대.
- **확장성:** 텍스트 설명(`caption`)을 함께 활용하여 "깔끔한 댄디룩 찾아줘"와 같은 **자연어 기반 스타일 검색(Text-to-Image Retrieval)** 기능으로 확장 가능.