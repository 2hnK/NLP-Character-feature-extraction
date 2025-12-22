# 커플 매칭 모델 아키텍처 및 학습 구조

> 📅 작성일: 2025-12-22  
> 📁 프로젝트: NLP-Character-feature-extraction

---

## 목차

1. [개요](#1-개요)
2. [모델 아키텍처](#2-모델-아키텍처)
3. [입력(Input) 처리](#3-입력input-처리)
4. [Feature 추출 과정](#4-feature-추출-과정)
5. [출력(Output) 구조](#5-출력output-구조)
6. [학습 방법](#6-학습-방법)
7. [평가 지표](#7-평가-지표)
8. [하이퍼파라미터 설정](#8-하이퍼파라미터-설정)

---

## 1. 개요

본 프로젝트는 **Vision-Language Model(VLM)**을 활용한 커플 매칭 예측 모델입니다.  
실제 커플 이미지 쌍을 학습하여, 여성 이미지가 주어졌을 때 매칭되는 남성 이미지를 검색(Retrieval)하는 것이 목표입니다.

### 핵심 구성요소

| 구성요소            | 역할                       | 파일 위치                          |
| ------------------- | -------------------------- | ---------------------------------- |
| **Backbone**        | Qwen3-VL 기반 특징 추출    | `src/models/qwen_backbone.py`      |
| **Projection Head** | 임베딩 차원 축소 및 정규화 | `src/models/projection.py`         |
| **학습 스크립트**   | InfoNCE Loss 기반 학습     | `scripts/train_couple_matching.py` |

---

## 2. 모델 아키텍처

### 2.1 전체 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Pipeline                            │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │ Female Image│    │ Male Image  │                             │
│  └──────┬──────┘    └──────┬──────┘                             │
│         │                  │                                     │
│         ▼                  ▼                                     │
│  ┌─────────────────────────────────────────┐                    │
│  │     ResizeLongestEdge (max_size=768)    │                    │
│  └─────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Qwen3-VL Backbone (Frozen)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Vision Encoder                        │   │
│  │     (ViT-based, frozen during training)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Language Model Layers                       │   │
│  │     (Transformer Decoder, frozen during training)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│         hidden_states[-1]: [B, seq_len, 2048]                   │
│                              │                                   │
│                              ▼                                   │
│              Mean Pooling: [B, 2048]                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Projection Head (Trainable)                     │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────┐ │
│  │Linear(2048 │──▶│ BatchNorm  │──▶│    ReLU    │──▶│Linear  │ │
│  │   →1024)   │   │   (1024)   │   │            │   │(1024   │ │
│  │            │   │            │   │            │   │ →256)  │ │
│  └────────────┘   └────────────┘   └────────────┘   └────────┘ │
│                                                          │      │
│                                                          ▼      │
│                                            L2 Normalize: [B,256]│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   Output Embedding: [B, 256]
```

### 2.2 Backbone: Qwen3VLFeatureExtractor

**파일**: `src/models/qwen_backbone.py`

```python
class Qwen3VLFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        embedding_dim=2048,
        freeze_vision_encoder=True,  # 학습 시 동결
        use_projection_head=False,   # 외부 Projection Head 사용
        device="cuda"
    ):
        super(Qwen3VLFeatureExtractor, self).__init__()

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
```

**핵심 특징**:

- **모델**: Qwen/Qwen3-VL-2B-Instruct (약 20억 파라미터)
- **정밀도**: bfloat16 (메모리 효율성)
- **동결 상태**: Vision Encoder + Language Model 모두 동결
- **학습 대상**: Projection Head만 학습

### 2.3 Projection Head

**파일**: `src/models/projection.py`

```python
class ProjectionHead(nn.Module):
    """
    Projection Head for Contrastive Learning.
    Structure: Linear -> BatchNorm -> ReLU -> Linear -> L2 Normalization
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)   # 2048 → 1024
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)  # 1024 → 256

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = F.normalize(x, p=2, dim=1)  # L2 정규화
        return x
```

**구조**:
| 레이어 | 입력 차원 | 출력 차원 | 활성화 함수 |
|--------|----------|----------|------------|
| Linear 1 | 2048 | 1024 | - |
| BatchNorm | 1024 | 1024 | - |
| ReLU | 1024 | 1024 | ReLU |
| Linear 2 | 1024 | 256 | - |
| L2 Norm | 256 | 256 | - |

---

## 3. 입력(Input) 처리

### 3.1 데이터셋 구조

**파일**: `scripts/train_couple_matching.py`

커플 데이터는 다음과 같이 구성됩니다:

```
~/data/mutual-like-validations/images/
├── couple_0/
│   ├── female.png
│   └── male.png
├── couple_1/
│   ├── female.png
│   └── male.png
└── ...
```

### 3.2 이미지 전처리

```python
class ResizeLongestEdge:
    """이미지의 긴 변을 max_size로 리사이즈"""
    def __init__(self, max_size: int, interpolation=Image.BICUBIC):
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.max_size / max(w, h)
        if scale >= 1:
            return img  # 이미 작으면 그대로
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), self.interpolation)
```

**전처리 과정**:

1. **파일 로드**: `Image.open().convert('RGB')`
2. **리사이즈**: 긴 변 기준 768px로 조정 (비율 유지)
3. **보간법**: BICUBIC (고품질)

### 3.3 VLM 입력 구성

**파일**: `src/models/qwen_backbone.py` (Line 174-210)

```python
def forward(self, images):
    if isinstance(images, list):
        # PIL Images를 VLM 입력 형식으로 변환
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this person's appearance."}
                    ]
                }
            ]
            for img in images
        ]

        # Chat template 적용
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversations
        ]

        # Vision 정보 처리
        image_inputs, video_inputs = process_vision_info(conversations)

        # Processor로 토큰화
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
```

**입력 구성 요소**:

| 요소                | 내용                                 | 목적                        |
| ------------------- | ------------------------------------ | --------------------------- |
| **이미지**          | PIL Image (RGB)                      | 시각적 특징 추출            |
| **텍스트 프롬프트** | "Describe this person's appearance." | 외모 특징에 집중하도록 유도 |
| **Chat Template**   | Qwen3 형식                           | 모델 입력 형식 맞춤         |

---

## 4. Feature 추출 과정

### 4.1 Hidden States 추출

**파일**: `src/models/qwen_backbone.py` (Line 122-156)

```python
def extract_vision_features(self, inputs):
    # 모델 forward pass (생성 없이 hidden states만 추출)
    outputs = self.model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

    # 마지막 레이어의 hidden states 사용
    hidden_states = outputs.hidden_states[-1]  # [B, seq_len, 2048]

    # Mean Pooling: 시퀀스 전체 평균
    pooled_features = hidden_states.mean(dim=1)  # [B, 2048]

    # float32로 변환 (Projection Head 호환성)
    pooled_features = pooled_features.to(torch.float32)

    return pooled_features
```

### 4.2 추출 위치 상세

```
Qwen3-VL Model Structure:
├── Vision Encoder (ViT)
│   └── Image → Visual Tokens
├── Token Embedding
│   └── Text → Text Tokens
├── Transformer Decoder Layers
│   ├── Layer 0
│   ├── Layer 1
│   ├── ...
│   └── Layer N-1 (Last Layer) ← hidden_states[-1] 사용
│       └── Shape: [batch_size, sequence_length, 2048]
└── LM Head (사용 안 함)
```

**선택 근거**:

- **마지막 레이어**: 가장 추상화된 고수준 의미(semantic) 표현
- **Mean Pooling**: 이미지+텍스트 토큰 전체의 평균 표현
- **차원**: 2048 (Qwen3-VL-2B의 hidden size)

---

## 5. 출력(Output) 구조

### 5.1 최종 임베딩

```
Input: PIL Image
    ↓
Backbone: [B, 2048] (raw features)
    ↓
Projection Head: [B, 256] (normalized embeddings)
    ↓
Output: 256차원 단위 벡터 (||v|| = 1)
```

### 5.2 임베딩 특성

| 특성            | 값          | 설명                       |
| --------------- | ----------- | -------------------------- |
| **차원**        | 256         | 메모리 효율적, 검색에 적합 |
| **정규화**      | L2 Norm = 1 | 코사인 유사도 = 내적       |
| **데이터 타입** | float32     | 수치 안정성                |

### 5.3 유사도 계산

```python
# 코사인 유사도 = 내적 (L2 정규화된 벡터의 경우)
similarity = torch.matmul(female_embs, male_embs.T)  # [B, B]
```

---

## 6. 학습 방법

### 6.1 학습 목표

**목표**: 실제 커플 쌍의 임베딩은 가깝게, 비커플 쌍은 멀게 학습

```
같은 커플: female_i ↔ male_i → 유사도 높게
다른 쌍:   female_i ↔ male_j (i≠j) → 유사도 낮게
```

### 6.2 InfoNCE Loss

**파일**: `scripts/train_couple_matching.py` (Line 166-199)

```python
class InfoNCELoss(nn.Module):
    """InfoNCE Loss for contrastive learning"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, female_embs: torch.Tensor, male_embs: torch.Tensor) -> torch.Tensor:
        batch_size = female_embs.size(0)

        # 유사도 행렬: [B, B]
        logits = torch.matmul(female_embs, male_embs.T) / self.temperature

        # 정답 레이블: 대각선 (i번 female ↔ i번 male)
        labels = torch.arange(batch_size, device=logits.device)

        # Female → Male 방향 loss
        loss_f2m = F.cross_entropy(logits, labels)

        # Male → Female 방향 loss
        loss_m2f = F.cross_entropy(logits.T, labels)

        # 양방향 평균
        loss = (loss_f2m + loss_m2f) / 2

        return loss
```

### 6.3 Loss 수식

$$\mathcal{L}_{InfoNCE} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{e^{sim(f_i, m_i)/\tau}}{\sum_{j=1}^{N} e^{sim(f_i, m_j)/\tau}} + \log \frac{e^{sim(m_i, f_i)/\tau}}{\sum_{j=1}^{N} e^{sim(m_i, f_j)/\tau}} \right]$$

- $f_i$: i번째 female 임베딩
- $m_i$: i번째 male 임베딩 (실제 커플)
- $\tau$: temperature (0.1)
- $sim(\cdot, \cdot)$: 코사인 유사도 (내적)

### 6.4 학습 과정 예시

```
Batch Size = 4인 경우:

유사도 행렬 (logits / temperature):
              Male_0   Male_1   Male_2   Male_3
Female_0  [  ⭐9.0     2.0      1.0      3.0  ]
Female_1  [   1.0    ⭐8.5     2.0      1.5  ]
Female_2  [   2.0     1.5    ⭐8.8     1.0  ]
Female_3  [   1.0     2.0      1.0    ⭐9.2  ]

CrossEntropy:
- Female_0의 정답 = 0 (Male_0)
- Female_1의 정답 = 1 (Male_1)
- ...

→ 대각선 값이 가장 높아지도록 학습!
```

### 6.5 학습 루프

**파일**: `scripts/train_couple_matching.py` (Line 255-304)

```python
def train_one_epoch(backbone, projection_head, dataloader, optimizer, criterion,
                    scaler, device, config, epoch):
    projection_head.train()
    backbone.eval()  # Backbone은 항상 eval (동결)

    for batch in pbar:
        female_imgs = batch['female_imgs']
        male_imgs = batch['male_imgs']

        optimizer.zero_grad()

        with autocast('cuda', enabled=config.use_amp):
            # Forward pass (Backbone은 gradient 계산 안 함)
            with torch.no_grad():
                female_features = backbone.forward(female_imgs)
                male_features = backbone.forward(male_imgs)

            # Projection Head는 gradient 계산
            female_embs = projection_head(female_features)
            male_embs = projection_head(male_features)

            # Loss 계산
            loss = criterion(female_embs, male_embs)

        # Backward (Projection Head만 업데이트)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

---

## 7. 평가 지표

### 7.1 Recall@K

**파일**: `scripts/train_couple_matching.py` (Line 232-252)

```python
def compute_recall_at_k(female_embs, male_embs, k_values=[1, 5, 10]):
    n = len(female_embs)
    similarity = np.dot(female_embs, male_embs.T)

    results = {}

    # Female → Male 검색
    ranks_f2m = []
    for i in range(n):
        sorted_idx = np.argsort(-similarity[i])  # 유사도 내림차순 정렬
        rank = np.where(sorted_idx == i)[0][0]   # 정답의 순위 (0-indexed)
        ranks_f2m.append(rank)
    ranks_f2m = np.array(ranks_f2m)

    for k in k_values:
        results[f'recall@{k}'] = np.mean(ranks_f2m < k)  # 상위 K개 안에 정답 비율

    results['mrr'] = np.mean(1.0 / (ranks_f2m + 1))  # Mean Reciprocal Rank

    return results
```

### 7.2 지표 정의

| 지표          | 수식                                      | 의미          |
| ------------- | ----------------------------------------- | ------------- |
| **Recall@1**  | $\frac{1}{N}\sum \mathbb{1}[rank_i = 0]$  | 1위 정확도    |
| **Recall@5**  | $\frac{1}{N}\sum \mathbb{1}[rank_i < 5]$  | Top-5 정확도  |
| **Recall@10** | $\frac{1}{N}\sum \mathbb{1}[rank_i < 10]$ | Top-10 정확도 |
| **MRR**       | $\frac{1}{N}\sum \frac{1}{rank_i + 1}$    | 평균 역순위   |

### 7.3 해석 예시

```
N = 100명의 female에 대해 100명의 male 중 검색:

Recall@1 = 15% → 15명이 1위로 정답을 맞춤
Recall@5 = 35% → 35명이 상위 5개 안에 정답 포함
MRR = 0.25     → 평균적으로 정답이 4위 근처
```

---

## 8. 하이퍼파라미터 설정

### 8.1 모델 설정

**파일**: `scripts/train_couple_matching.py` (Line 49-84)

```python
@dataclass
class TrainConfig:
    # 모델 설정
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    embedding_dim: int = 2048          # Backbone 출력 차원
    projection_hidden_dim: int = 1024  # Projection 중간 차원
    projection_output_dim: int = 256   # 최종 임베딩 차원

    # 학습 하이퍼파라미터
    batch_size: int = 48               # 큰 배치 (InfoNCE에 유리)
    learning_rate: float = 5e-5        # 낮은 학습률
    weight_decay: float = 1e-3         # 규제 강화
    epochs: int = 30
    temperature: float = 0.1           # InfoNCE temperature

    # 스케줄러
    warmup_epochs: int = 2

    # Early stopping
    patience: int = 10

    # 이미지
    image_size: int = 768

    # Mixed precision
    use_amp: bool = True
```

### 8.2 Optimizer 및 Scheduler

```python
# AdamW Optimizer (Projection Head만)
optimizer = torch.optim.AdamW(
    projection_head.parameters(),
    lr=config.learning_rate,    # 5e-5
    weight_decay=config.weight_decay  # 1e-3
)

# Cosine Annealing Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

# Mixed Precision Scaler
scaler = GradScaler('cuda', enabled=config.use_amp)
```

### 8.3 하이퍼파라미터 선택 근거

| 파라미터          | 값   | 근거                                               |
| ----------------- | ---- | -------------------------------------------------- |
| **Batch Size**    | 48   | InfoNCE는 큰 배치에서 더 많은 negative sample 확보 |
| **Learning Rate** | 5e-5 | 사전학습 가중치 보존, 안정적 수렴                  |
| **Temperature**   | 0.1  | 너무 낮으면 과적합, 너무 높으면 구분력 저하        |
| **Weight Decay**  | 1e-3 | 과적합 방지                                        |
| **Output Dim**    | 256  | 검색 효율성 & 표현력 균형                          |
| **Image Size**    | 768  | VLM 권장 크기, 세부 특징 보존                      |

---

## 9. 체크포인트 저장

**파일**: `scripts/train_couple_matching.py` (Line 448-457)

```python
checkpoint_path = os.path.join(config.checkpoint_dir, f"best_model_fold{args.fold}.pth")

torch.save({
    'epoch': epoch,
    'fold': args.fold,
    'backbone_state_dict': backbone.state_dict(),
    'projection_head_state_dict': projection_head.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_recall': best_recall,
    'metrics': recall_metrics,
    'config': config.__dict__
}, checkpoint_path)
```

**저장 항목**:

- 에폭 정보
- Backbone 가중치 (optional, 동결 상태)
- Projection Head 가중치 (핵심)
- Optimizer 상태
- 성능 메트릭
- 설정값

---

## 10. 요약

### 전체 파이프라인

```
1. Input
   └── Female/Male 이미지 쌍 (768px 리사이즈)

2. VLM Processing
   └── Qwen3-VL로 이미지+프롬프트 처리
   └── 마지막 레이어 hidden states 추출
   └── Mean Pooling → [B, 2048]

3. Projection
   └── Linear → BatchNorm → ReLU → Linear
   └── L2 Normalization → [B, 256]

4. Training
   └── InfoNCE Loss (temperature=0.1)
   └── 대각선(실제 커플) 유사도 최대화

5. Evaluation
   └── Recall@K, MRR로 검색 성능 측정
```

### 핵심 설계 결정

| 설계 결정              | 선택 | 이유                            |
| ---------------------- | ---- | ------------------------------- |
| Backbone 동결          | ✅   | 사전학습 지식 보존, 메모리 절약 |
| Projection Head만 학습 | ✅   | 빠른 학습, 과적합 방지          |
| InfoNCE Loss           | ✅   | SOTA contrastive learning 방법  |
| 양방향 Loss            | ✅   | 대칭적 유사도 학습              |
| Mean Pooling           | ✅   | 전체 특징 활용                  |
| L2 정규화              | ✅   | 코사인 유사도 = 내적            |
