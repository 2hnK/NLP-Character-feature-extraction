# 시스템 아키텍처

## 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dating App Frontend                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway / Load Balancer                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Inference Server                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Feature      │  │  Matching    │  │  Preference  │          │
│  │ Extraction   │  │  Engine      │  │  Update      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   ML Model   │    │  Vector DB   │    │   User DB    │
│  (PyTorch)   │    │   (Faiss)    │    │ (PostgreSQL) │
└──────────────┘    └──────────────┘    └──────────────┘
```

## 모델 아키텍처

### 1. Feature Extraction Network

```
Input Image (224x224x3)
        ↓
┌─────────────────────────────────────┐
│     EfficientNet-B0 Backbone        │
│  (Pretrained on ImageNet)           │
│                                     │
│  Conv2d → BatchNorm → Swish         │
│  MBConv Blocks × 16                 │
│  Global Average Pooling             │
└────────────────┬────────────────────┘
                 ↓
        Feature Map (1280-dim)
                 ↓
┌─────────────────────────────────────┐
│      Embedding Head                 │
│                                     │
│  Linear(1280 → 512)                 │
│  BatchNorm1d                        │
│  ReLU                               │
│  Dropout(0.3)                       │
│  Linear(512 → 512)                  │
│  L2 Normalization                   │
└────────────────┬────────────────────┘
                 ↓
        Embedding Vector (512-dim)
```

### 2. Metric Learning Pipeline

```
Triplet Sampling
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Anchor    │  │  Positive   │  │  Negative   │
│  (User A)   │  │ (Matched    │  │ (Rejected   │
│             │  │  with A)    │  │  by A)      │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
  ┌────────────────────────────────────────┐
  │       Feature Extraction Model         │
  └────────────────────────────────────────┘
       │                │                │
       ▼                ▼                ▼
    Emb_A           Emb_P           Emb_N
       │                │                │
       └────────┬───────┴────────┬───────┘
                │                │
                ▼                ▼
         d(A, P)          d(A, N)
                │                │
                └────────┬───────┘
                         ▼
              Triplet Loss = max(0, d(A,P) - d(A,N) + margin)
```

## 데이터 파이프라인

### 1. Training Pipeline

```
Raw Images
    ↓
┌─────────────────────────┐
│  Face Detection         │
│  (MTCNN / RetinaFace)   │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│  Face Alignment         │
│  (Landmark-based)       │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│  Crop & Resize          │
│  (224x224)              │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│  Data Augmentation      │
│  - Random Crop          │
│  - Color Jitter         │
│  - Random Flip          │
│  - Gaussian Blur        │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│  Normalization          │
│  (ImageNet stats)       │
└────────┬────────────────┘
         ↓
    Training Batch
```

### 2. Inference Pipeline

```
User Profile Image
         ↓
┌─────────────────────────┐
│  Face Detection         │
│  + Alignment            │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│  Preprocessing          │
│  (Resize + Normalize)   │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│  Feature Extraction     │
│  (Model Inference)      │
└────────┬────────────────┘
         ↓
    512-dim Embedding
         ↓
┌─────────────────────────┐
│  Store in Vector DB     │
│  (Faiss Index)          │
└─────────────────────────┘
```

## 매칭 알고리즘

### 1. Basic Matching Flow

```
User Query (user_id)
         ↓
┌─────────────────────────────┐
│  Load User Embedding        │
│  from Vector DB             │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│  Apply Filters              │
│  - Gender preference        │
│  - Age range                │
│  - Location radius          │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│  Vector Similarity Search   │
│  (Faiss ANN)                │
│  - Cosine Similarity        │
│  - Top-K retrieval          │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│  Re-ranking                 │
│  - User preference weight   │
│  - Diversity boost          │
│  - Recency factor           │
└────────┬────────────────────┘
         ↓
    Top-K Matches
```

### 2. Personalized Matching

```
User Feedback Loop

User Actions (Likes/Passes)
         ↓
┌─────────────────────────────┐
│  Collect Positive/Negative  │
│  Samples                    │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│  Compute Preference Vector  │
│  P = mean(liked_embeddings) │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│  Blend with Base Embedding  │
│  E_new = α*E_base + β*P     │
└────────┬────────────────────┘
         ↓
┌─────────────────────────────┐
│  Update User Profile        │
│  in Vector DB               │
└─────────────────────────────┘
```

## 학습 전략

### 1. Training Stages

```
Stage 1: Warmup (5 epochs)
├─ Freeze backbone
├─ Train embedding head only
└─ Learning rate: 1e-3

Stage 2: Fine-tuning (20 epochs)
├─ Unfreeze all layers
├─ Learning rate: 1e-4
└─ Cosine annealing

Stage 3: Refinement (25 epochs)
├─ Add user feedback data
├─ Hard negative mining
└─ Learning rate: 1e-5
```

### 2. Loss Function

```python
Total Loss = Triplet Loss + Regularization

Triplet Loss:
L_triplet = max(0, ||A - P||² - ||A - N||² + margin)

where:
- A: Anchor embedding
- P: Positive embedding
- N: Negative embedding
- margin: 0.5

Regularization:
L_reg = λ * ||W||²  (Weight decay)
```

### 3. Batch Sampling Strategy

```
Batch Construction (batch_size = 32)

For each batch:
1. Sample 8 unique users (classes)
2. For each user, sample:
   - 1 Anchor image
   - 2 Positive images (matches)
   - 1 Negative image (rejects)

Total: 8 × 4 = 32 images per batch

Ensures:
✓ Rich triplets per batch
✓ Balanced positive/negative ratio
✓ Efficient GPU utilization
```

## 벡터 검색 최적화

### Faiss Index Configuration

```
Index Type: IVF + PQ

Parameters:
├─ nlist: 100 (number of clusters)
├─ m: 8 (PQ subvectors)
├─ nbits: 8 (bits per subvector)
└─ metric: METRIC_INNER_PRODUCT

Trade-offs:
✓ Search speed: ~5ms for 100K vectors
✓ Memory: ~10MB for 100K vectors (vs 200MB full)
✓ Recall@10: ~95%
```

## 배포 아키텍처

### Production Setup

```
┌──────────────────────────────────────────┐
│           Application Load Balancer       │
└────────────┬──────────────┬──────────────┘
             │              │
    ┌────────▼─────┐   ┌───▼──────────┐
    │  API Server  │   │  API Server  │
    │  (Instance1) │   │  (Instance2) │
    └────────┬─────┘   └───┬──────────┘
             │              │
    ┌────────▼──────────────▼──────────┐
    │      Model Serving (TorchServe)  │
    │      - GPU Instance (g4dn.xlarge)│
    └────────┬─────────────────────────┘
             │
    ┌────────▼──────────────────────────┐
    │       Vector DB Cluster           │
    │       (Faiss + Redis Cache)       │
    └───────────────────────────────────┘
```

## 모니터링 및 로깅

### Metrics to Track

```
Model Performance:
├─ Embedding quality (silhouette score)
├─ Triplet loss value
└─ Validation accuracy

System Performance:
├─ Inference latency (P50, P95, P99)
├─ Throughput (requests/sec)
├─ GPU utilization
└─ Memory usage

Business Metrics:
├─ Match rate (successful matches / total)
├─ User engagement (swipes per session)
├─ Conversion rate (messages / matches)
└─ Retention rate (7-day, 30-day)
```

## 확장성 고려사항

### Horizontal Scaling

```
User Growth Strategy:

100K users → 1M users → 10M users
    ↓            ↓            ↓
Single GPU → Multi-GPU → Distributed
   Faiss  →  Sharded   →  Milvus/
            Faiss       Qdrant
```

### Model Update Strategy

```
Continuous Learning Cycle:

Week 1-2: Collect new data
   ↓
Week 3: Retrain model
   ↓
Week 4: A/B test new model
   ↓
Week 5: Deploy if improved
   ↓
(Repeat)
```

## 보안 및 프라이버시

### Data Protection

```
1. Image Storage
   ├─ Encryption at rest (AES-256)
   └─ Access control (IAM policies)

2. Embeddings
   ├─ No reverse mapping to original images
   └─ Anonymous user IDs

3. API Security
   ├─ JWT authentication
   ├─ Rate limiting (100 req/min per user)
   └─ HTTPS only
```

## 참고 아키텍처

- **FaceNet**: Embedding 네트워크 설계 참고
- **ArcFace**: Loss 함수 개선 아이디어
- **Airbnb ML Platform**: 유사도 검색 시스템 참고
- **Tinder AI Blog**: 매칭 알고리즘 인사이트
