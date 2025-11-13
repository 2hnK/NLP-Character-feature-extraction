# 데이터 명세서

## 데이터셋 구조

### 디렉토리 구성

```
data/
├── raw/                          # 원본 데이터
│   ├── profiles/                 # 실제 프로필 이미지
│   │   ├── user_00001.jpg
│   │   ├── user_00002.jpg
│   │   └── ...
│   └── metadata.csv              # 메타데이터
│
├── processed/                    # 전처리된 데이터
│   ├── train/
│   │   ├── images/               # 전처리된 이미지
│   │   └── embeddings.npy        # 사전 계산된 임베딩 (선택)
│   ├── val/
│   └── test/
│
└── augmented/                    # 생성형 AI 증강 데이터
    ├── synthetic/                # 완전 합성 이미지
    └── enhanced/                 # 원본 기반 증강
```

## 메타데이터 형식

### metadata.csv

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| user_id | string | 고유 사용자 ID (익명화) | "usr_a3f9d2e1" |
| image_path | string | 이미지 파일 경로 | "profiles/user_00001.jpg" |
| gender | string | 성별 | "M", "F", "O" |
| age | int | 나이 | 25 |
| image_quality | float | 이미지 품질 점수 (0-1) | 0.85 |
| face_detected | bool | 얼굴 검출 여부 | True |
| face_confidence | float | 얼굴 검출 신뢰도 | 0.92 |
| image_width | int | 이미지 너비 (픽셀) | 1024 |
| image_height | int | 이미지 높이 (픽셀) | 1024 |
| upload_date | datetime | 업로드 시간 | "2024-01-15 14:30:00" |
| is_synthetic | bool | 합성 이미지 여부 | False |
| source | string | 데이터 출처 | "real", "stable_diffusion", "midjourney" |

**예시:**
```csv
user_id,image_path,gender,age,face_detected,face_confidence,image_quality,is_synthetic,source
usr_a3f9d2e1,profiles/user_00001.jpg,F,28,True,0.95,0.87,False,real
usr_b7c4e8f2,profiles/user_00002.jpg,M,32,True,0.89,0.82,False,real
usr_syn_0001,augmented/synthetic/img_0001.jpg,F,25,True,0.91,0.79,True,stable_diffusion
```

## 행동 데이터 (User Interaction)

### interactions.csv

매칭 성공/실패 기록을 저장하여 학습에 활용

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| interaction_id | string | 상호작용 ID | "int_x8d9a2" |
| user_id | string | 행동한 사용자 | "usr_a3f9d2e1" |
| target_user_id | string | 대상 사용자 | "usr_b7c4e8f2" |
| action | string | 행동 유형 | "like", "pass", "match", "message" |
| timestamp | datetime | 행동 시간 | "2024-01-15 15:45:30" |
| is_mutual | bool | 상호 좋아요 여부 | True |
| conversation_started | bool | 대화 시작 여부 | True |

**예시:**
```csv
interaction_id,user_id,target_user_id,action,timestamp,is_mutual,conversation_started
int_x8d9a2,usr_a3f9d2e1,usr_b7c4e8f2,like,2024-01-15 15:45:30,True,True
int_y9e1b3,usr_a3f9d2e1,usr_c5f3g9h1,pass,2024-01-15 15:46:10,False,False
int_z1f2c4,usr_b7c4e8f2,usr_a3f9d2e1,like,2024-01-15 15:47:00,True,True
```

## 학습 데이터 구성

### Triplet 데이터셋

학습 시 동적으로 생성하거나 사전 구성

#### triplets.csv (사전 구성 시)

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| anchor_id | string | Anchor 사용자 ID |
| positive_id | string | Positive 사용자 ID (매칭 성공) |
| negative_id | string | Negative 사용자 ID (매칭 실패) |
| triplet_source | string | 생성 방법 |

**생성 전략:**

1. **Hard Negatives**: 유사하지만 매칭 실패한 프로필
2. **Random Negatives**: 무작위 선택
3. **Semi-hard Negatives**: 중간 거리의 네거티브

```python
# 동적 생성 예시
def create_triplet(user_id, interactions_df):
    # Anchor
    anchor = user_id

    # Positive: 매칭 성공한 사용자
    positives = interactions_df[
        (interactions_df['user_id'] == user_id) &
        (interactions_df['is_mutual'] == True)
    ]['target_user_id'].tolist()

    # Negative: 패스한 사용자
    negatives = interactions_df[
        (interactions_df['user_id'] == user_id) &
        (interactions_df['action'] == 'pass')
    ]['target_user_id'].tolist()

    if len(positives) > 0 and len(negatives) > 0:
        positive = random.choice(positives)
        negative = random.choice(negatives)
        return (anchor, positive, negative)

    return None
```

## 이미지 사양

### 입력 이미지 요구사항

| 속성 | 최소 요구사항 | 권장 사항 |
|------|--------------|-----------|
| 해상도 | 256 × 256 | 512 × 512 이상 |
| 파일 형식 | JPG, PNG | JPG (압축률 85%) |
| 파일 크기 | < 5MB | < 2MB |
| 얼굴 크기 | 이미지의 20% 이상 | 이미지의 40-60% |
| 조명 | 얼굴이 명확히 보임 | 자연광 또는 균일한 조명 |
| 각도 | 정면 ±45도 이내 | 정면 ±30도 이내 |

### 전처리 후 이미지 규격

```
표준 규격:
- 크기: 224 × 224 pixels
- 색상: RGB (3 channels)
- 정규화: ImageNet mean/std
  - mean: [0.485, 0.456, 0.406]
  - std: [0.229, 0.224, 0.225]
- 데이터 타입: float32
- 값 범위: [0, 1] → normalized
```

## 데이터 증강 사양

### 1. 기본 증강 (Augmentation)

```python
train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### 2. 생성형 AI 증강

#### Stable Diffusion 프롬프트 템플릿

```
Positive Prompt:
"professional dating profile photo of a {age}-year-old {gender},
{style} style, {lighting} lighting, {background} background,
high quality, 4k, detailed face"

Parameters:
- style: casual, professional, artistic, natural
- lighting: natural, studio, golden hour, soft
- background: plain, outdoor, urban, home

Negative Prompt:
"blurry, low quality, distorted face, multiple people,
nude, inappropriate, watermark"
```

#### 증강 데이터 비율

```
총 학습 데이터 구성:
├─ 실제 프로필 이미지: 60%
├─ 기본 증강 (Flip, Crop 등): 30%
└─ 생성형 AI 증강: 10%

목표: 다양성 확보 + 실제 데이터 품질 유지
```

## 데이터 분할

### Train / Validation / Test Split

```
전체 데이터셋 분할 비율:
├─ Train: 70% (학습)
├─ Validation: 15% (하이퍼파라미터 튜닝)
└─ Test: 15% (최종 평가)

분할 기준:
- 사용자 ID 기준 분할 (동일 사용자가 여러 split에 포함 방지)
- 성별/나이 분포 유지 (Stratified Split)
- 시간 기반 분할 (선택): 과거 데이터로 학습 → 최신 데이터로 테스트
```

### 데이터 개수 예시

```
Stage 1: MVP (Minimum Viable Product)
├─ Train: 5,000 users × 1 image = 5,000 images
├─ Val: 1,000 images
└─ Test: 1,000 images

Stage 2: Production
├─ Train: 50,000 users × 1-3 images = 100,000 images
├─ Val: 10,000 images
└─ Test: 10,000 images

Stage 3: Scale
├─ Train: 500,000+ images
├─ Val: 50,000 images
└─ Test: 50,000 images
```

## 데이터 품질 관리

### 자동 필터링 기준

```python
def filter_low_quality_images(metadata_df):
    """
    품질 기준에 미달하는 이미지 필터링
    """
    filtered = metadata_df[
        (metadata_df['face_detected'] == True) &           # 얼굴 검출 필수
        (metadata_df['face_confidence'] >= 0.8) &          # 검출 신뢰도 80% 이상
        (metadata_df['image_quality'] >= 0.6) &            # 이미지 품질 60% 이상
        (metadata_df['image_width'] >= 256) &              # 최소 해상도
        (metadata_df['image_height'] >= 256)
    ]
    return filtered
```

### 데이터 검증 체크리스트

```
□ 중복 이미지 제거 (perceptual hash 사용)
□ 손상된 파일 제거 (PIL.Image.verify)
□ 얼굴 미검출 이미지 제거
□ 저해상도 이미지 제거 (< 256×256)
□ 부적절한 콘텐츠 필터링 (NSFW classifier)
□ 메타데이터 일관성 검증
□ 파일명-메타데이터 매칭 확인
```

## 임베딩 벡터 저장

### 저장 형식

```python
# NumPy 형식
embeddings = {
    'user_ids': ['usr_a3f9d2e1', 'usr_b7c4e8f2', ...],  # List[str]
    'vectors': np.array([[...], [...]]),                  # shape: (N, 512)
    'metadata': {
        'model_version': 'v1.0.0',
        'extraction_date': '2024-01-15',
        'embedding_dim': 512
    }
}

# 저장
np.save('embeddings.npy', embeddings)

# 또는 HDF5 형식 (대용량)
import h5py
with h5py.File('embeddings.h5', 'w') as f:
    f.create_dataset('vectors', data=vectors)
    f.create_dataset('user_ids', data=user_ids)
```

### Faiss Index 저장

```python
import faiss

# 인덱스 구축
index = faiss.IndexFlatIP(512)  # Inner Product (Cosine similarity)
index.add(embeddings)

# 저장
faiss.write_index(index, 'faiss_index.bin')

# 로드
index = faiss.read_index('faiss_index.bin')
```

## 데이터 버전 관리

### DVC (Data Version Control)

```bash
# 데이터 추적
dvc add data/raw/profiles
dvc add data/processed

# Git에 메타데이터만 커밋
git add data/raw/profiles.dvc data/processed.dvc
git commit -m "Add dataset v1.0"

# 태그
git tag -a data-v1.0 -m "Initial dataset"

# 원격 스토리지 설정 (S3)
dvc remote add -d storage s3://mybucket/dvcstore
dvc push
```

## 프라이버시 및 규정 준수

### 개인정보 보호

```
1. 익명화
   - 사용자 ID: 해시 처리 (SHA-256)
   - 이미지 메타데이터: EXIF 제거

2. 암호화
   - 저장: AES-256 암호화
   - 전송: TLS 1.3

3. 접근 제어
   - 역할 기반 접근 (RBAC)
   - 감사 로그 (Audit Log)

4. 데이터 보관 정책
   - 삭제 요청 시 즉시 삭제
   - 백업 데이터 30일 후 자동 삭제
```

### GDPR / CCPA 준수

```
- Right to access: 사용자 데이터 조회 API 제공
- Right to deletion: 완전 삭제 보장
- Right to portability: 데이터 내보내기 기능
- Consent management: 명시적 동의 수집
```

## 참고 데이터셋

학습 및 벤치마킹에 활용 가능한 공개 데이터셋:

| 데이터셋 | 규모 | 용도 |
|---------|------|------|
| CelebA | 200K images, 10K identities | 얼굴 속성 분류 |
| VGGFace2 | 3.3M images, 9K identities | Face recognition |
| MS-Celeb-1M | 10M images, 100K identities | Large-scale training |
| LFW (Labeled Faces in the Wild) | 13K images | 벤치마크 평가 |

**주의**: 실제 프로덕션에서는 자체 수집 데이터 사용 필수 (저작권 및 프라이버시)
