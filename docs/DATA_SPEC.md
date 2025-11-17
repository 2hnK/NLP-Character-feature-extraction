# 데이터 명세서

## 데이터셋 구조

### 현재 프로젝트 데이터 규모

```
총 데이터: 3,300개 이미지
├── 실제 사용자 이미지: 100개 (검증용)
│   └── 사용자당 이미지: 1~3개 (평균 2장)
└── 증강 이미지: 3,200개 (학습용)
    └── 생성 방법: 생성형 AI (Stable Diffusion 등)

목표 분할:
├── Train: 약 2,800개 (실제 70개 + 증강 2,730개)
├── Validation: 약 300개 (실제 20개 + 증강 280개)
└── Test: 약 200개 (실제 10개 + 증강 190개)
```

### 디렉토리 구성

```
data/
├── raw/                          # 원본 데이터
│   ├── profiles/                 # 실제 프로필 이미지 (100개)
│   │   ├── user_001_1.jpg
│   │   ├── user_001_2.jpg
│   │   └── ...
│   └── augmented/                # 증강 이미지 (3,200개)
│       ├── gen_0001.jpg
│       ├── gen_0002.jpg
│       └── ...
│
└── processed/                    # 전처리된 데이터
    ├── train/
    │   └── images/               # 전처리된 학습 이미지
    ├── val/
    │   └── images/               # 전처리된 검증 이미지
    ├── train_metadata.csv        # 학습 메타데이터
    └── val_metadata.csv          # 검증 메타데이터
```

## 메타데이터 형식

### metadata.csv (간소화 버전)

프로젝트 초기 단계에서는 간단한 메타데이터만 사용합니다.

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| filename | string | 이미지 파일명 | "user_001_1.jpg" |
| user_id | string | 고유 사용자 ID | "user_001" |
| image_idx | int | 사용자별 이미지 인덱스 | 1 |
| filepath | string | 전체 파일 경로 | "data/processed/train/user_001_1.jpg" |
| is_synthetic | bool | 증강 이미지 여부 | False, True |

**예시:**

```csv
filename,user_id,image_idx,filepath,is_synthetic
user_001_1.jpg,user_001,1,data/processed/train/user_001_1.jpg,False
user_001_2.jpg,user_001,2,data/processed/train/user_001_2.jpg,False
gen_0001.jpg,gen_0001,1,data/processed/train/gen_0001.jpg,True
gen_0002.jpg,gen_0002,1,data/processed/train/gen_0002.jpg,True
```

### 확장 메타데이터 (향후 계획)

프로덕션 배포 시 추가할 수 있는 필드:

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| gender | string | 성별 ("M", "F", "O") |
| age | int | 나이 |
| image_quality | float | 이미지 품질 점수 (0-1) |
| face_detected | bool | 얼굴 검출 여부 |
| upload_date | datetime | 업로드 시간 |

## 학습 데이터 구성

### Triplet 데이터셋

현재 프로젝트에서는 **동적 Triplet 생성** 방식을 사용합니다.

#### 동적 생성 방법

학습 시 DataLoader에서 실시간으로 (Anchor, Positive, Negative) 조합 생성:

```python
# TripletDataset 예시
class TripletDataset:
    def __getitem__(self, idx):
        # 1. Anchor 선택
        anchor_user = random.choice(user_ids)
        anchor_img = random.choice(user_images[anchor_user])
        
        # 2. Positive 선택 (같은 사용자의 다른 이미지)
        positive_img = random.choice([img for img in user_images[anchor_user] 
                                      if img != anchor_img])
        
        # 3. Negative 선택 (다른 사용자)
        negative_user = random.choice([u for u in user_ids if u != anchor_user])
        negative_img = random.choice(user_images[negative_user])
        
        return anchor_img, positive_img, negative_img
```

**장점:**
- 사전 구성 불필요
- 메모리 효율적
- 매 epoch마다 다른 조합으로 학습

### 향후 확장: 행동 데이터 기반 Triplet

실제 사용자 피드백 데이터가 수집되면 활용 가능:

- **Positive**: 실제로 매칭된 사용자
- **Negative**: 패스한 사용자
- **Hard Negative**: 비슷해 보이지만 거절당한 사용자

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
현재 프로젝트 분할 (3,300개 기준):
├─ Train: 약 2,800개 (85%)
│  ├─ 실제: 70개
│  └─ 증강: 2,730개
├─ Validation: 약 300개 (9%)
│  ├─ 실제: 20개
│  └─ 증강: 280개
└─ Test: 약 200개 (6%)
   ├─ 실제: 10개
   └─ 증강: 190개

분할 기준:
- 사용자 ID 기준 분할 (동일 사용자가 여러 split에 포함 방지)
- 증강 데이터는 실제 데이터와 비율 유지
```

## 데이터 품질 관리

### 자동 필터링 기준 (선택사항)

현재 프로젝트에서는 기본적인 품질 검증만 수행합니다:

```python
def filter_low_quality_images(metadata_df):
    """
    기본 품질 기준 필터링
    """
    filtered = metadata_df[
        (metadata_df['image_width'] >= 224) &    # 최소 해상도
        (metadata_df['image_height'] >= 224) &
        (metadata_df['file_size'] > 10000)       # 최소 파일 크기 (10KB)
    ]
    return filtered
```

### 데이터 검증 체크리스트

```
□ 중복 이미지 제거
□ 손상된 파일 제거 (PIL.Image.verify)
□ 저해상도 이미지 제거 (< 224×224)
□ 메타데이터 일관성 검증
□ 파일명-메타데이터 매칭 확인
```

### 향후 확장 (프로덕션 단계)

- 얼굴 검출 신뢰도 기반 필터링
- 이미지 품질 점수 계산
- 부적절한 콘텐츠 필터링 (NSFW classifier)

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
