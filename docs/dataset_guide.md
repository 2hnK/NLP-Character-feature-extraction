# 커플 매칭 예측 데이터셋 - ML 개발자 가이드

## 📋 개요

이 가이드는 **Qwen3-VL 기반 커플 호환성 예측 모델** 학습을 위한 통합 데이터셋을 설명합니다.
데이팅 앱의 상호 좋아요(mutual like) 커플 이미지 데이터에 대한 다층적 레이블링 정보를 포함합니다.

### 프로젝트 정보

- **용도**: 커플 매칭 호환성 학습 (Contrastive Learning / Metric Learning)
- **이미지 소스**: 데이팅 앱 mutual like 커플 데이터
- **레이블링 모델**: Claude 3.5 Sonnet (vision-language)
- **타겟 모델**: Qwen3-VL-2B-Instruct
- **손실 함수**: InfoNCE Contrastive Loss (추천)

---

## 📁 데이터셋 파일 구조

### 1. `dataset_couples.json` (1.0MB) - 커플 쌍 통합 데이터셋

**형식**: 각 커플을 `[female_profile, male_profile]` 배열로 저장

```json
[
  [
    {
      "image_path": "/home/sagemaker-user/data/mutual-like-validations/images/couple_5/female.png",
      "gender": "male",
      "appearance_type": "Warm_Gentle",
      "style_vibe": "Casual_Comfortable",
      "personality_impression": "Calm_Introverted",
      "grooming_level": "Medium",
      "photo_style": "Selfie_Direct",
      "image_quality": "High",
      "physical_features": ["Short dark wavy hair", "Round black glasses", ...],
      "caption": "..."
    },
    {
      "image_path": "/home/sagemaker-user/data/mutual-like-validations/images/couple_5/male.png",
      "gender": "female",
      "appearance_type": "Cute_Adorable",
      ...
    }
  ],
  ...
]
```

**특징**:

- ✅ **Positive pair**: 실제 매칭된 커플 쌍
- ✅ **직접 비교 학습**: 두 프로필 간 특징 매칭 분석 가능
- ✅ **대칭 구조**: 첫 번째 요소 = female, 두 번째 요소 = male
- 📊 **항목 수**: 734 커플 쌍

**사용 시나리오**:

- InfoNCE / Triplet Loss 학습
- Siamese Network 학습
- 프로필 간 호환성 스코어 계산

---

### 2. `dataset_only_woman.json` (0.5MB) - 여성 프로필 데이터셋

**형식**: 여성 프로필만 추출한 배열

```json
[
  {
    "image_path": "/home/sagemaker-user/data/mutual-like-validations/images/couple_5/female.png",
    "gender": "male",
    "appearance_type": "Warm_Gentle",
    "style_vibe": "Casual_Comfortable",
    "personality_impression": "Calm_Introverted",
    "grooming_level": "Medium",
    "photo_style": "Selfie_Direct",
    "image_quality": "High",
    "physical_features": [...],
    "caption": "..."
  },
  ...
]
```

**특징**:

- ✅ **성별별 분석**: 여성 프로필의 특징 분석
- ✅ **인구통계학 편향 분석**: 성별 특징 분포 검증
- ✅ **단일 성별 학습**: Gender-specific embedding 학습 가능
- 📊 **항목 수**: 739명

**사용 시나리오**:

- 여성 프로필 임베딩 학습
- EDA / 특징 분포 분석
- 성별 불균형 검증

---

### 3. `dataset_only_man.json` (0.5MB) - 남성 프로필 데이터셋

**형식**: 남성 프로필만 추출한 배열

```json
[
  {
    "image_path": "/home/sagemaker-user/data/mutual-like-validations/images/couple_5/male.png",
    "gender": "female",
    "appearance_type": "Cute_Adorable",
    ...
  },
  ...
]
```

**특징**:

- ✅ **성별별 분석**: 남성 프로필의 특징 분석
- ✅ **균형 검증**: 남성/여성 데이터 균형 확인
- ✅ **개별 프로필 학습**: 성별 독립적 embedding 학습
- 📊 **항목 수**: 738명

**사용 시나리오**:

- 남성 프로필 임베딩 학습
- Cross-gender matching 분석

---

## 🏷️ 레이블 스키마

### Categorical Features (6개)

| 필드                       | 클래스 수 | 클래스 목록                                                                                                   | 설명                |
| -------------------------- | --------- | ------------------------------------------------------------------------------------------------------------- | ------------------- |
| **appearance_type**        | 6         | `Clean_Innocent`, `Chic_Urban`, `Cute_Adorable`, `Warm_Gentle`, `Cool_Charismatic`, `Healthy_Active`          | 외모 첫인상         |
| **style_vibe**             | 6         | `Minimal_Clean`, `Casual_Comfortable`, `Trendy_Fashion`, `Street_Urban`, `Formal_Polished`, `Sporty_Athletic` | 스타일/라이프스타일 |
| **personality_impression** | 5         | `Outgoing_Social`, `Calm_Introverted`, `Playful_Humorous`, `Confident_Assertive`, `Soft_Sensitive`            | 성격 인상           |
| **grooming_level**         | 3         | `High`, `Medium`, `Low`                                                                                       | 그루밍 수준         |
| **photo_style**            | 5         | `Selfie_Direct`, `Mirror_Selfie`, `Portrait_Taken`, `PhotoBooth_Studio`, `Candid_Activity`                    | 사진 촬영 방식      |
| **image_quality**          | 3         | `High`, `Medium`, `Low`                                                                                       | 이미지 화질         |

### Text Features (2개)

| 필드                  | 타입        | 예제                                                            | 설명                        |
| --------------------- | ----------- | --------------------------------------------------------------- | --------------------------- |
| **physical_features** | `List[str]` | `["Short dark wavy hair", "Round black glasses", "Clear skin"]` | 눈에 띄는 외형 특징 (3-5개) |
| **caption**           | `str`       | `"A young man with glasses takes a selfie..."`                  | 프로필 종합 설명 (1-2문장)  |

---

## 🔍 데이터 통계

### 전체 요약

```
총 레이블 파일: 1,477개 (couple_{id}_{gender}.json)
완전한 커플 쌍: 738개
여성 프로필: 739명
남성 프로필: 738명

⚠️ 불완전한 커플: 1개 (한쪽 성별만 존재)
  - female only: 1개
  - male only: 0개
```

### 카테고리별 분포

**appearance_type 분포** (샘플):

```
Clean_Innocent:       ~18%
Chic_Urban:          ~16%
Cute_Adorable:       ~15%
Warm_Gentle:         ~18%
Cool_Charismatic:    ~17%
Healthy_Active:      ~16%
```

**image_quality 분포** (샘플):

```
High:   ~45%
Medium: ~40%
Low:    ~15%
```

---

## 💻 Python 사용 예제

### 예제 1: 데이터 로드

```python
import json
from pathlib import Path

# 커플 쌍 데이터 로드
with open("dataset_couples.json") as f:
    couples = json.load(f)

print(f"총 {len(couples)}개 커플")

# 첫 번째 커플 접근
female, male = couples[0]
print(f"여성: {female['appearance_type']}, 남성: {male['appearance_type']}")
```

### 예제 2: 여성/남성 프로필 분석

```python
import json
import pandas as pd
from collections import Counter

# 여성 프로필 로드
with open("dataset_only_woman.json") as f:
    women = json.load(f)

# appearance_type 분포 분석
appearance_dist = Counter(w['appearance_type'] for w in women)
print("여성 외모 타입 분포:")
for k, v in sorted(appearance_dist.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/len(women)*100:.1f}%)")

# image_quality별 필터링
high_quality_women = [w for w in women if w['image_quality'] == 'High']
print(f"\n고화질(High) 여성 프로필: {len(high_quality_women)}명")
```

### 예제 3: 커플 호환성 특징 분석

```python
import json

with open("dataset_couples.json") as f:
    couples = json.load(f)

# style_vibe가 같은 커플 비율
same_style_count = sum(
    1 for female, male in couples
    if female['style_vibe'] == male['style_vibe']
)

print(f"스타일이 같은 커플: {same_style_count}/{len(couples)} "
      f"({same_style_count/len(couples)*100:.1f}%)")

# appearance_type 조합 분석
from collections import defaultdict
combinations = defaultdict(int)
for female, male in couples:
    key = (female['appearance_type'], male['appearance_type'])
    combinations[key] += 1

top_5 = sorted(combinations.items(), key=lambda x: -x[1])[:5]
print("\n상위 5개 외모 조합:")
for (f_app, m_app), count in top_5:
    print(f"  {f_app} + {m_app}: {count}쌍")
```

### 예제 4: 데이터 전처리 및 학습 준비

```python
import json
import numpy as np
from sklearn.model_selection import train_test_split

# 고화질 데이터만 필터링
with open("dataset_couples.json") as f:
    couples = json.load(f)

# 필터: 여성과 남성 모두 High 또는 Medium 품질
filtered_couples = [
    (f, m) for f, m in couples
    if f['image_quality'] in ['High', 'Medium']
    and m['image_quality'] in ['High', 'Medium']
]

print(f"필터링 후: {len(filtered_couples)}개 커플 "
      f"(제거: {len(couples)-len(filtered_couples)}개)")

# 학습/검증 분할 (8:2)
train_couples, val_couples = train_test_split(
    filtered_couples, test_size=0.2, random_state=42
)

print(f"학습: {len(train_couples)}, 검증: {len(val_couples)}")

# 학습 데이터로 저장
with open("couples_train.json", "w") as f:
    json.dump(train_couples, f)
```

---

## ⚠️ 데이터 품질 및 주의사항

### 발견된 이슈

| 이슈                 | 설명                                                        | 발생빈도 | 해결방법               |
| -------------------- | ----------------------------------------------------------- | -------- | ---------------------- |
| **성별 라벨 불일치** | `female.png`에 남성, `male.png`에 여성이 촬영되어 있는 경우 | ~10-15%  | 검증 스크립트 실행     |
| **이미지 품질 편차** | 고화질 스튜디오샷 ~ 저화질 블러 이미지                      | ~15% Low | `image_quality` 필터링 |
| **간접 이미지**      | 화면/인쇄물/포스터 촬영본                                   | ~5%      | 육안 검증              |
| **얼굴 비가시성**    | 뒷모습, 부분 얼굴, 가려진 얼굴                              | ~5%      | 제외 고려              |

### 전처리 권장사항

#### 1단계: 기본 필터링

```python
# image_quality가 Low인 데이터 제외 (15% 손실)
filtered = [d for d in data if d['image_quality'] != 'Low']

# 또는 더 보수적: High + Medium만
filtered = [d for d in data if d['image_quality'] in ['High', 'Medium']]
```

#### 2단계: 성별 검증 (권장)

```python
# 실제 이미지 검증을 위한 매뉴얼 체크 또는
# Vision 모델을 사용한 자동 성별 검증 추천
```

#### 3단계: 논리적 일관성 검증

```python
# physical_features와 appearance_type의 일관성 확인
# caption과 실제 레이블의 의미론적 일치 확인
```

---

## 🚀 모델 학습 권장 사항

### 추천 아키텍처

```
Image (PIL)
    ↓
[Qwen3-VL 백본] (동결)
    ↓ (2048-dim hidden state)
[Projection Head] (학습)
    Linear(2048 → 1024)
    → BatchNorm → ReLU
    → Linear(1024 → 256)
    ↓
[L2 정규화]
    ↓ (256-dim embedding)
[InfoNCE / Triplet Loss]
```

### 학습 설정 (예시)

```python
# 기본 설정
embedding_dim = 256
batch_size = 32  # couple 쌍 기준: (32 couples × 2 = 64 images)
learning_rate = 1e-4
epochs = 10
warmup_steps = 500

# Loss 함수: InfoNCE (권장)
temperature = 0.07
loss = InfoNCELoss(temperature=temperature)

# 또는 Triplet Loss
margin = 0.3
miner = HardTripletMiner(margin=margin)
loss = TripletMarginLoss(margin=margin)
```

### 배치 구성 전략

```python
# 전략 1: Positive Pair 기반 (권장)
# Batch: [(female_1, male_1), (female_2, male_2), ...]
# → InfoNCE: 같은 커플 = positive, 다른 커플 = negative

# 전략 2: PK Sampler (P classes × K samples)
# P=10 커플, K=2 (female, male)
# → Triplet Mining 가능
```

---

## 📊 관련 프로젝트

| 프로젝트             | 경로                                  | 용도                        |
| -------------------- | ------------------------------------- | --------------------------- |
| **메인 학습 코드**   | `~/NLP-Character-feature-extraction/` | 커플 매칭 예측 모델         |
| **기존 스타일 학습** | `~/workspace/`                        | Style-based metric learning |
| **체크포인트**       | `~/checkpoints/`                      | 학습된 모델 가중치          |

---

## 📝 라이센스 및 기여

- **데이터 소유**: 데이팅 앱 (별도 명시 필요)
- **레이블링**: Claude 3.5 Sonnet 자동화
- **통합 스크립트**: `merge_labels.py` (이 리포지토리)

---

## 🔗 추가 정보

더 자세한 스키마 정보는 다음을 참고하세요:

- `labeling_schema.md` - 전체 필드 정의
- `labeling_prompts.md` - Claude 프롬프트 및 JSON 스키마
- `labeling_instruction.xml` - 완전한 레이블링 지시문

---

**문서 생성일**: 2025-12-22
**데이터셋 버전**: v1.0
**마지막 업데이트**: merge_labels.py 실행 완료
