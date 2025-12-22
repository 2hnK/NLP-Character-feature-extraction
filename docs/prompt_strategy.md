# 프롬프트 전략 가이드 (Prompt Strategy Guide)

> 📅 작성일: 2025-12-22  
> 📁 프로젝트: NLP-Character-feature-extraction

---

## 1. 개요

본 문서는 커플 매칭 호환성 예측 모델에서 사용하는 **프롬프트 전략**을 설명합니다.
Qwen3-VL 모델에 입력되는 시스템 프롬프트와 유저 프롬프트의 설계 원칙과 구현 방법을 다룹니다.

---

## 2. 프롬프트 구조

### 2.1 시스템 프롬프트 (System Prompt) - 성별별

**여성용 (Female)**:

```python
SYSTEM_PROMPTS = {}
SYSTEM_PROMPTS["female"] = """You are a dating compatibility analyst.
Analyze this woman's visual features that attract male partners.
Focus on feminine charm, style, and overall appeal.
Ignore background and irrelevant objects."""
```

**남성용 (Male)**:

```python
SYSTEM_PROMPTS["male"] = """You are a dating compatibility analyst.
Analyze this man's visual features that attract female partners.
Focus on masculine appeal, style, and overall charm.
Ignore background and irrelevant objects."""
```

**기본값 (Default)**:

```python
SYSTEM_PROMPTS["default"] = """You are a dating compatibility analyst.
Focus on visual features that predict romantic matching success.
Ignore background and irrelevant objects."""
```

### 2.2 유저 프롬프트 템플릿 (User Prompt) - 성별별

**여성용 (Female)**:

```python
USER_PROMPT_TEMPLATES = {}
USER_PROMPT_TEMPLATES["female"] = """Dating profile (Woman):
- Appearance: {appearance_type}
- Style: {style_vibe}
- Personality: {personality_impression}
- Grooming: {grooming_level}
- Features: {physical_features_str}

Analyze her romantic appeal."""
```

**남성용 (Male)**:

```python
USER_PROMPT_TEMPLATES["male"] = """Dating profile (Man):
- Appearance: {appearance_type}
- Style: {style_vibe}
- Personality: {personality_impression}
- Grooming: {grooming_level}
- Features: {physical_features_str}

Analyze his romantic appeal."""
```

**메타데이터 변수**:

| 변수                     | 설명                     | 예시                               |
| ------------------------ | ------------------------ | ---------------------------------- |
| `appearance_type`        | 외모 유형 (6 클래스)     | Clean_Innocent, Cool_Charismatic   |
| `style_vibe`             | 스타일 분위기 (6 클래스) | Casual_Comfortable, Trendy_Fashion |
| `personality_impression` | 성격 인상 (5 클래스)     | Outgoing_Social, Calm_Introverted  |
| `grooming_level`         | 관리 수준 (3 클래스)     | High, Medium, Low                  |
| `physical_features_str`  | 외형 특징 (문자열)       | "Short dark hair, Round glasses"   |

### 2.3 기본 프롬프트 (메타데이터 없을 때)

```python
DEFAULT_USER_PROMPT = "Dating profile photo. Analyze for romantic compatibility."
```

---

## 3. 설계 원칙

### 3.1 역할 분담

| 컴포넌트           | 담당 정보        | 역할                                              |
| ------------------ | ---------------- | ------------------------------------------------- |
| **메타데이터**     | 명시적 범주 정보 | appearance_type, style_vibe 등 이미 라벨링된 정보 |
| **VLM (Qwen3-VL)** | 암묵적 시각 패턴 | 색 조화, 얼굴 비율, 분위기의 "느낌", 잠재적 매력  |
| **프롬프트**       | 컨텍스트 제공    | 메타데이터를 VLM에 전달하여 보완적 특징 추출 유도 |

### 3.2 프롬프트 최적화 결정

**❌ 피해야 할 것**:

- VLM에게 이미 라벨링된 정보를 "추출"하라고 지시 (중복 작업)
- 과도하게 긴 프롬프트 (Mean Pooling에서 정보 희석)
- 의미적으로 부적절한 속성 그룹핑

**✅ 적용한 것**:

- 메타데이터를 프롬프트에 포함하여 VLM이 **컨텍스트**로 활용
- 병렬 구조로 각 속성의 독립성 유지
- `physical_features` 리스트를 자연어 문자열로 변환

### 3.3 토큰 효율성

```
입력 구성:
├── System Prompt: ~30 토큰
├── User Prompt (메타데이터 포함): ~50 토큰
└── 이미지 토큰: ~256-1024 토큰 (주요)

→ 이미지 토큰 비중을 최대화하면서 필요한 컨텍스트만 제공
```

---

## 4. 구현

### 4.1 코드 위치

**파일**: `src/models/qwen_backbone.py`

### 4.2 프롬프트 생성 함수

```python
def _build_user_prompt(self, metadata: Optional[Dict] = None) -> str:
    """메타데이터를 기반으로 유저 프롬프트 생성"""
    if metadata is None:
        return DEFAULT_USER_PROMPT

    # physical_features 리스트를 문자열로 변환
    physical_features = metadata.get('physical_features', [])
    if isinstance(physical_features, list):
        physical_features_str = ', '.join(physical_features)
    else:
        physical_features_str = str(physical_features)

    return USER_PROMPT_TEMPLATE.format(
        appearance_type=metadata.get('appearance_type', 'Unknown'),
        style_vibe=metadata.get('style_vibe', 'Unknown'),
        personality_impression=metadata.get('personality_impression', 'Unknown'),
        grooming_level=metadata.get('grooming_level', 'Unknown'),
        physical_features_str=physical_features_str
    )
```

### 4.3 대화 구성

```python
conversation = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_prompt}
        ]
    }
]
```

---

## 5. 사용 예시

### 5.1 메타데이터 있는 경우

```python
metadata = {
    'appearance_type': 'Clean_Innocent',
    'style_vibe': 'Casual_Comfortable',
    'personality_impression': 'Outgoing_Social',
    'grooming_level': 'High',
    'physical_features': ['Short dark hair', 'Round glasses', 'Clear skin']
}

# forward 호출
embeddings = backbone.forward([image], metadata=[metadata])
```

**생성되는 프롬프트**:

```
Dating profile:
- Appearance: Clean_Innocent
- Style: Casual_Comfortable
- Personality: Outgoing_Social
- Grooming: High
- Features: Short dark hair, Round glasses, Clear skin

Analyze for romantic compatibility.
```

### 5.2 메타데이터 없는 경우

```python
# 메타데이터 없이 forward 호출
embeddings = backbone.forward([image])
```

**생성되는 프롬프트**:

```
Dating profile photo. Analyze for romantic compatibility.
```

---

## 6. 메타데이터 스키마

### 6.1 Categorical Features (6개)

| 변수                     | 클래스 수 | 옵션                                                                                              |
| ------------------------ | --------- | ------------------------------------------------------------------------------------------------- |
| `appearance_type`        | 6         | Clean_Innocent, Chic_Urban, Cute_Adorable, Warm_Gentle, Cool_Charismatic, Healthy_Active          |
| `style_vibe`             | 6         | Minimal_Clean, Casual_Comfortable, Trendy_Fashion, Street_Urban, Formal_Polished, Sporty_Athletic |
| `personality_impression` | 5         | Outgoing_Social, Calm_Introverted, Playful_Humorous, Confident_Assertive, Soft_Sensitive          |
| `grooming_level`         | 3         | High, Medium, Low                                                                                 |
| `photo_style`            | 5         | Selfie_Direct, Mirror_Selfie, Portrait_Taken, PhotoBooth_Studio, Candid_Activity                  |
| `image_quality`          | 3         | High, Medium, Low                                                                                 |

### 6.2 Text Features (2개)

| 변수                | 타입      | 설명                      |
| ------------------- | --------- | ------------------------- |
| `physical_features` | List[str] | 눈에 띄는 외형 특징 3-5개 |
| `caption`           | str       | 프로필 종합 설명 1-2문장  |

---

## 7. 향후 개선 방향

### 7.1 A/B 테스트 계획

| 실험     | 프롬프트 변형        | 측정 지표     |
| -------- | -------------------- | ------------- |
| Baseline | 현재 프롬프트        | Recall@1, MRR |
| Exp 1    | 시스템 프롬프트 제거 |               |
| Exp 2    | 메타데이터 순서 변경 |               |
| Exp 3    | 자연어 문장 형태     |               |

### 7.2 고려 중인 개선

1. **Attention 가중치 분석**: 프롬프트의 어떤 부분이 임베딩에 영향을 미치는지 분석
2. **프롬프트 튜닝**: Soft prompt 또는 prefix tuning 적용 검토
3. **다국어 프롬프트**: 한국어 프롬프트 효과 실험

---

## 8. 참고

- [Qwen3-VL 모델 문서](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [InfoNCE Loss 논문](https://arxiv.org/abs/1807.03748)
- 프로젝트 아키텍처: `docs/model_architecture.md`
