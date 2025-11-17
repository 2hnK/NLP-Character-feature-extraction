# Dating Profile Matcher - Project Context

> **Claude가 이 프로젝트를 이해하기 위한 핵심 문서**  
> 작성일: 2025-11-17

## 🎯 프로젝트 목표

### 비즈니스 목표
데이팅 앱에서 **외모 기반 매력도 유사도**를 활용하여 매칭률을 향상시키는 것

**핵심 가설:**
> "외적으로 뛰어난 사람은 상대의 외모가 뛰어나길 바라는 경향이 많다"
> 
> → 서로 선호할 만한 외모 스타일을 가진 사용자끼리 매칭하면  
> → 좋아요 비율이 증가할 것

### 기술적 목표
1. **프로필 사진 → 임베딩 벡터**: 각 사용자의 사진을 숫자 벡터로 변환
2. **유사도 기반 매칭**: 벡터 간 거리를 계산하여 유사한 사용자 추천
3. **실제 서비스 검증**: 좋아요 비율 개선 여부 측정

### 프로젝트 범위
- ✅ **포함**: 모델 학습, 임베딩 추출, 유사도 계산
- ❌ **제외**: 백엔드 통합, 실시간 서비스 배포, 모니터링 시스템
- 📌 **최종 결과물**: 학습된 모델 + 추론 API (프로토타입 수준)

---

## 📊 데이터 현황

### 데이터 구성
```
총 3,300개 이미지
├── 실제 사용자 이미지: 100개 (검증용)
│   ├── 남녀 비율: 미정 (증강 데이터는 5:5 목표)
│   └── 사용자당 이미지: 1~3개
└── 증강 이미지: 3,200개 (학습용)
    └── 생성 방법: 실제 이미지 기반 생성형 AI
```

### 데이터 특성
- **레이블링 없음**: 명시적인 "매력도" 점수나 카테고리 없음
- **Self-supervised 학습**: 같은 사용자 사진 = 유사, 다른 사용자 = 비유사
- **메타데이터 미정**: 나이, 성별, 좋아요 기록 등 활용 여부 추후 결정

### 데이터 구조 (예상)
```
data/
├── raw/
│   └── profiles/
│       ├── user_001_1.jpg
│       ├── user_001_2.jpg
│       ├── user_002_1.jpg
│       └── ...
├── augmented/
│   └── generated/
│       ├── gen_001.jpg
│       ├── gen_002.jpg
│       └── ...
└── processed/
    └── metadata.csv (선택사항)
        # user_id, image_path, gender, age, ...
```

---

## 🧠 모델 아키텍처

### 선택 모델: Qwen3-VL-2B-Instruct-FP8

**선택 이유:**
- 인물 사진 특징 추출 능력이 뛰어남
- Vision-Language Model로 이미지 이해 능력 우수
- SageMaker에서 사용 가능

**사용 방식:**
1. **Feature Extractor로 활용**
   - 이미지 입력 → 임베딩 벡터 출력 (예: 512차원 또는 모델 기본 차원)
   - Vision 인코더의 마지막 레이어 출력 사용

2. **Fine-tuning 전략**
   - **학습 데이터**: 증강된 3,200개 이미지
   - **검증 데이터**: 실제 사용자 100개 이미지
   - **학습 방법**: Metric Learning (Triplet Loss 또는 Contrastive Loss)

3. **출력 형태**
   ```python
   # 입력: 프로필 사진 (예: 224x224 RGB)
   # 출력: 임베딩 벡터 (예: [512] 차원의 float 배열)
   
   embedding = model.extract_features(image)
   # embedding.shape = (512,)
   # embedding = [0.23, -0.45, 0.67, ..., 0.12]
   ```

### 유사도 계산
```python
# 두 사용자의 임베딩 벡터
user_a_embedding = [0.1, 0.2, 0.3, ...]  # 512차원
user_b_embedding = [0.15, 0.18, 0.35, ...]  # 512차원

# 코사인 유사도 (Cosine Similarity)
similarity = cosine_similarity(user_a_embedding, user_b_embedding)
# similarity = 0.85 (0~1 사이 값, 1에 가까울수록 유사)
```

---

## 🔧 기술 스택

### 클라우드 환경
- **AWS SageMaker AI Studio**: 모델 학습 및 실험
  - GPU 인스턴스 사용 (예: ml.g5.xlarge)
  - Jupyter Notebook 환경
  - 실험 관리: SageMaker Experiments

### 프레임워크
- **PyTorch 2.0+**: 딥러닝 프레임워크
- **Transformers (Hugging Face)**: Qwen 모델 로드
- **FastAPI**: 추론 API 서버 (선택사항)

### 라이브러리
```python
# 핵심 라이브러리
torch>=2.0.0
transformers>=4.35.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# SageMaker
sagemaker>=2.190.0
boto3>=1.28.0

# 실험 관리 (선택)
wandb>=0.15.0

# API 서버 (선택)
fastapi>=0.104.0
uvicorn>=0.24.0
```

---

## 📈 평가 방식

### 주요 평가 지표
**비즈니스 KPI (실제 서비스 적용 후):**
- **좋아요 비율**: 매칭된 상대를 보고 좋아요를 누르는 비율
- **비교 기준**: 모델 도입 전 vs 후

**기술적 지표 (개발/연구 단계):**
1. **임베딩 품질**
   - 같은 사용자 이미지 간 거리: 작을수록 좋음
   - 다른 사용자 이미지 간 거리: 클수록 좋음

2. **검색 정확도** (선택사항)
   - Top-K Retrieval Accuracy
   - 쿼리 이미지와 유사한 이미지를 얼마나 잘 찾는가

3. **시각적 평가**
   - 유사하다고 판단된 사진들이 실제로 유사한가
   - t-SNE/UMAP 시각화로 클러스터링 확인

### 평가 방법
```python
# 예시: 같은 사용자의 다른 사진들이 가까운지 확인
user_images = ["user_001_1.jpg", "user_001_2.jpg", "user_001_3.jpg"]
embeddings = [model.extract(img) for img in user_images]

# 평균 거리 계산
avg_distance = calculate_pairwise_distance(embeddings)
# 목표: avg_distance < 0.3
```

---

## 🎓 학습자 수준 및 제약사항

### 기술 경험
- **Python**: 코드 읽기 가능, 간단한 스크립트 작성 가능
- **PyTorch**: 초급 수준 (튜토리얼 따라하기)
- **SageMaker**: 사용 경험 없음 (처음)
- **딥러닝 개념**: 기본적인 이해는 있으나 실전 경험 부족

### 불확실한 개념들
**우선순위 높은 학습 필요 항목:**
1. **임베딩 벡터 (Embedding Vector)**: 이미지를 숫자로 변환하는 개념
2. **Metric Learning**: 거리 기반 학습 방법
3. **Triplet Loss / Contrastive Loss**: 학습 손실 함수
4. **Fine-tuning**: 사전학습 모델 추가 학습
5. **SageMaker 사용법**: 환경 설정, 노트북 실행, 모델 저장

**현재 모르는 개념:**
- Online Triplet Mining
- Faiss 인덱스 타입
- Early Stopping
- 하이퍼파라미터 튜닝 전략

### 제약사항
- ⏰ **시간**: 학업 과제/연구 프로젝트 (마감 일정 있을 가능성)
- 💰 **비용**: SageMaker 사용료 고려 필요
- 🔧 **인프라**: 로컬 GPU 없음, SageMaker 의존
- 👥 **리소스**: 1인 프로젝트 (팀 협업 없음)

---

## 🚀 프로젝트 진행 단계

### Phase 1: 환경 설정 및 데이터 준비 (1주)
- [ ] SageMaker Studio 환경 구축
- [ ] Qwen3-VL-2B 모델 로드 테스트
- [ ] 데이터 전처리 파이프라인 구축
- [ ] 기본 데이터 탐색 (EDA)

### Phase 2: 베이스라인 모델 구축 (1-2주)
- [ ] Feature Extractor 구현
- [ ] 간단한 유사도 계산 테스트
- [ ] 임베딩 품질 시각화
- [ ] 초기 성능 평가

### Phase 3: Fine-tuning 실험 (2-3주)
- [ ] Metric Learning 손실 함수 구현
- [ ] 증강 데이터로 학습
- [ ] 하이퍼파라미터 실험
- [ ] 성능 개선 확인

### Phase 4: 검증 및 평가 (1주)
- [ ] 실제 사용자 데이터로 평가
- [ ] 유사도 기반 매칭 시뮬레이션
- [ ] 결과 분석 및 보고서 작성

### Phase 5: API 구축 (선택, 1주)
- [ ] FastAPI 엔드포인트 구현
- [ ] 추론 최적화
- [ ] 간단한 테스트 UI

---

## 🤝 Claude 활용 가이드

### Claude에게 도움 요청할 때

**✅ 효과적인 요청 방법:**
```
"SageMaker에서 Qwen3-VL-2B 모델을 로드하는 코드를 
단계별로 설명해줘. PyTorch 초급자 수준에 맞춰서."
```

```
"Triplet Loss가 뭔지 모르겠어. 
우리 프로젝트에서 어떻게 사용되는지 예제와 함께 설명해줘."
```

```
"증강 데이터 3,200개로 Fine-tuning할 때 
적절한 batch size와 learning rate는 얼마야?"
```

**❌ 비효과적인 요청:**
```
"코드 짜줘"  # 너무 모호함
"이게 왜 안돼?"  # 에러 메시지나 상황 설명 없음
"빨리 해결해줘"  # 구체적인 문제 명시 필요
```

### Claude가 알아야 할 핵심 정보
1. **프로젝트 목표**: 외모 유사도 기반 매칭, 좋아요 비율 향상
2. **모델**: Qwen3-VL-2B만 사용
3. **데이터**: 실제 100개 + 증강 3,200개
4. **환경**: SageMaker, PyTorch 초급
5. **제약**: 1인 프로젝트, 처음 해보는 작업

---

## 📚 주요 참고 자료

### 반드시 읽어볼 자료
1. **Qwen-VL 공식 문서**
   - https://github.com/QwenLM/Qwen-VL
   - 모델 로드 및 사용법

2. **Metric Learning 기초**
   - Triplet Loss 개념
   - Contrastive Learning 개념

3. **SageMaker 시작 가이드**
   - Jupyter Notebook 사용법
   - GPU 인스턴스 선택

### 유용한 튜토리얼
- PyTorch Metric Learning 라이브러리
- Face Recognition with Deep Learning (유사한 문제)
- Image Similarity Search 구현 예제