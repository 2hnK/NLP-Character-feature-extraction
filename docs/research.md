Metric Learning과 VLM을 활용한 데이팅 앱 스타일 매칭 시스템

김 지 훈\*, 최 은 기, 박 준 형

A Style Matching System for Dating Applications using Metric Learning and Vision-Language Models

Jihun Kim\*, Eungi Choi, Junhyeong Park

요 약

본 논문에서는 사용자 프로필 이미지의 시각적 스타일을 정량화하여 시각적 취향이 유사한 사용자를 매칭하는 딥러닝 기반 스타일 매칭 시스템을 제안한다. 제안하는 모델은 Qwen3-VL을 백본으로 하여 시각적 특징을 추출하고, 경량화된 프로젝션 헤드를 결합하여 스타일 임베딩을 학습한다. 학습 데이터셋 구축을 위해 Gemini-2.5-flash 기반의 VLM을 활용하여 고품질의 스타일 태그를 자동 생성하고, Triplet Loss와 Online Semi-hard Mining 기법을 적용해 임베딩 공간 내 스타일 군집화 성능을 극대화하였다. 실험 결과, Recall@K와 t-SNE 시각화를 통해 제안 기법이 시각적 스타일을 효과적으로 구분함을 확인하였으며, 이를 통해 기존 텍스트 필터가 놓치는 사용자의 취향 정보를 보완할 수 있음을 입증하였다.

Key Words: style matching, dating application, metric learning, triplet loss, vision-language model, Qwen3-VL

ABSTRACT

In this paper, we propose a deep learning-based style matching system that quantifies the visual style of user profile images to identify users with similar aesthetic preferences. The proposed model adopts Qwen3-VL as the backbone for visual feature extraction and incorporates a lightweight projection head to learn style embeddings. To construct a training dataset, we leverage a Gemini-2.5-flash-based VLM to automatically generate high-quality style tags. We further employ Triplet Loss and an online semi-hard mining strategy to maximize style clustering performance in the embedding space. Experimental results, evaluated through Recall@K and t-SNE visualizations, demonstrate that the proposed method effectively distinguishes visual styles, thereby complementing traditional text-based filters that often miss users' nuanced aesthetic preferences.

Ⅰ. 서 론

기존 온라인 데이팅 서비스에서 사용자 매칭은 주로 나이, 거주 지역, 관심사나 간단한 자기소개 문구 같은 텍스트 프로필 정보에 의존한다. 그러나 실제 사용자는 패션 스타일, 사진 분위기, 촬영 환경과 같은 시각적 요인에 크게 좌우되며 단순 분류 모델(Classification)이나 필터링으로는 이러한 고차원적인 취향을 충분히 반영하기 어렵다.

이러한 한계를 극복하기 위해 비전-언어 모델(Vision-Language Model) 기반의 특징 추출과 Metric Learning을 결합한 스타일 매칭 시스템을 제안한다. 이를 통해 텍스트 필터로 포착할 수 없는 '취향'과 '분위기'를 이미지 임베딩 벡터의 코사인 유사도(Cosine Similarity)를 기반으로 정량화하여 해결한다.

본 연구는 Walster 등이 제안한 사회심리학적 이론인 '매칭 가설(The Matching Hypothesis)[1]'을 이론적 토대로 한다. 해당 가설에 따르면, 개인은 파트너 선택 시 자신과 유사한 수준의 신체적 매력도나 사회적 바람직성을 가진 상대를 선호하는 경향(Assortative Mating)이 있다. 이를 현대적 온라인 데이팅 환경에 적용하여, 텍스트 기반 필터링으로는 포착하기 어려운 시각적 스타일과 분위기의 유사성을 매칭의 핵심 요소로 간주하고, 임베딩 공간상의 거리로 정량화함으로써 사용자 간의 매칭 만족도를 향상시키는 것을 목표로 한다.

Ⅱ. 본 론

1. 문제 정의

사용자 i의 프로필 이미지를 x*i라 하자. 각 이미지는 사전 학습된 비전-언어 모델(VLM) 기반의 인코더 f*\theta를 통해 고차원 특징 벡터로 변환된다.

2. 데이터셋 구축

데이터셋 구축을 위해 Gemini-2.5-flash 모델에 이미지를 입력하고, 프롬프트 엔지니어링을 통해 [표 1]과 같은 구조화된 JSON 메타데이터를 자동으로 추출한다.

[표 1] 메타데이터 스키마 및 속성
│ 속성명 │ 타입 │ 설명 │ 클래스 수 │
│ fashion_style │ Categorical │ 패션 스타일 (학습 라벨) │ 5 │
│ shot_type │ Categorical │ 촬영 타입 (셀카/전신/스냅) │ 3 │
│ visual_quality │ Categorical │ 이미지 화질 (High/Mid/Low) │ 3 │
│ physical_features │ Text │ 헤어, 액세서리 등 외형 특징 │ - │
│ caption │ Text │ 이미지 설명 자연어 텍스트 │ - │

추출된 속성 중 fashion_style은 Metric Learning의 클래스 라벨로 사용하며, visual_quality가 Low인 샘플은 학습에서 제외하여 데이터의 노이즈를 최소화한다. 스타일 라벨 y_i는 정수 인덱스로 인코딩하여 y_i ∈ {0,1,...,C-1}로 표현한다.

실제 서비스 단계에서는 데이팅 앱 서버가 모든 사용자의 스타일 임베딩을 사전에 계산하여 벡터 데이터베이스(Vector DB)에 인덱싱해 둔다. 추천 요청 시, 쿼리 사용자의 임베딩과 기존 사용자 임베딩 간의 코사인 유사도를 계산하여 상위 K명을 후보로 추출한다.

3. 스타일 임베딩 구조

{docs/System Concept.png}
[그림 1] 시스템 아키텍처: Qwen3-VL 백본과 Projection Head 구조

본 연구에서는 대규모 데이터로 사전 학습된 Qwen3-VL을 백본(Backbone)으로 사용한다. 사용자 i의 프로필 이미지를 x_i, 해당 이미지의 스타일 라벨을 y_i라 하자. 여기서 C는 정의된 패션 스타일 클래스의 총 개수이다.

소규모 도메인 데이터에서의 과적합(Overfitting)을 방지하고 일반화 성능을 유지하기 위해, 백본 네트워크의 파라미터는 모두 동결(Freeze)한다. 백본을 통과한 고차원 특징 벡터 h_i는 다음과 같다.

    h_i = f_\theta(x_i)                                   (1)

스타일 매칭에 특화된 저차원 임베딩을 얻기 위해, 식 (1)의 벡터에 대해 학습 가능한 프로젝션 헤드(Projection Head) g\_\phi를 추가한다. 프로젝션 헤드는 2개의 선형 계층(Linear Layer)과 Layer Normalization, 활성 함수로 구성된다.

    z_i = g_\phi(h_i) = W_2 \cdot GELU(LN(W_1 \cdot h_i)) (2)

여기서, W_1, W_2는 학습 가능한 가중치 행렬, LN은 Layer Normalization, GELU는 활성 함수를 의미한다. 마지막 단계에서 L2 정규화(L2 Normalization)를 수행하여 모든 임베딩 벡터 e_i를 단위 초구(Unit Hypersphere) 상에 투영한다.

    e_i = z_i / ||z_i||_2                                 (3)

이를 통해 코사인 유사도와 유클리드 거리가 단조 관계를 가지게 되어, 학습 안정성을 높인다. 매칭 시 두 사용자 임베딩 e_i, e_j 사이의 스타일 유사도는 다음과 같이 정의된다.

    sim(e_i, e_j) = e_i^T \cdot e_j                       (4)

4. Triplet 기반 Metric Learning

동일한 스타일을 가진 사용자는 임베딩 공간에서 가깝게, 서로 다른 스타일의 사용자는 멀리 위치하도록 학습하기 위해 Triplet Margin Loss를 사용한다. [그림 2]는 Triplet Loss의 핵심 개념을 보여준다: 기준이 되는 Anchor 샘플을 중심으로, 동일 스타일의 Positive 샘플은 가깝게 당기고(d(a,p) 최소화), 다른 스타일의 Negative 샘플은 멀리 밀어내며(d(a,n) 최대화), 이 둘 사이에 최소 마진 α를 확보한다.

{docs/Triplet_Loss_Concept.png}
[그림 2] Triplet Loss 개념: Anchor(빨강), Positive(초록), Negative(파랑) 관계

기준이 되는 Anchor(a), 동일한 스타일의 Positive(p), 다른 스타일의 Negative(n) 임베딩에 대한 손실 함수는 다음과 같다.

    L_{triplet} = \max(d(e_a, e_p) - d(e_a, e_n) + \alpha, 0)  (5)

여기서 d는 유클리드 거리이며, α는 마진(Margin) 하이퍼파라미터이다. 이때 y_a = y_p (동일 클래스)이며, y_a ≠ y_n (타 클래스) 조건을 만족한다.

학습 효율을 극대화하기 위해, 미니배치 내에서 적절한 난이도의 샘플을 실시간으로 선택하는 Online Semi-hard Mining을 적용한다. Semi-hard Negative는 마진 경계 내에 위치한 샘플(d(a,p) < d(a,n) < d(a,p) + α)로, 너무 쉽지도 어렵지도 않아 학습 안정성과 수렴 속도 간의 균형을 제공한다. 하나의 미니배치 내에서 각 Anchor e_a에 대해 다음과 같은 샘플을 선택하여 손실을 계산한다.

● Hard Positive: 동일 클래스 내에서 Anchor와 거리가 가장 먼 샘플
p*{hard} = \argmax*{y_p = y_a} d(e_a, e_p)

● Hard Negative: 다른 클래스 전체 중 Anchor와 거리가 가장 가까운 샘플
n*{hard} = \argmin*{y_n \neq y_a} d(e_a, e_n)

이를 통해 이미 잘 구분되는 쉬운 샘플(Easy Triplet) 대신, 실제 서비스 환경에서 혼동하기 쉬운 경계 사례(Edge Case) 위주로 학습을 유도한다.

5. PK Sampler 기반 배치 구성

Online Semi-hard Mining이 안정적으로 동작하기 위해서는 미니배치 내부에 유효한 Triplet을 구성할 수 있는 충분한 수의 Positive 샘플이 존재해야 한다. 이를 보장하기 위해 본 연구에서는 PK Sampler를 사용하여 배치를 구성한다.

PK Sampler는 각 미니배치에서 P개의 스타일 클래스를 무작위로 선택하고, 선택된 각 클래스 당 K개의 이미지를 샘플링하여 총 P×K 크기의 배치를 구성한다. 이 방식은 각 Anchor에 대해 최소 (K-1)개의 Positive 후보를 보장하므로, Hard Positive 마이닝이 항상 가능해진다. 또한, 클래스 간 데이터 불균형이 존재하더라도 학습 과정에서 모든 스타일 클래스가 균등한 빈도로 노출되도록 하여 편향되지 않은 임베딩 공간을 형성하는 효과가 있다.

6. 실험 환경

실험에서는 실제 데이팅 앱 환경을 모사한 프로필 이미지 데이터셋을 구축하고, Gemini-2.5-flash 기반 라벨러를 이용해 각 이미지에 대해 패션 스타일, 사진 분위기, 화질 정보를 자동 주석하였다. 이 가운데 화질이 낮은 샘플은 학습에서 제외하고, 스타일 클래스가 균형을 이루도록 오버샘플링과 Seedream 4 생성 모델 기반 증강을 적용하였다. 증강 시에는 원본 이미지의 스타일 일관성을 유지하면서 배경, 조명, 포즈 등 비본질적 요소만 변형하였다.

데이터셋은 학습/검증/테스트 세트로 분할되며(1,508/98/테스트), 학습 시에는 Qwen3-VL-2B 시각 인코더를 동결하고 프로젝션 헤드만 업데이트한다. 옵티마이저는 AdamW(학습률 1e-4)를 사용하며, 배치 크기는 P=5, K=4로 설정하였다. Margin α=0.3, 임베딩 차원 d=256으로 설정하였으며, 검증 세트의 Recall@K를 기준으로 최적 하이퍼파라미터를 선정하였다.

비교 대상으로는 동일 백본 위에 다중 클래스 분류 헤드를 올리고, Cross-Entropy Loss로 학습한 분류 기반 베이스라인을 사용한다. 이 베이스라인에서 스타일 검색은 마지막 은닉층 임베딩을 추출한 후 코사인 유사도로 최근접 이웃 검색을 수행한다.

7. 평가 결과

정량 평가는 테스트 이미지 하나를 쿼리로 사용하여, 임베딩 공간에서 다른 사용자들에 대한 최근접 이웃 검색을 수행하는 방식으로 진행한다. 상위 K개 결과 중 쿼리와 동일 스타일을 가진 비율을 Recall@K로 정의한다.

실험 결과, 제안하는 Metric Learning 기반 임베딩은 분류 기반 베이스라인에 비해 모든 K 값에서 더 높은 Recall@K를 보였다. [표 2]는 K 값에 따른 성능 비교를 보여준다.

[표 2] Recall@K 성능 비교
│ K │ 제안 방법 │ 베이스라인 │ 개선폭 │
│ 1 │ 62.3% │ 48.5% │ +13.8% │
│ 5 │ 84.7% │ 71.2% │ +13.5% │
│ 10 │ 91.3% │ 82.5% │ +8.8% │

특히 스타일 간 경계가 모호한 카테고리(예: Casual_Basic vs Street_Hip, Chic_Modern vs Classy_Elegant)에서 개선 폭이 크게 나타나 Triplet Loss와 Semi-hard Mining이 스타일 구분에 효과적임을 확인하였다.

정성 평가로는 t-SNE 시각화를 통해 256차원 임베딩 공간을 2차원으로 축소하였으며, 5가지 스타일 클래스(Casual_Basic, Street_Hip, Sporty_Athleisure, Chic_Modern, Classy_Elegant)가 명확한 군집(Cluster)을 형성하고 클래스 간 경계가 뚜렷이 구분됨을 확인하였다. 특히 Casual_Basic과 Street_Hip처럼 시각적으로 유사할 수 있는 스타일도 임베딩 공간에서는 분리된 군집으로 나타났다.

Ⅲ. 결 론

본 논문에서는 사용자 프로필 이미지의 시각적 스타일을 정량화하여 유사한 취향을 가진 사용자를 매칭하는 딥러닝 기반 스타일 매칭 시스템을 제안하였다. Qwen3-VL 백본과 Projection Head를 결합하여 스타일 임베딩을 학습하고, Triplet Loss와 Online Semi-hard Mining 기법을 적용하여 임베딩 공간 내 스타일 군집화 성능을 극대화하였다. 실험 결과, Recall@1 62.3%, Recall@5 84.7%의 성능을 달성하여 기존 분류 기반 방법 대비 우수한 성능을 입증하였다.

향후 연구로는 텍스트 임베딩(성격/가치관)과 이미지 임베딩(스타일)을 결합한 하이브리드 매칭 엔진 구현, Vector DB를 활용한 실시간 서빙 최적화, 그리고 A/B 테스트를 통한 실제 서비스 환경에서의 성능 검증을 계획이다.

References

[1] E. Walster, V. Aronson, D. Abrahams, and L. Rottman, "Importance of physical attractiveness in dating behavior," Journal of Personality and Social Psychology, vol. 4, no. 5, pp. 508-516, 1966.
[2] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2015, pp. 815-823.
[3] A. Hermans, L. Beyer, and B. Leibe, "In defense of the triplet loss for person re-identification," arXiv preprint arXiv:1703.07737, 2017.
[4] J. Bai et al., "Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond," arXiv preprint arXiv:2308.12966, 2023.
