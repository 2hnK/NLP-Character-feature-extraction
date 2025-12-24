# Metric Learning과 VLM을 활용한 데이팅 앱 커플 매칭 예측 시스템

## A Couple Matching Prediction System for Dating Applications using Metric Learning and Vision-Language Models

**저자**: 김지훈*, 최은기, 박준형  
**소속**: 국립한밭대학교  
**이메일**: 20227128@edu.hanbat.ac.kr, 20221991@edu.hanbat.ac.kr, 20197119@edu.hanbat.ac.kr

---

## 초록 (Abstract)

본 논문에서는 실제 데이팅 앱에서 성사된 커플 데이터를 활용하여, 사용자 프로필 이미지 기반의 매칭 예측 시스템을 제안한다. 제안하는 모델은 Qwen3-VL을 백본으로 하여 시각적 특징을 추출하고, 경량화된 프로젝션 헤드를 결합하여 매칭 호환성 임베딩을 학습한다. InfoNCE Loss를 적용하여 실제 커플 쌍을 가깝게, 비커플 쌍을 멀게 배치하도록 학습한다. 495쌍의 커플 데이터를 활용하여 실험을 수행하였다. 실험 결과, 제안 모델은 사전 학습된 VLM 베이스라인 대비 양방향 평균 Recall@5에서 2.23배, MRR에서 2.38배 향상된 성능을 달성하였다.

---

## I. 서 론

기존 온라인 데이팅 서비스에서 사용자 매칭은 주로 나이, 거주 지역, 관심사나 간단한 자기소개 문구 같은 텍스트 프로필 정보에 의존한다. 그러나 실제 사용자는 프로필 사진에서 드러나는 시각적 요인에 크게 좌우되며, 이러한 암묵적 선호를 정량화하는 것은 어려운 과제이다. 기존 연구[2][3]는 대부분 텍스트 프로필 유사도나 협업 필터링에 의존하였으나, 사용자의 시각적 선호를 반영하지 못하는 한계가 있다.

본 연구는 Walster 등이 제안한 사회심리학적 이론인 '매칭 가설(The Matching Hypothesis)'[1]을 이론적 토대로 한다. 해당 가설에 따르면, 개인은 파트너 선택 시 자신과 유사한 수준의 신체적 매력도나 사회적 바람직성을 가진 상대를 선호하는 경향(Assortative Mating)이 있다. 이를 현대적 온라인 데이팅 환경에 적용하여, 실제 매칭된 커플 데이터를 학습함으로써 매칭 호환성을 예측하는 모델을 제안한다.

본 연구에서는 Qwen3-VL[4]을 백본으로 활용하고, 경량화된 프로젝션 헤드를 결합하여 매칭 호환성 임베딩을 학습한다. CLIP[5]에서 검증된 InfoNCE Loss를 적용하여 실제 커플 쌍을 임베딩 공간에서 가깝게, 비커플 쌍을 멀게 배치하도록 학습한다. 본 연구의 주요 기여점은 다음과 같다: (1) 실제 커플 데이터 495쌍을 활용한 대조 학습 프레임워크 제안, (2) VLM 백본과 경량 프로젝션 헤드를 결합한 효율적 임베딩 구조 설계, (3) 사전 학습된 VLM 베이스라인 대비 Recall@5 2.23배, MRR 2.38배 성능 향상 달성.

---

## II. 본 론

### 2.1 문제 정의 및 데이터셋

사용자 $i$의 프로필 이미지를 $x_i$라 정의한다. 커플 쌍 $(x_f, x_m)$에 대해, 두 사용자의 임베딩 $e_f$, $e_m$이 임베딩 공간에서 가깝도록 학습하며, 비커플 쌍은 멀어지도록 대조 학습(Contrastive Learning)을 수행한다.

실제 데이팅 앱에서 상호 좋아요를 통해 매칭된 커플 495쌍의 프로필 이미지를 수집하였다. 각 커플 폴더에는 female.png와 male.png가 포함되어 있다. 전체 데이터를 표 1과 같이 분할하였다.

**표 1. 데이터셋 구성**

| 항목 | 값 |
|------|-----|
| 총 커플 수 | 495쌍 |
| Train | 346쌍 (70%) |
| Validation | 74쌍 (15%) |
| Test | 75쌍 (15%) |

### 2.2 모델 아키텍처

대규모 데이터로 사전 학습된 Qwen3-VL-2B[4]를 백본(Backbone)으로 사용한다. Qwen3-VL은 이미지와 텍스트를 함께 이해하는 비전-언어 모델(VLM)로, 다양한 시각적 태스크에서 뛰어난 일반화 성능을 보인다. 소규모 도메인 데이터에서의 과적합(Overfitting)을 방지하기 위해, 백본 네트워크의 약 20억 개 파라미터는 모두 동결(Freeze)한다.

백본을 통과한 고차원 특징 벡터 $h_i$는 식 (1)과 같다.

$$h_i = f_\theta(x_i) \in \mathbb{R}^{2048}$$

매칭 호환성 학습에 특화된 저차원 임베딩을 얻기 위해, 학습 가능한 프로젝션 헤드(Projection Head) $g_\phi$를 추가한다. 프로젝션 헤드는 2개의 선형 계층과 BatchNorm, ReLU 활성 함수로 구성된다.

$$z_i = g_\phi(h_i) = W_2 \cdot \text{ReLU}(\text{BN}(W_1 \cdot h_i))$$

$$e_i = \frac{z_i}{\|z_i\|_2} \in \mathbb{R}^{d}$$

L2 정규화를 통해 모든 임베딩 벡터를 단위 초구(Unit Hypersphere) 상에 투영하여, 코사인 유사도와 유클리드 거리가 단조 관계를 가지게 한다. 여기서 $d$는 프로젝션 차원으로, 실험을 통해 최적값을 탐색하였다.

### 2.3 Pooling 전략 선택

VLM의 Hidden States에서 단일 임베딩 벡터를 추출하기 위해 EOS Token Pooling과 Mean Pooling 두 가지 전략을 비교하였다. 표 2는 사전 학습된 Qwen3-VL 백본에서 각 Pooling 전략의 성능을 나타낸다.

**표 2. Pooling 전략별 베이스라인 성능 비교 (양방향 평균)**

| Pooling 전략 | R@5 | R@10 | R@20 | MRR |
|-------------|-----|------|------|-----|
| EOS Token | 8.0% | 14.0% | 29.3% | 0.069 |
| Mean Pooling | **11.3%** | **18.7%** | **30.0%** | **0.082** |
| 향상률 | 1.41배 | 1.34배 | 1.02배 | 1.19배 |

Mean Pooling이 EOS Token 대비 R@5에서 1.41배, MRR에서 1.19배 높은 성능을 보였다. 이는 단일 토큰보다 전체 시퀀스의 평균이 이미지의 전반적인 시각적 특징을 더 잘 표현함을 시사한다. 따라서 이후 모든 실험에서 Mean Pooling을 기본 전략으로 채택하였다.

### 2.4 손실 함수

실제 커플 쌍을 Positive, 배치 내 다른 사용자들을 Negative로 하여 대조 학습을 수행한다. CLIP[5]에서 검증된 InfoNCE Loss를 사용하며, 손실 함수는 식 (4)와 같다.

$$\mathcal{L} = -\log\frac{\exp(\text{sim}(e_f, e_m)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(e_f, e_{m_j})/\tau)}$$

여기서 $\tau$는 temperature 하이퍼파라미터이며, $\text{sim}$은 코사인 유사도, $N$은 배치 크기를 나타낸다.

---

## III. 실험 및 결과

### 3.1 실험 환경

백본 모델로 Qwen3-VL-2B를 사용하며, 과적합 방지를 위해 백본의 모든 파라미터를 동결하였다. 학습 환경은 AWS SageMaker의 NVIDIA A10G GPU(24GB)를 사용하였다. 기본 하이퍼파라미터는 표 3과 같다.

**표 3. 학습 하이퍼파라미터**

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| Backbone | Qwen3-VL-2B | 동결 (과적합 방지) |
| Batch Size | 48쌍 | A10G 24GB 최적화 |
| Learning Rate | 5e-5 | CosineAnnealing |
| Temperature (τ) | 0.1 | 기본값 |
| Weight Decay | 1e-3 | L2 정규화 |
| Epochs | 30 | Early Stopping (patience=10) |
| Projection Dim | 256 | 기본값 |
| Pooling | Mean | Hidden states 평균 |

### 3.2 베이스라인 비교

제안 모델의 효과를 검증하기 위해, 프로젝션 헤드 없이 사전 학습된 Qwen3-VL의 원본 임베딩(2048차원)을 직접 사용하는 베이스라인과 비교하였다. 평가 지표로는 Recall@K와 MRR(Mean Reciprocal Rank)을 사용하였다. 표 4는 테스트 세트(75쌍)에서 양방향(Female↔Male) 평균 성능을 나타낸다.

**표 4. 베이스라인 vs 제안 모델 성능 비교 (양방향 평균)**

| 모델 | R@5 | R@10 | R@20 | R@50 | MRR |
|------|-----|------|------|------|-----|
| Random Baseline | 6.7% | 13.3% | 26.7% | 66.7% | 0.040 |
| Qwen3-VL (Pre-trained) | 11.3% | 18.7% | 30.0% | 72.7% | 0.082 |
| **Proposed (Fine-tuned)** | **25.3%** | **36.7%** | **43.3%** | **74.7%** | **0.195** |
| 향상률 (vs Pre-trained) | **2.23배** | **1.96배** | **1.44배** | **1.03배** | **2.38배** |

사전 학습된 Qwen3-VL 베이스라인은 랜덤 기준 대비 전반적으로 높은 성능을 보였다. 이는 VLM의 일반적인 시각적 특징 추출 능력이 커플 매칭 태스크에 일부 유효함을 시사한다. 그러나 제안 모델은 베이스라인 대비 R@5에서 2.23배, MRR에서 2.38배 향상된 성능을 달성하여, 도메인 특화 프로젝션 헤드 학습의 효과를 명확히 입증하였다.

### 3.3 하이퍼파라미터 분석

최적의 모델 성능을 확보하기 위해 주요 하이퍼파라미터에 대한 실험을 수행하였다. 표 5는 프로젝션 차원과 Temperature 변화에 따른 성능 변화를 보여준다.

**표 5. 하이퍼파라미터 튜닝 결과 (양방향 평균)**

| 설정 | R@5 | R@10 | R@20 | R@50 | MRR |
|------|-----|------|------|------|-----|
| 기본 (dim=256, τ=0.1) | 25.3% | 36.7% | 43.3% | 74.7% | 0.195 |
| dim=512, τ=0.1 | **29.3%** | 36.7% | **49.3%** | **78.0%** | **0.237** |
| dim=256, τ=0.2 | **29.3%** | **37.3%** | **50.0%** | **82.0%** | 0.216 |

프로젝션 차원을 256에서 512로 증가시킨 경우, R@5가 25.3%에서 29.3%로 향상되었으며, MRR도 0.195에서 0.237로 21.5% 증가하였다. 이는 더 높은 차원의 임베딩 공간이 커플 간의 복잡한 관계를 더 잘 표현할 수 있음을 시사한다. Temperature를 0.1에서 0.2로 완화한 경우, R@50이 74.7%에서 82.0%로 크게 향상되어 유사도 분포의 완화가 일반화 성능 향상에 효과적임을 확인하였다.

---

## IV. 결 론

본 논문에서는 실제 데이팅 앱의 매칭 데이터를 활용하여 시각적 매칭 호환성을 예측하는 딥러닝 기반 시스템을 제안하였다. Qwen3-VL 백본의 파라미터를 동결하고 경량 프로젝션 헤드만 학습하는 효율적인 구조를 설계하였으며, InfoNCE Loss 기반 대조 학습을 통해 실제 커플 쌍을 임베딩 공간에서 가깝게 배치하도록 학습하였다.

실험 결과, 제안 모델은 사전 학습된 VLM 베이스라인 대비 R@5에서 2.23배, MRR에서 2.38배 향상된 성능을 달성하였다. 또한 하이퍼파라미터 튜닝을 통해 프로젝션 차원 512 및 Temperature 0.2 설정에서 R@50 82.0%의 최고 성능을 확인하였다.

본 연구를 통해 다음의 인사이트를 도출하였다. 첫째, 사전 학습된 VLM의 일반적 특징만으로는 커플 매칭 예측에 한계가 있으며, 도메인 특화 프로젝션 헤드 학습이 필수적이다. 둘째, 프로젝션 차원 및 Temperature 조정이 모델 성능에 큰 영향을 미친다.

향후 연구로는 텍스트 임베딩(성격, 가치관)과 이미지 임베딩을 결합한 하이브리드 모델 개발, 그리고 더 많은 커플 데이터 확보를 통한 일반화 성능 향상을 계획한다.

---

## 참고문헌

[1] E. Walster, V. Aronson, D. Abrahams, and L. Rottman, "Importance of physical attractiveness in dating behavior," *Journal of Personality and Social Psychology*, vol. 4, no. 5, pp. 508-516, 1966.

[2] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 815-823, 2015.

[3] A. Hermans, L. Beyer, and B. Leibe, "In defense of the triplet loss for person re-identification," *arXiv preprint arXiv:1703.07737*, 2017.

[4] J. Bai et al., "Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond," *arXiv preprint arXiv:2308.12966*, 2023.

[5] A. Radford et al., "Learning transferable visual models from natural language supervision," in *Proc. ICML*, 2021.

