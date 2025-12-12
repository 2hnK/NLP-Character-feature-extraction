Metric Learning과 VLM을 활용한 데이팅 앱 커플 매칭 예측 시스템

김 지 훈\*, 최 은 기, 박 준 형

A Couple Matching Prediction System for Dating Applications using Metric Learning and Vision-Language Models

Jihun Kim\*, Eungi Choi, Junhyeong Park

요 약

본 논문에서는 실제 데이팅 앱에서 성사된 커플 데이터를 활용하여, 사용자 프로필 이미지 기반의 매칭 예측 시스템을 제안한다. 제안하는 모델은 Qwen3-VL을 백본으로 하여 시각적 특징을 추출하고, 경량화된 프로젝션 헤드를 결합하여 매칭 호환성 임베딩을 학습한다. InfoNCE Loss와 Semi-Hard Negative Mining 기법을 적용하여 실제 커플 쌍을 가깝게, 비커플 쌍을 멀게 배치하도록 학습한다. 소규모 데이터(775쌍)의 한계를 극복하기 위해 5-Fold Cross Validation을 적용하여 모델의 일반화 성능과 신뢰구간을 확보하였다.

Key Words: couple matching, dating application, metric learning, contrastive learning, InfoNCE, vision-language model, Qwen3-VL

ABSTRACT

In this paper, we propose a couple matching prediction system based on user profile images, utilizing actual couple data from a dating application. The proposed model adopts Qwen3-VL as the backbone for visual feature extraction and incorporates a lightweight projection head to learn matching compatibility embeddings. We employ InfoNCE Loss with Semi-Hard Negative Mining to train the model such that actual couple pairs are embedded closely while non-couple pairs are pushed apart. To overcome the limitations of small-scale data (775 pairs), we apply 5-Fold Cross Validation to ensure model generalization and establish confidence intervals.

Ⅰ. 서 론

기존 온라인 데이팅 서비스에서 사용자 매칭은 주로 나이, 거주 지역, 관심사나 간단한 자기소개 문구 같은 텍스트 프로필 정보에 의존한다. 그러나 실제 사용자는 프로필 사진에서 드러나는 시각적 요인에 크게 좌우되며, 이러한 암묵적 선호를 정량화하는 것은 어려운 과제이다.

본 연구는 Walster 등이 제안한 사회심리학적 이론인 '매칭 가설(The Matching Hypothesis)[1]'을 이론적 토대로 한다. 해당 가설에 따르면, 개인은 파트너 선택 시 자신과 유사한 수준의 신체적 매력도나 사회적 바람직성을 가진 상대를 선호하는 경향(Assortative Mating)이 있다. 이를 현대적 온라인 데이팅 환경에 적용하여, 실제 매칭된 커플 데이터를 학습함으로써 매칭 호환성을 예측하는 모델을 제안한다.

Ⅱ. 본 론

1. 문제 정의

사용자 i의 프로필 이미지를 x_i라 하자. 커플 쌍 (x_f, x_m)에 대해, 두 사용자의 임베딩 e_f, e_m이 임베딩 공간에서 가깝도록 학습한다. 비커플 쌍은 멀어지도록 대조 학습(Contrastive Learning)을 수행한다.

2. 데이터셋

실제 데이팅 앱에서 상호 좋아요를 통해 매칭된 커플 775쌍의 프로필 이미지를 수집하였다. 각 커플 폴더에는 female.png와 male.png가 포함되어 있다.

[표 1] 데이터셋 구성
│ 항목 │ 값 │
│ 총 커플 수 │ 775쌍 │
│ Train+Valid (5-Fold) │ 620쌍 (80%) │
│ Test (Hold-out) │ 155쌍 (20%) │

3. 임베딩 구조

본 연구에서는 대규모 데이터로 사전 학습된 Qwen3-VL[4]을 백본(Backbone)으로 사용한다. 소규모 도메인 데이터에서의 과적합(Overfitting)을 방지하기 위해, 백본 네트워크의 파라미터는 모두 동결(Freeze)한다.

    h_i = f_θ(x_i)                                   (1)

매칭 호환성 학습에 특화된 저차원 임베딩을 얻기 위해, 학습 가능한 프로젝션 헤드(Projection Head) g_φ를 추가한다.

    z_i = g_φ(h_i) = W_2 · GELU(LN(W_1 · h_i))       (2)
    e_i = z_i / ||z_i||_2                            (3)

4. InfoNCE 기반 Contrastive Learning

실제 커플 쌍을 Positive, 배치 내 다른 사용자들을 Negative로 하여 대조 학습을 수행한다. CLIP[5]에서 검증된 InfoNCE Loss를 사용한다.

    L = -log(exp(sim(e_f, e_m)/τ) / Σ_j exp(sim(e_f, e_m_j)/τ))  (4)

여기서 τ는 temperature 하이퍼파라미터(0.07)이며, sim은 코사인 유사도를 의미한다.

학습 효율을 높이기 위해 Semi-Hard Negative Mining을 적용한다. 이는 마진 경계 내에 위치한 샘플을 선택하여 학습 안정성과 수렴 속도 간의 균형을 제공한다.

5. 5-Fold Cross Validation

775쌍의 소규모 데이터에서 신뢰성 있는 평가를 위해 5-Fold Cross Validation을 적용한다.

[표 2] 데이터 분할 전략
│ 방법 │ Train 데이터 활용 │ 평가 신뢰성 │
│ Hold-out 70/30 │ 70%만 사용 │ 1회 검증 │
│ 5-Fold CV │ 모든 데이터 활용 │ 5회 평균±표준편차 │

각 fold에서 최적 체크포인트를 저장하고, 최종 성능은 5개 fold의 평균과 표준편차로 보고한다.

6. 실험 환경

[표 3] 학습 하이퍼파라미터
│ 파라미터 │ 값 │ 근거 │
│ Backbone │ Qwen3-VL-2B │ 동결 │
│ Batch Size │ 48~64쌍 │ A10G 24GB 최적화 │
│ Learning Rate │ 1e-4 │ CosineAnnealing │
│ Temperature │ 0.07 │ CLIP 검증값 │
│ Epochs │ 30 │ Early Stopping │

Ⅲ. 결 론

본 논문에서는 실제 데이팅 앱의 매칭 데이터를 활용하여 시각적 매칭 호환성을 예측하는 딥러닝 기반 시스템을 제안하였다. Qwen3-VL 백본과 Projection Head를 결합하여 매칭 임베딩을 학습하고, InfoNCE Loss와 Semi-Hard Negative Mining 기법을 적용하였다. 소규모 데이터의 한계를 극복하기 위해 5-Fold Cross Validation을 적용하여 모델의 일반화 성능을 확보하였다.

향후 연구로는 텍스트 임베딩(성격/가치관)과 이미지 임베딩을 결합한 하이브리드 매칭 엔진 구현, 그리고 더 많은 커플 데이터를 확보하여 모델 성능을 향상시키는 것을 계획한다.

References

[1] E. Walster, V. Aronson, D. Abrahams, and L. Rottman, "Importance of physical attractiveness in dating behavior," Journal of Personality and Social Psychology, vol. 4, no. 5, pp. 508-516, 1966.
[2] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2015, pp. 815-823.
[3] A. Hermans, L. Beyer, and B. Leibe, "In defense of the triplet loss for person re-identification," arXiv preprint arXiv:1703.07737, 2017.
[4] J. Bai et al., "Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond," arXiv preprint arXiv:2308.12966, 2023.
[5] A. Radford et al., "Learning transferable visual models from natural language supervision," in Proc. ICML, 2021.
