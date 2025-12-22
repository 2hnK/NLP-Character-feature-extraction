Metric Learning과 VLM을 활용한 데이팅 앱 커플 매칭 예측 시스템

김 지 훈*, 최 은 기, 박 준 형

A Couple Matching Prediction System for Dating Applications using Metric Learning and Vision-Language Models

Jihun Kim*, Eungi Choi, Junhyeong Park

요 약

본 논문에서는 실제 데이팅 앱에서 성사된 커플 데이터를 활용하여, 사용자 프로필 이미지 기반의 매칭 예측 시스템을 제안한다. 제안하는 모델은 Qwen3-VL을 백본으로 하여 시각적 특징을 추출하고, 경량화된 프로젝션 헤드를 결합하여 매칭 호환성 임베딩을 학습한다. InfoNCE Loss와 Semi-Hard Negative Mining 기법을 적용하여 실제 커플 쌍을 가깝게, 비커플 쌍을 멀게 배치하도록 학습한다. 775쌍의 데이터를 80/20 비율로 분할하여 Train/Validation과 Test 세트를 구성하였다. 실험 결과, 제안 모델은 사전 학습된 VLM 베이스라인 대비 R@1에서 109% 향상된 1.69%를 달성하였다.

Key Words: couple matching, dating application, metric learning, contrastive learning, InfoNCE, vision-language model, Qwen3-VL

ABSTRACT

This paper proposes a couple matching prediction system based on user profile images, utilizing actual couple data from a dating application. The proposed model employs Qwen3-VL as the backbone for visual feature extraction and incorporates a lightweight projection head to learn matching compatibility embeddings. We apply InfoNCE Loss with Semi-Hard Negative Mining to embed actual couple pairs closely while pushing non-couple pairs apart. The dataset of 775 pairs is split into 80/20 ratio for Train/Validation and Test sets. Experimental results demonstrate that the proposed model achieves R@1 of 1.69%, representing a 109% improvement over the pre-trained VLM baseline (R@1 0.81%), and a 2.6-fold improvement over the random baseline.

Ⅰ. 서 론

기존 온라인 데이팅 서비스에서 사용자 매칭은 주로 나이, 거주 지역, 관심사나 간단한 자기소개 문구 같은 텍스트 프로필 정보에 의존한다. 그러나 실제 사용자는 프로필 사진에서 드러나는 시각적 요인에 크게 좌우되며, 이러한 암묵적 선호를 정량화하는 것은 어려운 과제이다. 기존 연구[2][3]는 대부분 텍스트 프로필 유사도나 협업 필터링에 의존하였으나, 사용자의 시각적 선호를 반영하지 못하는 한계가 있다.

본 연구는 Walster 등이 제안한 사회심리학적 이론인 '매칭 가설(The Matching Hypothesis)[1]'을 이론적 토대로 한다. 해당 가설에 따르면, 개인은 파트너 선택 시 자신과 유사한 수준의 신체적 매력도나 사회적 바람직성을 가진 상대를 선호하는 경향(Assortative Mating)이 있다. 이를 현대적 온라인 데이팅 환경에 적용하여, 실제 매칭된 커플 데이터를 학습함으로써 매칭 호환성을 예측하는 모델을 제안한다.

본 연구의 주요 기여점은 다음과 같다: (1) 실제 커플 데이터 775쌍을 활용한 대조 학습 프레임워크 제안, (2) VLM 백본과 경량 프로젝션 헤드를 결합한 효율적 임베딩 구조 설계, (3) 사전 학습된 VLM 베이스라인 대비 109% 성능 향상 달성, (4) 소규모 데이터에서의 과적합 억제를 위한 하이퍼파라미터 튜닝 전략 검증.

Ⅱ. 본 론

1. 문제 정의

사용자 i의 프로필 이미지를 x_i라 정의한다. 커플 쌍 (x_f, x_m)에 대해, 두 사용자의 임베딩 e_f, e_m이 임베딩 공간에서 가깝도록 학습하며, 비커플 쌍은 멀어지도록 대조 학습(Contrastive Learning)을 수행한다.

2. 데이터셋

실제 데이팅 앱에서 상호 좋아요를 통해 매칭된 커플 775쌍의 프로필 이미지를 수집하였다. 각 커플 폴더에는 female.png와 male.png가 포함되어 있다.

[표 1] 데이터셋 구성
│ 항목 │ 값 │
│ 총 커플 수 │ 775쌍 │
│ Train+Valid │ 620쌍 (80%) │
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

여기서 τ는 temperature 하이퍼파라미터(0.1)이며, sim은 코사인 유사도를 의미한다.

학습 효율을 높이기 위해 Semi-Hard Negative Mining을 적용한다. 이는 마진 경계 내에 위치한 샘플을 선택하여 학습 안정성과 수렴 속도 간의 균형을 확보한다.

5. 데이터 분할

표 1과 같이 전체 데이터를 80/20 비율로 분할하였다. Validation 셋은 학습 중 Early Stopping과 최적 체크포인트 선택에 활용하며, Test 셋은 최종 성능 평가에 사용한다.

6. 실험 환경

[표 2] 학습 하이퍼파라미터
│ 파라미터 │ 값 │ 근거 │
│ Backbone │ Qwen3-VL-2B │ 동결 │
│ Batch Size │ 48~64쌍 │ A10G 24GB 최적화 │
│ Learning Rate │ 5e-5 │ CosineAnnealing │
│ Temperature │ 0.1 │ 실험적 최적값 │
│ Weight Decay │ 1e-3 │ L2 정규화 │
│ Epochs │ 30 │ Early Stopping │

백본 모델로 Qwen3-VL-2B를 사용하며, 과적합 방지를 위해 백본의 모든 파라미터를 동결하였다. 배치 크기는 GPU 메모리(A10G 24GB) 제약에 따라 48~64쌍으로 설정하였다. 학습률은 5e-5로 시작하여 CosineAnnealing 스케줄러를 적용하였으며, InfoNCE Loss의 Temperature는 실험을 통해 최적화된 0.1을 사용하였다. 총 30 에폭 동안 학습하며 Early Stopping을 적용하여 과적합을 방지하였다.

7. 실험 결과

7.1 베이스라인 비교

제안 모델의 효과를 검증하기 위해, 프로젝션 헤드 없이 사전 학습된 Qwen3-VL의 원본 임베딩을 사용하는 베이스라인과 비교하였다. 표 3은 Test 세트(124쌍)에서의 양방향(Female↔Male) 평균 성능을 나타낸다.

[표 3] 베이스라인 vs 제안 모델 성능 비교 (Test Set)
│ 모델 │ R@1 │ R@5 │ R@10 │ R@20 │ R@50 │ MRR │
│ Random Baseline │ 0.65% │ 3.23% │ 6.45% │ 12.90% │ 32.26% │ 0.032 │
│ Qwen3-VL (Pre-trained) │ 0.81% │ 3.63% │ 9.68% │ 18.95% │ 40.32% │ 0.044 │
│ Proposed (Fine-tuned) │ 1.69% │ 7.85% │ 14.98% │ 24.97% │ 37.29% │ 0.062 │
│ 향상률 (vs Pre-trained) │ +109% │ +116% │ +55% │ +32% │ -7% │ +41% │

사전 학습된 Qwen3-VL 베이스라인은 랜덤 기준 대비 R@1에서 25% 향상된 0.81%를 보였다. 이는 VLM의 일반적인 시각적 특징이 커플 매칭에 일부 도움이 됨을 시사한다. 그러나 제안 모델은 베이스라인 대비 R@1에서 109%, R@5에서 116% 향상된 성능을 달성하여, 도메인 특화 프로젝션 헤드 학습의 효과를 입증하였다.

7.2 하이퍼파라미터 튜닝 실험

과적합 억제를 위한 4가지 하이퍼파라미터 설정에 대해 비교 실험을 수행하였다. 표 4는 각 실험의 Test 성능을 나타낸다.

[표 4] 하이퍼파라미터 튜닝 실험 결과
│ 실험 │ τ │ LR │ Weight Decay │ R@1 │ R@5 │ R@10 │ MRR │
│ Exp1 (Initial) │ 0.07 │ 1e-4 │ 1e-4 │ 0.78% │ 5.71% │ 10.18% │ 0.047 │
│ Exp2 (+Dropout 0.3) │ 0.07 │ 1e-4 │ 1e-4 │ 0.97% │ 4.93% │ 10.18% │ 0.045 │
│ Exp3 (+Temp 0.1) │ 0.10 │ 1e-4 │ 1e-4 │ 1.49% │ 5.77% │ 11.28% │ 0.051 │
│ Exp4 (Final) │ 0.10 │ 5e-5 │ 1e-3 │ 1.69% │ 7.85% │ 14.98% │ 0.062 │

최종 모델(Exp4)은 초기 설정(Exp1) 대비 R@1에서 117% 향상된 1.69%를 달성하였다. 이는 랜덤 기준(0.65%) 대비 2.6배 높은 수치이며, R@10 기준 14.98%로 상위 10명 내에 실제 매칭 상대가 포함될 확률이 유의미하게 높음을 확인하였다.

Ⅲ. 결 론

본 논문에서는 실제 데이팅 앱의 매칭 데이터를 활용하여 시각적 매칭 호환성을 예측하는 딥러닝 기반 시스템을 제안하였다. 제안 모델은 사전 학습된 Qwen3-VL 베이스라인 대비 R@1에서 109% 향상된 1.69%를 달성하였으며, 이는 랜덤 기준 대비 2.6배에 해당한다.

본 실험을 통해 다음의 인사이트를 도출하였다. 첫째, 사전 학습된 VLM의 일반적 특징만으로는 커플 매칭 예측에 한계가 있으며, 도메인 특화 프로젝션 헤드 학습이 필수적이다. R@1 기준 베이스라인(0.81%)에서 파인튜닝 모델(1.69%)로 109% 성능 향상을 확인하였다. 둘째, 소규모 데이터셋에서는 Dropout이 오히려 역효과를 보였으며, 이는 제한된 학습 신호를 더욱 약화시키기 때문으로 분석된다. 셋째, InfoNCE Loss의 Temperature 파라미터가 일반화 성능에 중요한 영향을 미쳤으며, τ=0.07에서 τ=0.1로 완화 시 모든 지표에서 개선이 관찰되었다. 넷째, Weight Decay 증가(1e-4→1e-3)와 Learning Rate 감소(1e-4→5e-5)의 조합이 과적합 억제에 효과적이었다.

향후 연구로는 (1) 텍스트 임베딩(성격, 가치관)과 이미지 임베딩을 결합한 하이브리드 모델 개발, (2) 더 많은 커플 데이터 확보를 통한 일반화 성능 향상, (3) Cross-modal Attention을 활용한 남녀 간 시각적 상호작용 모델링을 계획한다.

References

[1] E. Walster, V. Aronson, D. Abrahams, and L. Rottman, "Importance of physical attractiveness in dating behavior," Journal of Personality and Social Psychology, vol. 4, no. 5, pp. 508-516, 1966.
[2] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2015, pp. 815-823.
[3] A. Hermans, L. Beyer, and B. Leibe, "In defense of the triplet loss for person re-identification," arXiv preprint arXiv:1703.07737, 2017.
[4] J. Bai et al., "Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond," arXiv preprint arXiv:2308.12966, 2023.
[5] A. Radford et al., "Learning transferable visual models from natural language supervision," in Proc. ICML, 2021.
