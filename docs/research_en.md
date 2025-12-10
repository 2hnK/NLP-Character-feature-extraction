# A Style Matching System for Dating Applications using Metric Learning and Vision-Language Models

Jihun Kim\*, Eungi Choi, Junhyeong Park

## Abstract

In this paper, we propose a deep learning-based style matching system that quantifies the visual style of user profile images to identify users with similar aesthetic preferences. The proposed model adopts Qwen3-VL as the backbone for visual feature extraction and incorporates a lightweight projection head to learn style embeddings. To construct a training dataset, we leverage a Gemini-2.5-flash-based VLM to automatically generate high-quality style tags. We further employ Triplet Loss and an online semi-hard mining strategy to maximize style clustering performance in the embedding space. Experimental results, evaluated through Recall@K and t-SNE visualizations, demonstrate that the proposed method effectively distinguishes visual styles, thereby complementing traditional text-based filters that often miss users' nuanced aesthetic preferences.

**Keywords:** style matching, dating application, metric learning, triplet loss, vision-language model, Qwen3-VL

## Ⅰ. Introduction

In conventional online dating services, user matching primarily relies on text-based profile information such as age, location, interests, and brief self-descriptions. However, users are significantly influenced by visual factors including fashion style, photo atmosphere, and shooting environment. Simple classification models or filtering mechanisms cannot adequately capture these high-dimensional preferences.

To overcome these limitations, we propose a style matching system that combines Vision-Language Model (VLM)-based feature extraction with Metric Learning. This approach quantifies the 'taste' and 'atmosphere' that cannot be captured by text filters, using cosine similarity of image embedding vectors.

This research is theoretically grounded in the 'Matching Hypothesis[1]', a social psychological theory proposed by Walster et al. According to this hypothesis, individuals tend to prefer partners with similar levels of physical attractiveness or social desirability (Assortative Mating). Applying this to modern online dating environments, we consider the similarity of visual style and atmosphere—difficult to capture through text-based filtering—as a core matching element, and aim to improve user matching satisfaction by quantifying it as distance in the embedding space.

## Ⅱ. Methodology

### 1. Problem Definition

Let x_i denote the profile image of user i. Each image is transformed into a high-dimensional feature vector through an encoder f_θ based on a pre-trained Vision-Language Model (VLM).

### 2. Dataset Construction

For dataset construction, we input images into the Gemini-2.5-flash model and automatically extract structured JSON metadata as shown in Table 1 through prompt engineering.

**Table 1.** Metadata Schema and Attributes
| Attribute | Type | Description | Classes |
|-----------|------|-------------|---------|
| fashion_style | Categorical | Fashion style (training label) | 5 |
| shot_type | Categorical | Shot type (selfie/full-body/snapshot) | 3 |
| visual_quality | Categorical | Image quality (High/Mid/Low) | 3 |
| physical_features | Text | Hair, accessories, and other appearance features | - |
| caption | Text | Natural language image description | - |

Among the extracted attributes, fashion_style is used as the class label for Metric Learning, and samples with Low visual_quality are excluded from training to minimize data noise. The style label y_i is encoded as an integer index, represented as y_i ∈ {0,1,...,C-1}.

In the actual service stage, the dating app server pre-computes style embeddings for all users and indexes them in a Vector Database (Vector DB). Upon recommendation request, the cosine similarity between the query user's embedding and existing user embeddings is calculated to extract the top K candidates.

### 3. Style Embedding Architecture

{docs/System Concept.png}
**Figure 1.** System Architecture: Qwen3-VL Backbone and Projection Head Structure

In this study, we use Qwen3-VL[4], pre-trained on large-scale data, as the backbone. Let x_i be the profile image of user i, and y_i be the style label of that image. Here, C is the total number of defined fashion style classes.

To prevent overfitting on small-scale domain data and maintain generalization performance, all backbone network parameters are frozen. The high-dimensional feature vector h_i after passing through the backbone is:

    h_i = f_θ(x_i)                                   (1)

To obtain a low-dimensional embedding specialized for style matching, we add a learnable Projection Head g_φ to the vector from Equation (1). The projection head consists of two linear layers, Layer Normalization, and an activation function.

    z_i = g_φ(h_i) = W_2 · GELU(LN(W_1 · h_i))       (2)

Here, W_1, W_2 are learnable weight matrices, LN is Layer Normalization, and GELU is the activation function. In the final stage, L2 normalization is performed to project all embedding vectors e_i onto the unit hypersphere.

    e_i = z_i / ||z_i||_2                            (3)

This ensures a monotonic relationship between cosine similarity and Euclidean distance, improving training stability. The style similarity between two user embeddings e_i, e_j during matching is defined as:

    sim(e_i, e_j) = e_i^T · e_j                      (4)

### 4. Triplet-based Metric Learning

To train users with the same style to be close in the embedding space and users with different styles to be far apart, we use Triplet Margin Loss[2]. Figure 2 illustrates the core concept of Triplet Loss: centered on the Anchor sample, the Positive sample with the same style is pulled closer (minimizing d(a,p)), the Negative sample with a different style is pushed farther (maximizing d(a,n)), and a minimum margin α is maintained between them.

{docs/Triplet_Loss_Concept.png}
**Figure 2.** Triplet Loss Concept: Anchor (red), Positive (green), Negative (blue) Relationship

The loss function for the Anchor (a), Positive (p) with the same style, and Negative (n) with a different style embeddings is:

    L_{triplet} = max(d(e_a, e_p) - d(e_a, e_n) + α, 0)  (5)

Here, d is the Euclidean distance, and α is the margin hyperparameter. The conditions y_a = y_p (same class) and y_a ≠ y_n (different class) must be satisfied.

To maximize training efficiency, we apply Online Semi-hard Mining[3], which selects samples of appropriate difficulty in real-time within the mini-batch. Semi-hard Negatives are samples located within the margin boundary (d(a,p) < d(a,n) < d(a,p) + α), providing a balance between training stability and convergence speed as they are neither too easy nor too hard. Within a mini-batch, for each Anchor e_a, the following samples are selected for loss computation:

● Hard Positive: The sample with the greatest distance from the Anchor within the same class
p_{hard} = argmax_{y_p = y_a} d(e_a, e_p)

● Hard Negative: The sample with the smallest distance from the Anchor among all other classes
n_{hard} = argmin_{y_n ≠ y_a} d(e_a, e_n)

This approach guides learning to focus on edge cases that are easily confused in real service environments, rather than easy samples (Easy Triplets) that are already well-distinguished.

### 5. PK Sampler-based Batch Construction

For Online Semi-hard Mining to operate stably, there must be a sufficient number of Positive samples within the mini-batch to construct valid Triplets. To ensure this, we use the PK Sampler for batch construction.

The PK Sampler randomly selects P style classes for each mini-batch and samples K images from each selected class to construct a batch of size P×K. This approach guarantees at least (K-1) Positive candidates for each Anchor, ensuring Hard Positive mining is always possible. Additionally, even when class imbalance exists in the data, all style classes are exposed with equal frequency during training, forming an unbiased embedding space.

### 6. Experimental Setup

In the experiments, we constructed a profile image dataset simulating a real dating app environment and used a Gemini-2.5-flash-based labeler to automatically annotate each image with fashion style, photo atmosphere, and quality information. Samples with low quality were excluded from training, and oversampling and Seedream 4 generative model-based augmentation were applied to balance style classes. During augmentation, only non-essential factors such as background, lighting, and pose were varied while maintaining style consistency with the original image.

The dataset is split into training/validation sets (1,508/98). During training, the Qwen3-VL-2B visual encoder is frozen, and only the projection head is updated. The optimizer is AdamW (learning rate 1e-4), with batch size P=5, K=4. Margin α=0.3 and embedding dimension d=256 were set, and the optimal checkpoint was selected based on Recall@1 on the validation set.

### 7. Evaluation Results

Quantitative evaluation is conducted by using a validation image as a query and performing nearest neighbor search on other users in the embedding space. Table 2 shows the performance metrics measured on the validation set.

**Table 2.** Validation Set Performance Metrics
| Metric | Value |
|--------|-------|
| Recall@1 | 87.76% |
| MAP@R | 78.52% |
| Silhouette Score | 0.34 |

Recall@1 indicates the probability that a sample with the same style as the query is retrieved as the nearest neighbor, and MAP@R measures the mean average precision of related samples. The positive Silhouette Score (0.34) confirms that style clusters are well-formed in the embedding space.

For qualitative evaluation, t-SNE visualization was used to reduce the 256-dimensional embedding space to 2 dimensions, confirming that the five style classes (Casual_Basic, Street_Hip, Sporty_Athleisure, Chic_Modern, Classy_Elegant) form distinct clusters with clear boundaries between classes.

## Ⅲ. Conclusion

In this paper, we proposed a deep learning-based style matching system that quantifies the visual style of user profile images to match users with similar preferences. We combined the Qwen3-VL backbone with a Projection Head to learn style embeddings and applied Triplet Loss and Online Semi-hard Mining techniques to maximize style clustering performance in the embedding space. Experimental results achieved Recall@1 of 87.76% and MAP@R of 78.52% on the validation set, with a Silhouette Score of 0.34 confirming effective style cluster formation in the embedding space.

Future research directions include implementing a hybrid matching engine combining text embeddings (personality/values) with image embeddings (style), real-time serving optimization using Vector DB, and performance validation in real service environments through A/B testing.

## References

[1] E. Walster, V. Aronson, D. Abrahams, and L. Rottman, "Importance of physical attractiveness in dating behavior," Journal of Personality and Social Psychology, vol. 4, no. 5, pp. 508-516, 1966.

[2] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2015, pp. 815-823.

[3] A. Hermans, L. Beyer, and B. Leibe, "In defense of the triplet loss for person re-identification," arXiv preprint arXiv:1703.07737, 2017.

[4] J. Bai et al., "Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond," arXiv preprint arXiv:2308.12966, 2023.
