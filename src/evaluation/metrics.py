"""
Evaluation metrics for profile matching
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple


def compute_embedding_quality(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute embedding quality metrics

    Args:
        embeddings: Embedding vectors of shape (N, embedding_dim)
        labels: Class labels of shape (N,)

    Returns:
        metrics: Dictionary of metrics
    """
    unique_labels = np.unique(labels)

    intra_class_distances = []
    inter_class_distances = []

    # Compute distances
    for label in unique_labels:
        # Get embeddings for this class
        class_mask = labels == label
        class_embeddings = embeddings[class_mask]

        if len(class_embeddings) < 2:
            continue

        # Intra-class distances (within same class)
        intra_dist = pairwise_distances(class_embeddings, metric='euclidean')
        # Take upper triangle (exclude diagonal)
        intra_dist = intra_dist[np.triu_indices_from(intra_dist, k=1)]
        intra_class_distances.extend(intra_dist)

        # Inter-class distances (to other classes)
        other_mask = labels != label
        other_embeddings = embeddings[other_mask]

        if len(other_embeddings) > 0:
            inter_dist = pairwise_distances(
                class_embeddings,
                other_embeddings,
                metric='euclidean'
            )
            inter_class_distances.extend(inter_dist.flatten())

    # Compute statistics
    metrics = {
        'intra_class_mean': np.mean(intra_class_distances),
        'intra_class_std': np.std(intra_class_distances),
        'inter_class_mean': np.mean(inter_class_distances),
        'inter_class_std': np.std(inter_class_distances),
        'separation_ratio': np.mean(inter_class_distances) / (np.mean(intra_class_distances) + 1e-8)
    }

    return metrics


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Compute retrieval metrics (Top-K accuracy)

    Args:
        query_embeddings: Query embedding vectors
        gallery_embeddings: Gallery embedding vectors
        query_labels: Query labels
        gallery_labels: Gallery labels
        k_values: K values for Top-K accuracy

    Returns:
        metrics: Dictionary of metrics
    """
    # Compute similarity matrix (cosine similarity)
    similarity = cosine_similarity(query_embeddings, gallery_embeddings)

    metrics = {}

    for k in k_values:
        # Get top-k indices for each query
        top_k_indices = np.argsort(-similarity, axis=1)[:, :k]

        # Check if any of top-k matches the query label
        correct = 0
        for i, query_label in enumerate(query_labels):
            top_k_labels = gallery_labels[top_k_indices[i]]
            if query_label in top_k_labels:
                correct += 1

        accuracy = correct / len(query_labels)
        metrics[f'top_{k}_accuracy'] = accuracy

    return metrics


def compute_map(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray
) -> float:
    """
    Compute Mean Average Precision (mAP)

    Args:
        query_embeddings: Query embedding vectors
        gallery_embeddings: Gallery embedding vectors
        query_labels: Query labels
        gallery_labels: Gallery labels

    Returns:
        mAP: Mean Average Precision
    """
    # Compute similarity matrix
    similarity = cosine_similarity(query_embeddings, gallery_embeddings)

    # Get sorted indices
    sorted_indices = np.argsort(-similarity, axis=1)

    aps = []

    for i, query_label in enumerate(query_labels):
        # Get sorted gallery labels
        sorted_labels = gallery_labels[sorted_indices[i]]

        # Find relevant items
        relevant = (sorted_labels == query_label)

        if not np.any(relevant):
            continue

        # Compute average precision
        precisions = []
        num_relevant = 0

        for j, is_relevant in enumerate(relevant):
            if is_relevant:
                num_relevant += 1
                precision = num_relevant / (j + 1)
                precisions.append(precision)

        ap = np.mean(precisions) if precisions else 0.0
        aps.append(ap)

    mAP = np.mean(aps) if aps else 0.0

    return mAP


def compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score

    Args:
        embeddings: Embedding vectors
        labels: Class labels

    Returns:
        silhouette_score: Average silhouette score
    """
    from sklearn.metrics import silhouette_score as sklearn_silhouette

    if len(np.unique(labels)) < 2:
        return 0.0

    score = sklearn_silhouette(embeddings, labels, metric='euclidean')

    return score


def evaluate_model(
    model,
    dataloader,
    device='cuda',
    k_values=[1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Comprehensive model evaluation

    Args:
        model: Feature extraction model
        dataloader: Validation/test dataloader
        device: Device to run on
        k_values: K values for Top-K accuracy

    Returns:
        metrics: Dictionary of all metrics
    """
    import torch

    model.eval()

    all_embeddings = []
    all_labels = []

    # Extract all embeddings
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']

            embeddings = model(images)
            embeddings = embeddings.cpu().numpy()

            all_embeddings.append(embeddings)
            all_labels.append(labels.numpy())

    # Concatenate
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    # Normalize embeddings
    all_embeddings = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8)

    # Split into query and gallery (use 50-50 split)
    n = len(all_embeddings)
    split_idx = n // 2

    query_embeddings = all_embeddings[:split_idx]
    query_labels = all_labels[:split_idx]
    gallery_embeddings = all_embeddings[split_idx:]
    gallery_labels = all_labels[split_idx:]

    # Compute metrics
    metrics = {}

    # Embedding quality
    quality_metrics = compute_embedding_quality(all_embeddings, all_labels)
    metrics.update(quality_metrics)

    # Retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        k_values=k_values
    )
    metrics.update(retrieval_metrics)

    # mAP
    mAP = compute_map(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels
    )
    metrics['mAP'] = mAP

    # Silhouette score
    silhouette = compute_silhouette_score(all_embeddings, all_labels)
    metrics['silhouette_score'] = silhouette

    return metrics


if __name__ == "__main__":
    # Test the metrics
    print("Testing evaluation metrics...")

    # Generate dummy data
    np.random.seed(42)

    n_samples = 100
    n_classes = 10
    embedding_dim = 512

    # Create embeddings with some structure
    embeddings = []
    labels = []

    for i in range(n_classes):
        # Create cluster
        cluster_center = np.random.randn(embedding_dim)
        cluster_embeddings = cluster_center + np.random.randn(n_samples // n_classes, embedding_dim) * 0.1

        embeddings.append(cluster_embeddings)
        labels.extend([i] * (n_samples // n_classes))

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Compute metrics
    quality_metrics = compute_embedding_quality(embeddings, labels)
    print("\nEmbedding Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Retrieval metrics
    split_idx = len(embeddings) // 2
    retrieval_metrics = compute_retrieval_metrics(
        embeddings[:split_idx],
        embeddings[split_idx:],
        labels[:split_idx],
        labels[split_idx:],
        k_values=[1, 5, 10]
    )

    print("\nRetrieval Metrics:")
    for key, value in retrieval_metrics.items():
        print(f"  {key}: {value:.4f}")

    # mAP
    mAP = compute_map(
        embeddings[:split_idx],
        embeddings[split_idx:],
        labels[:split_idx],
        labels[split_idx:]
    )
    print(f"\nmAP: {mAP:.4f}")

    # Silhouette
    silhouette = compute_silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
