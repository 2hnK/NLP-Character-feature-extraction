"""
Loss functions for metric learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning

    L = max(0, ||anchor - positive||^2 - ||anchor - negative||^2 + margin)
    """

    def __init__(self, margin=0.5, distance_metric='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Tensor of shape (batch_size, embedding_dim)
            positive: Tensor of shape (batch_size, embedding_dim)
            negative: Tensor of shape (batch_size, embedding_dim)

        Returns:
            loss: Scalar tensor
        """
        if self.distance_metric == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss with batch hard mining
    Automatically mines hard triplets from the batch
    """

    def __init__(self, margin=0.5, mining_strategy='hard'):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,)

        Returns:
            loss: Scalar tensor
        """
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)

        if self.mining_strategy == 'hard':
            return self._batch_hard_triplet_loss(labels, pairwise_dist)
        elif self.mining_strategy == 'all':
            return self._batch_all_triplet_loss(labels, pairwise_dist)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")

    def _pairwise_distances(self, embeddings):
        """Compute pairwise distances between embeddings"""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute dot product
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 distances
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)

        return distances

    def _batch_hard_triplet_loss(self, labels, pairwise_dist):
        """Build the triplet loss over a batch with hard negative mining"""
        # Get the hardest positive and negative for each anchor
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)

        # Hardest positive: max distance among positives
        anchor_positive_dist = pairwise_dist * mask_anchor_positive.float()
        hardest_positive_dist = anchor_positive_dist.max(dim=1, keepdim=True)[0]

        # Hardest negative: min distance among negatives
        # Add max value to positives to ignore them
        max_anchor_negative_dist = pairwise_dist + (1.0 - mask_anchor_negative.float()) * 1e9
        hardest_negative_dist = max_anchor_negative_dist.min(dim=1, keepdim=True)[0]

        # Combine to get triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        return triplet_loss.mean()

    def _batch_all_triplet_loss(self, labels, pairwise_dist):
        """Build the triplet loss over a batch using all valid triplets"""
        # Get valid triplet mask
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Get valid triplets
        mask = self._get_triplet_mask(labels)
        triplet_loss = triplet_loss * mask.float()

        # Remove negative losses
        triplet_loss = F.relu(triplet_loss)

        # Count valid triplets
        num_positive_triplets = (triplet_loss > 1e-16).float().sum()

        # Average over positive triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss

    def _get_anchor_positive_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, p] is True if a and p have same label"""
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        mask = indices_not_equal & labels_equal

        return mask

    def _get_anchor_negative_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True if a and n have different labels"""
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = ~labels_equal

        return mask

    def _get_triplet_mask(self, labels):
        """Return a 3D mask for valid triplets"""
        # Check that i, j, k are distinct
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = i_equal_j & ~i_equal_k

        mask = distinct_indices & valid_labels

        return mask


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for pairs
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1: Tensor of shape (batch_size, embedding_dim)
            embedding2: Tensor of shape (batch_size, embedding_dim)
            label: Tensor of shape (batch_size,) - 1 for similar, 0 for dissimilar

        Returns:
            loss: Scalar tensor
        """
        distance = F.pairwise_distance(embedding1, embedding2)

        loss_similar = label * torch.pow(distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(F.relu(self.margin - distance), 2)

        loss = loss_similar + loss_dissimilar
        return loss.mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Additive Angular Margin Loss)

    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698
    """

    def __init__(self, embedding_dim, num_classes, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,)

        Returns:
            loss: Scalar tensor
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight_norm)

        # Numerical stability
        cosine = cosine.clamp(-1, 1)

        # Get theta
        theta = torch.acos(cosine)

        # Add angular margin
        target_theta = theta.clone()
        target_theta.scatter_(1, labels.unsqueeze(1),
                            theta.gather(1, labels.unsqueeze(1)) + self.margin)

        # Convert back to cosine
        target_cosine = torch.cos(target_theta)

        # Scale
        output = target_cosine * self.scale

        # Cross entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


if __name__ == "__main__":
    # Test the loss functions
    batch_size = 32
    embedding_dim = 512

    # Test Triplet Loss
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negative = torch.randn(batch_size, embedding_dim)

    triplet_loss = TripletLoss(margin=0.5)
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss.item():.4f}")

    # Test Online Triplet Loss
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, 10, (batch_size,))

    online_triplet_loss = OnlineTripletLoss(margin=0.5, mining_strategy='hard')
    loss = online_triplet_loss(embeddings, labels)
    print(f"Online Triplet Loss: {loss.item():.4f}")

    # Test Contrastive Loss
    emb1 = torch.randn(batch_size, embedding_dim)
    emb2 = torch.randn(batch_size, embedding_dim)
    pair_labels = torch.randint(0, 2, (batch_size,)).float()

    contrastive_loss = ContrastiveLoss(margin=1.0)
    loss = contrastive_loss(emb1, emb2, pair_labels)
    print(f"Contrastive Loss: {loss.item():.4f}")
