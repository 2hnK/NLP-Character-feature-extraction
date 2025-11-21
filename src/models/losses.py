import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners, distances, reducers

class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss wrapper using pytorch-metric-learning.
    Uses TripletMarginLoss with a Miner.
    """
    def __init__(self, margin=0.2, type_of_triplets="semihard"):
        super().__init__()
        self.distance = distances.CosineSimilarity()
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss_func = losses.TripletMarginLoss(
            margin=margin, 
            distance=self.distance, 
            reducer=self.reducer
        )
        self.miner = miners.TripletMarginMiner(
            margin=margin, 
            distance=self.distance, 
            type_of_triplets=type_of_triplets
        )
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (Batch_Size, Embedding_Dim)
            labels: (Batch_Size)
        """
        # Mine triplets
        hard_pairs = self.miner(embeddings, labels)
        
        # Calculate loss
        loss = self.loss_func(embeddings, labels, hard_pairs)
        
        # Calculate number of active triplets for logging
        num_triplets = self.miner.num_triplets
        
        return loss, num_triplets
