import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    Projection Head for Triplet Loss.
    Structure: Linear -> ReLU -> Linear -> L2 Normalization
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer2(x)
        # L2 Normalization
        x = F.normalize(x, p=2, dim=1)
        return x
