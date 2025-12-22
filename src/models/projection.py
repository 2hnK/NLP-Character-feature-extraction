import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection Head for Contrastive Learning.
    Structure: Linear -> BatchNorm -> ReLU -> Linear -> L2 Normalization
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


class GenderSpecificProjection(nn.Module):
    """
    성별별 Projection Head
    
    구조:
    - Female: Backbone → Female Projection Head → Female Embedding
    - Male: Backbone → Male Projection Head → Male Embedding
    
    각 성별에 특화된 임베딩 공간을 학습하여:
    - 여성 이미지: 남성에게 매력적인 특징을 강조하는 공간으로 변환
    - 남성 이미지: 여성에게 매력적인 특징을 강조하는 공간으로 변환
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # 여성용 Projection Head
        self.female_head = ProjectionHead(input_dim, hidden_dim, output_dim)
        
        # 남성용 Projection Head
        self.male_head = ProjectionHead(input_dim, hidden_dim, output_dim)
        
    def forward_female(self, x):
        """여성 이미지에 대한 projection"""
        return self.female_head(x)
    
    def forward_male(self, x):
        """남성 이미지에 대한 projection"""
        return self.male_head(x)
    
    def forward(self, female_features, male_features):
        """
        양쪽 성별 동시 처리
        
        Args:
            female_features: [B, input_dim] 여성 Backbone 출력
            male_features: [B, input_dim] 남성 Backbone 출력
            
        Returns:
            female_embs: [B, output_dim] 여성 임베딩
            male_embs: [B, output_dim] 남성 임베딩
        """
        female_embs = self.female_head(female_features)
        male_embs = self.male_head(male_features)
        return female_embs, male_embs
