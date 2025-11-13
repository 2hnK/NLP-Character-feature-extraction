"""
Backbone networks for feature extraction
"""

import torch
import torch.nn as nn
from torchvision import models


class BackboneFactory:
    """Factory class to create different backbone networks"""

    @staticmethod
    def create_backbone(backbone_name, pretrained=True):
        """
        Create a backbone network

        Args:
            backbone_name: Name of the backbone (efficientnet_b0, resnet50, vit_base)
            pretrained: Whether to load pretrained weights

        Returns:
            backbone: nn.Module
            feature_dim: int (output feature dimension)
        """
        if backbone_name == 'efficientnet_b0':
            return BackboneFactory._create_efficientnet_b0(pretrained)
        elif backbone_name == 'efficientnet_b3':
            return BackboneFactory._create_efficientnet_b3(pretrained)
        elif backbone_name == 'resnet50':
            return BackboneFactory._create_resnet50(pretrained)
        elif backbone_name == 'resnet101':
            return BackboneFactory._create_resnet101(pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

    @staticmethod
    def _create_efficientnet_b0(pretrained=True):
        """Create EfficientNet-B0 backbone"""
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Remove classification head
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        return backbone, feature_dim

    @staticmethod
    def _create_efficientnet_b3(pretrained=True):
        """Create EfficientNet-B3 backbone"""
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # Remove classification head
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        return backbone, feature_dim

    @staticmethod
    def _create_resnet50(pretrained=True):
        """Create ResNet50 backbone"""
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove classification head
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        return backbone, feature_dim

    @staticmethod
    def _create_resnet101(pretrained=True):
        """Create ResNet101 backbone"""
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        backbone = models.resnet101(weights=weights)

        # Remove classification head
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        return backbone, feature_dim


class EmbeddingHead(nn.Module):
    """
    Embedding head to project features to embedding space
    """

    def __init__(self, feature_dim, embedding_dim, dropout=0.3):
        super(EmbeddingHead, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.embedding(x)


class ProfileFeatureExtractor(nn.Module):
    """
    Complete model for profile feature extraction

    Architecture:
        Input Image -> Backbone -> Embedding Head -> L2 Normalization -> Embedding
    """

    def __init__(
        self,
        backbone_name='efficientnet_b0',
        embedding_dim=512,
        pretrained=True,
        dropout=0.3,
        normalize=True
    ):
        super(ProfileFeatureExtractor, self).__init__()

        # Create backbone
        self.backbone, feature_dim = BackboneFactory.create_backbone(
            backbone_name, pretrained
        )

        # Embedding head
        self.embedding_head = EmbeddingHead(feature_dim, embedding_dim, dropout)

        # L2 normalization flag
        self.normalize = normalize

        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Extract features with backbone
        features = self.backbone(x)

        # Project to embedding space
        embeddings = self.embedding_head(features)

        # L2 normalization
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_embedding(self, x):
        """Alias for forward, for clarity"""
        return self.forward(x)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device='cpu'):
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on

        Returns:
            model: Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get model config from checkpoint
        config = checkpoint.get('config', {})

        # Create model
        model = cls(
            backbone_name=config.get('backbone_name', 'efficientnet_b0'),
            embedding_dim=config.get('embedding_dim', 512),
            pretrained=False,  # Don't load ImageNet weights
            dropout=config.get('dropout', 0.3),
            normalize=config.get('normalize', True)
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Set to eval mode
        model.eval()

        return model

    def save_checkpoint(self, path, optimizer=None, epoch=None, **kwargs):
        """
        Save model checkpoint

        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            **kwargs: Additional metadata to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'feature_dim': self.feature_dim,
                'normalize': self.normalize,
            }
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        # Add any additional metadata
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)


class MultiTaskModel(nn.Module):
    """
    Multi-task model for joint learning
    Main task: Embedding extraction
    Auxiliary tasks: Age prediction, style classification, etc.
    """

    def __init__(
        self,
        backbone_name='efficientnet_b0',
        embedding_dim=512,
        pretrained=True,
        dropout=0.3,
        auxiliary_tasks=None
    ):
        super(MultiTaskModel, self).__init__()

        # Main embedding model
        self.feature_extractor = ProfileFeatureExtractor(
            backbone_name=backbone_name,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            dropout=dropout,
            normalize=True
        )

        # Auxiliary task heads
        self.auxiliary_heads = nn.ModuleDict()

        if auxiliary_tasks:
            for task in auxiliary_tasks:
                task_name = task['name']
                num_classes = task['num_classes']

                # Simple classification head
                self.auxiliary_heads[task_name] = nn.Sequential(
                    nn.Linear(embedding_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, num_classes)
                )

    def forward(self, x):
        """
        Args:
            x: Input tensor

        Returns:
            outputs: Dict with embeddings and auxiliary task outputs
        """
        # Get embeddings
        embeddings = self.feature_extractor(x)

        outputs = {'embeddings': embeddings}

        # Auxiliary task predictions
        for task_name, head in self.auxiliary_heads.items():
            outputs[task_name] = head(embeddings)

        return outputs


if __name__ == "__main__":
    # Test the models
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    # Test ProfileFeatureExtractor
    print("Testing ProfileFeatureExtractor...")
    model = ProfileFeatureExtractor(
        backbone_name='efficientnet_b0',
        embedding_dim=512,
        pretrained=False  # Set to False for testing
    )

    embeddings = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding norm: {embeddings.norm(dim=1).mean().item():.4f}")

    # Test freezing/unfreezing
    model.freeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (frozen backbone): {trainable_params:,}")

    model.unfreeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (unfrozen backbone): {trainable_params:,}")

    # Test MultiTaskModel
    print("\nTesting MultiTaskModel...")
    auxiliary_tasks = [
        {'name': 'age', 'num_classes': 10},
        {'name': 'style', 'num_classes': 5}
    ]

    multi_model = MultiTaskModel(
        backbone_name='efficientnet_b0',
        embedding_dim=512,
        pretrained=False,
        auxiliary_tasks=auxiliary_tasks
    )

    outputs = multi_model(input_tensor)
    print(f"Embeddings shape: {outputs['embeddings'].shape}")
    print(f"Age output shape: {outputs['age'].shape}")
    print(f"Style output shape: {outputs['style'].shape}")
