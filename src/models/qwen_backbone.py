"""
Qwen3-VL Vision-Language Model Integration for Profile Feature Extraction
"""

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen3VLFeatureExtractor(nn.Module):
    """Feature extractor using Qwen3-VL model.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        embedding_dim=512,
        freeze_vision_encoder=False,
        use_projection_head=True,
        device="cuda"
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Target embedding dimension
            freeze_vision_encoder: Whether to freeze the vision encoder
            use_projection_head: Whether to add a projection head
            device: Device to load model on
        """
        super(Qwen3VLFeatureExtractor, self).__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device

        print(f"Loading Qwen3-VL model: {model_name}")

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Hidden size will be determined lazily from the first forward pass
        # by inspecting the hidden state tensor shape. Different Qwen3-VL
        # configs may not expose a standard hidden_size field.
        self.vision_hidden_size = None

        # Freeze vision encoder if requested
        if freeze_vision_encoder:
            self.freeze_vision_model()

        # Projection head will be lazily initialized once we know the
        # actual hidden size from a forward pass.
        self.use_projection_head = use_projection_head
        self.projection_head = None

        self.normalize = True

    def freeze_vision_model(self):
        """Freeze vision encoder parameters"""
        visual_module = None
        for attr in ("visual", "vision_tower", "vision_model"):
            if hasattr(self.model, attr):
                visual_module = getattr(self.model, attr)
                break

        if visual_module is None:
            print("[WARN] Could not find vision submodule to freeze; skipping.")
            return

        for param in visual_module.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")

    def unfreeze_vision_model(self):
        """Unfreeze vision encoder parameters"""
        visual_module = None
        for attr in ("visual", "vision_tower", "vision_model"):
            if hasattr(self.model, attr):
                visual_module = getattr(self.model, attr)
                break

        if visual_module is None:
            print("[WARN] Could not find vision submodule to unfreeze; skipping.")
            return

        for param in visual_module.parameters():
            param.requires_grad = True
        print("Vision encoder unfrozen")

    def _ensure_projection_head(self, hidden_dim: int) -> None:
        """Lazily initialize projection head based on hidden dimension."""
        if self.projection_head is not None:
            return

        self.vision_hidden_size = hidden_dim

        if self.use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(self.vision_hidden_size, self.embedding_dim * 2),
                nn.LayerNorm(self.embedding_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            )
        else:
            self.projection_head = nn.Identity()
            self.embedding_dim = self.vision_hidden_size

    def extract_vision_features(self, inputs):
        """
        Extract features from vision encoder using the full model

        Args:
            inputs: Processed inputs from processor

        Returns:
            vision_features: Extracted visual features
        """
        # Use the full model forward pass to get hidden states
        # This is more reliable than calling vision encoder directly
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

        # Get vision features from hidden states
        # For Qwen2-VL, we use the last hidden state and pool it
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

        # Mean pooling over sequence length
        # Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        pooled_features = hidden_states.mean(dim=1)

        # Initialize projection head based on hidden dimension
        hidden_dim = pooled_features.shape[-1]
        self._ensure_projection_head(hidden_dim)

        return pooled_features

    def forward(self, images):
        """
        Forward pass for feature extraction

        Args:
            images: PIL Images or image tensors (already preprocessed)
                   If PIL Images: list of PIL.Image objects
                   If tensors: [batch_size, 3, H, W]

        Returns:
            embeddings: Feature embeddings [batch_size, embedding_dim]
        """
        # Process images through Qwen3-VL processor
        if isinstance(images, list):
            # PIL Images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this person's appearance."}
                    ]
                }
                for img in images
            ]

            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

        else:
            # Already processed tensors
            # For batch processing, we need to use the processor
            # This is a simplified version - adjust based on actual usage
            inputs = {
                'pixel_values': images.to(self.device),
                'image_grid_thw': torch.tensor([[1, images.shape[2] // 14, images.shape[3] // 14]]).to(self.device)
            }

        # Extract vision features
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            vision_features = self.extract_vision_features(inputs)

        # Project to embedding space
        embeddings = self.projection_head(vision_features)

        # L2 normalization
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_embedding(self, image_path):
        """
        Get embedding for a single image

        Args:
            image_path: Path to image file

        Returns:
            embedding: Feature vector
        """
        from PIL import Image

        image = Image.open(image_path).convert('RGB')

        with torch.no_grad():
            embedding = self.forward([image])

        return embedding[0].cpu().numpy()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device='cuda'):
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load on

        Returns:
            model: Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        config = checkpoint.get('config', {})

        model = cls(
            model_name=config.get('model_name', 'Qwen/Qwen3-VL-2B-Instruct-FP8'),
            embedding_dim=config.get('embedding_dim', 512),
            freeze_vision_encoder=False,
            device=device
        )

        # Load projection head weights (vision model already loaded from HF)
        if 'projection_head_state_dict' in checkpoint:
            model.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Load full state dict
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        model.eval()
        return model

    def save_checkpoint(self, path, optimizer=None, epoch=None, **kwargs):
        """
        Save model checkpoint

        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state
            epoch: Current epoch
            **kwargs: Additional metadata
        """
        checkpoint = {
            'config': {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'vision_hidden_size': self.vision_hidden_size,
            },
            'projection_head_state_dict': self.projection_head.state_dict(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


class Qwen3VLWithTextFeatureExtractor(Qwen3VLFeatureExtractor):
    """
    Extended version that can use both visual and text features
    for multimodal profile matching
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional projection for text features (lazy init)
        self.text_projection = None

    def forward_with_text(self, images, text_descriptions):
        """
        Extract features using both image and text

        Args:
            images: List of PIL Images
            text_descriptions: List of text descriptions

        Returns:
            embeddings: Combined embeddings
        """
        # Prepare multimodal inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": desc}
                ]
            }
            for img, desc in zip(images, text_descriptions)
        ]

        # Process through model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Forward pass
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

            # Get features from hidden states (already includes both vision and text)
            # For multimodal, we use the combined representation
            hidden_states = outputs.hidden_states[-1]  # Last layer
            pooled_features = hidden_states.mean(dim=1)

            # Split into vision and text features
            # This is simplified - in practice, you might want to separate them differently
            vision_features = pooled_features
            text_features = pooled_features

        # Ensure projection head is initialized based on hidden dimension
        hidden_dim = vision_features.shape[-1]
        self._ensure_projection_head(hidden_dim)

        # Lazily initialize text projection if needed
        if self.text_projection is None:
            self.text_projection = nn.Sequential(
                nn.Linear(self.vision_hidden_size, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.GELU()
            )

        # Project features
        vision_emb = self.projection_head(vision_features)
        text_emb = self.text_projection(text_features)

        # Combine (weighted average)
        embeddings = 0.7 * vision_emb + 0.3 * text_emb

        # Normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings


if __name__ == "__main__":
    # Test the model
    print("Testing Qwen3VLFeatureExtractor...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen3VLFeatureExtractor(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        embedding_dim=512,
        device=device
    )

    print(f"Model loaded on {device}")
    print(f"Embedding dimension: {model.embedding_dim}")
    print(f"Vision hidden size: {model.vision_hidden_size}")

    # Test with dummy image
    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    with torch.no_grad():
        embedding = model.forward([dummy_image])

    print(f"Output embedding shape: {embedding.shape}")
    print(f"Embedding norm: {embedding.norm(dim=1).item():.4f}")
