"""
Qwen3-VL Vision-Language Model Integration for Profile Feature Extraction
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


# ============================================================================
# 프롬프트 설정
# ============================================================================

# 성별별 시스템 프롬프트
SYSTEM_PROMPTS = {
    "female": """You are a dating compatibility analyst.
Analyze this woman's visual features that attract male partners.
Focus on feminine charm, style, and overall appeal.
Ignore background and irrelevant objects.""",
    
    "male": """You are a dating compatibility analyst.
Analyze this man's visual features that attract female partners.
Focus on masculine appeal, style, and overall charm.
Ignore background and irrelevant objects.""",
    
    # 성별 불명시 기본값
    "default": """You are a dating compatibility analyst.
Focus on visual features that predict romantic matching success.
Ignore background and irrelevant objects."""
}

# 성별별 유저 프롬프트 템플릿
USER_PROMPT_TEMPLATES = {
    "female": """Dating profile (Woman):
- Appearance: {appearance_type}
- Style: {style_vibe}
- Personality: {personality_impression}
- Grooming: {grooming_level}
- Features: {physical_features_str}

Analyze her romantic appeal.""",
    
    "male": """Dating profile (Man):
- Appearance: {appearance_type}
- Style: {style_vibe}
- Personality: {personality_impression}
- Grooming: {grooming_level}
- Features: {physical_features_str}

Analyze his romantic appeal.""",
    
    # 성별 불명시 기본값
    "default": """Dating profile:
- Appearance: {appearance_type}
- Style: {style_vibe}
- Personality: {personality_impression}
- Grooming: {grooming_level}
- Features: {physical_features_str}

Analyze for romantic compatibility."""
}

# 메타데이터 없을 때 사용하는 기본 프롬프트
DEFAULT_USER_PROMPTS = {
    "female": "Dating profile photo of a woman. Analyze her romantic appeal.",
    "male": "Dating profile photo of a man. Analyze his romantic appeal.",
    "default": "Dating profile photo. Analyze for romantic compatibility."
}


class Qwen3VLFeatureExtractor(nn.Module):
    """Feature extractor using Qwen3-VL model.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        embedding_dim=512,
        freeze_vision_encoder=False,
        use_projection_head=True,
        pooling_mode="mean",
        device="cuda"
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Target embedding dimension
            freeze_vision_encoder: Whether to freeze the vision encoder
            use_projection_head: Whether to add a projection head
            pooling_mode: 'mean' for mean pooling, 'eos' for last token (EOS)
            device: Device to load model on
        """
        super(Qwen3VLFeatureExtractor, self).__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pooling_mode = pooling_mode  # 'mean' or 'eos'
        self.device = device

        print(f"Loading Qwen3-VL model: {model_name}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
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
            # Keep projection head in float32 for numerical stability
            # 3-layer MLP with wider hidden dims to prevent embedding collapse
            hidden_dim = max(self.embedding_dim * 4, 2048)
            self.projection_head = nn.Sequential(
                nn.Linear(self.vision_hidden_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.embedding_dim),
            ).to(self.device).float()
        else:
            self.projection_head = nn.Identity().to(self.device)
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
        # Shape: [batch_size, seq_len, hidden_size]

        # Pooling strategy
        if self.pooling_mode == "eos":
            # EOS Pooling: Use the last token (typically EOS/end token)
            # Shape: [batch_size, hidden_size]
            pooled_features = hidden_states[:, -1, :]
        else:
            # Mean Pooling: Average over sequence length (default)
            # Shape: [batch_size, hidden_size]
            pooled_features = hidden_states.mean(dim=1)

        # Work in float32 for the projection head to avoid
        # dtype mismatches with the half-precision backbone.
        pooled_features = pooled_features.to(torch.float32)

        # Initialize projection head based on hidden dimension
        hidden_dim = pooled_features.shape[-1]
        self._ensure_projection_head(hidden_dim)

        return pooled_features

    def _build_user_prompt(self, metadata: Optional[Dict] = None) -> str:
        """
        메타데이터를 기반으로 유저 프롬프트 생성
        
        Args:
            metadata: 메타데이터 딕셔너리
                - gender: str ('female', 'male')
                - appearance_type: str
                - style_vibe: str
                - personality_impression: str
                - grooming_level: str
                - physical_features: List[str]
        
        Returns:
            user_prompt: 포맷팅된 유저 프롬프트 문자열
        """
        # 성별 결정 (기본값: 'default')
        gender = metadata.get('gender', 'default') if metadata else 'default'
        if gender not in USER_PROMPT_TEMPLATES:
            gender = 'default'
        
        if metadata is None:
            return DEFAULT_USER_PROMPTS.get(gender, DEFAULT_USER_PROMPTS['default'])
        
        # physical_features 리스트를 문자열로 변환
        physical_features = metadata.get('physical_features', [])
        if isinstance(physical_features, list):
            physical_features_str = ', '.join(physical_features)
        else:
            physical_features_str = str(physical_features)
        
        return USER_PROMPT_TEMPLATES[gender].format(
            appearance_type=metadata.get('appearance_type', 'Unknown'),
            style_vibe=metadata.get('style_vibe', 'Unknown'),
            personality_impression=metadata.get('personality_impression', 'Unknown'),
            grooming_level=metadata.get('grooming_level', 'Unknown'),
            physical_features_str=physical_features_str
        )
    
    def _get_system_prompt(self, metadata: Optional[Dict] = None) -> str:
        """
        성별에 따른 시스템 프롬프트 반환
        
        Args:
            metadata: 메타데이터 딕셔너리 (gender 필드 포함)
        
        Returns:
            system_prompt: 성별에 맞는 시스템 프롬프트
        """
        gender = metadata.get('gender', 'default') if metadata else 'default'
        if gender not in SYSTEM_PROMPTS:
            gender = 'default'
        return SYSTEM_PROMPTS[gender]

    def forward(self, images, metadata: Optional[List[Dict]] = None):
        """
        Forward pass for feature extraction

        Args:
            images: PIL Images or image tensors (already preprocessed)
                   If PIL Images: list of PIL.Image objects
                   If tensors: [batch_size, 3, H, W]
            metadata: Optional list of metadata dictionaries (one per image)
                   Each dict should contain:
                   - appearance_type, style_vibe, personality_impression
                   - grooming_level, physical_features

        Returns:
            embeddings: Feature embeddings [batch_size, embedding_dim]
        """
        # Process images through Qwen3-VL processor
        if isinstance(images, list):
            # PIL Images
            # Create a list of conversations (one per image) to enable batch processing
            conversations = []
            for i, img in enumerate(images):
                # 메타데이터가 있으면 해당 인덱스의 메타데이터 사용
                meta = metadata[i] if metadata and i < len(metadata) else None
                user_prompt = self._build_user_prompt(meta)
                system_prompt = self._get_system_prompt(meta)
                
                conversation = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
                conversations.append(conversation)

            # Prepare inputs for batch
            texts = [
                self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
                for conv in conversations
            ]

            # process_vision_info expects a list of messages (conversations) for batching?
            # Or we need to flatten/handle carefully. 
            # Qwen-VL utils usually handle a list of messages (single convo).
            # For batching, we typically pass the list of conversations if supported, 
            # or we need to verify qwen_vl_utils behavior.
            # Assuming standard Qwen2-VL/Qwen3-VL pattern:
            image_inputs, video_inputs = process_vision_info(conversations)

            inputs = self.processor(
                text=texts,
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
        # Extract vision features
        # Autocast removed to prevent "Unexpected floating ScalarType" error
        # Ensure floating point inputs are cast to model's dtype (float16)
        if 'pixel_values' in inputs and inputs['pixel_values'].dtype != self.model.dtype:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.model.dtype)

        vision_features = self.extract_vision_features(inputs)

        # Ensure features are on the same device as projection head
        vision_features = vision_features.to(self.device)

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

        # Projection head is now lazily initialized based on hidden size.
        # To load its weights, we must ensure it's created first.
        if 'projection_head_state_dict' in checkpoint:
            # Trigger lazy init with the stored hidden size if available,
            # otherwise run a tiny dummy forward once.
            vision_hidden_size = config.get('vision_hidden_size')
            if vision_hidden_size is not None and model.projection_head is None:
                model._ensure_projection_head(int(vision_hidden_size))

            if model.projection_head is None:
                # As a fallback, run a minimal dummy forward to infer hidden size
                from PIL import Image
                import numpy as np

                dummy_image = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )

                with torch.no_grad():
                    _ = model.forward([dummy_image])

            model.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])

        elif 'model_state_dict' in checkpoint:
            # Load full state dict (backbone + projection)
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
        # Prepare multimodal inputs - each sample as a separate conversation
        # conversations is a list of lists (batch of conversations)
        conversations = [
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": desc}
                    ]
                }
            ]
            for img, desc in zip(images, text_descriptions)
        ]

        # Process through model - apply chat template to each conversation
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversations
        ]

        image_inputs, video_inputs = process_vision_info(conversations)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Forward pass
        # Forward pass
        # Ensure inputs are in correct dtype
        if 'pixel_values' in inputs and inputs['pixel_values'].dtype != self.model.dtype:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.model.dtype)

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        # Get features from hidden states (already includes both vision and text)
        # For multimodal, we use the combined representation
        hidden_states = outputs.hidden_states[-1]  # Last layer
        pooled_features = hidden_states.mean(dim=1)

        # Split into vision and text features
        # This is simplified - in practice, you might want to separate them differently
        # Ensure float32 for projection head consistency
        vision_features = pooled_features.to(dtype=torch.float32)
        text_features = pooled_features.to(dtype=torch.float32)

        # Ensure projection head is initialized based on hidden dimension
        hidden_dim = vision_features.shape[-1]
        self._ensure_projection_head(hidden_dim)

        # Lazily initialize text projection if needed
        if self.text_projection is None:
            # Text projection also uses float32, matching vision projection structure
            text_hidden_dim = max(self.embedding_dim * 4, 2048)
            self.text_projection = nn.Sequential(
                nn.Linear(self.vision_hidden_size, text_hidden_dim),
                nn.LayerNorm(text_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(text_hidden_dim, text_hidden_dim // 2),
                nn.LayerNorm(text_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(text_hidden_dim // 2, self.embedding_dim),
            ).to(self.device).float()

        # Move to correct device before projection
        vision_features = vision_features.to(self.device)
        text_features = text_features.to(self.device)

        # Project features
        vision_emb = self.projection_head(vision_features)
        text_emb = self.text_projection(text_features)

        # Combine (weighted average)
        embeddings = 0.7 * vision_emb + 0.3 * text_emb

        # Normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-8)

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
