"""
Matching engine for profile recommendations
"""

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from PIL import Image
import faiss
from pathlib import Path

from src.models.backbone import ProfileFeatureExtractor


class MatchingEngine:
    """
    Matching engine for profile recommendations using vector similarity
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        index_type: str = 'Flat'
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run model on
            index_type: Faiss index type ('Flat', 'IVF', 'HNSW')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = ProfileFeatureExtractor.load_from_checkpoint(
            model_path,
            device=self.device
        )
        self.model.eval()

        self.embedding_dim = self.model.embedding_dim

        # Initialize Faiss index
        self.index_type = index_type
        self.index = None
        self.user_ids = []  # Keep track of user IDs

        # User embeddings cache
        self.user_embeddings = {}

        # Personalization
        self.user_preferences = {}

    def _build_index(self, embeddings: np.ndarray):
        """
        Build Faiss index

        Args:
            embeddings: Embeddings array of shape (N, embedding_dim)
        """
        if self.index_type == 'Flat':
            # Exact search using inner product (cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        elif self.index_type == 'IVF':
            # Approximate search with inverted file index
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

            # Train index
            self.index.train(embeddings)

        elif self.index_type == 'HNSW':
            # Hierarchical navigable small world graph
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Add embeddings to index
        self.index.add(embeddings)

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from image

        Args:
            image_path: Path to image

        Returns:
            embedding: Feature vector
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))

        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            embedding = self.model(image_tensor)

        embedding = embedding.cpu().numpy()[0]

        # Normalize for inner product (cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def add_user(self, user_id: str, image_path: str):
        """
        Add a user to the matching database

        Args:
            user_id: Unique user ID
            image_path: Path to user's profile image
        """
        # Extract features
        embedding = self.extract_features(image_path)

        # Store
        self.user_embeddings[user_id] = embedding
        self.user_ids.append(user_id)

    def build_index_from_users(self):
        """Build Faiss index from all added users"""
        if len(self.user_embeddings) == 0:
            raise ValueError("No users added. Call add_user() first.")

        # Stack embeddings
        embeddings = np.stack([self.user_embeddings[uid] for uid in self.user_ids])

        # Build index
        self._build_index(embeddings.astype('float32'))

        print(f"Built index with {len(self.user_ids)} users")

    def find_matches(
        self,
        user_id: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Find matching profiles for a user

        Args:
            user_id: User ID to find matches for
            top_k: Number of matches to return
            min_similarity: Minimum similarity threshold

        Returns:
            matches: List of (user_id, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index_from_users() first.")

        if user_id not in self.user_embeddings:
            raise ValueError(f"User {user_id} not found")

        # Get user embedding
        query_embedding = self.user_embeddings[user_id]

        # Apply personalization if available
        if user_id in self.user_preferences:
            pref_vector = self.user_preferences[user_id]
            query_embedding = 0.7 * query_embedding + 0.3 * pref_vector
            # Re-normalize
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search (k+1 to exclude self)
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, top_k + 1)

        # Process results
        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            matched_user_id = self.user_ids[idx]

            # Skip self
            if matched_user_id == user_id:
                continue

            # Check minimum similarity
            if dist >= min_similarity:
                matches.append((matched_user_id, float(dist)))

            if len(matches) >= top_k:
                break

        return matches

    def update_preferences(
        self,
        user_id: str,
        liked_users: List[str],
        passed_users: Optional[List[str]] = None
    ):
        """
        Update user preferences based on their interactions

        Args:
            user_id: User ID
            liked_users: List of user IDs they liked
            passed_users: List of user IDs they passed on
        """
        # Get embeddings of liked profiles
        liked_embeddings = [
            self.user_embeddings[uid]
            for uid in liked_users
            if uid in self.user_embeddings
        ]

        if len(liked_embeddings) == 0:
            return

        # Compute preference vector as mean of liked profiles
        preference_vector = np.mean(liked_embeddings, axis=0)

        # Normalize
        preference_vector = preference_vector / np.linalg.norm(preference_vector)

        # Store
        self.user_preferences[user_id] = preference_vector

        print(f"Updated preferences for user {user_id} based on {len(liked_embeddings)} likes")

    def save_index(self, index_path: str, metadata_path: str):
        """
        Save Faiss index and metadata

        Args:
            index_path: Path to save Faiss index
            metadata_path: Path to save metadata (user IDs)
        """
        # Save Faiss index
        faiss.write_index(self.index, index_path)

        # Save metadata
        metadata = {
            'user_ids': self.user_ids,
            'user_embeddings': self.user_embeddings,
            'user_preferences': self.user_preferences
        }

        np.save(metadata_path, metadata, allow_pickle=True)

        print(f"Saved index to {index_path}")
        print(f"Saved metadata to {metadata_path}")

    def load_index(self, index_path: str, metadata_path: str):
        """
        Load Faiss index and metadata

        Args:
            index_path: Path to Faiss index
            metadata_path: Path to metadata
        """
        # Load Faiss index
        self.index = faiss.read_index(index_path)

        # Load metadata
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.user_ids = metadata['user_ids']
        self.user_embeddings = metadata['user_embeddings']
        self.user_preferences = metadata.get('user_preferences', {})

        print(f"Loaded index with {len(self.user_ids)} users")


if __name__ == "__main__":
    # Example usage
    print("Testing MatchingEngine...")

    # Note: Requires actual model and images to run
    # Uncomment and modify paths to test

    # model_path = "models/saved_models/best_model.pth"
    # matcher = MatchingEngine(model_path, device='cuda')

    # # Add users
    # matcher.add_user('user_001', 'data/processed/user_001.jpg')
    # matcher.add_user('user_002', 'data/processed/user_002.jpg')
    # matcher.add_user('user_003', 'data/processed/user_003.jpg')

    # # Build index
    # matcher.build_index_from_users()

    # # Find matches
    # matches = matcher.find_matches('user_001', top_k=5)
    # print(f"\nMatches for user_001:")
    # for user_id, score in matches:
    #     print(f"  {user_id}: {score:.4f}")

    # # Update preferences
    # matcher.update_preferences('user_001', liked_users=['user_002'])

    # # Find matches again (personalized)
    # matches = matcher.find_matches('user_001', top_k=5)
    # print(f"\nPersonalized matches for user_001:")
    # for user_id, score in matches:
    #     print(f"  {user_id}: {score:.4f}")
