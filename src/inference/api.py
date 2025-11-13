"""
FastAPI server for profile matching inference
"""

import os
import io
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image

from src.inference.matcher import MatchingEngine


# Initialize FastAPI app
app = FastAPI(
    title="Dating Profile Matcher API",
    description="API for profile feature extraction and matching",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global matching engine (loaded on startup)
matcher: Optional[MatchingEngine] = None


# Pydantic models
class EmbeddingResponse(BaseModel):
    user_id: Optional[str] = None
    embedding: List[float]
    embedding_dim: int


class Match(BaseModel):
    user_id: str
    similarity: float


class MatchResponse(BaseModel):
    query_user_id: str
    matches: List[Match]
    total: int


class PreferenceUpdate(BaseModel):
    user_id: str
    liked_users: List[str]
    passed_users: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_users: int


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global matcher

    # Load configuration
    model_path = os.getenv('MODEL_PATH', 'models/saved_models/best_model.pth')
    device = os.getenv('DEVICE', 'cuda')
    index_type = os.getenv('INDEX_TYPE', 'Flat')

    print(f"Loading model from: {model_path}")

    try:
        matcher = MatchingEngine(
            model_path=model_path,
            device=device,
            index_type=index_type
        )

        # Load existing index if available
        index_path = os.getenv('INDEX_PATH', 'models/faiss_index.bin')
        metadata_path = os.getenv('METADATA_PATH', 'models/index_metadata.npy')

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            matcher.load_index(index_path, metadata_path)
            print(f"Loaded existing index with {len(matcher.user_ids)} users")
        else:
            print("No existing index found. Will build on-the-fly.")

    except Exception as e:
        print(f"Error loading model: {e}")
        matcher = None


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if matcher is not None else "unhealthy",
        model_loaded=matcher is not None,
        num_users=len(matcher.user_ids) if matcher else 0
    )


@app.post("/extract_features", response_model=EmbeddingResponse)
async def extract_features(
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(None, description="Optional user ID")
):
    """
    Extract features from uploaded image

    Args:
        file: Image file
        user_id: Optional user ID

    Returns:
        Embedding vector
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Save temporarily
        temp_path = f"/tmp/{file.filename}"
        image.save(temp_path)

        # Extract features
        embedding = matcher.extract_features(temp_path)

        # Clean up
        os.remove(temp_path)

        return EmbeddingResponse(
            user_id=user_id,
            embedding=embedding.tolist(),
            embedding_dim=len(embedding)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/add_user")
async def add_user(
    user_id: str = Query(..., description="User ID"),
    file: UploadFile = File(..., description="Profile image")
):
    """
    Add a user to the matching database

    Args:
        user_id: Unique user ID
        file: Profile image

    Returns:
        Success message
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Save temporarily
        temp_path = f"/tmp/{user_id}_{file.filename}"
        image.save(temp_path)

        # Add user
        matcher.add_user(user_id, temp_path)

        # Rebuild index
        matcher.build_index_from_users()

        # Clean up
        os.remove(temp_path)

        return {
            "status": "success",
            "message": f"Added user {user_id}",
            "total_users": len(matcher.user_ids)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding user: {str(e)}")


@app.get("/matches/{user_id}", response_model=MatchResponse)
async def get_matches(
    user_id: str,
    top_k: int = Query(10, ge=1, le=100, description="Number of matches to return"),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """
    Get matching recommendations for a user

    Args:
        user_id: User ID
        top_k: Number of matches to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of matches
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if user_id not in matcher.user_embeddings:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        # Find matches
        matches = matcher.find_matches(user_id, top_k, min_similarity)

        # Format response
        match_list = [
            Match(user_id=uid, similarity=score)
            for uid, score in matches
        ]

        return MatchResponse(
            query_user_id=user_id,
            matches=match_list,
            total=len(match_list)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding matches: {str(e)}")


@app.post("/update_preferences")
async def update_preferences(preference: PreferenceUpdate):
    """
    Update user preferences based on their interactions

    Args:
        preference: Preference update data

    Returns:
        Success message
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        matcher.update_preferences(
            user_id=preference.user_id,
            liked_users=preference.liked_users,
            passed_users=preference.passed_users
        )

        return {
            "status": "success",
            "message": f"Updated preferences for user {preference.user_id}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating preferences: {str(e)}")


@app.post("/save_index")
async def save_index(
    index_path: str = Query("models/faiss_index.bin"),
    metadata_path: str = Query("models/index_metadata.npy")
):
    """
    Save current Faiss index to disk

    Args:
        index_path: Path to save index
        metadata_path: Path to save metadata

    Returns:
        Success message
    """
    if matcher is None or matcher.index is None:
        raise HTTPException(status_code=503, detail="Model or index not loaded")

    try:
        matcher.save_index(index_path, metadata_path)

        return {
            "status": "success",
            "message": "Index saved successfully",
            "index_path": index_path,
            "metadata_path": metadata_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving index: {str(e)}")


@app.get("/users")
async def list_users():
    """
    Get list of all users in the database

    Returns:
        List of user IDs
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "users": matcher.user_ids,
        "total": len(matcher.user_ids)
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
