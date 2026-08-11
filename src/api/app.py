from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle 
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import using absolute import that works when run as module
try:
    from src.models.recommend import recommend_items
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.models.recommend import recommend_items

app = FastAPI(
    title="Collaborative Filtering Recommendation API",
    description="Matrix factorization-based recommendation system using TruncatedSVD",
    version="1.0.0"
)

# Response models
class RecommendationItem(BaseModel):
    item_id: int = Field(..., description="ID of the recommended item")
    score: float = Field(..., description="Predicted preference score")
    rank: int = Field(..., description="Rank in recommendation list (1-based)")

class RecommendResponse(BaseModel):
    user_id: int = Field(..., description="User ID for whom recommendations were generated")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")
    strategy: str = Field(..., description="Recommendation strategy used (collaborative/popularity)")
    total_items: int = Field(..., description="Total number of items in catalog")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_users: int
    num_items: int
    matrix_shape: tuple

# Load model and data on startup
try:
    model_path = Path("artifacts/model.pkl")
    data_path = Path("data/processed/train.csv")
    
    logger.info(f"Loading model from {model_path}")
    model = pickle.load(open(model_path, "rb"))
    
    logger.info(f"Loading training data from {data_path}")
    data = pd.read_csv(data_path)
    matrix = data.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)
    
    logger.info(f"Model loaded successfully. Matrix shape: {matrix.shape}")
    logger.info(f"Users: {len(matrix.index)}, Items: {len(matrix.columns)}")
    
except Exception as e:
    logger.error(f"Failed to load model or data: {e}")
    model = None
    matrix = None
    data = None

@app.get("/", tags=["General"])
def root():
    """Root endpoint with API information"""
    return {
        "message": "Collaborative Filtering Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "recommend": "/recommend/{user_id}"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint"""
    if model is None or matrix is None:
        raise HTTPException(status_code=503, detail="Service not ready - model not loaded")
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        num_users=len(matrix.index),
        num_items=len(matrix.columns),
        matrix_shape=matrix.shape
    )

@app.get("/recommend/{user_id}", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(
    user_id: int,
    n: int = Query(default=5, ge=1, le=50, description="Number of recommendations to return"),
    exclude_rated: bool = Query(default=True, description="Exclude already-rated items")
):
    """
    Get top-N recommendations for a user
    
    - **user_id**: User ID to get recommendations for
    - **n**: Number of recommendations (default: 5, max: 50)
    - **exclude_rated**: Whether to exclude already-rated items (default: true)
    
    Returns personalized recommendations using collaborative filtering (SVD).
    Falls back to popularity-based recommendations for unknown users.
    """
    if model is None or matrix is None:
        raise HTTPException(status_code=503, detail="Service not ready - model not loaded")
    
    # Check if user exists in training set
    if user_id not in matrix.index:
        logger.warning(f"User {user_id} not found, using popularity-based fallback")
        
        # Fallback: popularity-based recommendations
        avg_ratings = matrix.mean(axis=0).sort_values(ascending=False)
        n_items = min(n, len(avg_ratings))
        top_items = avg_ratings.head(n_items)
        
        recommendations = [
            RecommendationItem(
                item_id=int(item_id),
                score=float(score),
                rank=i + 1
            )
            for i, (item_id, score) in enumerate(top_items.items())
        ]
        
        return RecommendResponse(
            user_id=user_id,
            recommendations=recommendations,
            strategy="popularity",
            total_items=len(matrix.columns)
        )
    
    # Collaborative filtering recommendations
    try:
        items, scores = recommend_items(
            user_id=user_id,
            user_item_matrix=matrix,
            model=model,
            n=n,
            exclude_rated=exclude_rated
        )
        
        recommendations = [
            RecommendationItem(
                item_id=int(item_id),
                score=float(score),
                rank=i + 1
            )
            for i, (item_id, score) in enumerate(zip(items, scores))
        ]
        
        return RecommendResponse(
            user_id=user_id,
            recommendations=recommendations,
            strategy="collaborative",
            total_items=len(matrix.columns)
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)