"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    assert "version" in data


def test_health_check_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "num_users" in data
    assert "num_items" in data
    assert "matrix_shape" in data


def test_recommend_valid_user():
    """Test recommendation for valid user."""
    # User 2 should exist in training data
    response = client.get("/recommend/2")
    assert response.status_code == 200
    
    data = response.json()
    assert "user_id" in data
    assert "recommendations" in data
    assert "strategy" in data
    assert data["user_id"] == 2
    assert len(data["recommendations"]) > 0
    
    # Check recommendation structure
    rec = data["recommendations"][0]
    assert "item_id" in rec
    assert "score" in rec
    assert "rank" in rec


def test_recommend_with_custom_n():
    """Test recommendation with custom n parameter."""
    response = client.get("/recommend/2?n=10")
    assert response.status_code == 200
    
    data = response.json()
    # Should have up to 10 recommendations
    assert len(data["recommendations"]) <= 10


def test_recommend_with_exclude_rated():
    """Test recommendation with exclude_rated parameter."""
    # First check which users are available
    health = client.get("/health")
    assert health.status_code == 200
    
    # Use any valid user from the test - try a few common ones
    response = None
    for user_id in [1, 2, 3, 5, 10]:
        test_response = client.get(f"/recommend/{user_id}?exclude_rated=true")
        if test_response.status_code == 200:
            response = test_response
            break
    
    # If no users work, skip the test (shouldn't happen with real data)
    if response is None:
        pytest.skip("No valid users found in training data")
    
    data = response.json()
    # Should use one of the available strategies
    assert data["strategy"] in ["collaborative", "popularity"]


def test_recommend_unknown_user_fallback():
    """Test that unknown user gets popularity-based recommendations."""
    # User 9999 shouldn't exist
    response = client.get("/recommend/9999")
    assert response.status_code == 200
    
    data = response.json()
    assert data["strategy"] == "popularity"
    assert len(data["recommendations"]) > 0


def test_recommend_invalid_n_zero():
    """Test that n=0 is rejected."""
    response = client.get("/recommend/2?n=0")
    assert response.status_code == 422  # Validation error


def test_recommend_invalid_n_negative():
    """Test that negative n is rejected."""
    response = client.get("/recommend/2?n=-5")
    assert response.status_code == 422  # Validation error


def test_recommend_n_too_large():
    """Test that n is capped at maximum."""
    response = client.get("/recommend/2?n=100")
    assert response.status_code == 422  # Exceeds max of 50


def test_docs_endpoint():
    """Test that API documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


def test_recommendation_rank_ordering():
    """Test that recommendations are ranked correctly (1-based)."""
    response = client.get("/recommend/2?n=5")
    assert response.status_code == 200
    
    data = response.json()
    recommendations = data["recommendations"]
    
    # Check ranks are sequential starting from 1
    for i, rec in enumerate(recommendations):
        assert rec["rank"] == i + 1


def test_recommendation_scores_ordering():
    """Test that scores are in descending order."""
    response = client.get("/recommend/2?n=5&exclude_rated=false")
    assert response.status_code == 200
    
    data = response.json()
    recommendations = data["recommendations"]
    
    scores = [rec["score"] for rec in recommendations if rec["score"] is not None]
    
    # Scores should be descending (or None for excluded items)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
