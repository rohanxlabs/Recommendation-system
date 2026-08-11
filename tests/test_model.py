"""
Tests for model training and recommendation module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import train_model
from src.models.recommend import recommend_items
from sklearn.decomposition import TruncatedSVD


def test_train_model_returns_correct_types():
    """Test that train_model returns SVD model and latent matrix."""
    # Create a simple user-item matrix
    matrix = pd.DataFrame({
        101: [5, 4, 0],
        102: [4, 0, 3],
        103: [0, 5, 4]
    }, index=[1, 2, 3])
    
    model, latent_matrix = train_model(matrix)
    
    assert isinstance(model, TruncatedSVD)
    assert isinstance(latent_matrix, np.ndarray)
    assert latent_matrix.shape[0] == 3  # 3 users


def test_train_model_component_size():
    """Test that n_components is set correctly based on matrix size."""
    # Small matrix: 3 users x 3 items
    matrix = pd.DataFrame(np.random.rand(3, 3))
    model, _ = train_model(matrix)
    
    # n_components should be min(shape) - 1 = 2
    assert model.n_components == 2


def test_recommend_items_returns_correct_length():
    """Test that recommend_items returns requested number of items."""
    matrix = pd.DataFrame({
        101: [5, 4, 0],
        102: [4, 0, 3],
        103: [0, 5, 4],
        104: [3, 3, 2],
        105: [2, 1, 5]
    }, index=[1, 2, 3])
    
    model, _ = train_model(matrix)
    
    items, scores = recommend_items(1, matrix, model, n=3)
    
    assert len(items) == 3
    assert len(scores) == 3


def test_recommend_items_returns_valid_items():
    """Test that recommended items are from the catalog."""
    matrix = pd.DataFrame({
        101: [5, 4, 0],
        102: [4, 0, 3],
        103: [0, 5, 4]
    }, index=[1, 2, 3])
    
    model, _ = train_model(matrix)
    items, scores = recommend_items(1, matrix, model, n=2)
    
    # All recommended items should be in the column index
    assert all(item in matrix.columns for item in items)


def test_recommend_items_excludes_rated():
    """Test that exclude_rated parameter works correctly."""
    # Create a larger matrix for better SVD decomposition
    matrix = pd.DataFrame({
        101: [5, 1, 0, 0, 0, 0, 0, 0],
        102: [4, 0, 0, 0, 0, 0, 0, 0],
        103: [0, 5, 0, 0, 0, 0, 0, 0],
        104: [0, 4, 0, 0, 0, 0, 0, 0],
        105: [0, 0, 5, 0, 0, 0, 0, 0],
        106: [0, 0, 4, 0, 0, 0, 0, 0],
        107: [0, 0, 0, 5, 4, 3, 2, 1],
        108: [0, 0, 0, 4, 5, 2, 3, 1]
    }, index=[1, 2, 3, 4, 5, 6, 7, 8])
    
    model, _ = train_model(matrix)
    
    # User 1 has rated items 101 and 102
    items_excluded, scores_excluded = recommend_items(1, matrix, model, n=6, exclude_rated=True)
    items_included, scores_included = recommend_items(1, matrix, model, n=6, exclude_rated=False)
    
    # When exclude_rated=True, should not recommend already-rated items
    assert 101 not in items_excluded
    assert 102 not in items_excluded
    
    # When exclude_rated=False, might include rated items
    # (This is expected behavior - we're testing the exclusion works)


def test_recommend_items_scores_descending():
    """Test that recommendations are sorted by score descending."""
    matrix = pd.DataFrame({
        101: [5, 4, 0],
        102: [4, 0, 3],
        103: [0, 5, 4],
        104: [3, 3, 2]
    }, index=[1, 2, 3])
    
    model, _ = train_model(matrix)
    items, scores = recommend_items(1, matrix, model, n=4, exclude_rated=False)
    
    # Scores should be in descending order
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


def test_recommend_items_with_small_n():
    """Test edge case with n=1."""
    matrix = pd.DataFrame({
        101: [5, 4, 0],
        102: [4, 0, 3],
        103: [0, 5, 4]
    }, index=[1, 2, 3])
    
    model, _ = train_model(matrix)
    items, scores = recommend_items(1, matrix, model, n=1)
    
    assert len(items) == 1
    assert len(scores) == 1


def test_recommend_items_invalid_n():
    """Test that invalid n raises error."""
    matrix = pd.DataFrame({
        101: [5, 4],
        102: [4, 3]
    }, index=[1, 2])
    
    model, _ = train_model(matrix)
    
    with pytest.raises(ValueError):
        recommend_items(1, matrix, model, n=0)
    
    with pytest.raises(ValueError):
        recommend_items(1, matrix, model, n=-1)


def test_recommend_items_caps_at_available():
    """Test that n is capped at number of available items."""
    matrix = pd.DataFrame({
        101: [5, 0],
        102: [4, 0],
        103: [3, 0]
    }, index=[1, 2])
    
    model, _ = train_model(matrix)
    
    # Request more items than available
    items, scores = recommend_items(1, matrix, model, n=100)
    
    # Should return only available items
    assert len(items) <= 3
