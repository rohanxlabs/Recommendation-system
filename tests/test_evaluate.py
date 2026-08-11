"""
Tests for evaluation metrics.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.evaluate import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    evaluate_recommendations
)


def test_precision_at_k_perfect():
    """Test precision with perfect recommendations."""
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 2, 3, 4, 5]
    
    assert precision_at_k(recommended, relevant, 5) == 1.0


def test_precision_at_k_half():
    """Test precision with half relevant."""
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 3, 5]
    
    # 3 out of 5 recommendations are relevant
    assert precision_at_k(recommended, relevant, 5) == 0.6


def test_precision_at_k_none():
    """Test precision with no relevant items."""
    recommended = [1, 2, 3]
    relevant = [4, 5, 6]
    
    assert precision_at_k(recommended, relevant, 3) == 0.0


def test_recall_at_k_perfect():
    """Test recall with all relevant items found."""
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 2, 3]
    
    assert recall_at_k(recommended, relevant, 5) == 1.0


def test_recall_at_k_partial():
    """Test recall with some relevant items found."""
    recommended = [1, 2, 6, 7, 8]
    relevant = [1, 2, 3, 4, 5]
    
    # Found 2 out of 5 relevant items
    assert recall_at_k(recommended, relevant, 5) == 0.4


def test_recall_at_k_none():
    """Test recall with no relevant items found."""
    recommended = [1, 2, 3]
    relevant = [4, 5, 6]
    
    assert recall_at_k(recommended, relevant, 3) == 0.0


def test_hit_rate_at_k_hit():
    """Test hit rate when at least one item is relevant."""
    recommended = [1, 2, 3, 4, 5]
    relevant = [3, 6, 7]
    
    assert hit_rate_at_k(recommended, relevant, 5) == 1.0


def test_hit_rate_at_k_miss():
    """Test hit rate with no relevant items."""
    recommended = [1, 2, 3]
    relevant = [4, 5, 6]
    
    assert hit_rate_at_k(recommended, relevant, 3) == 0.0


def test_ndcg_at_k_perfect():
    """Test NDCG with perfect ranking."""
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 2, 3, 4, 5]
    
    # Perfect ranking should give NDCG = 1.0
    assert ndcg_at_k(recommended, relevant, 5) == 1.0


def test_ndcg_at_k_worst():
    """Test NDCG with worst ranking (relevant items at end)."""
    recommended = [6, 7, 8, 9, 1]
    relevant = [1, 2, 3]
    
    # Only 1 relevant item at last position
    score = ndcg_at_k(recommended, relevant, 5)
    assert 0 < score < 1.0  # Not perfect, but not zero


def test_ndcg_at_k_none():
    """Test NDCG with no relevant items."""
    recommended = [1, 2, 3]
    relevant = [4, 5, 6]
    
    assert ndcg_at_k(recommended, relevant, 3) == 0.0


def test_ndcg_at_k_position_matters():
    """Test that NDCG rewards early relevant items."""
    recommended_early = [1, 2, 6, 7, 8]
    recommended_late = [6, 7, 8, 1, 2]
    relevant = [1, 2]
    
    score_early = ndcg_at_k(recommended_early, relevant, 5)
    score_late = ndcg_at_k(recommended_late, relevant, 5)
    
    # Early relevant items should have higher NDCG
    assert score_early > score_late


def test_evaluate_recommendations_multiple_users():
    """Test evaluation across multiple users."""
    user_recommendations = {
        1: [101, 102, 103],
        2: [104, 105, 106],
        3: [107, 108, 109]
    }
    
    user_relevant_items = {
        1: [101, 102],  # 2/3 precision @3
        2: [104],       # 1/3 precision @3
        3: [110]        # 0/3 precision @3
    }
    
    results = evaluate_recommendations(
        user_recommendations,
        user_relevant_items,
        k_values=[3]
    )
    
    assert len(results) == 1
    assert results.iloc[0]['K'] == 3
    
    # Average precision should be (2/3 + 1/3 + 0/3) / 3 = 1/3
    assert abs(results.iloc[0]['Precision@K'] - 1/3) < 0.01


def test_edge_case_empty_recommendations():
    """Test with empty recommendation list."""
    recommended = []
    relevant = [1, 2, 3]
    
    assert precision_at_k(recommended, relevant, 5) == 0.0
    assert recall_at_k(recommended, relevant, 5) == 0.0
    assert hit_rate_at_k(recommended, relevant, 5) == 0.0
    assert ndcg_at_k(recommended, relevant, 5) == 0.0


def test_edge_case_empty_relevant():
    """Test with empty relevant list."""
    recommended = [1, 2, 3]
    relevant = []
    
    # Recall is 0 when no relevant items exist
    assert recall_at_k(recommended, relevant, 3) == 0.0
    assert hit_rate_at_k(recommended, relevant, 3) == 0.0
    assert ndcg_at_k(recommended, relevant, 3) == 0.0
