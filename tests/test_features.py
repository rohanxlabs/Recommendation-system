"""
Tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import create_user_item_matrix


def test_create_user_item_matrix_shape():
    """Test that matrix has correct shape."""
    df = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'item_id': [101, 102, 101, 103, 102],
        'rating': [5, 4, 3, 5, 2]
    })
    
    matrix = create_user_item_matrix(df)
    
    # 3 users, 3 items
    assert matrix.shape == (3, 3)
    assert len(matrix.index) == 3
    assert len(matrix.columns) == 3


def test_create_user_item_matrix_fills_zeros():
    """Test that missing values are filled with zeros."""
    df = pd.DataFrame({
        'user_id': [1, 2],
        'item_id': [101, 102],
        'rating': [5, 4]
    })
    
    matrix = create_user_item_matrix(df)
    
    # User 1 rated item 101 but not 102
    assert matrix.loc[1, 101] == 5
    assert matrix.loc[1, 102] == 0
    
    # User 2 rated item 102 but not 101
    assert matrix.loc[2, 102] == 4
    assert matrix.loc[2, 101] == 0


def test_create_user_item_matrix_preserves_ratings():
    """Test that actual ratings are preserved correctly."""
    df = pd.DataFrame({
        'user_id': [1, 1, 2, 2],
        'item_id': [101, 102, 101, 102],
        'rating': [5.0, 4.5, 3.0, 2.5]
    })
    
    matrix = create_user_item_matrix(df)
    
    assert matrix.loc[1, 101] == 5.0
    assert matrix.loc[1, 102] == 4.5
    assert matrix.loc[2, 101] == 3.0
    assert matrix.loc[2, 102] == 2.5


def test_create_user_item_matrix_handles_duplicates():
    """Test behavior with duplicate user-item pairs (should average)."""
    df = pd.DataFrame({
        'user_id': [1, 1, 1],
        'item_id': [101, 101, 102],
        'rating': [4, 5, 3]
    })
    
    matrix = create_user_item_matrix(df)
    
    # Pivot table averages duplicates by default
    assert matrix.loc[1, 101] == 4.5  # (4 + 5) / 2
    assert matrix.loc[1, 102] == 3.0


def test_create_user_item_matrix_index_labels():
    """Test that index and column labels are correct."""
    df = pd.DataFrame({
        'user_id': [10, 20, 30],
        'item_id': [101, 102, 103],
        'rating': [5, 4, 3]
    })
    
    matrix = create_user_item_matrix(df)
    
    assert list(matrix.index) == [10, 20, 30]
    assert list(matrix.columns) == [101, 102, 103]
