"""
Tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import preprocess_data


def test_preprocess_data_splits_correctly():
    """Test that data is split into train/test with correct proportions."""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("user_id,item_id,rating\n")
        for i in range(100):
            f.write(f"{i % 10},{i % 20},{(i % 5) + 1}\n")
        temp_path = f.name
    
    try:
        train, test = preprocess_data(temp_path)
        
        # Check split ratio (approximately 80/20)
        assert len(train) == 80
        assert len(test) == 20
        
        # Check columns exist
        assert 'user_id' in train.columns
        assert 'item_id' in train.columns
        assert 'rating' in train.columns
        
    finally:
        Path(temp_path).unlink()


def test_preprocess_data_no_overlap():
    """Test that train and test sets don't overlap."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("user_id,item_id,rating\n")
        for i in range(50):
            f.write(f"{i},{i},{5}\n")
        temp_path = f.name
    
    try:
        train, test = preprocess_data(temp_path)
        
        # Create unique identifiers for interactions
        train_ids = set(zip(train['user_id'], train['item_id']))
        test_ids = set(zip(test['user_id'], test['item_id']))
        
        # No overlap expected (assuming random split)
        assert len(train_ids & test_ids) == 0
        
    finally:
        Path(temp_path).unlink()


def test_preprocess_data_preserves_values():
    """Test that ratings are preserved during split."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("user_id,item_id,rating\n")
        f.write("1,101,5.0\n")
        f.write("2,102,4.5\n")
        f.write("3,103,3.0\n")
        f.write("4,104,2.0\n")
        f.write("5,105,1.0\n")
        temp_path = f.name
    
    try:
        train, test = preprocess_data(temp_path)
        
        # Check that all ratings are valid
        all_data = pd.concat([train, test])
        assert all_data['rating'].min() >= 1.0
        assert all_data['rating'].max() <= 5.0
        assert len(all_data) == 5
        
    finally:
        Path(temp_path).unlink()
