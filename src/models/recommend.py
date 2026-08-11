import numpy as np
from typing import Tuple, List

def recommend_items(
    user_id: int,
    user_item_matrix,
    model,
    n: int = 5,
    exclude_rated: bool = True
) -> Tuple[List[int], List[float]]:
    """
    Generate top-N recommendations for a user using collaborative filtering.
    
    Args:
        user_id: User ID to generate recommendations for
        user_item_matrix: User-item rating matrix (pandas DataFrame)
        model: Trained TruncatedSVD model
        n: Number of recommendations to return
        exclude_rated: Whether to exclude items the user has already rated
    
    Returns:
        Tuple of (item_ids, scores) where both are lists of length n
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    # Get user index and transform matrix to latent space
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_vector = model.transform(user_item_matrix)
    
    # Calculate predicted scores for all items
    scores = np.dot(user_vector[user_idx], model.components_)
    
    # Optionally exclude already-rated items
    if exclude_rated:
        rated_mask = user_item_matrix.iloc[user_idx] > 0
        scores = scores.copy()
        scores[rated_mask.values] = -np.inf
    
    # Get top-N recommendations
    n = min(n, len(scores))  # Cap at available items
    top_indices = scores.argsort()[::-1][:n]
    
    # Get item IDs and their scores
    recommended_items = user_item_matrix.columns[top_indices].tolist()
    recommended_scores = scores[top_indices].tolist()
    
    return recommended_items, recommended_scores