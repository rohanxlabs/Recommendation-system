"""
Evaluation metrics for recommendation systems.

Implements ranking-based metrics suitable for top-N collaborative filtering:
- Precision@K: Fraction of recommended items that are relevant
- Recall@K: Fraction of relevant items that are recommended
- Hit Rate@K: Whether any recommended item is relevant
- NDCG@K: Normalized Discounted Cumulative Gain (position-aware)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def precision_at_k(recommended_items: List, relevant_items: List, k: int) -> float:
    """
    Calculate Precision@K: fraction of recommended items that are relevant.
    
    Args:
        recommended_items: List of recommended item IDs (ordered by rank)
        relevant_items: List of ground-truth relevant item IDs
        k: Number of top recommendations to consider
    
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k <= 0 or len(recommended_items) == 0:
        return 0.0
    
    recommended_at_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    
    num_relevant_recommended = len(recommended_at_k & relevant_set)
    return num_relevant_recommended / min(k, len(recommended_items))


def recall_at_k(recommended_items: List, relevant_items: List, k: int) -> float:
    """
    Calculate Recall@K: fraction of relevant items that are recommended.
    
    Args:
        recommended_items: List of recommended item IDs (ordered by rank)
        relevant_items: List of ground-truth relevant item IDs
        k: Number of top recommendations to consider
    
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    if k <= 0 or len(recommended_items) == 0:
        return 0.0
    
    recommended_at_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    
    num_relevant_recommended = len(recommended_at_k & relevant_set)
    return num_relevant_recommended / len(relevant_set)


def hit_rate_at_k(recommended_items: List, relevant_items: List, k: int) -> float:
    """
    Calculate Hit Rate@K: whether at least one recommended item is relevant.
    
    Args:
        recommended_items: List of recommended item IDs (ordered by rank)
        relevant_items: List of ground-truth relevant item IDs
        k: Number of top recommendations to consider
    
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if k <= 0 or len(recommended_items) == 0 or len(relevant_items) == 0:
        return 0.0
    
    recommended_at_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    
    return 1.0 if len(recommended_at_k & relevant_set) > 0 else 0.0


def ndcg_at_k(recommended_items: List, relevant_items: List, k: int) -> float:
    """
    Calculate NDCG@K: Normalized Discounted Cumulative Gain.
    
    Rewards relevant items appearing earlier in the recommendation list.
    
    Args:
        recommended_items: List of recommended item IDs (ordered by rank)
        relevant_items: List of ground-truth relevant item IDs
        k: Number of top recommendations to consider
    
    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if k <= 0 or len(recommended_items) == 0 or len(relevant_items) == 0:
        return 0.0
    
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in relevant_set:
            # Relevance = 1 for relevant items, 0 otherwise
            # Discount by log2(position + 2) to favor early positions
            dcg += 1.0 / np.log2(i + 2)
    
    # Calculate IDCG (Ideal DCG) - if all top-k were relevant
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(
    user_recommendations: Dict[int, List[int]],
    user_relevant_items: Dict[int, List[int]],
    k_values: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Evaluate recommendation quality across multiple users and K values.
    
    Args:
        user_recommendations: Dict mapping user_id to list of recommended items
        user_relevant_items: Dict mapping user_id to list of relevant items (ground truth)
        k_values: List of K values to evaluate
    
    Returns:
        DataFrame with evaluation results for each metric and K value
    """
    results = []
    
    for k in k_values:
        precision_scores = []
        recall_scores = []
        hit_rate_scores = []
        ndcg_scores = []
        
        for user_id in user_recommendations:
            if user_id not in user_relevant_items:
                continue
            
            recommended = user_recommendations[user_id]
            relevant = user_relevant_items[user_id]
            
            if len(relevant) == 0:
                continue
            
            precision_scores.append(precision_at_k(recommended, relevant, k))
            recall_scores.append(recall_at_k(recommended, relevant, k))
            hit_rate_scores.append(hit_rate_at_k(recommended, relevant, k))
            ndcg_scores.append(ndcg_at_k(recommended, relevant, k))
        
        results.append({
            'K': k,
            'Precision@K': np.mean(precision_scores) if precision_scores else 0.0,
            'Recall@K': np.mean(recall_scores) if recall_scores else 0.0,
            'Hit_Rate@K': np.mean(hit_rate_scores) if hit_rate_scores else 0.0,
            'NDCG@K': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'Num_Users': len(precision_scores)
        })
    
    return pd.DataFrame(results)


def evaluate_model(model, train_matrix: pd.DataFrame, test_df: pd.DataFrame, 
                   k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Evaluate a trained recommendation model on test data.
    
    For each user in the test set:
    1. Generate recommendations using the model
    2. Compare against items they rated highly in the test set (rating >= 4.0)
    
    Args:
        model: Trained TruncatedSVD model
        train_matrix: User-item matrix used for training (pandas DataFrame)
        test_df: Test interactions DataFrame with columns [user_id, item_id, rating]
        k_values: List of K values to evaluate
    
    Returns:
        DataFrame with evaluation metrics
    """
    from src.models.recommend import recommend_items
    
    logger.info("Starting model evaluation...")
    
    # Get relevant items per user from test set (items rated >= 4.0)
    user_relevant_items = {}
    for user_id, group in test_df.groupby('user_id'):
        relevant = group[group['rating'] >= 4.0]['item_id'].tolist()
        if len(relevant) > 0:
            user_relevant_items[user_id] = relevant
    
    # Generate recommendations for users in test set
    user_recommendations = {}
    max_k = max(k_values)
    
    evaluated_users = 0
    skipped_users = 0
    
    for user_id in user_relevant_items.keys():
        if user_id not in train_matrix.index:
            skipped_users += 1
            continue
        
        try:
            recommended, _ = recommend_items(
                user_id=user_id,
                user_item_matrix=train_matrix,
                model=model,
                n=max_k,
                exclude_rated=True  # Don't recommend items from training set
            )
            user_recommendations[user_id] = recommended
            evaluated_users += 1
        except Exception as e:
            logger.warning(f"Failed to generate recommendations for user {user_id}: {e}")
            skipped_users += 1
    
    logger.info(f"Evaluated {evaluated_users} users, skipped {skipped_users} users")
    
    # Compute metrics
    results = evaluate_recommendations(user_recommendations, user_relevant_items, k_values)
    
    return results


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (for rating prediction).
    
    Note: RMSE is less relevant for ranking-based collaborative filtering.
    Use Precision@K, Recall@K, and NDCG@K instead.
    """
    return np.sqrt(np.mean((true - pred) ** 2))