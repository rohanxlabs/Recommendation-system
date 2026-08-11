"""
Evaluate the trained recommendation model on test data.

Usage:
    python scripts/evaluate_model.py
"""
import pickle
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.evaluate import evaluate_model

# Paths
MODEL_PATH = Path("artifacts/model.pkl")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")

def main():
    print("=" * 60)
    print("RECOMMENDATION MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded (SVD components: {model.n_components})")
    
    # Load training data
    print(f"\nLoading training data from {TRAIN_DATA_PATH}...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_matrix = train_df.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating'
    ).fillna(0)
    print(f"✓ Training matrix shape: {train_matrix.shape}")
    print(f"  Users: {len(train_matrix.index)}, Items: {len(train_matrix.columns)}")
    
    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"✓ Test interactions: {len(test_df)}")
    print(f"  Users: {test_df['user_id'].nunique()}, Items: {test_df['item_id'].nunique()}")
    
    # Evaluate model
    print("\n" + "-" * 60)
    print("EVALUATING MODEL...")
    print("-" * 60)
    
    k_values = [5, 10, 20]
    results = evaluate_model(model, train_matrix, test_df, k_values)
    
    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()
    print(results.to_string(index=False))
    print()
    
    # Interpretation
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print()
    print("Precision@K: Fraction of recommended items that are relevant")
    print("             Higher is better (max: 1.0)")
    print()
    print("Recall@K:    Fraction of relevant items that are recommended")
    print("             Higher is better (max: 1.0)")
    print()
    print("Hit Rate@K:  Percentage of users who got at least 1 relevant item")
    print("             Higher is better (max: 1.0)")
    print()
    print("NDCG@K:      Position-aware metric (rewards early relevant items)")
    print("             Higher is better (max: 1.0)")
    print()
    print("=" * 60)
    
    # Save results
    output_path = Path("artifacts/evaluation_results.csv")
    results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    print()

if __name__ == "__main__":
    main()
