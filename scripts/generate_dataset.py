"""
Generate a synthetic movie rating dataset for collaborative filtering.

Creates realistic user-item interactions with patterns:
- User preferences (some users like action, others like drama)
- Item characteristics (some items are popular, others niche)
- Rating patterns (sparse matrix, realistic rating distribution)
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Dataset configuration
NUM_USERS = 50
NUM_ITEMS = 80
NUM_INTERACTIONS = 500  # Sparse matrix: ~12.5% density

# User preferences (latent factors)
# 3 types of users: action fans, drama fans, and mixed
user_types = np.random.choice(['action', 'drama', 'mixed'], NUM_USERS, p=[0.4, 0.3, 0.3])

# Item characteristics (latent factors)
# 3 types of items: action, drama, and mixed
item_types = np.random.choice(['action', 'drama', 'mixed'], NUM_ITEMS, p=[0.35, 0.35, 0.3])

# Generate interactions
interactions = []

# Each user rates some items
for user_id in range(1, NUM_USERS + 1):
    user_type = user_types[user_id - 1]
    
    # Number of ratings per user (between 5 and 20)
    num_ratings = np.random.randint(5, 21)
    
    # Select items to rate (biased by user preferences)
    item_probs = np.ones(NUM_ITEMS)
    
    for item_id in range(NUM_ITEMS):
        item_type = item_types[item_id]
        
        # Increase probability of rating items that match user preference
        if user_type == 'action' and item_type == 'action':
            item_probs[item_id] *= 3.0
        elif user_type == 'drama' and item_type == 'drama':
            item_probs[item_id] *= 3.0
        elif user_type == 'mixed':
            item_probs[item_id] *= 1.5
    
    # Normalize probabilities
    item_probs /= item_probs.sum()
    
    # Sample items for this user
    rated_items = np.random.choice(NUM_ITEMS, size=min(num_ratings, NUM_ITEMS), 
                                   replace=False, p=item_probs)
    
    # Generate ratings
    for item_id in rated_items:
        item_type = item_types[item_id]
        
        # Base rating depends on user-item compatibility
        if user_type == item_type:
            base_rating = 4.5
        elif user_type == 'mixed' or item_type == 'mixed':
            base_rating = 3.5
        else:
            base_rating = 2.5
        
        # Add some noise
        rating = base_rating + np.random.normal(0, 0.8)
        
        # Clip to 1-5 range and round to nearest 0.5
        rating = np.clip(rating, 1, 5)
        rating = np.round(rating * 2) / 2  # Round to nearest 0.5
        
        interactions.append({
            'user_id': user_id,
            'item_id': int(item_id + 101),  # Item IDs start at 101
            'rating': rating
        })

# Create DataFrame
df = pd.DataFrame(interactions)

# Remove duplicates (in case any were created)
df = df.drop_duplicates(subset=['user_id', 'item_id'])

# Sort by user_id, then item_id
df = df.sort_values(['user_id', 'item_id']).reset_index(drop=True)

# Print statistics
print("Dataset Generation Complete!")
print(f"\nStatistics:")
print(f"  Total interactions: {len(df)}")
print(f"  Unique users: {df['user_id'].nunique()}")
print(f"  Unique items: {df['item_id'].nunique()}")
print(f"  Sparsity: {len(df) / (NUM_USERS * NUM_ITEMS) * 100:.2f}%")
print(f"\nRating distribution:")
print(df['rating'].value_counts().sort_index())
print(f"\nAverage ratings per user: {len(df) / df['user_id'].nunique():.1f}")
print(f"Average ratings per item: {len(df) / df['item_id'].nunique():.1f}")

# Save to file
output_path = Path("data/raw/interactions.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\nDataset saved to {output_path}")

# Show sample
print(f"\nFirst 10 interactions:")
print(df.head(10))
