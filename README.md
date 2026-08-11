# Collaborative Filtering Recommendation System

A production-ready recommendation engine using matrix factorization (SVD) to deliver personalized item recommendations through a FastAPI service.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-45%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project demonstrates a **collaborative filtering recommendation system** using matrix factorization. It learns user preferences from historical rating data and generates personalized top-N recommendations.

**Key Features:**
- ✅ Matrix factorization using Truncated SVD
- ✅ Personalized user-based recommendations
- ✅ Cold-start handling with popularity fallback
- ✅ FastAPI REST API with OpenAPI documentation
- ✅ Comprehensive evaluation metrics (Precision@K, Recall@K, NDCG@K)
- ✅ 45 unit & integration tests with 100% pass rate
- ✅ Configurable top-K recommendations with score filtering

---

## Problem Statement

Recommendation systems help users discover relevant items from large catalogs. This system addresses:

- **Information Overload**: Filtering 80+ items to show the most relevant ones
- **Personalization**: Tailoring recommendations based on individual user preferences
- **Cold Start**: Handling new users without rating history via popularity-based fallback
- **Evaluation**: Measuring recommendation quality with standard metrics

---

## Recommendation Approach

**Collaborative Filtering (Matrix Factorization)**

This implementation uses **Truncated SVD** to decompose the user-item rating matrix into latent factors:

```
User-Item Matrix (50×80)  →  SVD  →  User Factors (50×49) × Item Factors (49×80)
```

**How it works:**
1. Learn latent user preferences and item characteristics from ratings
2. Project users and items into shared latent space
3. Calculate predicted scores: `score = user_vector · item_components`
4. Rank items by predicted score, return top-N

**Why SVD?**
- Captures latent patterns in user-item interactions
- Handles sparse matrices effectively (~16% density)
- Fast inference with precomputed decomposition
- Proven approach for collaborative filtering

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User-Item Ratings                        │
│                    (interactions.csv)                        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Data Preprocessing                          │
│              • Train/Test Split (80/20)                      │
│              • CSV → DataFrame                               │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                Feature Engineering                           │
│           • Pivot to User-Item Matrix                        │
│           • Fill missing values with 0                       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Matrix Factorization (SVD)                      │
│           • Decompose into latent factors                    │
│           • n_components = min(users, items) - 1             │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Recommendation Engine                       │
│           • Calculate: user_vector · item_components         │
│           • Filter already-rated items (optional)            │
│           • Rank by predicted score                          │
│           • Return top-K items with scores                   │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Service                          │
│           • GET /recommend/{user_id}?n=5                     │
│           • Cold-start → Popularity fallback                 │
│           • Health check & monitoring                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Synthetic Collaborative Filtering Dataset**

- **647 interactions** (user-item-rating triples)
- **50 users** × **80 items**
- **Sparsity**: ~16% (realistic for collaborative filtering)
- **Ratings**: 1.0 to 5.0 (0.5 increments)
- **Pattern**: Items have implicit genre categories (action, drama, mixed)
- **User preferences**: Some users prefer action, others drama, some mixed

**Train/Test Split:**
- Train: 517 interactions (80%)
- Test: 130 interactions (20%)

**Generation:**
```bash
python scripts/generate_dataset.py
```

---

## API

### Endpoints

#### **GET /** 
Root endpoint with API information.

#### **GET /health**
Health check with system status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_users": 50,
  "num_items": 80,
  "matrix_shape": [50, 80]
}
```

#### **GET /recommend/{user_id}**
Get personalized recommendations for a user.

**Parameters:**
- `user_id` (path): User ID to get recommendations for
- `n` (query, optional): Number of recommendations (default: 5, max: 50)
- `exclude_rated` (query, optional): Exclude already-rated items (default: true)

**Response:**
```json
{
  "user_id": 15,
  "recommendations": [
    {
      "item_id": 134,
      "score": 4.82,
      "rank": 1
    },
    {
      "item_id": 127,
      "score": 4.56,
      "rank": 2
    },
    {
      "item_id": 145,
      "score": 4.21,
      "rank": 3
    }
  ],
  "strategy": "collaborative",
  "total_items": 80
}
```

**Strategy Types:**
- `collaborative`: Personalized recommendations using SVD
- `popularity`: Fallback for unknown users (items sorted by average rating)

---

## Example Usage

### Start the API

```bash
# Start development server
uvicorn src.api.app:app --reload

# Or with module syntax
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Make Requests

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations for user 15
curl http://localhost:8000/recommend/15

# Get 10 recommendations, including already-rated items
curl "http://localhost:8000/recommend/15?n=10&exclude_rated=false"

# Unknown user (gets popularity-based recommendations)
curl http://localhost:8000/recommend/99999
```

### API Documentation

Visit **http://localhost:8000/docs** for interactive Swagger UI.

---

## Evaluation Metrics

The system is evaluated on held-out test data using standard ranking metrics:

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Precision@K** | 2.42% | 3.03% | 2.88% |
| **Recall@K** | 6.16% | 15.51% | 28.84% |
| **Hit Rate@K** | 12.12% | 30.30% | 45.45% |
| **NDCG@K** | 3.60% | 7.36% | 11.46% |

**Interpretation:**
- **Hit Rate@10 = 30%**: 30% of users receive at least one relevant item in top-10
- **Recall@20 = 29%**: Top-20 recommendations capture 29% of all relevant items
- **NDCG rewards early relevant items**: Higher scores when relevant items appear first

**Run Evaluation:**
```bash
python scripts/evaluate_model.py
```

Results saved to `artifacts/evaluation_results.csv`.

---

## Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/rohanxlabs/Recommendation-system.git
cd Recommendation-system

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset (optional - already included)
python scripts/generate_dataset.py

# 5. Train model
python run_pipeline.py
```

**Expected output:**
```
Pipeline executed successfully
```

This creates:
- `artifacts/model.pkl` (trained SVD model)
- `data/processed/train.csv` (training data)
- `data/processed/test.csv` (test data)

---

## Running the System

### 1. Train the Model

```bash
python run_pipeline.py
```

### 2. Evaluate the Model

```bash
python scripts/evaluate_model.py
```

### 3. Start the API

```bash
# Development
uvicorn src.api.app:app --reload

# Production
gunicorn src.api.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 4. Run Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

---

## Project Structure

```
Recommendation-system/
├── data/
│   ├── raw/
│   │   └── interactions.csv          # Raw user-item ratings
│   └── processed/
│       ├── train.csv                 # Training set (80%)
│       └── test.csv                  # Test set (20%)
├── artifacts/
│   ├── model.pkl                     # Trained SVD model
│   └── evaluation_results.csv        # Evaluation metrics
├── src/
│   ├── api/
│   │   └── app.py                    # FastAPI application
│   ├── data/
│   │   └── preprocess.py             # Data preprocessing
│   ├── features/
│   │   └── build_features.py         # Feature engineering
│   ├── models/
│   │   ├── train.py                  # Model training
│   │   ├── recommend.py              # Recommendation logic
│   │   └── evaluate.py               # Evaluation metrics
│   └── utils/
│       └── config.py                 # Configuration
├── scripts/
│   ├── generate_dataset.py           # Dataset generation
│   └── evaluate_model.py             # Model evaluation
├── tests/
│   ├── test_api.py                   # API tests
│   ├── test_model.py                 # Model tests
│   ├── test_evaluate.py              # Evaluation tests
│   ├── test_features.py              # Feature tests
│   └── test_preprocessing.py         # Preprocessing tests
├── run_pipeline.py                   # Main pipeline script
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Pytest configuration
└── README.md                         # This file
```

---

## Implementation Details

### Matrix Factorization (SVD)

```python
from sklearn.decomposition import TruncatedSVD

# Decompose user-item matrix
svd = TruncatedSVD(n_components=49, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)

# Generate recommendations
user_vector = svd.transform(user_item_matrix[user_idx])
scores = user_vector @ svd.components_
top_items = scores.argsort()[::-1][:k]
```

### Recommendation Algorithm

```python
def recommend_items(user_id, user_item_matrix, model, n=5, exclude_rated=True):
    # 1. Get user's latent vector
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_vector = model.transform(user_item_matrix)
    
    # 2. Score all items
    scores = np.dot(user_vector[user_idx], model.components_)
    
    # 3. Optionally exclude already-rated items
    if exclude_rated:
        rated_mask = user_item_matrix.iloc[user_idx] > 0
        scores[rated_mask] = -np.inf
    
    # 4. Rank and return top-N
    top_indices = scores.argsort()[::-1][:n]
    return items[top_indices], scores[top_indices]
```

---

## Limitations

### Current Limitations

1. **Static Model**: Model is trained offline and not updated with new ratings
2. **Cold Start (Items)**: New items without ratings cannot be recommended
3. **Cold Start (Users)**: New users receive popularity-based recommendations only
4. **Scalability**: Full matrix decomposition doesn't scale to millions of users/items
5. **Sparsity**: Requires sufficient ratings per user/item for quality recommendations
6. **No Context**: Doesn't consider temporal, geographical, or contextual factors
7. **Implicit Feedback**: Designed for explicit ratings, not implicit signals (clicks, views)

### What This Project Demonstrates

✅ **Collaborative Filtering Fundamentals**  
✅ **Matrix Factorization (SVD)**  
✅ **Recommendation Ranking & Evaluation**  
✅ **API Design for ML Systems**  
✅ **Testing & Reproducibility**  
✅ **Cold-Start Handling**  

❌ **Not Demonstrated:**  
- Real-time model updates
- Neural collaborative filtering
- Hybrid recommenders (content + collaborative)
- Large-scale distributed systems

---

## Future Improvements

### Short Term
- [ ] Add user/item metadata for content-based hybrid recommendations
- [ ] Implement negative sampling for implicit feedback
- [ ] Add A/B testing framework for recommendation strategies
- [ ] Cache recommendations for popular users
- [ ] Add recommendation explanations (e.g., "Users like you also liked...")

### Medium Term
- [ ] Implement incremental SVD for online learning
- [ ] Add neural collaborative filtering (NCF) approach
- [ ] Support session-based recommendations
- [ ] Add diversity and novelty metrics
- [ ] Implement multi-armed bandit for exploration/exploitation

### Long Term
- [ ] Scale to millions of users/items with approximate nearest neighbors
- [ ] Add contextual bandits for context-aware recommendations
- [ ] Implement reinforcement learning for long-term user satisfaction
- [ ] Build real-time streaming pipeline with Kafka
- [ ] Deploy on Kubernetes with auto-scaling

---

## Technology Stack

**Core:**
- Python 3.11
- NumPy & Pandas (data processing)
- scikit-learn (TruncatedSVD)

**API:**
- FastAPI (web framework)
- Pydantic (data validation)
- Uvicorn (ASGI server)
- Gunicorn (production server)

**Testing:**
- pytest (testing framework)
- httpx (API testing)

**Development:**
- Git (version control)
- Virtual environments (isolation)

---

## Performance

**Model Training:**
- Dataset: 647 interactions, 50 users, 80 items
- Training time: <1 second
- Model size: ~50 KB (serialized)

**Inference:**
- Cold start: ~100ms (average ratings)
- Collaborative: ~50ms (matrix multiplication)
- API response time: ~100-200ms (including network)

**Scalability Considerations:**

Current implementation suitable for:
- ✅ Thousands of users
- ✅ Thousands of items
- ✅ Millions of interactions (if sparse)

For larger scale, consider:
- Approximate nearest neighbors (FAISS, Annoy)
- Model serving infrastructure (TensorFlow Serving, Triton)
- Distributed training (Spark MLlib, Dask)

---

## Contributing

Contributions are welcome! Areas for contribution:
- Adding new evaluation metrics
- Implementing alternative algorithms (ALS, NCF)
- Improving API features
- Adding deployment configurations
- Writing tutorials and documentation

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by production recommender systems at Netflix, Amazon, and Spotify
- Built as a portfolio project demonstrating ML systems design
- Dataset synthetically generated with realistic collaborative filtering patterns

---

## Contact

**Rohan**  
GitHub: [@rohanxlabs](https://github.com/rohanxlabs)  
Repository: [Recommendation-system](https://github.com/rohanxlabs/Recommendation-system)

---

**⭐ If you find this project helpful, please consider giving it a star!**
=======
Live link : https://recommendation-system-8z0w.onrender.com

🎯 Recommendation System

⚡ Intelligent Item Suggestion using Similarity & Ranking

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=22&duration=3000&color=FF6F61&center=true&vCenter=true&width=700&lines=Recommendation+System;Similarity-Based+Suggestions;From+Data+→+Insight+→+Recommendation" />
</p><p align="center">
  <img src="https://img.shields.io/badge/System-Recommendation-blue">
  <img src="https://img.shields.io/badge/Type-Content Based-orange">
  <img src="https://img.shields.io/badge/Backend-FastAPI-green">
  <img src="https://img.shields.io/badge/Status-Working-success">
</p>---

🎯 Problem Statement

Modern platforms like:

- Amazon
- Netflix
- YouTube

rely heavily on recommendation systems to:

- Improve user experience
- Increase engagement
- Boost conversions

This project implements a basic recommendation engine to demonstrate how such systems work at a foundational level.

---

💡 System Type

This project is based on:

👉 Content-Based / Similarity-Based Recommendation

It works by:

- Finding similar items
- Ranking them based on relevance
- Returning top matches

---

🏗️ System Architecture

User Input
   ↓
Feature Extraction
   ↓
Similarity Calculation
   ↓
Ranking
   ↓
Top-N Recommendations

---

⚙️ Core Components

🔍 Feature Processing

- Extract item features
- Convert into comparable format

📊 Similarity Engine

- Compute similarity between items
- (e.g., cosine similarity / distance metrics)

📈 Ranking System

- Rank items based on similarity score
- Select top-N recommendations

🌐 API Layer

- FastAPI backend
- Handles user queries

---

🔄 Workflow

1. Load dataset
2. Process features
3. Compute similarity matrix
4. Receive user query
5. Retrieve similar items
6. Return recommendations

---

🛠️ Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,fastapi,git" />
</p>- Python
- Pandas / NumPy
- Scikit-learn
- FastAPI

---

📂 Project Structure

recommendation-system/
│
├── data/                → Dataset
├── model/               → Similarity logic
├── utils/               → Helper functions
├── main.py              → API server
├── frontend.html        → UI
└── requirements.txt

---

📊 Key Concepts Demonstrated

Concept| Explanation
Content-Based Filtering| Recommend similar items
Similarity Metrics| Measure item closeness
Ranking| Select best results
Feature Engineering| Represent items numerically

---

📌 Limitations

- ❌ No personalization (same result for all users)
- ❌ No learning from user behavior
- ❌ No collaborative filtering
- ❌ No real-time feedback loop

---

🚀 Future Improvements

- Add collaborative filtering
- Add user-based personalization
- Hybrid recommendation system
- Integrate with real dataset
- Add ML-based ranking

---

▶️ Run Locally

git clone https://github.com/rohanxlabs/Recommendation-system
cd Recommendation-system
pip install -r requirements.txt
uvicorn main:app --reload

---

🌐 API

http://localhost:8000

---

🧑‍💻 Author

Rohan
GitHub: https://github.com/rohanxlabs

---

⭐ Why This Project Matters

This project focuses on core recommendation system fundamentals, which are the foundation of all large-scale recommender systems.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6F61,100:FFB347&height=120&section=footer"/>
</p>---

