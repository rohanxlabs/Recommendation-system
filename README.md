Live link : https://recommendation-system-8z0w.onrender.com


🎯 AI-Powered Recommendation System

⚡ Intelligent Product Recommendations with RAG & LLMs

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=22&duration=3000&color=FF6F61&center=true&vCenter=true&width=700&lines=AI+Recommendation+System;RAG+Enhanced+Suggestions;From+Search+→+Understanding+→+Recommendation" />
</p><p align="center">
  <img src="https://img.shields.io/badge/System-Recommendation-blue">
  <img src="https://img.shields.io/badge/AI-RAG-orange">
  <img src="https://img.shields.io/badge/LLM-Powered-green">
  <img src="https://img.shields.io/badge/API-FastAPI-purple">
</p>---

🎯 Problem Statement

Traditional recommendation systems:

- ❌ Only match keywords or basic similarity
- ❌ Cannot explain recommendations
- ❌ Lack contextual understanding

This project solves these limitations using:

👉 Retrieval-Augmented Generation (RAG)
👉 LLM-based reasoning for better recommendations

---

💡 What Makes This System Unique

Unlike basic recommenders, this system:

✔ Retrieves relevant products using similarity search
✔ Uses LLM to enhance and explain recommendations
✔ Supports both standard + RAG-based recommendations
✔ Provides context-aware suggestions instead of static results

---

🏗️ System Architecture

User Query
    ↓
Query Processing
    ↓
Embedding Generation
    ↓
Similarity Search (Vector DB)
    ↓
Top-K Products Retrieved
    ↓
LLM Enhancement (RAG)
    ↓
Final Recommendations + Explanation

---

⚙️ Core Components

🔍 Recommendation Engine

- Matches user queries with products
- Uses similarity-based ranking

🧠 RAG Layer

- Enhances recommendations using LLM
- Generates human-like explanations

🗂️ Vector Database

- Stores embeddings of product data
- Enables fast semantic search

🌐 API Layer

- FastAPI backend
- Handles user queries and responses

---

🔄 System Workflow

1. User inputs query
2. System retrieves relevant products
3. Top-K results selected
4. LLM enhances recommendations
5. Final response returned

---

🛠️ Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,fastapi,git,docker" />
</p>- Python
- FastAPI
- Vector Database (FAISS)
- LLM APIs (OpenAI / HuggingFace)
- Embeddings (Word2Vec / Transformers)

---

📂 Project Structure

recommendation-system/
│
├── data/                → Product dataset
├── vector_db/           → FAISS index & embeddings
├── recommend.py         → Basic recommendation logic
├── rag_recommend.py     → RAG-based enhancement
├── create_vector_db.py  → Embedding pipeline
├── main.py              → FastAPI backend
├── frontend.html        → User interface
└── requirements.txt

---

📊 Why RAG in Recommendation?

Traditional systems rank items.
RAG systems understand + explain recommendations.

LLMs can:

- Act as ranking engines
- Generate explanations
- Improve personalization dynamically

---

▶️ Run Locally

git clone https://github.com/rohanxlabs/Recommendation-system
cd Recommendation-system
pip install -r requirements.txt
uvicorn main:app --reload

---

🌐 API Endpoints

POST /recommend        → Basic recommendations
POST /rag_recommend    → AI-enhanced recommendations

---

🚀 Future Improvements

- Personalized recommendations (user history)
- Hybrid filtering (collaborative + content-based)
- Real product dataset integration
- Real-time feedback learning
- Scalable cloud deployment

---

⚠️ Limitations

- Uses mock/static product data
- Limited personalization
- Depends on retrieval quality
- LLM cost & latency considerations

---

🧑‍💻 Author

Rohan
GitHub: https://github.com/rohanxlabs

---

⭐ Why This Project Stands Out

This project bridges the gap between:

👉 Traditional Recommendation Systems
👉 Modern AI-powered intelligent systems

It demonstrates how to build next-generation recommendation engines.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6F61,100:FFB347&height=120&section=footer"/>
</p>---