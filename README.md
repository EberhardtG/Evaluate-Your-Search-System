
# 🔍 Evaluate Your Search System

This project implements a small semantic‑search evaluation framework using sentence embeddings. It measures how well a search system retrieves relevant documents by computing **precision** and **recall** across multiple queries and multiple retrieval settings. The goal is to build intuition for evaluating and tuning search quality — a skill you’ll use in the module project and future RAG systems.

---

## 📘 Overview

The script creates:

- A **12‑document knowledge base** across 4 topics  
- A **search function** using cosine similarity  
- An **evaluation set** of 6 test queries  
- **Precision and recall** scoring  
- An `evaluate()` function that runs the full experiment  
- Three evaluation runs at different thresholds  
- A short written analysis of the results  

This provides a complete, minimal framework for understanding how retrieval quality changes with different search parameters.

---

## 📂 Project Structure

```
my_eval.py
├── docs (12 documents)
├── embeddings (SentenceTransformer)
├── search() function
├── precision() and recall()
├── eval_set (6 queries)
├── evaluate() function
└── analysis output
```

---

## 📄 Knowledge Base

The knowledge base contains 12 short documents across:

- **Python**
- **FastAPI**
- **Databases / SQL**
- **Machine Learning**

Each document has an ID, text, and topic label. These IDs are used to compute precision and recall.

---

## 🧪 Evaluation Set

Six test queries are defined, each mapped to the document IDs that should be considered relevant. Example:

```python
"How do SQL joins work?" → ["db1"]
```

This allows the evaluation function to compare retrieved results against expected ones.

---

## ⚙️ How It Works

### 1. Embed all documents  
Using `all-MiniLM-L6-v2`, each document is converted into a vector.

### 2. Search function  
The search function:

- Encodes the query  
- Computes cosine similarity  
- Filters results by threshold  
- Returns the top‑k document IDs  

### 3. Precision and Recall  
For each query:

- **Precision** = relevant retrieved / total retrieved  
- **Recall** = relevant retrieved / total relevant  

### 4. Evaluation Loop  
The `evaluate()` function:

- Runs all queries  
- Prints per‑query precision and recall  
- Computes average scores  
- Repeats the experiment at three thresholds  

---

## 📊 Example Output

```
=== Evaluation at threshold=0.3, n_results=3 ===
Query 1: P=100.0% R=50.0%
Query 2: P=66.7% R=100.0%
...
AVERAGE: P=72.3% R=65.4%
```

Higher thresholds produce fewer results (higher precision, lower recall).  
Lower thresholds produce more results (lower precision, higher recall).

---

## 🚀 Running the Script

Install dependencies:

```bash
pip install sentence-transformers
```

Run the evaluation:

```bash
python my_eval.py
```

You will see three evaluation blocks — one for each threshold.

---

## 🧠 Analysis Summary

- **Low thresholds (0.3)** return more results, improving recall but sometimes lowering precision.  
- **High thresholds (0.7)** improve precision but may miss relevant documents entirely.  
- Queries with clear one‑to‑one mappings perform well across all thresholds.  
- More nuanced queries would benefit from tuning `n_results`, adjusting document text, or experimenting with different embedding models.  

This evaluation framework provides a foundation for analyzing retrieval quality in larger RAG systems.

