from sentence_transformers import SentenceTransformer, util
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

#---knowledge base---
docs = [
    {"id": "py1", "text": "Python functions allow reusable blocks of code defined with def.", "topic": "python"},
    {"id": "py2", "text": "List comprehensions provide a concise way to create lists in Python.", "topic": "python"},
    {"id": "py3", "text": "Virtual environments isolate project dependencies using venv.", "topic": "python"},
]

docs += [
    {"id": "fa1", "text": "FastAPI uses Pydantic models for request validation and serialization.", "topic": "fastapi"},
    {"id": "fa2", "text": "FastAPI supports dependency injection through the Depends function.", "topic": "fastapi"},
    {"id": "fa3", "text": "FastAPI automatically generates OpenAPI documentation for your API.", "topic": "fastapi"},
]

docs += [
    {"id": "db1", "text": "SQL JOIN operations combine rows from multiple tables based on related columns.", "topic": "database"},
    {"id": "db2", "text": "Indexes improve query performance by speeding up lookups on specific columns.", "topic": "database"},
    {"id": "db3", "text": "Transactions ensure atomic operations using COMMIT and ROLLBACK.", "topic": "database"},
]

docs += [
    {"id": "ml1", "text": "A neural network learns patterns using layers of interconnected neurons.", "topic": "machine_learning"},
    {"id": "ml2", "text": "Gradient descent optimizes model parameters by minimizing loss.", "topic": "machine_learning"},
    {"id": "ml3", "text": "Overfitting occurs when a model memorizes training data instead of generalizing.", "topic": "machine_learning"},
]

#---embed the documents---

doc_ids = [d["id"] for d in docs]
doc_texts = [d["text"] for d in docs]

doc_embeddings = model.encode(doc_texts)



def search(query, top_k=3, threshold=0.3):
    """Search documents and return document IDs above threshold."""
    query_emb = model.encode(query)
    scores = util.cos_sim(query_emb, doc_embeddings)[0]

    # Collect (id, score) pairs above threshold
    results = [
        (doc_ids[i], float(scores[i]))
        for i in range(len(scores))
        if scores[i] >= threshold
    ]

    # Sort by similarity score
    results.sort(key=lambda x: x[1], reverse=True)

    # Return ONLY the document IDs
    return [doc_id for doc_id, _ in results[:top_k]]

def precision(predicted_ids, expected_ids):
    if len(predicted_ids) == 0:
        return 0.0
    correct = len(set(predicted_ids) & set(expected_ids))
    return correct / len(predicted_ids)

def recall(predicted_ids, expected_ids):
    if len(expected_ids) == 0:
        return 0.0
    correct = len(set(predicted_ids) & set(expected_ids))
    return correct / len(expected_ids)

eval_set = {
    "How do I create reusable code in Python?": ["py1"],
    "What is the benefit of list comprehensions?": ["py2"],
    "How do I manage Python project dependencies?": ["py3"],
    "How does FastAPI validate incoming requests?": ["fa1"],
    "Explain dependency injection in FastAPI.": ["fa2"],
    "How do SQL joins work?": ["db1"],
}


def evaluate(eval_set, threshold=0.3, n_results=3):
    """
    Runs evaluation for all queries in eval_set using search().
    Computes precision and recall per query and prints formatted results.
    Returns average precision and recall.
    """
    print(f"\n=== Evaluation at threshold={threshold}, n_results={n_results} ===")

    precisions = []
    recalls = []

    for i, (query, expected_ids) in enumerate(eval_set.items(), start=1):
        predicted_ids = search(query, top_k=n_results, threshold=threshold)

        p = precision(predicted_ids, expected_ids)
        r = recall(predicted_ids, expected_ids)

        precisions.append(p)
        recalls.append(r)

        print(f"Query {i}: P={p*100:.1f}% R={r*100:.1f}%")

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)

    print(f"\nAVERAGE: P={avg_p*100:.1f}% R={avg_r*100:.1f}%\n")
    
    return avg_p, avg_r

if __name__ == "__main__":
    evaluate(eval_set, threshold=0.3, n_results=3)
    evaluate(eval_set, threshold=0.5, n_results=3)
    evaluate(eval_set, threshold=0.7, n_results=3)

    print("ANALYSIS:")
    print("- Lower thresholds (0.3) return more results, improving recall but sometimes lowering precision.")
    print("- Higher thresholds (0.7) improve precision but can miss relevant documents, hurting recall.")
    print("- Some queries (like simple one-to-one mappings) perform well at all thresholds,")
    print("  while more nuanced queries would benefit from tuning n_results or adjusting document wording.")