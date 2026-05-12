from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# model
model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("rag/faiss_index.bin")
documents = pickle.load(open("rag/documents.pkl", "rb"))


# keyword
def keyword_score(query, doc):
    query_words = query.lower().split()
    doc_lower = doc.lower()
    return sum(1 for w in query_words if w in doc_lower)


# retrieval
def retrieve(query, top_k=5):
    query_emb = model.encode([query]).astype("float32")
    # normalize before searching
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k * 2)
    results = []

    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        doc = documents[idx]
        boost = keyword_score(query, doc) * 0.1
        final_score = float(score) + boost
        results.append((final_score, doc))

    results.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in results[:top_k]]


# chatbot
def get_response(query):
    query = query.strip().lower()

    # greet
    if any(greet in query for greet in ["hi", "hello", "hey"]):
        return (
            "Hello! I’m Qyro, your AI medical assistant.\n"
            "Describe your symptoms or condition, and I’ll help analyze them."
        )

    if "who are you" in query:
        return (
            "I am Qyro, built inside Qyronix AI.\n"
            "I use AI + RAG to provide medical information for education purposes."
        )

    if "thank" in query:
        return " You're welcome! Take care of your health."

    # rag
    docs = retrieve(query)
    if not docs:
        return (
            "I couldn't find a strong match.\n"
            "Try describing symptoms more clearly or use a medical term."
        )

    context = "\n".join([f"• {doc}" for doc in docs])

    response = f"""
   Qyronix AI Analysis

 Relevant Medical Information:
{context}

 Summary:
Based on retrieved medical knowledge, these conditions may be related to your query.

 Recommendation:
Please consult a certified healthcare professional for accurate diagnosis.

---
Qyronix AI is for educational purposes only.
"""

    return response