from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import re

model = SentenceTransformer("all-MiniLM-L6-v2")


# smart chunking
def chunk_text(text, chunk_size=3):

    sentences = re.split(r'\.|\n', text)

    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []

    for i in range(0, len(sentences), chunk_size):

        chunk = ". ".join(sentences[i:i+chunk_size])

        chunks.append(chunk)

    return chunks


folder = "medical_data"

documents = []

for file in os.listdir(folder):

    path = os.path.join(folder, file)

    with open(path, "r", encoding="utf-8") as f:

        text = f.read()

        documents.extend(chunk_text(text))


# embeddings
embeddings = model.encode(documents)

embeddings = np.array(embeddings).astype("float32")

# normalize for cosine similarity
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])

index.add(embeddings)

os.makedirs("rag", exist_ok=True)

faiss.write_index(index, "rag/faiss_index.bin")

pickle.dump(documents, open("rag/documents.pkl", "wb"))

print("✅ PRO INDEX CREATED")