import os
import json
import faiss
import numpy as np
from datetime import datetime
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# ===== Config =====
OUTPUT_PATH = "output/output.json"
INDEX_PATH = "faiss_output/index.index"
METADATA_PATH = "faiss_output/index_metadata.json"
EMBEDDINGS_PATH = "faiss_output/embeddings.npy"
BM25_TOP_K = 50
FAISS_TOP_K = 5
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
T5_MODEL_NAME = "t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Initialize Models =====
tok = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)
# SentenceTransformer will be loaded later

# ===== Step 1: User Input =====
persona = input("Enter the persona: ").strip()
job = input("Enter the job to be done: ").strip()
raw_query = f"As a {persona}, your task is to {job}."

# ===== Helpers =====
def rephrase(text: str) -> str:
    input_ids = tok(f"summarize: {text}", return_tensors="pt").input_ids.to(DEVICE)
    outputs = t5_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tok.decode(outputs[0], skip_special_tokens=True)

# We will use same model for summarization later
def summarize(text: str) -> str:
    input_ids = tok(f"summarize: {text}", return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
    outputs = t5_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    return tok.decode(outputs[0], skip_special_tokens=True)

# ===== Step 2: Rephrase Query =====
print("✏️ Rephrasing query using T5...")
query = rephrase(raw_query)
print(f"🔁 Rephrased Query: {query}")

# ===== Step 3: Load FAISS + Data =====
print("🔁 Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
embeddings = np.load(EMBEDDINGS_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ===== Step 4: BM25 Filtering =====
print("🔎 Running BM25 filtering...")
token_lists = [chunk["tokens"] for chunk in chunks]
bm25 = BM25Okapi(token_lists)
query_tokens = query.lower().split()
bm25_scores = bm25.get_scores(query_tokens)
top_bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
top_embeddings = np.array([embeddings[i] for i in top_bm25_indices])

# ===== Step 5: FAISS Reranking =====
print("🔍 FAISS reranking...")
sbert = SentenceTransformer(MODEL_NAME)
query_emb = sbert.encode([query], convert_to_numpy=True).astype("float32")
temp_index = faiss.IndexFlatL2(EMBEDDING_DIM)
temp_index.add(top_embeddings)
D, I = temp_index.search(query_emb, FAISS_TOP_K)

# ===== Step 6: Collect & Summarize =====
results = []
for rank, local_idx in enumerate(I[0]):
    true_idx = top_bm25_indices[local_idx]
    chunk = chunks[true_idx]
    text = chunk['text']
    summary = summarize(text)
    results.append({
        "rank": rank + 1,
        "document": os.path.basename(chunk['document']),
        "page_number": chunk['page'],
        "section_title": chunk.get('section', "Unknown Section"),
        "text": text,
        "score": round(1 - D[0][rank], 4),
        "summary": summary
    })

# ===== Step 7: Build Output =====
input_docs = sorted({os.path.basename(chunk['document']) for chunk in chunks})
output = {
    "metadata": {
        "input_documents": input_docs,
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.utcnow().isoformat()
    },
    "extracted_sections": [],
    "subsection_analysis": []
}
for item in results:
    output["extracted_sections"].append({
        "document": item['document'],
        "section_title": item['section_title'],
        "importance_rank": item['rank'],
        "page_number": item['page_number']
    })
    output["subsection_analysis"].append({
        "document": item['document'],
        "refined_text": item['summary'],
        "page_number": item['page_number']
    })

# ===== Step 8: Save Output =====
print("💾 Saving output to JSON...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
print(f"\n✅ Top {FAISS_TOP_K} results written to '{OUTPUT_PATH}'")
