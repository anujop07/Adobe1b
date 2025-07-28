import os
import json
import numpy as np
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# ===== Configuration =====
INPUT_DIR = "input"
OUTPUT_DIR = "faiss_output"
INDEX_PATH = os.path.join(OUTPUT_DIR, "index.index")
METADATA_PATH = os.path.join(OUTPUT_DIR, "index_metadata.json")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ===== Step 1: Load PDFs =====
def load_pdfs(input_dir=INPUT_DIR):
    loader = DirectoryLoader(
        path=input_dir,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages from '{input_dir}'")
    return docs

# ===== Step 2: Chunking =====
def split_into_chunks(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks")
    return chunks

# ===== Step 3: Embed & Build FAISS =====
def embed_chunks_and_build_index(chunks, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings

# ===== Step 4: Save index, embeddings + metadata =====
def save_outputs(index, embeddings, chunks, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, "index.index"))
    # Save embeddings array for downstream use
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    # Extract one main heading per (doc, page)
    page_heading: dict[tuple[str,int], str] = {}

    def extract_heading(text: str) -> str | None:
        for line in text.split("\n"):
            line = line.strip()
            if 5 < len(line) < 100 and line[0].isupper() and "." not in line:
                return line
        return None

    metadata = []
    for i, chunk in enumerate(chunks):
        doc = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", -1)
        key = (doc, page)

        if key not in page_heading:
            candidate = extract_heading(chunk.page_content)
            page_heading[key] = candidate or "Unknown Section"

        section = page_heading[key]
        text = chunk.page_content
        tokens = text.lower().split()

        metadata.append({
            "chunk_id": i + 1,
            "section": section,
            "text": text,
            "tokens": tokens,
            "document": doc,
            "page": page
        })

    with open(os.path.join(output_dir, "index_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved FAISS index, embeddings & metadata to '{output_dir}'")

# ===== Run All Steps =====
if __name__ == "__main__":
    all_docs = load_pdfs()
    all_chunks = split_into_chunks(all_docs)
    index, embeddings = embed_chunks_and_build_index(all_chunks)
    save_outputs(index, embeddings, all_chunks)
