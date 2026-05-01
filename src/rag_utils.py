"""
Lightweight RAG retriever: sentence-transformers + FAISS index.
Owner: Person B (Stage 3)
"""
import os
import pickle
from pathlib import Path
from typing import List, Optional

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "models/rag_index.faiss"
CHUNKS_PATH = "models/rag_chunks.pkl"
CHUNK_SIZE = 200   # words per chunk


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]


def build_index(corpus_path: str, index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH):
    """
    Build a FAISS index from a plain-text corpus file (one document per line or full text).
    TODO (Person B): run once in 04_Prompt_Engineering.ipynb after downloading PubMed subset.

    Usage:
        build_index("data/processed/pubmed_corpus.txt")
    """
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(EMBED_MODEL)

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = _chunk_text(raw)
    print(f"Indexing {len(chunks)} chunks...")

    embeddings = encoder.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product on normalized vecs = cosine similarity
    index.add(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index saved to {index_path} ({index.ntotal} vectors)")


def load_index(index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH):
    """Load a pre-built FAISS index and chunk list."""
    import faiss
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(query: str, k: int = 3, index=None, chunks: Optional[List[str]] = None) -> str:
    """
    Retrieve the top-k most relevant chunks for a query.
    Returns a single string of joined chunks for injection into a prompt.
    TODO (Person B): call this from prompt_templates.answer_rag_few_shot.
    """
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if index is None or chunks is None:
        index, chunks = load_index()

    encoder = SentenceTransformer(EMBED_MODEL)
    q_vec = encoder.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(q_vec, k)

    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n---\n\n".join(retrieved)
