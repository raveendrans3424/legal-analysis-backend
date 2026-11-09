"""
build_faiss_index.py
One-time (re-run whenever provisions change) script that:
  1. Reads all rows from legal_provisions in MySQL
  2. Embeds their full_text with the sentence-transformers model
  3. Builds a FAISS index and saves it to disk, along with an id_map
     that translates FAISS row index -> legal_provisions.id

Usage:
    python build_faiss_index.py
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from db import SessionLocal
from models import LegalProvision
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_DIR, FAISS_INDEX_PATH, FAISS_ID_MAP_PATH


def build_index():
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    db = SessionLocal()
    try:
        provisions = db.query(LegalProvision).all()
    finally:
        db.close()

    if not provisions:
        print("No provisions found in the database. Run seed_provisions.py first.")
        return

    print(f"Embedding {len(provisions)} provisions with {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [p.full_text or "" for p in provisions]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product; embeddings should be normalized for cosine-sim behavior
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    id_map = {str(i): p.id for i, p in enumerate(provisions)}

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_ID_MAP_PATH, "w") as f:
        json.dump(id_map, f)

    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
    print(f"Saved id map to {FAISS_ID_MAP_PATH}")


if __name__ == "__main__":
    build_index()
