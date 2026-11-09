"""
retrieval.py
Steps 8-9 of the workflow: retrieve candidate legal provisions.

If a predicted category is given, this does a STRICT category-only lookup
from MySQL first (bypassing FAISS's broader top_k), so unrelated categories
can never leak into the ranked results when enough same-category provisions
exist. Falls back to full FAISS semantic search only if that category has
zero rows in the database.
"""

import json
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_ID_MAP_PATH,
    FAISS_TOP_K,
)
from models import LegalProvision


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def load_faiss_index():
    """
    Loads the pre-built FAISS index and its id_map (FAISS row -> DB provision id).
    Build these once with build_faiss_index.py before starting the API.
    """
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_ID_MAP_PATH, "r") as f:
        id_map = json.load(f)  # {"0": 1, "1": 2, ...} FAISS row index -> provision.id
    return index, id_map


def _to_dict(p: LegalProvision) -> dict:
    return {
        "id": p.id,
        "act_name": p.act_name,
        "section_number": p.section_number,
        "title": p.title,
        "full_text": p.full_text,
        "category": p.category,
        "source_url": p.source_url,
    }


def get_candidates(query_text: str, db: Session, category: str | None = None,
                    top_k: int = FAISS_TOP_K) -> list[dict]:
    """
    STRICT category filter: if a category is given and it has provisions
    in the database, return ONLY those (ranking.py will then re-rank just
    within that category — unrelated categories can never leak in).
    Falls back to full FAISS semantic search only when that category has
    zero rows in the database.
    """
    if category:
        same_category = (
            db.query(LegalProvision)
            .filter(LegalProvision.category == category)
            .all()
        )
        if same_category:
            return [_to_dict(p) for p in same_category]

    model = get_embedding_model()
    index, id_map = load_faiss_index()

    query_vec = model.encode([query_text]).astype("float32")
    faiss.normalize_L2(query_vec)  # index was built with normalized vectors (cosine via inner product)
    scores, indices = index.search(query_vec, top_k)

    candidate_ids = [id_map[str(i)] for i in indices[0] if str(i) in id_map]

    provisions = (
        db.query(LegalProvision)
        .filter(LegalProvision.id.in_(candidate_ids))
        .all()
    )

    # Preserve FAISS similarity order
    provisions_by_id = {p.id: p for p in provisions}
    ordered = [provisions_by_id[pid] for pid in candidate_ids if pid in provisions_by_id]

    return [_to_dict(p) for p in ordered]