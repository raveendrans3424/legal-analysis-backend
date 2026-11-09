"""
ranking.py
Step 10 of the workflow: compare candidate provisions against the actual
case facts and rank them by case-specific relevance using a cross-encoder.

This is what prevents "popular section bias" — the model scores each
(case_facts, provision_text) pair jointly, so a rarely-used but factually
precise section can outrank a commonly-cited generic one.
"""

from functools import lru_cache
from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL_NAME


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def rerank(case_facts: str, candidates: list[dict]) -> list[dict]:
    """
    Adds a 'relevance_score' to each candidate dict and returns the list
    sorted descending by that score.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(case_facts, c["full_text"] or "") for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["relevance_score"] = float(score)

    return sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)
