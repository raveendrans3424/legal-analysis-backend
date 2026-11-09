"""
judgments.py - runtime part of the "related judgments" feature.

Finds Supreme Court judgments related to an analysed case:
  1. judgments that CITE the matched provisions (IPC citations were converted to BNS at ingest time)
  2. judgments whose text is similar to the case facts (FAISS semantic search)
and lets the owner download at most N of the recommended PDFs per case.

Nothing big is stored here: the database holds metadata + a snippet, the PDF is
streamed from the public bucket when the user clicks Download. That is why it keeps
working after hosting: it only needs MySQL, the two judgments_* files in faiss_index/,
and outbound HTTPS to the bucket.

Judgments data: Indian Supreme Court Judgments (eCourts), licence CC-BY-4.0.
"""

import json
import os
import threading
from urllib.parse import quote

from sqlalchemy.orm import Session

from config import FAISS_INDEX_DIR, EMBEDDING_MODEL_NAME
from judgment_utils import act_key_for
from models import Judgment, JudgmentSection, CaseJudgment, JudgmentDownload

INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "judgments.index")
IDMAP_PATH = os.path.join(FAISS_INDEX_DIR, "judgments_id_map.json")

JUDGMENTS_PER_CASE = int(os.getenv("JUDGMENTS_PER_CASE", "3"))
MAX_DOWNLOADS_PER_CASE = int(os.getenv("MAX_JUDGMENT_DOWNLOADS_PER_CASE", "2"))
SIMILARITY_FLOOR = float(os.getenv("JUDGMENT_SIMILARITY_FLOOR", "0.30"))
PDF_BASE_URL = os.getenv("JUDGMENT_PDF_BASE_URL", "https://indian-supreme-court-judgments.s3.amazonaws.com/")
ATTRIBUTION = ("Source: Indian Supreme Court Judgments (eCourts, via AWS Open Data), "
               "licensed CC-BY-4.0. For reference only - not legal advice.")

_lock = threading.Lock()
_model = None
_index_cache = {"index": None, "ids": None, "loaded": False}


def embed_texts(texts: list[str]):
    """Normalised float32 embeddings (same model family as the provisions index)."""
    global _model
    import numpy as np
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def _load_index():
    if _index_cache["loaded"]:
        return _index_cache["index"], _index_cache["ids"]
    with _lock:
        if not _index_cache["loaded"]:
            if os.path.exists(INDEX_PATH) and os.path.exists(IDMAP_PATH):
                import faiss
                _index_cache["index"] = faiss.read_index(INDEX_PATH)
                with open(IDMAP_PATH, encoding="utf-8") as f:
                    _index_cache["ids"] = json.load(f)
            _index_cache["loaded"] = True
    return _index_cache["index"], _index_cache["ids"]


def reset_cache():
    _index_cache.update({"index": None, "ids": None, "loaded": False})


def pdf_url(judgment: Judgment) -> str:
    return PDF_BASE_URL.rstrip("/") + "/" + quote(judgment.pdf_key, safe="/=")


def _provision_weight(rank: int) -> float:
    return max(0.4, 1.0 - 0.15 * rank)


def _find_related(db: Session, facts: str, provisions: list[dict], limit: int) -> list[dict]:
    import numpy as np
    index, ids = _load_index()

    # 1. judgments citing the matched provisions
    cite_score: dict[int, float] = {}
    cited_by: dict[int, list[str]] = {}
    for rank, p in enumerate(provisions):
        key = act_key_for(p.get("act_name", ""))
        sec = str(p.get("section_number", "")).strip().upper()
        if not key or not sec:
            continue
        rows = (db.query(JudgmentSection.judgment_id)
                .filter(JudgmentSection.act_key == key, JudgmentSection.section_number == sec).all())
        for (jid,) in rows:
            cite_score[jid] = cite_score.get(jid, 0.0) + _provision_weight(rank)
            cited_by.setdefault(jid, []).append(f"{p.get('act_name', '')} Section {p.get('section_number', '')}")

    # 2. semantic similarity with the case facts
    sims: dict[int, float] = {}
    q = None
    if index is not None and ids:
        q = embed_texts([facts[:1500]])
        k = min(30, len(ids))
        scores, pos = index.search(q, k)
        for s, i in zip(scores[0], pos[0]):
            if i >= 0:
                sims[ids[i]] = float(s)
        pos_of = {jid: n for n, jid in enumerate(ids)}
        for jid in cite_score:                      # similarity for cited ones that missed the top-30
            if jid not in sims and jid in pos_of:
                sims[jid] = float(np.dot(q[0], index.reconstruct(pos_of[jid])))

    scored = []
    for jid in set(cite_score) | set(sims):
        sim = sims.get(jid, 0.0)
        cites = cite_score.get(jid, 0.0)
        if cites == 0 and sim < SIMILARITY_FLOOR:
            continue                                # neither cites our sections nor similar enough
        scored.append((sim + 0.25 * cites, jid))
    scored.sort(reverse=True)
    top = [jid for _, jid in scored[:limit]]
    if not top:
        return []

    by_id = {j.id: j for j in db.query(Judgment).filter(Judgment.id.in_(top)).all()}
    out = []
    for jid in top:
        j = by_id.get(jid)
        if not j:
            continue
        out.append({
            "judgment_id": j.id,
            "title": j.title,
            "petitioner": j.petitioner,
            "respondent": j.respondent,
            "court": j.court or "Supreme Court of India",
            "decision_date": j.decision_date,
            "citation": j.citation,
            "year": j.year,
            "cites_matched_sections": cited_by.get(jid, []),
            "reason": ("Cites " + ", ".join(cited_by[jid][:3])) if jid in cited_by else "Similar facts",
        })
    return out


def find_related(db: Session, facts: str, provisions: list[dict], limit: int = JUDGMENTS_PER_CASE) -> list[dict]:
    """Never raises: the main analysis must not fail because the judgments part has a problem."""
    try:
        return _find_related(db, facts, provisions, limit)
    except Exception as e:                      # noqa: BLE001
        print(f"judgments: skipped ({type(e).__name__})")
        return []


def store_case_judgments(db: Session, case_id: int, items: list[dict]) -> None:
    for rank, it in enumerate(items):
        db.add(CaseJudgment(case_id=case_id, judgment_id=it["judgment_id"], rank=rank, reason=(it.get("reason") or "")[:400]))
    db.commit()


def downloads_used(db: Session, case_id: int) -> int:
    return db.query(JudgmentDownload.judgment_id).filter(JudgmentDownload.case_id == case_id).distinct().count()


def case_judgments_view(db: Session, case_id: int) -> dict:
    """Judgments stored for a case (for GET /cases/{id}) + downloads left."""
    rows = (db.query(CaseJudgment, Judgment)
            .join(Judgment, Judgment.id == CaseJudgment.judgment_id)
            .filter(CaseJudgment.case_id == case_id).order_by(CaseJudgment.rank).all())
    items = [{
        "judgment_id": j.id, "title": j.title, "petitioner": j.petitioner, "respondent": j.respondent,
        "court": j.court or "Supreme Court of India", "decision_date": j.decision_date,
        "citation": j.citation, "year": j.year, "reason": cj.reason or "",
    } for cj, j in rows]
    return {
        "judgments": items,
        "downloads_left": max(0, MAX_DOWNLOADS_PER_CASE - downloads_used(db, case_id)),
        "download_limit": MAX_DOWNLOADS_PER_CASE,
        "attribution": ATTRIBUTION,
    }
