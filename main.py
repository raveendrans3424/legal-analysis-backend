"""
main.py
FastAPI application entry point. Wires together the full pipeline:

  upload -> extract text -> classify -> retrieve candidates -> re-rank
  -> generate explanations -> store -> respond

Uploaded files are stored AES-256-GCM encrypted; extracted text is encrypted in
the database too. Plaintext only ever exists in memory.

Every endpoint that touches case data requires a logged-in user (JWT), and
every case query is filtered by that user's id.

Run with:
    uvicorn main:app --reload
"""

import os
import time
from urllib.parse import quote_plus

import requests
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import engine, Base, get_db
import models  # noqa: F401  (ensures models are registered before create_all)
from models import Case, CaseProvisionMatch, LegalProvision, User, Judgment, CaseJudgment, JudgmentDownload

from auth import router as auth_router, get_current_user
from extraction import extract_text_from_bytes, ALLOWED_EXTS
from crypto_utils import decrypt_bytes, decrypt_text, DecryptionError
from classification import classify_case
from retrieval import get_candidates
from ranking import rerank
from explanation import generate_explanations_batch
from utils import save_encrypted_upload, store_case, store_matches
from config import FINAL_RESULT_COUNT
from judge import judge_reply, judge_summary
import judgments as jdg


# Creates tables if they don't already exist (schema.sql covers manual setup too)
Base.metadata.create_all(bind=engine)


MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def build_source_link(act_name: str, section_number: str) -> str:
    query = f"{act_name} section {section_number}"
    return f"https://indiankanoon.org/search/?formInput={quote_plus(query)}"


app = FastAPI(title="Legal Case Analysis API")

# The frontend is served by this same app (public/ folder), so browsers call it
# from the same address and CORS is not even needed. Only these origins are allowed
# in addition. On the server set  ALLOWED_ORIGINS=https://your-domain.com  in .env
_origins = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_case(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    t0 = time.perf_counter(); marks = []

    def mark(label):
        marks.append((label, time.perf_counter()))

    # 1-2. Receive file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF, PNG or JPG.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum is {MAX_UPLOAD_MB} MB.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # 3-5. Extract complete case text (direct or OCR) from the bytes IN MEMORY
    try:
        text, method = extract_text_from_bytes(content, ext)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file")

    mark("extract")

    # Only the AES-256-GCM encrypted copy is written to disk (and only after
    # extraction succeeded, so failed uploads leave nothing behind).
    path = save_encrypted_upload(content, ext, current_user.id)

    # 6. Classify into a broad category
    category, confidence = classify_case(text)

    mark("classify")

    # 7. Use the raw extracted text as the query context (no extra LLM call)
    facts = text[:2000]

    # Persist the case record, owned by the logged-in user
    case = store_case(db, path, text, method, category, confidence, user_id=current_user.id)

    mark("save")

    # 8-9. Retrieve candidate provisions
    candidates = get_candidates(facts, db, category=category)
    mark("retrieve")

    if not candidates:
        return {
            "case_id": case.id,
            "category": category,
            "confidence": confidence,
            "provisions": [],
            "message": "No candidate provisions found in the legal database.",
        }

    # 10. Rank by case-specific relevance
    ranked = rerank(facts, candidates)
    mark("rerank")

    # 11-12. Top provisions, with one batched LLM call for all explanations
    top_results = ranked[:FINAL_RESULT_COUNT]
    explanations = generate_explanations_batch(facts, top_results)
    mark("explain")

    results = []
    for r, expl in zip(top_results, explanations):
        results.append({
            "provision_id": r["id"],
            "section": f"{r['act_name']} Section {r['section_number']}",
            "title": r["title"],
            "source": build_source_link(r["act_name"], r["section_number"]),
            "relevance_score": r["relevance_score"],
            "explanation": expl,
        })

    store_matches(db, case.id, results)
    mark("store")

    # Related Supreme Court judgments (never breaks the analysis if anything goes wrong)
    related = jdg.find_related(db, facts, top_results)
    if related:
        jdg.store_case_judgments(db, case.id, related)
    mark("judgments")

    prev, parts = t0, []
    for label, t in marks:
        parts.append(f"{label}={t - prev:.1f}s")
        prev = t
    print("TIMING " + " ".join(parts) + f" | total={prev - t0:.1f}s | provisions={len(candidates)} candidates")

    # Don't leak internal provision_id needed only for DB storage
    response_results = [
        {k: v for k, v in r.items() if k != "provision_id"} for r in results
    ]

    return {
        "case_id": case.id,
        "category": category,
        "confidence": confidence,
        "extraction_method": method,
        "provisions": response_results,
        "judgments": related,
        "downloads_left": jdg.MAX_DOWNLOADS_PER_CASE,
        "download_limit": jdg.MAX_DOWNLOADS_PER_CASE,
        "attribution": jdg.ATTRIBUTION,
    }


# ---------------- per-user case access (authorization) ----------------

def _get_owned_case_or_404(db: Session, case_id: int, user: User) -> Case:
    """
    Returns the case only if it belongs to `user`. Someone else's case gives
    the same 404 as a case that doesn't exist, so IDs can't be probed.
    """
    case = (
        db.query(Case)
        .filter(Case.id == case_id, Case.user_id == user.id)
        .first()
    )
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


def _safe_decrypt_text(value, user_id):
    try:
        return decrypt_text(value, user_id)
    except DecryptionError:
        raise HTTPException(status_code=500, detail="Stored case data could not be decrypted.")


@app.get("/cases")
def list_my_cases(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cases = (
        db.query(Case)
        .filter(Case.user_id == current_user.id)
        .order_by(Case.id.desc())
        .all()
    )
    return [
        {
            "id": c.id,
            "category": c.predicted_category,
            "confidence": c.category_confidence,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in cases
    ]


@app.get("/cases/{case_id}")
def get_my_case(case_id: int, current_user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    case = _get_owned_case_or_404(db, case_id, current_user)

    rows = (
        db.query(CaseProvisionMatch, LegalProvision)
        .join(LegalProvision, LegalProvision.id == CaseProvisionMatch.provision_id)
        .filter(CaseProvisionMatch.case_id == case.id)
        .order_by(CaseProvisionMatch.relevance_score.desc())
        .all()
    )
    return {
        "id": case.id,
        "category": case.predicted_category,
        "confidence": case.category_confidence,
        "extraction_method": case.extraction_method,
        "extracted_text": _safe_decrypt_text(case.extracted_text, current_user.id),
        "provisions": [
            {
                "section": f"{p.act_name} Section {p.section_number}",
                "title": p.title,
                "source": build_source_link(p.act_name, p.section_number),
                "relevance_score": m.relevance_score,
                "explanation": m.explanation,
            }
            for m, p in rows
        ],
        **jdg.case_judgments_view(db, case.id),
    }


@app.get("/cases/{case_id}/file")
def download_my_case_file(case_id: int, current_user: User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    """Decrypts the stored file in memory and returns the original. Owner only."""
    case = _get_owned_case_or_404(db, case_id, current_user)
    if not case.file_path or not os.path.exists(case.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(case.file_path, "rb") as f:
        blob = f.read()
    try:
        data = decrypt_bytes(blob, current_user.id)
    except DecryptionError:
        raise HTTPException(status_code=500, detail="Stored file failed its integrity check.")

    original_ext = os.path.splitext(case.file_path[:-4])[1] or ".bin"
    media = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
             ".jpeg": "image/jpeg", ".webp": "image/webp", ".bmp": "image/bmp",
             ".tiff": "image/tiff"}.get(original_ext, "application/octet-stream")
    return Response(
        content=data,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="case_{case.id}{original_ext}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


MAX_JUDGMENT_PDF_BYTES = 60 * 1024 * 1024


@app.get("/cases/{case_id}/judgments/{judgment_id}/download")
def download_judgment(case_id: int, judgment_id: int, current_user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    """
    Streams one recommended judgment PDF. Only judgments recommended for THIS case
    and owned by the logged-in user, and only up to MAX_JUDGMENT_DOWNLOADS_PER_CASE
    different judgments per case (downloading the same one again is free).
    """
    case = _get_owned_case_or_404(db, case_id, current_user)
    allowed = (db.query(CaseJudgment)
               .filter(CaseJudgment.case_id == case.id, CaseJudgment.judgment_id == judgment_id).first())
    judgment = db.query(Judgment).filter(Judgment.id == judgment_id).first() if allowed else None
    if judgment is None:
        raise HTTPException(status_code=404, detail="Judgment not found for this case")

    already = (db.query(JudgmentDownload)
               .filter(JudgmentDownload.case_id == case.id, JudgmentDownload.judgment_id == judgment_id).first())
    if not already and jdg.downloads_used(db, case.id) >= jdg.MAX_DOWNLOADS_PER_CASE:
        raise HTTPException(
            status_code=403,
            detail=f"Download limit reached ({jdg.MAX_DOWNLOADS_PER_CASE} judgments per case).",
        )

    try:
        upstream = requests.get(jdg.pdf_url(judgment), stream=True, timeout=(5, 30))
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Could not reach the judgment source. Try again later.")
    size = int(upstream.headers.get("content-length") or 0)
    if upstream.status_code != 200 or size > MAX_JUDGMENT_PDF_BYTES:
        upstream.close()
        raise HTTPException(status_code=502, detail="The judgment file is not available right now.")

    if not already:   # counted only after the source really answered
        db.add(JudgmentDownload(case_id=case.id, user_id=current_user.id, judgment_id=judgment_id))
        db.commit()

    return StreamingResponse(
        upstream.iter_content(chunk_size=65536),
        media_type="application/pdf",
        background=BackgroundTask(upstream.close),
        headers={
            "Content-Disposition": f'attachment; filename="judgment_{judgment.id}.pdf"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Attribution": "Indian Supreme Court Judgments (eCourts) CC-BY-4.0",
        },
    )


@app.delete("/cases/{case_id}")
def delete_my_case(case_id: int, current_user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Deletes the case record, its matches, and the uploaded file from disk."""
    case = _get_owned_case_or_404(db, case_id, current_user)
    file_path = case.file_path

    db.query(CaseProvisionMatch).filter(CaseProvisionMatch.case_id == case.id).delete()
    db.query(CaseJudgment).filter(CaseJudgment.case_id == case.id).delete()
    db.query(JudgmentDownload).filter(JudgmentDownload.case_id == case.id).delete()
    db.delete(case)
    db.commit()

    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass

    return {"deleted": True}


# ---------------- AI judge (login required) ----------------

class JudgeMessage(BaseModel):
    role: str
    text: str


class JudgeChatRequest(BaseModel):
    history: list[JudgeMessage] = []
    message: str


class JudgeSummaryRequest(BaseModel):
    history: list[JudgeMessage]


@app.post("/judge/chat")
def judge_chat(req: JudgeChatRequest, _user: User = Depends(get_current_user)):
    reply = judge_reply([h.dict() for h in req.history], req.message)
    return {"reply": reply}


@app.post("/judge/summary")
def judge_verdict(req: JudgeSummaryRequest, _user: User = Depends(get_current_user)):
    summary = judge_summary([h.dict() for h in req.history])
    return {"summary": summary}


# ---- Frontend: serve login.html + lexindia_app.html from the "public" folder ----
# Keep this LAST (after all API routes). If the folder is missing the API still starts.
if os.path.isdir("public"):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory="public", html=True), name="frontend")
else:
    print("WARNING: 'public' folder not found - API only (create public\\ with login.html + lexindia_app.html)")
