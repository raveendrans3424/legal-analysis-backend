"""
utils.py
Helpers used by main.py: saving ENCRYPTED uploads and persisting case / match
records. Plaintext case files and plaintext extracted text never reach disk/DB.
"""

import os
import uuid

from sqlalchemy.orm import Session

from config import UPLOAD_DIR
from crypto_utils import encrypt_bytes, encrypt_text
from models import Case, CaseProvisionMatch


def save_encrypted_upload(content: bytes, original_ext: str, user_id: int) -> str:
    """
    Encrypts `content` with AES-256-GCM and writes uploads/<random>.<ext>.enc
    (the original extension is kept inside the name so we know the file type
    when decrypting). Written atomically: temp file first, then rename.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = f"{uuid.uuid4().hex}{original_ext.lower()}.enc"
    path = os.path.join(UPLOAD_DIR, name)
    tmp = path + ".part"

    with open(tmp, "wb") as f:
        f.write(encrypt_bytes(content, user_id))
    os.replace(tmp, path)
    return path


def store_case(db: Session, file_path: str, text: str, method: str,
               category: str, confidence: float, user_id: int) -> Case:
    case = Case(
        user_id=user_id,
        file_path=file_path,
        extracted_text=encrypt_text(text, user_id),
        extraction_method=method,
        predicted_category=category,
        category_confidence=confidence,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def store_matches(db: Session, case_id: int, results: list[dict]) -> None:
    for r in results:
        db.add(CaseProvisionMatch(
            case_id=case_id,
            provision_id=r["provision_id"],
            relevance_score=r["relevance_score"],
            explanation=r["explanation"],
        ))
    db.commit()
