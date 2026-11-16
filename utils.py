"""
utils.py
Helpers used by main.py: saving ENCRYPTED uploads and persisting case / match
records. Plaintext case files and plaintext extracted text never reach disk/DB.

File storage strategy:
  - If AWS S3 env vars are set  → upload to S3 (persistent, survives EC2 restarts)
  - Otherwise                   → save to local disk (dev / local use)
"""

import os
import uuid
import io

from sqlalchemy.orm import Session

from config import UPLOAD_DIR, USE_S3, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_REGION
from crypto_utils import encrypt_bytes, encrypt_text, decrypt_bytes
from models import Case, CaseProvisionMatch

# Lazy-import boto3 only when S3 is actually used
_s3_client = None


def _get_s3():
    """Return a cached boto3 S3 client (created on first call)."""
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    return _s3_client


def save_encrypted_upload(content: bytes, original_ext: str, user_id: int) -> str:
    """
    Encrypts `content` with AES-256-GCM.
    - S3 mode  : uploads to S3 and returns "s3://<bucket>/<key>"
    - Local mode: writes uploads/<random>.<ext>.enc and returns the path.
    """
    name = f"{uuid.uuid4().hex}{original_ext.lower()}.enc"
    encrypted = encrypt_bytes(content, user_id)

    if USE_S3:
        key = f"uploads/{name}"
        _get_s3().upload_fileobj(
            io.BytesIO(encrypted),
            AWS_S3_BUCKET,
            key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        return f"s3://{AWS_S3_BUCKET}/{key}"

    # Local fallback
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, name)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(encrypted)
    os.replace(tmp, path)
    return path


def load_encrypted_upload(file_path: str) -> bytes:
    """
    Reads the raw encrypted bytes from wherever they were saved.
    Handles both S3 paths ("s3://bucket/key") and local paths.
    """
    if file_path.startswith("s3://"):
        _, rest = file_path[5:].split("/", 1)
        bucket, key = rest.split("/", 1) if "/" in rest else (rest, "")
        buf = io.BytesIO()
        _get_s3().download_fileobj(bucket, key, buf)
        return buf.getvalue()

    with open(file_path, "rb") as f:
        return f.read()


def delete_encrypted_upload(file_path: str) -> None:
    """Deletes file from S3 or local disk. Silently ignores missing files."""
    try:
        if file_path.startswith("s3://"):
            _, rest = file_path[5:].split("/", 1)
            bucket, key = rest.split("/", 1) if "/" in rest else (rest, "")
            _get_s3().delete_object(Bucket=bucket, Key=key)
        elif os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass


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
