"""
models.py
SQLAlchemy ORM models. Added: User table, and Case.user_id so every case
belongs to exactly one user (this is what makes authorization possible).
"""

import datetime
from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, TIMESTAMP, Enum, Index
from sqlalchemy.dialects.mysql import LONGTEXT

from db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)  # Argon2id hash, never the password
    role = Column(String(50), default="advocate")
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)


class LegalProvision(Base):
    __tablename__ = "legal_provisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    act_name = Column(String(255))
    section_number = Column(String(50))
    title = Column(String(500))
    full_text = Column(Text)
    category = Column(String(100))
    source_url = Column(String(500))
    embedding_id = Column(Integer)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    file_path = Column(String(500))
    # AES-256-GCM encrypted + base64 text is ~1.4x longer than the original, so use LONGTEXT on MySQL
    extracted_text = Column(Text().with_variant(LONGTEXT(), "mysql"))
    extraction_method = Column(Enum("direct", "ocr", name="extraction_method_enum"))
    predicted_category = Column(String(100))
    category_confidence = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)


class CaseProvisionMatch(Base):
    __tablename__ = "case_provision_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"))
    provision_id = Column(Integer, ForeignKey("legal_provisions.id"))
    relevance_score = Column(Float)
    explanation = Column(Text)


# ---------------- court judgments (Supreme Court open dataset, CC-BY-4.0) ----------------
# Only metadata + a short text snippet live in our database. The PDF itself is
# fetched on demand from the public bucket (nothing big is stored on our server).

class Judgment(Base):
    __tablename__ = "judgments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500))
    petitioner = Column(String(500))
    respondent = Column(String(500))
    judge = Column(String(500))
    court = Column(String(150))
    decision_date = Column(String(30))
    citation = Column(String(255))
    year = Column(Integer, index=True)
    pdf_key = Column(String(500), unique=True)     # e.g. data/pdf/year=2019/english/judgment12.pdf
    snippet = Column(Text)                          # first part of the judgment text
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)


class JudgmentSection(Base):
    """Which statute sections a judgment cites (IPC citations are stored as the matching BNS section)."""
    __tablename__ = "judgment_sections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    judgment_id = Column(Integer, ForeignKey("judgments.id"), index=True)
    act_key = Column(String(100))                   # normalised act name prefix, e.g. bharatiyanyayasanhita
    section_number = Column(String(50))
    __table_args__ = (Index("ix_js_act_section", "act_key", "section_number"),)


class CaseJudgment(Base):
    """Judgments recommended for one analysed case. Downloads are only allowed for these."""
    __tablename__ = "case_judgments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"), index=True)
    judgment_id = Column(Integer, ForeignKey("judgments.id"))
    rank = Column(Integer)
    reason = Column(String(400))


class JudgmentDownload(Base):
    __tablename__ = "judgment_downloads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    judgment_id = Column(Integer, ForeignKey("judgments.id"))
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
