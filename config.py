"""
config.py
Central place for model names, constants, and the broad case-category list.
Change values here instead of hunting through every module.
"""

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# ---------- Database ----------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "legal_analysis_db")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ---------- File storage ----------
UPLOAD_DIR = "uploads"
FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "provisions.index")
FAISS_ID_MAP_PATH = os.path.join(FAISS_INDEX_DIR, "id_map.json")

# ---------- Models ----------
# Embedding model used for retrieval (FAISS). Swap for a legal-domain model if available.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative (legal-domain, heavier): "law-ai/InLegalBERT"

# Zero-shot classification model (broad case category)
ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"

# Cross-encoder used for re-ranking candidate provisions against case facts
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------- Broad case categories ----------
# This is a config list only — NOT a keyword-to-section map.
# The model decides which category fits; this list just defines the label space.
CASE_CATEGORIES = [
    # criminal
    "murder",
    "theft",
    "cheating",
    "assault",
    "fraud",
    "kidnapping",
    "sexual offense",
    "cybercrime",
    "criminal breach of trust",
    "defamation",
    "extortion",
    "forgery",
    "drug offense",
    "corruption",
    "traffic offense",
    "human trafficking",
    "atrocities",
    # family and women
    "domestic violence",
    "marriage and divorce",
    "inheritance and succession",
    "guardianship and custody",
    "senior citizen maintenance",
    "workplace harassment",
    "juvenile justice",
    # civil, commercial, cyber, public law
    "property dispute",
    "cyber law",
    "cheque bounce",
    "contract dispute",
    "consumer dispute",
    "labour and wages",
    "right to information",
    "constitutional rights",
]

# ---------- Local LLM (used for fact extraction + explanations) ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
INDIAN_KANOON_API_KEY = os.getenv("INDIAN_KANOON_API_KEY", "")

# ---------- Retrieval / ranking tuning ----------
FAISS_TOP_K = 20          # candidates pulled from vector search before re-ranking
FINAL_RESULT_COUNT = 5    # provisions returned to the frontend
MIN_TEXT_LENGTH_FOR_DIRECT_EXTRACTION = 50  # char threshold to decide direct-vs-OCR