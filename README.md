# Legal Case Analysis Backend

FastAPI + MySQL backend that extracts text from an uploaded PDF/image,
classifies the case, and returns the most relevant legal provisions
with sources and explanations.

## 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need **Tesseract OCR** and **poppler** (for pdf2image) installed
at the OS level:

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

Optional: install [Ollama](https://ollama.com) locally for fact extraction
and explanation generation (free, local LLM):

```bash
ollama pull llama3
```

## 2. Configure environment

```bash
cp .env.example .env
# edit .env with your MySQL credentials
```

## 3. Set up the database

```sql
-- in MySQL shell, or just let main.py auto-create tables on first run
SOURCE schema.sql;
```

## 4. Seed legal provisions

Replace the sample data in `seed_provisions.py`, or provide your own
JSON file:

```bash
python seed_provisions.py your_provisions.json
```

## 5. Build the FAISS vector index

Re-run this any time provisions data changes:

```bash
python build_faiss_index.py
```

## 6. Run the API

```bash
uvicorn main:app --reload
```

Test it:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@sample_case.pdf"
```

## Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, `/analyze` endpoint, orchestrates the pipeline |
| `config.py` | Model names, categories, tunables |
| `db.py` | SQLAlchemy engine/session |
| `models.py` | ORM models (LegalProvision, Case, CaseProvisionMatch) |
| `schema.sql` | Raw SQL schema (reference) |
| `extraction.py` | PDF/OCR text extraction |
| `classification.py` | Zero-shot case category classification |
| `fact_analysis.py` | Structured fact extraction via local LLM |
| `retrieval.py` | FAISS vector search for candidate provisions |
| `ranking.py` | Cross-encoder re-ranking by case-specific relevance |
| `explanation.py` | Generates the "why this applies" text |
| `utils.py` | File save + DB persistence helpers |
| `seed_provisions.py` | One-time script to load provisions into MySQL |
| `build_faiss_index.py` | One-time script to build/rebuild the FAISS index |
