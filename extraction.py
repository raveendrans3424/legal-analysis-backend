"""
extraction.py
Extract text from an uploaded PDF or image, entirely IN MEMORY.
The plaintext file is never written to disk (only the AES-256 encrypted copy is).

- PDF with selectable text -> PyMuPDF directly.
- Scanned PDF / image      -> Tesseract OCR.
"""

import io
import os

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

from config import MIN_TEXT_LENGTH_FOR_DIRECT_EXTRACTION

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
ALLOWED_EXTS = (".pdf",) + IMAGE_EXTS


def extract_text_from_bytes(data: bytes, ext: str) -> tuple[str, str]:
    """Returns (text, method) where method is 'direct' or 'ocr'."""
    ext = ext.lower()

    if ext == ".pdf":
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception:
            raise ValueError("The PDF file is damaged or not a real PDF.")
        try:
            text = "".join(page.get_text() for page in doc)
        finally:
            doc.close()
        if len(text.strip()) > MIN_TEXT_LENGTH_FOR_DIRECT_EXTRACTION:
            return text, "direct"
        images = convert_from_bytes(data)
        return "\n".join(pytesseract.image_to_string(img) for img in images), "ocr"

    if ext in IMAGE_EXTS:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:
            raise ValueError("The image file is damaged or not a real image.")
        return pytesseract.image_to_string(img), "ocr"

    raise ValueError(f"Unsupported file type: {ext}. Please upload a PDF, PNG or JPG.")


def extract_text(path: str) -> tuple[str, str]:
    """Old path-based entry point (kept for compatibility / scripts)."""
    with open(path, "rb") as f:
        return extract_text_from_bytes(f.read(), os.path.splitext(path)[1])
