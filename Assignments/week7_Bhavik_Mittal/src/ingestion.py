"""
Document Ingestion Module
--------------------------
Loads raw text out of PDF, TXT, and DOCX files.
No external APIs are used — all parsing happens locally.
"""

import os
from pypdf import PdfReader
import docx


def load_pdf(path: str) -> str:
    """Extract text from a PDF file, page by page."""
    reader = PdfReader(path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def load_txt(path: str) -> str:
    """Load raw text from a .txt file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_docx(path: str) -> str:
    """Extract text from a Word document."""
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_document(path: str) -> str:
    """
    Dispatch to the correct loader based on file extension.
    Supported: .pdf, .txt, .docx
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = load_pdf(path)
    elif ext == ".txt":
        text = load_txt(path)
    elif ext == ".docx":
        text = load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .txt, or .docx")

    if not text or not text.strip():
        raise ValueError(
            f"No extractable text found in '{path}'. "
            "The file may be a scanned/image-only PDF."
        )
    return text


def load_multiple_documents(paths: list) -> dict:
    """
    Load several documents at once.
    Returns: {filename: raw_text}
    """
    docs = {}
    for path in paths:
        filename = os.path.basename(path)
        try:
            docs[filename] = load_document(path)
        except Exception as e:
            print(f"[WARN] Skipping '{filename}': {e}")
    return docs
