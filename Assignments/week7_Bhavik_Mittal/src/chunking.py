"""
Text Chunking Module
---------------------
Splits raw document text into smaller overlapping chunks to improve
retrieval accuracy. Uses a sentence-aware sliding window so chunks
don't cut sentences in half whenever possible.
"""

import re


def split_into_sentences(text: str) -> list:
    """Naive but effective sentence splitter (no external NLP API needed)."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    """
    Split text into overlapping chunks measured in characters.

    Args:
        text: raw input text
        chunk_size: target max characters per chunk
        chunk_overlap: number of trailing characters repeated in the next
                        chunk, so context isn't lost at chunk boundaries

    Returns:
        List of text chunks (strings)
    """
    sentences = split_into_sentences(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying over overlap from the end of previous chunk
            overlap_text = current[-chunk_overlap:] if current else ""
            current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def chunk_documents(docs: dict, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    """
    Chunk a dictionary of {filename: raw_text} documents.

    Returns:
        List of dicts: [{"text": chunk, "source": filename, "chunk_id": int}, ...]
    """
    all_chunks = []
    for filename, text in docs.items():
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for i, c in enumerate(chunks):
            all_chunks.append({
                "text": c,
                "source": filename,
                "chunk_id": i
            })
    return all_chunks
