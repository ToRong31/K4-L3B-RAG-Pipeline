"""Build the exact chunk corpus handed over by Task 4 for other indexes."""

from .task4_chunking_indexing import chunk_documents, load_documents


def load_shared_chunks() -> list[dict]:
    chunks = chunk_documents(load_documents())
    ids = [item["id"] for item in chunks]
    if not chunks:
        raise RuntimeError("No Task 4 chunks found in data/standardized")
    if len(ids) != len(set(ids)):
        raise ValueError("Task 4 corpus contains duplicate chunk IDs")
    return chunks
