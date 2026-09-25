"""Second dense branch: multilingual E5-large over the shared Task 4 chunks.

E5 has its own Chroma collection because its vectors must not be mixed with
the primary embedding model, even when both happen to have 1024 dimensions.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

from .task4_chunking_indexing import CHROMA_DIR

load_dotenv()

E5_MODEL = os.getenv("E5_MODEL", "intfloat/multilingual-e5-large")
E5_COLLECTION_NAME = os.getenv("E5_COLLECTION_NAME", "rag_documents_e5")
E5_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(E5_MODEL)


def embed_e5(texts: list[str], *, is_query: bool) -> list[list[float]]:
    """Apply the model's required query/passsage prefixes and normalize vectors."""
    if not texts:
        return []
    prefix = "query: " if is_query else "passage: "
    vectors = _model().encode(
        [prefix + text for text in texts],
        batch_size=E5_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def get_e5_collection():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=E5_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": E5_MODEL},
    )


def index_e5_chunks(chunks: list[dict]) -> int:
    """Upsert exactly the same content, IDs, and metadata as the primary index."""
    collection = get_e5_collection()
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("shared corpus contains duplicate chunk IDs")
    for start in range(0, len(chunks), E5_BATCH_SIZE):
        batch = chunks[start:start + E5_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=embed_e5([chunk["content"] for chunk in batch], is_query=False),
            metadatas=[chunk["metadata"] for chunk in batch],
        )
    existing = set(collection.get(include=[])["ids"])
    stale = sorted(existing - set(ids))
    if stale:
        collection.delete(ids=stale)
    return len(chunks)


def index_from_shared_corpus() -> int:
    """Build E5 over Task 4's chunking output and stable IDs."""
    from .shared_corpus import load_shared_chunks

    return index_e5_chunks(load_shared_chunks())


def e5_search(query: str, top_k: int = 10) -> list[dict]:
    """Return E5 cosine similarities in the normal dense SearchResult schema."""
    if not query.strip() or top_k <= 0:
        return []
    collection = get_e5_collection()
    if collection.count() == 0:
        return []
    response = collection.query(
        query_embeddings=embed_e5([query], is_query=True),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    seen = set()
    for item_id, content, metadata, distance in zip(
        response["ids"][0], response["documents"][0],
        response["metadatas"][0], response["distances"][0],
    ):
        if item_id in seen or not content:
            continue
        seen.add(item_id)
        results.append({
            "id": item_id,
            "content": content,
            "score": float(1.0 - distance),
            "metadata": {"url": None, **metadata},
            "retrieval_method": "dense",
        })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    print(f"Indexed {index_from_shared_corpus()} shared chunks with {E5_MODEL}")
