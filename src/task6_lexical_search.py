"""BM25 over the same chunk IDs as Chroma, backed by Elasticsearch.

``CORPUS`` supports a small in-memory BM25 during isolated contract tests. In
normal use it is empty and Elasticsearch is the only lexical search backend.
"""

import json
import math
import os
import re
from collections import Counter
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

CORPUS: list[dict] = []
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200").rstrip("/")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "rag_chunks")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER", "")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _local_bm25(query: str, corpus: list[dict], top_k: int) -> list[dict]:
    """Deterministic BM25 for injected test data; production uses Elasticsearch."""
    tokens = [_tokens(item["content"]) for item in corpus]
    query_terms = set(_tokens(query))
    if not tokens or not query_terms:
        return []
    n = len(tokens)
    avg_length = sum(map(len, tokens)) / n or 1
    doc_frequency = Counter(term for doc in tokens for term in set(doc))
    results = []
    for item, terms in zip(corpus, tokens):
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            idf = math.log(1 + (n - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * len(terms) / avg_length))
        if score > 0:
            results.append({**item, "score": float(score), "retrieval_method": "bm25"})
    return sorted(results, key=lambda item: (-item["score"], item["id"]))[:top_k]


def _request(method: str, path: str, **kwargs):
    import requests

    headers = kwargs.pop("headers", {})
    timeout = kwargs.pop("timeout", 15)
    if ELASTICSEARCH_API_KEY:
        headers["Authorization"] = f"ApiKey {ELASTICSEARCH_API_KEY}"
    auth = (ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD) if ELASTICSEARCH_USER else None
    response = requests.request(
        method, f"{ELASTICSEARCH_URL}/{path.lstrip('/')}",
        headers=headers, auth=auth, timeout=timeout, **kwargs,
    )
    response.raise_for_status()
    return response.json() if response.content else {}


def build_bm25_index(corpus: list[dict]):
    """Upsert the supplied Task 4 chunks into Elasticsearch with stable IDs."""
    from urllib.parse import quote

    if not corpus:
        return 0
    index = quote(ELASTICSEARCH_INDEX, safe="")
    mapping = {
        "mappings": {"properties": {
            "content": {"type": "text", "analyzer": "standard"},
            "metadata": {"type": "object", "enabled": True},
        }}
    }
    import requests

    response = requests.request(
        "PUT", f"{ELASTICSEARCH_URL}/{index}", json=mapping,
        headers={"Authorization": f"ApiKey {ELASTICSEARCH_API_KEY}"} if ELASTICSEARCH_API_KEY else {},
        auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD) if ELASTICSEARCH_USER else None,
        timeout=15,
    )
    if response.status_code not in (200, 201, 400):
        response.raise_for_status()
    if response.status_code == 400 and "resource_already_exists_exception" not in response.text:
        response.raise_for_status()
    lines = []
    for item in corpus:
        lines.append(json.dumps({"index": {"_id": item["id"]}}, ensure_ascii=False))
        lines.append(json.dumps({"content": item["content"], "metadata": item["metadata"]}, ensure_ascii=False))
    result = _request(
        "POST", f"{index}/_bulk?refresh=true",
        data="\n".join(lines) + "\n",
        headers={"Content-Type": "application/x-ndjson"},
    )
    if result.get("errors"):
        raise RuntimeError("Elasticsearch bulk indexing reported item errors")
    return len(corpus)


def chunks_from_chroma() -> list[dict]:
    """Read Dương's indexed chunks so lexical and dense IDs match exactly."""
    from .task4_chunking_indexing import get_collection

    response = get_collection().get(include=["documents", "metadatas"])
    return [
        {"id": item_id, "content": content, "metadata": {"url": None, **metadata}}
        for item_id, content, metadata in zip(
            response["ids"], response["documents"], response["metadatas"]
        )
        if content
    ]


@lru_cache(maxsize=1)
def _shared_corpus() -> list[dict]:
    from .shared_corpus import load_shared_chunks

    return load_shared_chunks()


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return BM25 SearchResults sorted by Elasticsearch relevance score."""
    if not query.strip() or top_k <= 0:
        return []
    if CORPUS:
        return _local_bm25(query, CORPUS, top_k)
    from urllib.parse import quote

    try:
        response = _request(
            "POST", f"{quote(ELASTICSEARCH_INDEX, safe='')}/_search",
            json={"size": top_k, "query": {"match": {"content": {"query": query}}}},
            timeout=3,
        )
    except Exception:
        if os.getenv("BM25_LOCAL_FALLBACK", "true").lower() != "true":
            raise
        return _local_bm25(query, _shared_corpus(), top_k)
    results = []
    seen = set()
    for hit in response.get("hits", {}).get("hits", []):
        item_id = hit["_id"]
        if item_id in seen:
            continue
        seen.add(item_id)
        source = hit["_source"]
        results.append({
            "id": item_id,
            "content": source["content"],
            "score": float(hit["_score"]),
            "metadata": source["metadata"],
            "retrieval_method": "bm25",
        })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    from .shared_corpus import load_shared_chunks

    chunks = load_shared_chunks()
    print(f"Indexed {build_bm25_index(chunks)} shared chunks into Elasticsearch")
