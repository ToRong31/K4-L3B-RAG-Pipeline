"""Optional Cohere reranking after RRF; SearchResult identity is preserved."""

import os
from contextvars import ContextVar

from dotenv import load_dotenv

load_dotenv()
_last_rerank_status: ContextVar[str] = ContextVar("cohere_rerank_status", default="not_run")


def get_last_rerank_status() -> str:
    return _last_rerank_status.get()


def cohere_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    _last_rerank_status.set("not_run")
    if not candidates or top_k <= 0:
        _last_rerank_status.set("empty_candidates")
        return []
    if os.getenv("COHERE_RERANK_ENABLED", "false").lower() != "true":
        _last_rerank_status.set("disabled")
        return candidates[:top_k]
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        _last_rerank_status.set("missing_key")
        return candidates[:top_k]
    requests = None
    try:
        import requests

        response = requests.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-fast"),
                "query": query,
                "documents": [item["content"] for item in candidates],
                "top_n": min(top_k, len(candidates)),
            },
            timeout=15,
        )
        response.raise_for_status()
        results = []
        seen = set()
        for item in response.json()["results"]:
            index = int(item["index"])
            if index < 0 or index >= len(candidates) or index in seen:
                continue
            seen.add(index)
            results.append({
                **candidates[index],
                "score": float(item["relevance_score"]),
            })
        if not results:
            _last_rerank_status.set("empty_fallback")
            return candidates[:top_k]
        _last_rerank_status.set("applied")
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    except Exception as exc:
        if isinstance(exc, getattr(requests, "Timeout", ())):
            status = "timeout_fallback"
        elif isinstance(exc, getattr(requests, "HTTPError", ())):
            status = "http_error_fallback"
        else:
            status = "error_fallback"
        _last_rerank_status.set(status)
        return candidates[:top_k]
