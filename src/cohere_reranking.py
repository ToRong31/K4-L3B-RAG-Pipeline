"""Optional Cohere reranking after RRF; SearchResult identity is preserved."""

import os

from dotenv import load_dotenv

load_dotenv()


def cohere_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates or top_k <= 0:
        return []
    if os.getenv("COHERE_RERANK_ENABLED", "false").lower() != "true":
        return candidates[:top_k]
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return candidates[:top_k]
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
            return candidates[:top_k]
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    except Exception:
        return candidates[:top_k]
