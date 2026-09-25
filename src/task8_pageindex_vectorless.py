"""PageIndex vectorless fallback using the API of pinned SDK version 0.2.8.

Uploads are explicit and cached; searches read PageIndex retrieval nodes as
evidence. REST calls use timeouts because the pinned SDK does not expose them.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
LEGAL_DIR = ROOT / "data" / "landing" / "legal"
CACHE_PATH = ROOT / "pageindex_doc_ids.json"
BASE_URL = "https://api.pageindex.ai"


def _headers() -> dict[str, str]:
    key = os.getenv("PAGEINDEX_API_KEY")
    if not key:
        raise RuntimeError("PAGEINDEX_API_KEY is required")
    return {"api_key": key}


def _request(method: str, path: str, **kwargs) -> dict:
    import requests

    response = requests.request(
        method, f"{BASE_URL}{path}", headers=_headers(), timeout=20, **kwargs
    )
    response.raise_for_status()
    return response.json()


def _cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def upload_documents() -> None:
    """Upload policy PDFs with stable cache keys; processing continues remotely."""
    if not LEGAL_DIR.exists():
        return
    cache = _cache()
    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        fingerprint = f"{path.stat().st_size}:{path.stat().st_mtime_ns}"
        if cache.get(path.name, {}).get("fingerprint") == fingerprint:
            continue
        with path.open("rb") as file:
            import requests

            response = requests.post(
                f"{BASE_URL}/doc/", headers=_headers(),
                files={"file": (path.name, file, "application/pdf")},
                data={"if_retrieval": "true"}, timeout=60,
            )
        response.raise_for_status()
        cache[path.name] = {"doc_id": response.json()["doc_id"], "fingerprint": fingerprint}
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _retrieved_nodes(doc_id: str, query: str) -> list[dict]:
    ready = _request("GET", f"/doc/{doc_id}/?type=tree")
    if not ready.get("retrieval_ready"):
        return []
    task = _request("POST", "/retrieval/", json={
        "doc_id": doc_id, "query": query, "thinking": False,
    })
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        result = _request("GET", f"/retrieval/{task['retrieval_id']}/")
        if result.get("status") == "completed":
            return result.get("retrieved_nodes", [])
        if result.get("status") == "failed":
            return []
        time.sleep(1)
    return []


def _flatten_contents(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_contents(item)
    elif isinstance(value, (dict, str)):
        yield value


def _page_number(part: dict, node: dict) -> int:
    value = part.get("page_index") or part.get("physical_index") or node.get("page_index") or 0
    if isinstance(value, int):
        return max(0, value)
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else 0


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Return PageIndex retrieved snippets as contract-valid SearchResults."""
    if not query.strip() or top_k <= 0:
        return []
    results = []
    seen = set()
    documents = [(source, item["doc_id"]) for source, item in _cache().items() if item.get("doc_id")]
    if not documents:
        return []
    query_terms = set(re.findall(r"\w+", query.casefold(), flags=re.UNICODE))
    with ThreadPoolExecutor(max_workers=min(5, len(documents))) as executor:
        futures = {
            executor.submit(_retrieved_nodes, doc_id, query): (source, doc_id)
            for source, doc_id in documents
        }
        for future in as_completed(futures):
            source, doc_id = futures[future]
            try:
                nodes = future.result()
            except Exception:
                continue
            for rank, node in enumerate(nodes, 1):
                contents = node.get("relevant_contents") or []
                if not contents and node.get("text"):
                    contents = [{"relevant_content": node["text"], "page_index": node.get("page_index", 0)}]
                for part in _flatten_contents(contents):
                    if isinstance(part, str):
                        part = {"relevant_content": part}
                    content = str(part.get("relevant_content") or "").strip()
                    page = _page_number(part, node)
                    node_id = node.get("node_id") or node.get("id") or page
                    item_id = f"pageindex:{doc_id}:{node_id}:{page}"
                    if not content or item_id in seen:
                        continue
                    seen.add(item_id)
                    terms = set(re.findall(r"\w+", (content + " " + str(node.get("title", ""))).casefold(), flags=re.UNICODE))
                    overlap = len(query_terms & terms) / max(1, len(query_terms))
                    results.append({
                        "id": item_id,
                        "content": content,
                        "score": float(overlap + 0.01 / rank),
                        "metadata": {
                            "source": source,
                            "title": node.get("title") or Path(source).stem,
                            "doc_type": "legal",
                            "url": None,
                            "chunk_index": int(page),
                        },
                        "retrieval_method": "pageindex",
                    })
    return sorted(results, key=lambda item: (-item["score"], item["id"]))[:top_k]


if __name__ == "__main__":
    upload_documents()
