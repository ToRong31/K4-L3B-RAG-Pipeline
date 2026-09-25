"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

import os
from contextvars import ContextVar

from dotenv import load_dotenv

from .cohere_reranking import cohere_rerank, get_last_rerank_status
from .query_formulation import formulate_query
from .task5_e5_search import e5_search
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf, rerank_rrf_weighted
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or "0.3")
DEFAULT_TOP_K = 5
_last_retrieval_trace: ContextVar[dict] = ContextVar("last_retrieval_trace", default={})


def get_last_retrieval_trace() -> dict:
    """Return query details from the latest retrieve call in this execution context."""
    return dict(_last_retrieval_trace.get())


def _update_retrieval_trace(**changes) -> None:
    _last_retrieval_trace.set({**_last_retrieval_trace.get(), **changes})


def _search_bilingual(search, query_vi: str, query_en: str, top_k: int) -> list[dict]:
    """Merge translations within one model branch before weighted RRF."""
    results = search(query_vi, top_k=top_k)
    if query_en and query_en.casefold() != query_vi.casefold():
        by_id = {item["id"]: item for item in results}
        for item in search(query_en, top_k=top_k):
            if item["id"] not in by_id or item["score"] > by_id[item["id"]]["score"]:
                by_id[item["id"]] = item
        results = sorted(by_id.values(), key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    _last_retrieval_trace.set({
        "query_vi": "", "query_en": "", "dense_queries": [], "bm25_query": "",
        "best_dense_score": None, "score_type": None, "cohere_status": "not_run",
    })
    if not query.strip() or top_k <= 0:
        return []
    query_vi, query_en = formulate_query(query)
    dense_queries = [query_vi]
    if query_en and query_en.casefold() != query_vi.casefold():
        dense_queries.append(query_en)
    _update_retrieval_trace(
        query_vi=query_vi,
        query_en=query_en,
        dense_queries=dense_queries,
        bm25_query=query_vi,
    )
    dense_backend = os.getenv("DENSE_BACKEND", "shared").lower()
    if dense_backend == "e5":
        primary_search, secondary_search = e5_search, semantic_search
    elif dense_backend == "shared":
        primary_search, secondary_search = semantic_search, e5_search
    else:
        raise ValueError("DENSE_BACKEND must be 'e5' or 'shared'")
    dense = _search_bilingual(primary_search, query_vi, query_en, top_k * 2)
    secondary_dense = []
    if os.getenv("MULTI_DENSE_ENABLED", "false").lower() == "true":
        try:
            secondary_dense = _search_bilingual(secondary_search, query_vi, query_en, top_k * 2)
        except Exception:
            secondary_dense = []
    try:
        sparse = lexical_search(query_vi, top_k=top_k * 2)
    except Exception:
        sparse = []
    use_cohere = (
        use_reranking
        and os.getenv("COHERE_RERANK_ENABLED", "false").lower() == "true"
        and bool(os.getenv("COHERE_API_KEY"))
    )
    candidate_k = top_k * 3 if use_cohere else top_k
    if use_reranking:
        dense_branches = [branch for branch in (dense, secondary_dense) if branch]
        ranked_lists = dense_branches + [sparse]
        if os.getenv("HYBRID_WEIGHTED_ENABLED", "true").lower() == "true":
            dense_weight = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
            keyword_weight = float(os.getenv("HYBRID_BM25_WEIGHT", "0.3"))
            weights = (
                [dense_weight / len(dense_branches)] * len(dense_branches) + [keyword_weight]
                if dense_branches else [1.0]
            )
            hybrid = rerank_rrf_weighted(
                ranked_lists, weights,
                top_k=candidate_k,
            )
        else:
            hybrid = rerank_rrf(ranked_lists, top_k=candidate_k)
    else:
        hybrid = (dense or secondary_dense)[:top_k]
    openai_dense = dense if dense_backend == "shared" else secondary_dense
    fallback_dense = openai_dense or dense or secondary_dense
    best_dense_score = max((item["score"] for item in fallback_dense), default=0.0)
    _update_retrieval_trace(best_dense_score=best_dense_score)
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                _update_retrieval_trace(score_type="pageindex", cohere_status="skipped_pageindex")
                return fallback[:top_k]
        except Exception:
            pass
    if use_cohere:
        reranked = cohere_rerank(query_vi, hybrid, top_k)
        cohere_status = get_last_rerank_status()
        _update_retrieval_trace(
            cohere_status=cohere_status,
            score_type="cohere_relevance" if cohere_status == "applied" else "rrf_rank",
        )
        return reranked
    _update_retrieval_trace(
        cohere_status="disabled",
        score_type="dense_cosine" if not use_reranking else "rrf_rank",
    )
    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
