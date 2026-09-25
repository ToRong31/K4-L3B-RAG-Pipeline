"""Offline checks for the optional retrieval services and bilingual fusion."""

import json
import sys
import types

from src.contracts import validate_search_results


def _result(item_id, score, method="dense"):
    return {
        "id": item_id,
        "content": f"Evidence {item_id}",
        "score": score,
        "metadata": {
            "source": "policy.pdf", "title": "Policy", "doc_type": "legal",
            "url": None, "chunk_index": 0,
        },
        "retrieval_method": method,
    }


def test_formulation_one_request_two_languages_and_error_fallback(monkeypatch):
    import src.query_formulation as formulation

    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(output_text=json.dumps({
                "query_vi": "học phí học kỳ", "query_en": "semester tuition"
            }))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setenv("QUERY_FORMULATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    assert formulation.formulate_query("Học phí?") == ("học phí học kỳ", "semester tuition")
    assert len(calls) == 1
    assert calls[0]["text"]["format"]["schema"]["required"] == ["query_vi", "query_en"]
    assert "VinUni" in calls[0]["instructions"]
    assert "AACC" in calls[0]["instructions"]

    class BrokenResponses:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    class BrokenOpenAI:
        def __init__(self, **kwargs):
            self.responses = BrokenResponses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=BrokenOpenAI))
    assert formulation.formulate_query("Học phí?") == ("Học phí?", "")


def test_bilingual_dense_deduplicates_before_single_rrf(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    monkeypatch.setattr(pipeline, "formulate_query", lambda q: ("học phí", "tuition"))
    monkeypatch.setattr(pipeline, "semantic_search", lambda q, top_k: (
        [_result("shared", 0.7), _result("vi", 0.6)] if q == "học phí"
        else [_result("shared", 0.8), _result("en", 0.5)]
    ))
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: [_result("vi", 3.0, "bm25")])
    monkeypatch.setattr(pipeline, "pageindex_search", lambda q, top_k: [])
    calls = []
    real_rrf = pipeline.rerank_rrf

    def traced_rrf(lists, top_k):
        calls.append(lists)
        return real_rrf(lists, top_k)

    monkeypatch.setattr(pipeline, "rerank_rrf", traced_rrf)
    output = pipeline.retrieve("Học phí?", top_k=3, score_threshold=0.3)
    assert len(calls) == 1
    assert [item["id"] for item in calls[0][0]].count("shared") == 1
    assert calls[0][0][0]["score"] == 0.8
    validate_search_results(output, top_k=3, expected_method="hybrid")


def test_cohere_scores_preserve_ids_and_failure_returns_rrf(monkeypatch):
    import src.cohere_reranking as reranking

    candidates = [_result("a", 0.03, "hybrid"), _result("b", 0.02, "hybrid")]
    monkeypatch.setenv("COHERE_RERANK_ENABLED", "true")
    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]}

    fake_requests = types.SimpleNamespace(post=lambda *args, **kwargs: Response())
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    output = reranking.cohere_rerank("học phí", candidates, 2)
    assert [item["id"] for item in output] == ["b", "a"]
    validate_search_results(output, top_k=2, expected_method="hybrid")

    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fail))
    assert reranking.cohere_rerank("học phí", candidates, 2) == candidates


def test_elasticsearch_hits_preserve_shared_chunk_ids(monkeypatch):
    import src.task6_lexical_search as lexical

    monkeypatch.setattr(lexical, "CORPUS", [])
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"hits": {"hits": [
            {"_id": "chunk-7", "_score": 4.2, "_source": {
                "content": "Evidence", "metadata": _result("chunk-7", 1)["metadata"]
            }}
        ]}}

    monkeypatch.setattr(lexical, "_request", fake_request)
    output = lexical.lexical_search("học phí", top_k=3)
    assert calls[0][2]["json"]["query"]["match"]["content"]["query"] == "học phí"
    assert output[0]["id"] == "chunk-7"
    validate_search_results(output, top_k=3, expected_method="bm25")


def test_bm25_uses_handover_corpus_when_elasticsearch_is_down(monkeypatch):
    import src.task6_lexical_search as lexical

    monkeypatch.setenv("BM25_LOCAL_FALLBACK", "true")
    monkeypatch.setattr(lexical, "CORPUS", [])
    monkeypatch.setattr(lexical, "_request", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError()))
    monkeypatch.setattr(lexical, "_shared_corpus", lambda: [
        {"id": "chunk-1", "content": "học phí VinUni", "metadata": _result("chunk-1", 1)["metadata"]},
        {"id": "chunk-2", "content": "lịch học", "metadata": _result("chunk-2", 1)["metadata"]},
    ])
    output = lexical.lexical_search("học phí", top_k=2)
    assert output[0]["id"] == "chunk-1"
    validate_search_results(output, top_k=2, expected_method="bm25")


def test_pageindex_parses_retrieval_nodes_without_network(monkeypatch):
    import src.task8_pageindex_vectorless as pageindex

    monkeypatch.setattr(pageindex, "_cache", lambda: {"policy.pdf": {"doc_id": "pi-1"}})

    def fake_request(method, path, **kwargs):
        if path.startswith("/doc/"):
            return {"retrieval_ready": True}
        if path == "/retrieval/":
            return {"retrieval_id": "ret-1"}
        return {"status": "completed", "retrieved_nodes": [{
            "node_id": "n1", "title": "Điều 1", "relevant_contents": [
                {"page_index": 2, "relevant_content": "Quy định học phí."}
            ],
        }]}

    monkeypatch.setattr(pageindex, "_request", fake_request)
    output = pageindex.pageindex_search("học phí", top_k=2)
    assert output[0]["content"] == "Quy định học phí."
    validate_search_results(output, top_k=2, expected_method="pageindex")


def test_pageindex_handles_nested_live_shape_across_documents(monkeypatch):
    import src.task8_pageindex_vectorless as pageindex

    monkeypatch.setattr(pageindex, "_cache", lambda: {
        "unrelated.pdf": {"doc_id": "pi-1"},
        "fees.pdf": {"doc_id": "pi-2"},
    })

    def fake_nodes(doc_id, query):
        content = "Thời tiết hôm nay" if doc_id == "pi-1" else "VinUni quy định mức học phí năm học"
        return [{"id": "0002", "title": "Học phí" if doc_id == "pi-2" else "Khác",
                 "relevant_contents": [[{"physical_index": "<physical_index_2>",
                                          "relevant_content": content}]]}]

    monkeypatch.setattr(pageindex, "_retrieved_nodes", fake_nodes)
    output = pageindex.pageindex_search("Mức học phí VinUni", top_k=2)
    assert output[0]["metadata"]["source"] == "fees.pdf"
    assert output[0]["metadata"]["chunk_index"] == 2
    validate_search_results(output, top_k=2, expected_method="pageindex")


def test_weighted_rrf_gives_dense_seventy_percent_of_rank_contribution():
    from src.task7_reranking import rerank_rrf_weighted

    dense = [_result("a", 0.9), _result("b", 0.8)]
    sparse = [_result("b", 5.0, "bm25"), _result("a", 4.0, "bm25")]
    output = rerank_rrf_weighted([dense, sparse], [0.7, 0.3], top_k=2)
    assert [item["id"] for item in output] == ["a", "b"]
    assert output[0]["score"] == 0.7 / 61 + 0.3 / 62
    validate_search_results(output, top_k=2, expected_method="hybrid")


def test_e5_uses_required_prefixes_and_shared_chunk_ids(monkeypatch):
    import src.task5_e5_search as e5

    encoded = []

    class Vectors:
        def tolist(self):
            return [[0.1, 0.2]]

    class Model:
        def encode(self, texts, **kwargs):
            encoded.append((texts, kwargs))
            return Vectors()

    class Collection:
        def __init__(self):
            self.upserts = []

        def upsert(self, **kwargs):
            self.upserts.append(kwargs)

        def get(self, **kwargs):
            return {"ids": ["shared-1"]}

        def count(self):
            return 1

        def query(self, **kwargs):
            assert kwargs["query_embeddings"] == [[0.1, 0.2]]
            return {"ids": [["shared-1"]], "documents": [["Evidence"]],
                    "metadatas": [[_result("shared-1", 1)["metadata"]]],
                    "distances": [[0.1]]}

    collection = Collection()
    monkeypatch.setattr(e5, "_model", lambda: Model())
    monkeypatch.setattr(e5, "get_e5_collection", lambda: collection)
    chunk = {"id": "shared-1", "content": "Evidence", "metadata": _result("shared-1", 1)["metadata"]}
    assert e5.index_e5_chunks([chunk]) == 1
    output = e5.e5_search("học phí", 2)
    assert collection.upserts[0]["ids"] == ["shared-1"]
    assert encoded[0][0] == ["passage: Evidence"]
    assert encoded[1][0] == ["query: học phí"]
    assert all(kwargs["normalize_embeddings"] for _, kwargs in encoded)
    validate_search_results(output, top_k=2, expected_method="dense")


def test_multi_dense_fuses_three_branches_once_at_35_35_30(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    monkeypatch.setenv("MULTI_DENSE_ENABLED", "true")
    monkeypatch.setenv("HYBRID_WEIGHTED_ENABLED", "true")
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "0.7")
    monkeypatch.setenv("HYBRID_BM25_WEIGHT", "0.3")
    monkeypatch.setattr(pipeline, "formulate_query", lambda q: ("học phí", "tuition"))
    monkeypatch.setattr(pipeline, "semantic_search", lambda q, top_k: [_result("a", 0.9)])
    monkeypatch.setattr(pipeline, "e5_search", lambda q, top_k: [_result("b", 0.85)])
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: [_result("c", 5, "bm25")])
    monkeypatch.setattr(pipeline, "pageindex_search", lambda q, top_k: [])
    calls = []
    real_fuse = pipeline.rerank_rrf_weighted

    def traced_fuse(lists, weights, top_k):
        calls.append((lists, weights))
        return real_fuse(lists, weights, top_k)

    monkeypatch.setattr(pipeline, "rerank_rrf_weighted", traced_fuse)
    output = pipeline.retrieve("Học phí?", top_k=3, score_threshold=0.3)
    assert len(calls) == 1
    assert calls[0][1] == [0.35, 0.35, 0.3]
    assert [[item["id"] for item in branch] for branch in calls[0][0]] == [["a"], ["b"], ["c"]]
    validate_search_results(output, top_k=3, expected_method="hybrid")


def test_e5_and_openai_fuse_35_each_and_openai_handles_missing_e5(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    monkeypatch.setenv("DENSE_BACKEND", "e5")
    monkeypatch.setenv("MULTI_DENSE_ENABLED", "true")
    monkeypatch.setenv("HYBRID_WEIGHTED_ENABLED", "true")
    monkeypatch.setattr(pipeline, "formulate_query", lambda q: (q, ""))
    monkeypatch.setattr(pipeline, "e5_search", lambda q, top_k: [_result("e5", 0.85)])
    monkeypatch.setattr(pipeline, "semantic_search", lambda q, top_k: [_result("openai", 0.75)])
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: [_result("bm25", 4, "bm25")])
    monkeypatch.setattr(pipeline, "pageindex_search", lambda q, top_k: [])
    calls = []
    real_fuse = pipeline.rerank_rrf_weighted

    def traced(lists, weights, top_k):
        calls.append((lists, weights))
        return real_fuse(lists, weights, top_k)

    monkeypatch.setattr(pipeline, "rerank_rrf_weighted", traced)
    pipeline.retrieve("học phí", top_k=3, score_threshold=0.5)
    assert [branch[0]["id"] for branch in calls[0][0]] == ["e5", "openai", "bm25"]
    assert calls[0][1] == [0.35, 0.35, 0.3]

    monkeypatch.setattr(pipeline, "e5_search", lambda q, top_k: [])
    calls.clear()
    pipeline.retrieve("học phí", top_k=3, score_threshold=0.5)
    assert [branch[0]["id"] for branch in calls[0][0]] == ["openai", "bm25"]
    assert calls[0][1] == [0.7, 0.3]


def test_generation_uses_preselected_chunks_without_retrieving_again(monkeypatch):
    import src.task10_generation as generation

    chunk = _result("selected", 0.9)
    monkeypatch.setattr(generation, "retrieve", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieved twice")))
    monkeypatch.setattr(generation, "call_llm", lambda system, prompt: "Có bằng chứng [Document 1].")

    output = generation.generate_from_chunks("Học phí?", [chunk])

    assert output["sources"] == [chunk]
    assert output["retrieval_source"] == "hybrid"
    assert "ID: selected" in generation.format_context(output["sources"])


def test_retrieval_trace_records_formulated_queries_without_second_request(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    calls = []

    def formulate(query):
        calls.append(query)
        return "thời hạn nộp học phí", "tuition payment deadline"

    monkeypatch.setattr(pipeline, "formulate_query", formulate)
    monkeypatch.setattr(pipeline, "semantic_search", lambda query, top_k: [_result("a", 0.9)])
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k: [])
    pipeline.retrieve("Quy định và thời hạn nộp học phí?", top_k=1, score_threshold=0.3)

    assert calls == ["Quy định và thời hạn nộp học phí?"]
    trace = pipeline.get_last_retrieval_trace()
    assert trace["query_vi"] == "thời hạn nộp học phí"
    assert trace["query_en"] == "tuition payment deadline"
    assert trace["dense_queries"] == ["thời hạn nộp học phí", "tuition payment deadline"]
    assert trace["bm25_query"] == "thời hạn nộp học phí"
    assert trace["best_dense_score"] == 0.9
    assert trace["score_type"] == "rrf_rank"


def test_cohere_timeout_reports_rrf_fallback(monkeypatch):
    import requests
    import src.task9_retrieval_pipeline as pipeline

    monkeypatch.setenv("COHERE_RERANK_ENABLED", "true")
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setattr(pipeline, "formulate_query", lambda q: (q, ""))
    monkeypatch.setattr(pipeline, "semantic_search", lambda q, top_k: [_result("dense", 0.8)])
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: [_result("bm25", 2, "bm25")])
    fake_requests = types.SimpleNamespace(
        post=lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout()),
        Timeout=requests.Timeout,
        HTTPError=requests.HTTPError,
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    output = pipeline.retrieve("học phí", top_k=2, score_threshold=0.3)
    trace = pipeline.get_last_retrieval_trace()

    assert output[0]["score"] < 0.02
    assert trace["cohere_status"] == "timeout_fallback"
    assert trace["score_type"] == "rrf_rank"
    assert trace["best_dense_score"] == 0.8
