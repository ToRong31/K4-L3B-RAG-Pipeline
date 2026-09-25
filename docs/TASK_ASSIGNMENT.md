# Phân công nhóm RAG Pipeline (3 người)

## Đã bao hết việc chưa?

**Có, nếu giao thêm generation cho người 3 và ghi rõ fallback cho người 2.** Ba nhãn ban đầu (data/chunking, retrieval/rerank, UI/eval) chưa nêu ai làm bước tạo câu trả lời có citation, safe refusal và ghép toàn bộ pipeline. Bảng dưới đây bao phủ Task 1–10, giao diện, đánh giá và hồ sơ nộp bài theo [hướng dẫn triển khai](STEP_BY_STEP.md) và [rubric](GRADING_RUBRIC.md).

| Người phụ trách | Phạm vi | File chính | Kết quả bàn giao / điều kiện hoàn thành |
| --- | --- | --- | --- |
| **1. Dương — Data, chunking, embedding, indexing** | Chốt corpus theo đề tài nhóm; thu thập ≥3 PDF/DOCX chính sách và ≥5 bài/page; lưu URL, tiêu đề, ngày crawl; chuyển sang Markdown; làm sạch, chia chunk; tạo embedding và index ChromaDB. | `src/task1_collect_legal_docs.py` → `src/task4_chunking_indexing.py`; `data/landing/`, `data/standardized/` | Có đủ dữ liệu thật và metadata nguồn; chunk không rỗng, ID ổn định, `chunk_index` đúng; chạy index lại không sinh bản trùng. Bàn giao `load_documents()`, `chunk_documents()`, `embed_texts()`, `get_collection()` và cách đọc cùng corpus chunks cho Trọng. Ghi rõ embedding model, dimension, chunk size/overlap và cách tái tạo index. |
| **2. Trọng — Retrieval, RRF, fallback** | Dense search dùng chung `embed_texts()` và Chroma collection của Dương; BM25 trên **cùng** corpus chunks; gộp dense + BM25 bằng RRF; làm PageIndex fallback và `retrieve()`; hiệu chỉnh ngưỡng trên query trong/ngoài domain. | `src/task5_semantic_search.py` → `src/task9_retrieval_pipeline.py` | Mỗi nhánh trả `SearchResult` đúng schema, không trùng ID, sort giảm dần và tuân thủ `top_k`; RRF chạy đúng một lần; fallback xét **cosine score gốc của dense**, không xét RRF score; lỗi PageIndex không làm hỏng luồng chat. Bàn giao `retrieve(query, top_k, score_threshold, use_reranking)` cho người 3. Reranker nâng cao (Jina/BGE) chỉ làm sau phần bắt buộc nếu còn thời gian. |
| **3. UI & Eval — Generation, giao diện, đánh giá** | Hoàn thiện generation theo provider cấu hình, context có title/source, citation truy về `sources`, safe refusal khi thiếu bằng chứng; nối `retrieve()` vào Streamlit; tạo golden dataset ≥15 câu dựa trên corpus; chạy 4 metric và A/B dense-only so với hybrid + RRF; phân tích lỗi, hoàn thiện báo cáo kết quả. | `src/task10_generation.py`, `app.py`, `group_project/evaluation/golden_dataset.json`, `group_project/evaluation/RESULT.md` | Chatbot chạy end-to-end, hiện câu trả lời, nguồn, retrieval method và score; citation khớp source; câu ngoài domain hoặc provider lỗi được xử lý an toàn. Báo cáo có faithfulness, answer relevance, context recall, context precision, A/B trên cùng dataset/generator/prompt/`top_k`, các ca lỗi và kết luận; không còn `TODO`. |

## Điểm phối hợp bắt buộc

1. **Cả nhóm:** thống nhất đề tài, phạm vi câu hỏi và nguồn dữ liệu trước khi làm. Mỗi người ghi commit/PR của mình và nộp báo cáo cá nhân theo `reports/INDIVIDUAL_REPORT.md` vào `reports/<student-id>-<short-name>.md`.
2. **Dương → Trọng:** dùng chung ID, metadata (`source`, `title`, `doc_type`, `url`, `chunk_index`), corpus chunks và embedding model. Hai nhánh dense/BM25 phải trả cùng schema trong `docs/MODULE_CONTRACTS.md`.
3. **Trọng → người 3:** chốt chữ ký `retrieve()` và ý nghĩa `score`/`retrieval_method`; cung cấp query mẫu đúng domain, ngoài domain và trường hợp fallback lỗi để kiểm tra tích hợp.
4. **Người 3 ↔ Dương, Trọng:** golden answers và `expected_context` phải kiểm chứng được từ tài liệu thật; nếu A/B cho thấy lỗi dữ liệu hoặc retrieval, chuyển ca lỗi cho đúng người sửa rồi chạy lại đánh giá.
5. **Cả nhóm trước khi nộp:** chạy `pytest -q`, demo một câu đúng domain, một câu ngoài domain và kết quả A/B; kiểm tra README/cách chạy lại, không commit `.env` hay API key. Một người có thể điều phối việc nộp, nhưng mỗi thành viên tự xác nhận phần mình.

## Lưu ý đường dẫn trong repo

`tests/test_acceptance.py` yêu cầu báo cáo ở `group_project/evaluation/RESULT.md`, trong khi template hiện nằm ở `reports/RESULT.md`. Người 3 dùng template đó để tạo file tại **đường dẫn test yêu cầu**. Template báo cáo cá nhân hiện ở `reports/INDIVIDUAL_REPORT.md`; phần đường dẫn cá nhân trong README đang khác với cấu trúc repo thực tế, nên dùng đường dẫn `reports/` nêu ở trên.
