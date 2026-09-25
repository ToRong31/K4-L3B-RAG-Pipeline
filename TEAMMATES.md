# DANH SÁCH THÀNH VIÊN VÀ PHÂN CÔNG DỰ ÁN (TEAMMATES)

**Dự án:** Hệ thống RAG Chatbot Tra cứu & Tư vấn Học vụ VinUni (VinUniversity AI Advisor)  
**Nhóm:** sloppers  
**Repository:** [ToRong31/K4-L3B-RAG-Pipeline](https://github.com/ToRong31/K4-L3B-RAG-Pipeline)  

---

## 👥 BẢNG TỔNG HỢP PHÂN CÔNG THÀNH VIÊN

| STT | Họ và tên | Mã học viên | Vai trò chính | Nhánh Git chính | Báo cáo cá nhân |
| :-: | :--- | :-: | :--- | :--- | :--- |
| 1 | **Phạm Hoàng Trọng** | `2A202602765` | **Team Leader**<br>Retrieval Pipeline, RRF & Fallback | `main` | [K4-L3B-2A202602765-PhamHoangTrong.md](reports/K4-L3B-2A202602765-PhamHoangTrong.md) |
| 2 | **Lâm Hải Dương** | `2A202602676` | **Member 1**<br>Data, Chunking, Embedding & Indexing | `main` | [2A202602676-duong.md](reports/2A202602676-duong.md) |
| 3 | **Lê Thị Thuỳ Trang** | `2A202602678` | **Member 3**<br>Generation, Streamlit UI & Evaluation | `trangltt/chunking_n_UI`<br>`trangltt/golden_dataset`<br>`trangltt/updateUI`<br>`main` | [K4-L3B-2A202602678-LeThiThuyTrang.md](reports/K4-L3B-2A202602678-LeThiThuyTrang.md) |

---

## 📋 CHI TIẾT PHẦN VIỆC TỪNG THÀNH VIÊN

### 1. Phạm Hoàng Trọng — Leader
- **Mã học viên:** `2A202602765`
- **Vai trò:** Điều phối chung, Thiết kế Kiến trúc Retrieval, Reciprocal Rank Fusion & Vectorless Fallback.
- **Nhánh phụ trách:** `main`
- **Các phần việc / Deliverables đảm nhiệm:**
  - **Task 5 (Dense Semantic Search):** Hiện thực tìm kiếm ngữ nghĩa dense dùng chung embedding (`keepitreal/vietnamese-sbert`, dim 768) và collection ChromaDB `rag_documents`, chuẩn hóa khoảng cách cosine `1.0 - distance`; mở rộng nhánh Dense thứ 2 với mô hình `intfloat/multilingual-e5-large` (`rag_documents_e5`).
  - **Task 6 (Lexical Search - BM25):** Hiện thực tìm kiếm từ khóa Okapi BM25 chuẩn hóa theo chunk ID từ Task 4; hỗ trợ Elasticsearch và cơ chế Local BM25 fallback tự động không phụ thuộc hạ tầng ngoài.
  - **Task 7 (Reciprocal Rank Fusion & Reranking):** Hiện thực thuật toán RRF ($k=60$) gộp kết quả Dense và Lexical; mở rộng hàm `rerank_rrf_weighted` phân bổ trọng số động (70% dense / 30% BM25); tích hợp Cohere Rerank API với cơ chế tự phục hồi về RRF khi timeout.
  - **Task 8 (PageIndex Vectorless Fallback):** Hiện thực fallback phi vector dùng PageIndex REST API cho các tài liệu PDF pháp lý; hỗ trợ cache ID văn bản, bóc tách cấu trúc phân cấp đa luồng và trích xuất số trang.
  - **Task 9 (Retrieval Pipeline & Telemetry Trace):** Tích hợp hoàn chỉnh pipeline `retrieve(query, top_k)`: Query formulation song ngữ Việt - Anh, điều phối đa nhánh dense + sparse, RRF fusion, kiểm tra ngưỡng fallback dựa trên cosine score gốc và ghi log ngữ cảnh (`ContextVar`) cho UI.
  - **Bộ kiểm thử mở rộng (14 tests):** Xây dựng `tests/test_retrieval_extensions.py` kiểm tra toàn bộ kịch bản lỗi mạng, fallback, song ngữ và deduplication.
- **Files chính phụ trách:**
  - `src/task5_semantic_search.py`, `src/task5_e5_search.py`
  - `src/task6_lexical_search.py`, `src/shared_corpus.py`
  - `src/task7_reranking.py`, `src/cohere_reranking.py`
  - `src/task8_pageindex_vectorless.py`
  - `src/task9_retrieval_pipeline.py`, `src/query_formulation.py`
  - `tests/test_retrieval_extensions.py`

---

### 2. Lâm Hải Dương — Member 1
- **Mã học viên:** `2A202602676`
- **Vai trò:** Dữ liệu, Xử lý văn bản, Chunking, Embedding & Indexing ChromaDB.
- **Nhánh phụ trách:** `main`
- **Các phần việc / Deliverables đảm nhiệm:**
  - **Task 1 (Thu thập tài liệu pháp lý):** Thu thập và thẩm định 5 văn bản PDF chính thức về tuyển sinh và học vụ của VinUniversity (3 quy chế gốc và 2 văn bản chính sách/biểu phí chi tiết).
  - **Task 2 (Crawl bài viết tin tức):** Xây dựng crawler trích xuất 5 bài báo tin tức tuyển sinh VinUni kèm đầy đủ 4 trường metadata bắt buộc (`url`, `title`, `date_crawled`, `content_markdown`).
  - **Task 3 (Chuẩn hóa Markdown):** Hiện thực MarkItDown chuyển đổi tài liệu PDF và JSON sang 8 file Markdown chuẩn hóa (độ dài 5.000 – 75.000 ký tự, cấu trúc rõ ràng).
  - **Task 4 (Chunking & Indexing):** Chia văn bản thành 396 chunks bằng `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 50), nhúng vector bằng `keepitreal/vietnamese-sbert` (dim 768) và upsert vào ChromaDB collection `rag_documents`.
  - **Tài liệu bàn giao (Handoff):** Soạn thảo `DATA_HANDOVER.md` mô tả cấu trúc dữ liệu, model embedding, dimension, collection ChromaDB và cơ chế dùng chung corpus cho nhóm.
- **Files chính phụ trách:**
  - `src/task1_collect_legal_docs.py`, `data/landing/legal/`
  - `src/task2_crawl_news.py`, `data/landing/news/*.json`
  - `src/task3_convert_markdown.py`, `data/standardized/`
  - `src/task4_chunking_indexing.py`, `chroma_db/`
  - `DATA_HANDOVER.md`

---

### 3. Lê Thị Thuỳ Trang — Member 3
- **Mã học viên:** `2A202602678`
- **Vai trò:** Generation có Citation, Giao diện người dùng Streamlit 3 Tabs, Đánh giá Benchmark A/B.
- **Nhánh phụ trách:** `trangltt/chunking_n_UI`, `trangltt/golden_dataset`, `trangltt/updateUI`, `main`
- **Các phần việc / Deliverables đảm nhiệm:**
  - **Task 10 (Generation có Citation & Safe Refusal):** Hiện thực thuật toán Front-Back reordering (`reorder_for_llm`), định dạng context (`format_context`), bộ điều phối gọi đa LLM (`call_llm`), sinh câu trả lời kèm trích dẫn minh bạch `[Document n | ID]` và cơ chế Safe Refusal tự động khi không đủ bằng chứng.
  - **Giao diện Chatbot Streamlit 3 Tabs (`app.py`):**
    - *Tab 1 (Live Chat):* Giao diện hỏi đáp thời gian thực, hiển thị trích dẫn nguồn có thể mở rộng (source, score, method).
    - *Tab 2 (RAG Observability & Live Telemetry):* Thẻ KPI đo độ trễ, best score, kênh retrieval và sơ đồ trực quan 5 tầng pipeline.
    - *Tab 3 (A/B Query Formulation Workbench - Bonus +3đ):* Bộ công cụ đối sánh trực quan truy vấn Trước (Raw Query) và Sau (Bilingual Formulated Queries), tính Delta Metrics (+5 chunks mới khám phá), hiển thị side-by-side answers và ma trận chunk mapping.
  - **Golden Dataset (27 câu grounded):** Xây dựng bộ test dataset 27 câu hỏi — đáp bao phủ 3 nhóm câu hỏi phức tạp (*14 Keyword-heavy, 8 Semantic similarity, 5 Multi-source confusion*), 100% bám sát văn bản VinUni.
  - **Đánh giá A/B & Báo cáo kết quả (`RESULT.md`):** Thực nghiệm đo lường so sánh A/B giữa Config A (Dense-only) và Config B (Hybrid + RRF) trên toàn bộ 27 test cases; đo 4 metric Ragas (Faithfulness: 0.937, Answer Relevance: 0.993, Context Recall: 0.989, Context Precision: 0.829); phân tích sâu 3 ca lỗi tệ nhất và giải quyết 100% `TODO`.
- **Files chính phụ trách:**
  - `src/task10_generation.py`
  - `app.py`, `.streamlit/config.toml`
  - `group_project/evaluation/golden_dataset.json`
  - `group_project/evaluation/RESULT.md`, `reports/RESULT.md`
  - `reports/K4-L3B-2A202602678-LeThiThuyTrang.md`

---

## 🎯 KẾT QUẢ NGHIỆM THU CHUNG CỦA DỰ ÁN

- **Unit & Contract Tests:** `100% PASSED` (`pytest tests/test_contracts.py -q`)
- **Acceptance Tests:** `100% PASSED` (`pytest tests/test_acceptance.py -q`)
- **Total Test Suite:** `34/34 tests PASSED` (`pytest -q`)
- **Báo cáo kết quả:** Hoàn thành tại `group_project/evaluation/RESULT.md` và `reports/RESULT.md` (không còn `TODO`).
- **Giao diện Streamlit:** Hoàn chỉnh 3 Tabs, tích hợp live chatbot, observability và công cụ đối sánh Query Formulation (đạt điểm thưởng rubric +3).
