# Day 8 — RAG Pipeline

## Mục tiêu

Mỗi nhóm xây dựng một chatbot RAG trả lời câu hỏi từ bộ tài liệu do nhóm thu thập. Sản phẩm phải có hybrid retrieval, citation, giao diện chat và báo cáo đánh giá.

Nhóm tự chọn bài toán và thu thập dữ liệu phù hợp; repo không cung cấp dữ liệu mẫu.

## Sản phẩm phải nộp

- Repository nhóm chạy được.
- Tối thiểu 3 tài liệu chính sách và 5 bài viết/page do nhóm tự thu thập.
- Pipeline: convert → chunk → index → dense + BM25 → RRF → fallback → generation có citation.
- Chatbot Streamlit hiển thị câu trả lời và nguồn đã dùng.
- Golden dataset tối thiểu 15 câu; đánh giá 4 metric và so sánh A/B.
- `group_project/evaluation/RESULT.md`.
- Mỗi thành viên nộp báo cáo cá nhân theo template trong `group_project/ịndividual/INDIVIDUAL_REPORT.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Điền API key cần dùng trong `.env`; không commit file này.

```bash
# 1. Thu thập và chuẩn hoá
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index và kiểm tra contract
python -m src.task4_chunking_indexing
pytest -q

# 3. Chạy sản phẩm
streamlit run app.py
```

## Lộ trình 3 giờ

| Mốc                  | Thời gian | Kết quả cần có                           |
| -------------------- | --------: | ---------------------------------------- |
| 0. Setup             |   10 phút | Môi trường và `.env` sẵn sàng            |
| 1. Data              |   25 phút | ≥3 legal, ≥5 news, Markdown đã chuẩn hoá |
| 2. Index & search    |   30 phút | ChromaDB, dense search và BM25 chạy được |
| 3. Fusion & fallback |   25 phút | RRF và fallback tuân thủ contract        |
| 4. Generation & UI   |   30 phút | Chatbot trả lời có citation              |
| 5. Evaluation        |   30 phút | 15+ Q&A, 4 metric, A/B comparison        |
| 6. Demo & handoff    |   30 phút | Test, report, demo và push repository    |

## Lưu ý quy tắc để có code quality tốt:

- Dense và BM25 nên cùng trả về `SearchResult` theo một schema.
- RRF chỉ nên dùng để gộp thứ hạng và chỉ chạy một lần.
- Fallback dùng cosine score gốc của dense retrieval.
- Threshold phải được hiệu chỉnh trên query in domain và out of domain, không có một con số đúng cho mọi corpus.

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md): schema, interface và invariant mà code/test nên tuân theo.
- [Step-by-step guide](docs/STEP_BY_STEP.md): thứ tự triển khai và tiêu chí hoàn thành từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md): Rubric thang điểm.
- [Individual report](group_project/ịndividual/INDIVIDUAL_REPORT.md): template báo cáo cá nhân.
- [Suggested topics](docs/SUGGESTED_TOPICS.md): danh sách chủ đề tham khảo, không bắt buộc.

## Kiểm tra

```bash
# Contract tests
pytest tests/test_contracts.py -q

# Acceptance tests
pytest tests/test_acceptance.py -q

# Toàn bộ
pytest -q
```

## Thiết lập retrieval của Trọng

Hiện tại dùng OpenAI `text-embedding-3-large` làm nhánh dense. Đặt
`DENSE_BACKEND=shared`, `MULTI_DENSE_ENABLED=false`, `EMBEDDING_PROVIDER=openai` và
`EMBEDDING_MODEL=text-embedding-3-large` trong `.env`, rồi chạy Task 4 để tạo
Chroma collection OpenAI trên 396 chunk chung:

```bash
python -m src.task4_chunking_indexing
```

Khởi động Docker Desktop, chạy `docker compose up -d elasticsearch`, rồi
đồng bộ chính các chunk từ `chunk_documents(load_documents())` vào BM25 bằng
`python -m src.task6_lexical_search`. Nếu Elasticsearch tạm thời không chạy,
`BM25_LOCAL_FALLBACK=true` dùng BM25 trong bộ nhớ trên cùng corpus.

Khi cần bật dense thứ hai `intfloat/multilingual-e5-large`, tạo E5 collection riêng bằng:

```bash
python -m pip install -e ".[e5]"
python -m src.task5_e5_search
```

Lệnh index E5 dùng cùng chunk ID/content/metadata của Task 4. Đặt
`DENSE_BACKEND=e5` và `MULTI_DENSE_ENABLED=true`: weighted RRF gộp E5 **0.35**,
OpenAI **0.35**, BM25 **0.30** đúng một lần. Nếu E5 chưa có index, OpenAI nhận
toàn bộ trọng số dense **0.70**. Mỗi nhánh dense gộp trùng ID giữa query Việt
và Anh trước RRF. Fallback so `SCORE_THRESHOLD` với cosine OpenAI gốc khi có
nhánh này, không so với điểm RRF hoặc Cohere.

`retrieve(query, top_k, score_threshold, use_reranking)` giữ nguyên chữ ký.
Muốn dùng PageIndex, điền `PAGEINDEX_API_KEY`, sau đó chạy
`python -m src.task8_pageindex_vectorless` để upload PDF chính sách một lần.
Chờ PageIndex xử lý xong trước khi demo fallback. Nếu PageIndex không sẵn sàng,
retrieval trả kết quả hybrid.

Để bật một lần gọi OpenAI tạo `query_vi` và `query_en`, đặt
`QUERY_FORMULATION_ENABLED=true`. Dense tìm cả hai query rồi gộp trùng ID;
BM25 dùng `query_vi`. Để bật Cohere sau RRF, điền `COHERE_API_KEY` và đặt
`COHERE_RERANK_ENABLED=true`. Nếu hai dịch vụ này lỗi, retrieval tiếp tục với
query gốc hoặc thứ hạng RRF. `use_reranking=False` giữ chế độ dense-only cho A/B.
