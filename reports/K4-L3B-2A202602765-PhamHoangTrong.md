# Individual contribution report

## Thông tin

- Họ và tên: Phạm Hoàng Trọng
- Mã học viên: 2A202602765
- Nhóm: sloppers
- Repository/branch: ToRong31/K4-L3B-RAG-Pipeline / main

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Task 5 — Semantic Search & Multilingual Dense Search** | Hiện thực tìm kiếm ngữ nghĩa dense dùng chung embedding (`keepitreal/vietnamese-sbert`, dim 768) và collection ChromaDB của Dương, chuẩn hóa điểm tương đồng `1.0 - distance`; mở rộng thêm nhánh dense thứ hai sử dụng `intfloat/multilingual-e5-large` với collection riêng (`rag_documents_e5`) và tiền tố chuẩn `query: ` / `passage: ` | `src/task5_semantic_search.py`, `src/task5_e5_search.py` / Commit `d2bc677` | Done |
| **Task 6 — Lexical Search (BM25)** | Hiện thực tìm kiếm từ khóa BM25 trên cùng tập chunk ID và metadata từ Task 4; hỗ trợ production Elasticsearch (`rag_chunks`, bulk upsert) kết hợp cơ chế Local BM25 fallback tự động (Okapi BM25 chuẩn) khi không có Elasticsearch | `src/task6_lexical_search.py`, `src/shared_corpus.py` / Commit `d2bc677` | Done |
| **Task 7 — Reciprocal Rank Fusion (RRF) & Reranking** | Hiện thực thuật toán RRF thuần gộp dense và lexical rank ($RRF(d) = \sum \frac{1}{k + rank}$, $k=60$); mở rộng hàm `rerank_rrf_weighted` phân bổ trọng số động (70% dense / 30% BM25); tích hợp Cohere Rerank API (`rerank-v4.0-fast`) với cơ chế fallback an toàn về RRF khi lỗi mạng hoặc timeout | `src/task7_reranking.py`, `src/cohere_reranking.py` / Commit `d2bc677`, `964b16a` | Done |
| **Task 8 — PageIndex Vectorless Fallback** | Hiện thực fallback phi vector dùng PageIndex REST API cho các tài liệu PDF pháp lý; hỗ trợ cache mã văn bản (`pageindex_doc_ids.json`), truy vấn cây phân cấp đa luồng (`ThreadPoolExecutor`), bóc tách cấu trúc lồng nhau và trích xuất số trang chuẩn hóa về `SearchResult` | `src/task8_pageindex_vectorless.py` / Commit `d2bc677` | Done |
| **Task 9 — Retrieval Pipeline & Telemetry Trace** | Tích hợp toàn bộ pipeline: Query formulation song ngữ Việt - Anh, điều phối đa nhánh dense + sparse, RRF fusion, kiểm tra ngưỡng fallback dựa trên **cosine score gốc của dense**, kích hoạt PageIndex khi dưới ngưỡng và tự phục hồi khi có lỗi; lưu vết ngữ cảnh truy vấn (`ContextVar`) phục vụ giao diện quan sát UI | `src/task9_retrieval_pipeline.py`, `src/query_formulation.py` / Commit `d2bc677`, `da3f652`, `78ede9f`, `964b16a` | Done |
| **Bàn giao Pipeline & Hỗ trợ UI** | Bàn giao chữ ký hàm `retrieve()` và hiện thực `generate_from_chunks` trong Task 10 giúp tránh gọi truy vấn hai lần; bổ sung cấu hình Streamlit và tích hợp telemetry trace vào bảng điều khiển giám sát pipeline | `src/task10_generation.py`, `app.py`, `.streamlit/config.toml` / Commit `da3f652`, `78ede9f`, `964b16a` | Done |
| **Kiểm thử Retrieval Extensions (14 tests)** | Xây dựng 14 bài kiểm thử đơn vị và tích hợp kiểm tra toàn diện các kịch bản: song ngữ, RRF deduplication, Cohere fallback khi timeout/lỗi HTTP, BM25 local fallback, cấu trúc lồng PageIndex, E5 prefixing và trace logging | `tests/test_retrieval_extensions.py` / Commit `d2bc677`, `da3f652`, `78ede9f`, `964b16a` | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Sử dụng điểm tương đồng cosine gốc của nhánh dense (`best_dense_score`) làm điều kiện duy nhất để kích hoạt PageIndex Fallback (`best_dense_score < score_threshold`), tuyệt đối không so sánh ngưỡng với điểm RRF.  
   **Lý do/evidence:** Điểm RRF chỉ phản ánh thứ hạng tương đối của tài liệu trong tập kết quả gộp theo công thức cộng nghịch đảo thứ hạng ($1/(k + rank)$), không đại diện cho độ tương quan ngữ nghĩa tuyệt đối với kho tri thức. Một câu hỏi hoàn toàn ngoài domain vẫn có thể nhận điểm RRF cao nếu tình cờ đứng đầu cả hai nhánh. Chỉ có cosine similarity gốc từ mô hình embedding mới phân biệt chính xác câu hỏi trong hay ngoài domain để kích hoạt fallback an toàn.  
   **Trade-off:** Phải lưu vết và tách riêng điểm dense gốc trước khi đưa vào hàm fusion, đồng thời bổ sung logic ghi nhận loại điểm (`score_type`) trong telemetry để UI hiển thị minh bạch.

2. **Quyết định:** Thiết kế kiến trúc Hybrid hai tầng: ưu tiên Production Backend (Elasticsearch cho lexical, Cohere API cho reranking, PageIndex cho fallback) nhưng luôn trang bị Zero-dependency Local Fallback (Local BM25 trên shared corpus, bỏ qua Cohere giữ RRF khi timeout, nuốt lỗi PageIndex giữ hybrid).  
   **Lý do/evidence:** Môi trường demo hoặc máy chấm bài của giảng viên có thể không có sẵn cụm Elasticsearch hoặc gặp gián đoạn kết nối mạng / hết hạn API key. Nhờ cơ chế fallback tự động này, toàn bộ hệ thống retrieval và giao diện chatbot vẫn vận hành trơn tru 100% offline mà không bao giờ bị crash hoặc trả về lỗi 500.  
   **Trade-off:** Tăng khối lượng mã nguồn xử lý ngoại lệ và đòi hỏi viết nhiều test mock giả lập các tình huống mất mạng/timeout, nhưng mang lại độ tin cậy và tính ổn định tối đa cho sản phẩm.

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:**  
  - `pytest tests/test_contracts.py -k "semantic or lexical or rrf or pageindex or retrieve" -v`
  - `pytest tests/test_retrieval_extensions.py -v`
  - Kiểm thử trực tiếp với các query thực tế: query trong domain (*"Mức học phí VinUni là bao nhiêu?", "Điều kiện duy trì học bổng VinUni"*) và query ngoài domain (*"Thời tiết Hà Nội hôm nay thế nào?", "Học phí trường Đại học Bách Khoa"*) để xác thực ngưỡng kích hoạt fallback.
- **Kết quả trước/sau nếu có:**  
  - Trước khi triển khai: Các bài test contract retrieval đều fail do chưa có hàm hoặc chưa đáp ứng đúng chữ ký và kiểu dữ liệu `SearchResult`.
  - Sau khi hoàn thành: **6/6 Contract tests retrieval PASSED**, **14/14 bài test chuyên sâu trong `tests/test_retrieval_extensions.py` PASSED (100%)**, toàn bộ 5/5 Acceptance tests PASSED. Các truy vấn đúng domain đạt cosine score > 0.65 không bị rơi vào fallback, còn câu hỏi ngoài domain có cosine score < 0.3 kích hoạt fallback PageIndex / Safe Refusal chính xác.
- **Lỗi đã phát hiện và cách xử lý:**  
  - *Lỗi trùng lặp chunk ID khi tìm kiếm song ngữ:* Khi query cả tiếng Việt và bản dịch tiếng Anh, cùng một chunk xuất hiện ở cả hai lượt tìm; đã xử lý deduplication lấy max score trong `_search_bilingual` trước khi chuyển sang RRF.  
  - *Lỗi cấu trúc phản hồi PageIndex lồng nhau:* API PageIndex thực tế trả về `relevant_contents` dạng mảng 2 chiều và thẻ `physical_index` dạng chuỗi (ví dụ `<physical_index_2>`); đã viết hàm đệ quy `_flatten_contents` và regex trích xuất số trang an toàn.  
  - *Lỗi timeout từ Cohere API:* Khi Cohere phản hồi chậm hoặc nghẽn mạng, luồng retrieval bị gián đoạn; đã thiết lập timeout 15s và bọc khối bắt ngoại lệ trả về danh sách RRF gốc kèm trạng thái `timeout_fallback` ghi vào ContextVar.

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** Nhánh PageIndex vectorless fallback phụ thuộc vào API trực tuyến từ xa nên có độ trễ phản hồi (latency) cao hơn so với tìm kiếm vector nội bộ; bộ phân tích từ khóa BM25 hiện tại mới tách từ dựa trên biểu thức chính quy (`\w+`) thay vì bộ tách từ ngữ pháp tiếng Việt chuyên dụng.
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Tích hợp bộ tách từ tiếng Việt chuyên dụng (như `pyvi` hoặc `underthesea`) vào Elasticsearch analyzer để cải thiện độ chuẩn xác của BM25, đồng thời xây dựng cơ chế caching cục bộ kết quả PageIndex để tối ưu thời gian phản hồi.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Phạm Hoàng Trọng
