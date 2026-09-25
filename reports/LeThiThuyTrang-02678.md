# Individual contribution report

## Thông tin

- Họ và tên: Lại Thị Thu Trang
- Mã học viên: 2A202602678
- Nhóm: sloppers
- Repository/branch: ToRong31/K4-L3B-RAG-Pipeline / trangltt/chunking_n_UI

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Task 10 — Generation có Citation & Safe Refusal** | Hiện thực `reorder_for_llm` (chống lost-in-the-middle), `format_context` gắn nhãn định danh nguồn, `call_llm` hỗ trợ đa provider (OpenAI, Gemini, Anthropic), và cơ chế tự động Safe Refusal khi thiếu dữ liệu hoặc lỗi provider | `src/task10_generation.py` / Commit `3b56fa8` | Done |
| **Giao diện Chatbot & RAG Observability (Streamlit)** | Thiết kế giao diện Chatbot tư vấn tuyển sinh VinUni hiện đại với cấu trúc 2 tabs: Tab 1 (Live Chat, citation expander, quick starter prompts) và Tab 2 (Bảng điều khiển giám sát Pipeline, KPI latency/score/method, invariants visualization) | `app.py` / Commit `3b56fa8` | Done |
| **Golden Dataset (≥ 15 câu)** | Xây dựng bộ dữ liệu đánh giá chuẩn gồm 15 cặp Q&A bám sát 100% cơ sở dữ liệu thật của VinUni (học phí, học bổng, quy chế tín chỉ, mốc tuyển sinh) có trích dẫn ngữ cảnh đối chiếu | `group_project/evaluation/golden_dataset.json` | Done |
| **Đánh giá A/B & Báo cáo kết quả** | Thực hiện so sánh A/B giữa Config A (Dense-only) và Config B (Hybrid + RRF), đo 4 metrics (Faithfulness, Relevance, Recall, Precision), phân tích 3 ca lỗi tệ nhất (Worst performers) và hoàn thiện báo cáo không còn `TODO` | `group_project/evaluation/RESULT.md`, `reports/RESULT.md` | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Áp dụng thuật toán **Front-Back Reordering** trong `reorder_for_llm` để sắp xếp lại vị trí các chunk quan trọng nhất ra đầu (vị trí #1) và cuối (vị trí #2) của context trước khi gửi đến LLM, thay vì giữ nguyên thứ tự giảm dần đơn thuần.  
   **Lý do/evidence:** Khắc phục triệt để hiện tượng *lost-in-the-middle* (mô hình ngôn ngữ lớn thường chú ý nhiều nhất ở phần đầu và phần cuối của prompt context, dễ bỏ sót thông tin ở giữa).  
   **Trade-off:** Đảo thứ tự hiển thị tuyến tính của điểm số ban đầu, nhưng giải quyết được bằng cách bảo tồn đầy đủ ID, metadata và score để hiển thị minh bạch trên giao diện UI cho người dùng.

2. **Quyết định:** Thiết kế giao diện Streamlit theo kiến trúc **2 Tabs tách biệt (Live Chat vs RAG Observability Dashboard)** và gắn nhãn trích dẫn có cấu trúc cho từng câu trả lời.  
   **Lý do/evidence:** Bài Lab đòi hỏi người chấm phải kiểm chứng được các khâu kỹ thuật (Chunking, Dense, BM25, RRF, Fallback). Tab Observability cho phép trực quan hóa toàn bộ hành trình của một truy vấn (query trace, KPI latency, best dense score so với fallback threshold, bảng chunks nạp vào LLM).  
   **Trade-off:** Tăng khối lượng code frontend và CSS tùy biến trong `app.py`, nhưng nâng cao trải nghiệm người dùng và giúp việc bảo vệ bài lab đạt điểm tối đa (bao gồm tiêu chí điểm thưởng bonus UI).

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:** 
  - `pytest tests/test_contracts.py -k "generation or reorder or safe_refusal" -v`
  - `pytest tests/test_acceptance.py -v`
- **Kết quả trước/sau nếu có:** 
  - Trước khi triển khai: `test_golden_dataset_has_15_grounded_cases` và `test_evaluation_report_is_completed` đều FAILED do file rỗng và còn placeholder `TODO`.
  - Sau khi hoàn thành: **5/5 Acceptance Tests đều PASSED (100%)** và toàn bộ Contract Tests thuộc phạm vi Generation đều PASSED.
- **Lỗi đã phát hiện và cách xử lý:** Phát hiện lỗi UnicodeEncodeError khi in câu trả lời tiếng Việt trên Windows console (`cp1252`), đã cấu hình `$env:PYTHONIOENCODING="utf-8"` và xử lý an toàn lỗi chuỗi rỗng khi gọi API LLM.

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** Giao diện Streamlit hiện tại nạp toàn bộ câu trả lời sau khi hoàn tất (chưa hiển thị dạng streaming từng từ).
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Tích hợp tính năng streaming response (`st.write_stream`) kết hợp highlight trực tiếp câu từ trích dẫn trong văn bản gốc.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Lại Thị Thu Trang
