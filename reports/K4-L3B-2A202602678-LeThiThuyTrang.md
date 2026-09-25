# Individual contribution report

## Thông tin

- Họ và tên: Lê Thị Thuỳ Trang
- Mã học viên: 2A202602678
- Nhóm: sloppers
- Repository/branch: ToRong31/K4-L3B-RAG-Pipeline / main

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Task 10 — Generation có Citation & Safe Refusal** | Hiện thực `reorder_for_llm` (chống lost-in-the-middle bằng Front-Back reordering), `format_context` định danh tài liệu và nguồn trích dẫn, `call_llm` hỗ trợ đa nhà cung cấp (OpenAI, Gemini, Anthropic), `generate_with_citation` và `generate_from_chunks` với cơ chế Safe Refusal tự động khi thiếu bằng chứng hoặc provider lỗi để loại bỏ ảo giác (hallucination) | `src/task10_generation.py` / Commit `3b56fa8`, `da3f652` | Done |
| **Giao diện Chatbot & RAG Observability 3 Tabs (Streamlit)** | Thiết kế UI Chatbot tư vấn học vụ VinUni hiện đại: **Tab 1** (Live Chat, trích dẫn minh bạch `[Document n \| ID]`, expander nguồn, câu hỏi gợi ý nhanh); **Tab 2** (Bảng điều khiển giám sát Pipeline, thẻ KPI đo độ trễ/score/kênh, Stage 0 Query Formulation, biểu diễn 5 invariants kỹ thuật); **Tab 3** (A/B Query Formulation Workbench) | `app.py`, `.streamlit/config.toml` / Commit `3b56fa8`, `d36f424`, `78a2a41` | Done |
| **Tab 3 — Đối sánh Query Formulation (Rubric Bonus +3đ)** | Xây dựng công cụ đối sánh trực quan A/B giữa Trước (Raw Query) và Sau (Bilingual Formulated Queries); tính toán Delta Metrics (+5 chunks mới khám phá nhờ song ngữ, chênh lệch latency/score), hiển thị song song 2 cột kèm badge `⭐ CHUNK MỚI` và ma trận đối chiếu từng đoạn văn bản | `app.py` / Commit `d36f424`, `78a2a41` | Done |
| **Golden Dataset chuẩn hóa (27 grounded cases)** | Xây dựng và hoàn thiện bộ dữ liệu chuẩn 27 câu hỏi — đáp bao phủ 3 nhóm phân loại RAG nâng cao (*14 Keyword-heavy, 8 Semantic similarity, 5 Multi-source confusion*), 100% bám sát văn bản quy chế thật của VinUni với `expected_answer` và `expected_context` đối soát chi tiết | `group_project/evaluation/golden_dataset.json` / Commit `74426df`, `80ba862` | Done |
| **Đánh giá A/B & Báo cáo kết quả Benchmark** | Thực hiện đo lường và lập báo cáo thực nghiệm so sánh A/B giữa Config A (Dense-only) và Config B (Hybrid + RRF) trên cùng 27 test cases; đánh giá 4 metrics chuẩn Ragas (Faithfulness 0.937, Relevance 0.993, Recall 0.989, Precision 0.826); phân tích sâu 3 ca lỗi tệ nhất và giải quyết 100% `TODO` | `group_project/evaluation/RESULT.md`, `reports/RESULT.md` / Commit `74426df`, `80ba862` | Done |
| **Kiểm thử Contracts & Acceptance (Task 10 & Eval)** | Kiểm thử tự động bảo đảm tuân thủ nghiêm ngặt chữ ký hàm contract (`generate_with_citation`), kiểm tra tính bảo toàn dữ liệu (non-mutating), xử lý an toàn lỗi chuỗi rỗng/mạng và đạt 100% pass toàn bộ 5/5 Acceptance tests | `tests/test_contracts.py`, `tests/test_acceptance.py` / Commit `3b56fa8`, `74426df` | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Áp dụng thuật toán **Front-Back Reordering** trong `reorder_for_llm` để sắp xếp lại vị trí các chunk: chunk quan trọng nhất (rank #1) ở đầu context, chunk quan trọng thứ hai (rank #2) ở cuối context, và các chunk có độ ưu tiên thấp hơn xen kẽ ở giữa, thay vì chuyển tuyến tính danh sách giảm dần sang LLM.  
   **Lý do/evidence:** Khắc phục triệt để hiện tượng *lost-in-the-middle* (nghiên cứu của Liu et al. chứng minh LLM chú ý tốt nhất ở phần mở đầu và kết thúc của prompt context, dễ bỏ sót hoặc suy giảm khả năng trích xuất thông tin nằm ở đoạn giữa).  
   **Trade-off:** Thứ tự hiển thị vật lý của văn bản nạp vào prompt bị đảo so với thứ tự ban đầu, nhưng giải quyết bằng cách đánh số `[Document n | ID]` cố định và trả về danh sách `sources` gốc có đầy đủ điểm score để hiển thị minh bạch cho người dùng trên giao diện.

2. **Quyết định:** Thiết kế giao diện Streamlit theo kiến trúc **3 Tabs chuyên biệt (Live Chat vs RAG Observability Dashboard vs Query Formulation A/B Workbench)** kết hợp cơ chế Safe Refusal Guardrail cưỡng bức.  
   **Lý do/evidence:** Vừa đóng vai trò một trợ lý AI tuyển sinh trực quan cho người dùng cuối (Chatbot thân thiện có trích dẫn minh bạch), vừa cung cấp bảng điều khiển kỹ thuật cho giảng viên/người chấm bài kiểm chứng từng mắt xích trong pipeline. Đặc biệt, Tab 3 đối kháng chứng minh rõ ràng giá trị vượt trội của mở rộng truy vấn song ngữ (bắt được thêm 5 chunk tài liệu tài chính mà câu hỏi gốc bị sót), đáp ứng trực tiếp tiêu chí điểm thưởng sáng tạo (+3 điểm) trong Grading Rubric.  
   **Trade-off:** Tăng khối lượng mã nguồn giao diện và CSS tùy biến (`app.py` hơn 880 dòng), nhưng đổi lại hệ thống có độ hoàn thiện cao, trực quan và bảo vệ đồ án thuyết phục.

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:**  
  - `pytest tests/test_contracts.py -k "generation or reorder or safe_refusal" -v`
  - `pytest tests/test_acceptance.py -v`
  - Chạy script benchmark thực nghiệm: `python group_project/evaluation/evaluate.py`
  - Thử nghiệm truy vấn thực tế trên UI: query đúng domain (*"Mức học phí Cử nhân Điều dưỡng VinUni?", "Tiêu chuẩn xét học bổng Tài năng?"*) và query ngoài domain (*"Thời tiết hôm nay ở Hà Nội thế nào?"*) để kiểm chứng cơ chế Safe Refusal (`Tôi không thể xác minh thông tin này từ nguồn hiện có.`).
- **Kết quả trước/sau nếu có:**  
  - Trước khi triển khai: Các bài kiểm thử generation và acceptance đều FAILED do chưa có hàm, thiếu golden dataset và báo cáo còn placeholder `TODO`.
  - Sau khi hoàn thành: **34/34 bài kiểm thử toàn hệ thống đều PASSED (100%)**, trong đó 5/5 Acceptance tests và toàn bộ Contract tests thế hệ câu trả lời đạt chuẩn.
  - Kết quả đánh giá A/B: Config B (Hybrid + RRF) nâng Average Score từ **0.910 lên 0.936** (+2.6%), đặc biệt Context Precision tăng vọt **+7.3%** và nhóm câu hỏi gây nhiễu đa nguồn tăng **+20.4% Precision**, triệt tiêu hoàn toàn hiện tượng nhầm lẫn giữa quy chế ĐH và Sau ĐH.
- **Lỗi đã phát hiện và cách xử lý:**  
  - *Lỗi UnicodeEncodeError trên Windows console:* Khi in câu trả lời tiếng Việt có dấu ra terminal mặc định (`cp1252`), hệ thống bị lỗi mã hóa; đã xử lý triệt để bằng cấu hình `$env:PYTHONIOENCODING="utf-8"`.
  - *Lỗi truy vấn hai lần (Double Retrieval Overhead):* Ban đầu giao diện Streamlit gọi `generate_with_citation` độc lập khiến pipeline phải chạy retrieve lại từ đầu; đã phối hợp bổ sung hàm `generate_from_chunks(query, chunks)` nhận trực tiếp kết quả từ chặng telemetry giúp giảm 50% thời gian phản hồi.
  - *Lỗi xung đột merge nhánh:* Khi merge nhánh `trangltt/updateUI` vào `main`, xuất hiện xung đột trong `app.py` do Leader cập nhật thêm ContextVar trace; đã chủ động đối soát và tích hợp toàn bộ tính năng mới của cả hai bên mà không làm mất bất kỳ mã nguồn nào.

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** Giao diện Streamlit hiện tại phản hồi toàn bộ văn bản sau khi LLM xử lý xong (chưa hỗ trợ streaming token theo thời gian thực); thời gian gọi đối sánh A/B ở Tab 3 phụ thuộc vào tốc độ mạng khi gọi đồng thời OpenAI API cho cả 2 nhánh.
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Tích hợp `st.write_stream` để hiển thị câu trả lời dạng gõ chữ tự nhiên (streaming), bổ sung tính năng highlight đoạn văn bản tương ứng trực tiếp trên file PDF gốc khi người dùng bấm vào trích dẫn.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Lê Thị Thùy Trang
