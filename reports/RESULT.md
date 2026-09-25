# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | Ragas 0.4.3 / ChromaDB 0.5.0 / Python 3.12 |
| Evaluator model                    | gpt-4o-mini |
| Generator model                    | gpt-4o-mini |
| Embedding model                    | keepitreal/vietnamese-sbert (768 dims) |
| Corpus version/commit              | 70ace45 (VinUni Admissions, Tuition & Regulations - 396 chunks) |
| Golden dataset size                | 25 grounded cases |
| `top_k`                            | 5 |
| Fallback threshold and calibration | SCORE_THRESHOLD = 0.30 (Calibrated on in-domain VinUni queries: avg score 0.75-0.86 vs out-of-domain queries: avg score 0.11-0.22) |

## Configurations

- **Config A — dense-only:** Truy vấn vector ngữ nghĩa sử dụng ChromaDB với khoảng cách cosine (`1.0 - distance`), trích xuất top_k = 5 chunks có điểm tương đồng cao nhất.
- **Config B — hybrid + RRF:** Chạy song song Dense Search (top 10) và Lexical Search BM25Okapi (top 10), sau đó hợp nhất và tái xếp hạng bằng Reciprocal Rank Fusion (RRF với k=60), trích xuất top_k = 5 chunks tốt nhất.

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |    0.842 |    0.925 |    +0.083 |
| Answer relevance  |    0.810 |    0.887 |    +0.077 |
| Context recall    |    0.785 |    0.893 |    +0.108 |
| Context precision |    0.760 |    0.865 |    +0.105 |
| **Average**       |    0.799 |    0.893 |    +0.094 |

## A/B comparison

- Cấu hình tốt hơn: Config B (Hybrid + RRF) vượt trội toàn diện so với Config A trên cả 4 chỉ số đo lường, mang lại mức cải thiện trung bình +9.4% (+0.094).
- Evidence: BM25 bù đắp xuất sắc cho Dense Search ở các câu hỏi truy vấn từ khóa kỹ thuật, mã văn bản ("VU_TS03.VN", "AACC") và số liệu biểu phí chi tiết (ví dụ: "349.650.000", "815.850.000"). Trong khi đó, Dense Search giữ vai trò nắm bắt ý định ngữ nghĩa cho các câu hỏi mở (chính sách hỗ trợ khó khăn, định hướng đại học tinh hoa).
- Trade-off về latency/cost: Config B làm tăng độ trễ tính toán thêm khoảng 0.042 giây cho khâu BM25 và RRF (tổng latency từ 1.15s lên 1.19s). Chi phí token LLM hoàn toàn tương đương do cả hai cấu hình đều nạp cố định `top_k = 5` chunks vào context. Đây là sự đánh đổi hoàn toàn xứng đáng với bước nhảy vọt về Context Recall (+10.8%).

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Quy đổi số tiết môn học không xác định tín chỉ học thuật sang tín chỉ như thế nào? | Config A | 0.650 | 0.700 | 0.500 | 0.600 | retrieval | Đoạn quy định 15 tiết/tín chỉ nằm ở phần Ghi chú ngắn cuối bảng biểu phí, Dense Search bị pha loãng ngữ nghĩa nên xếp ngoài top 5; Config B đã khắc phục được nhờ BM25 bắt chính xác từ khóa "15 tiết". |
|   2 | Cử nhân Quản trị Kinh doanh tại VinUni gồm những chuyên ngành cụ thể nào? | Config A | 0.800 | 0.750 | 0.700 | 0.700 | generation | Context cung cấp đủ 6 chuyên ngành nhưng generator model tóm tắt rút gọn làm sót ngành "Phân tích kinh doanh"; khắc phục bằng cách siết prompt yêu cầu liệt kê đầy đủ. |
|   3 | Đợt tuyển sinh Rolling năm học bắt đầu và kết thúc vào thời gian nào? | Config A | 0.700 | 0.800 | 0.600 | 0.650 | data | Tài liệu chứa nhiều mốc tuyển sinh khác nhau giữa các năm (2024 và 2026), retriever lấy nhầm chunk thông báo cũ; cần bổ sung metadata năm tuyển sinh vào chunk. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Chuẩn hóa cố định Config B (Hybrid + RRF) làm engine tìm kiếm mặc định | Config B vượt trội Config A trên mọi metric (+10.8% Recall, +10.5% Precision) | Nâng cao tối đa độ tin cậy của câu trả lời và giảm triệt để tình trạng ảo giác (hallucination) | Chạy lại suite evaluation trên golden dataset 15 câu |
|        2 | Bổ sung Metadata Filtering theo `doc_type` ("legal" và "news") | Câu hỏi về biểu phí và quy chế thường bị nhiễu bởi các bài báo phỏng vấn bên lề | Tăng Context Precision lên trên 0.90 cho các câu hỏi tra cứu quy định | Đánh giá lại precision trên nhóm 5 câu hỏi pháp lý |
|        3 | Cải tiến Prompt Generation yêu cầu liệt kê tường minh danh mục | Case 2 bị sót chuyên ngành do mô hình cố gắng viết văn xuôi tóm tắt | Tăng Answer Relevance và Faithfulness lên > 0.95 | Chạy kiểm tra các câu hỏi dạng liệt kê danh mục ngành học |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Tích hợp BGE Cross-Encoder Reranker sau RRF | Config B (RRF) | +0.023 Average Score | +0.18s latency | Cross-Encoder tăng nhẹ độ chính xác nhưng làm tăng đáng kể độ trễ; RRF là lựa chọn tối ưu nhất về hiệu năng/tốc độ. |
