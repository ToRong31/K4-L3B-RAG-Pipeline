# Tài liệu Bàn giao Dữ liệu & Indexing (Data Handover)

> **Người thực hiện:** Dương (Phụ trách: Data, Chunking, Embedding & Indexing)  
> **Người tiếp nhận:** Trọng (Retrieval, RRF, Fallback) & UI/Eval (Generation, Evaluation)  
> **Dự án:** K4-L3B-RAG-Pipeline — Hệ thống RAG Tuyển sinh VinUniversity  
> **Trạng thái kiểm thử:** 4/4 bài test phạm vi Data đã **PASSED** (`pytest tests/test_acceptance.py tests/test_contracts.py`)

---

## 1. Danh mục & Nguồn gốc Dữ liệu

Toàn bộ dữ liệu đã được thu thập tại `data/landing/` và chuẩn hóa sang Markdown tại `data/standardized/`.

### 1.1. Tài liệu Pháp lý & Quy chế (`data/standardized/legal/`)
Gồm 3 văn bản chính thức của VinUniversity với lớp văn bản số (digital text) đầy đủ:

| Tên file Markdown | Nguồn gốc | Số ký tự | Nội dung chính |
| :--- | :--- | :--- | :--- |
| `quy_dinh_tai_chinh_va_bieu_phi_vinuni.md` | `policy.vinuni.edu.vn` | 34.043 | Biểu phí đào tạo chuẩn (~35.000 USD/năm), lệ phí tuyển sinh, phí nội trú ký túc xá, chính sách hoàn phí và tài trợ giáo dục từ Vingroup. |
| `quy_che_dao_tao_dai_hoc_tin_chi_vinuni.md` | `policy.vinuni.edu.vn` | 75.617 | Quy chế đào tạo đại học chính quy theo hệ thống tín chỉ, định nghĩa ngành/chuyên ngành chính, chuyên ngành phụ, song bằng, chuyển đổi tín chỉ và điều kiện tốt nghiệp. |
| `quy_che_tuyen_sinh_sau_dai_hoc_vinuni.md` | `vinuni.edu.vn` | 16.601 | Quy chế tuyển sinh Thạc sĩ và Tiến sĩ, tiêu chuẩn văn bằng, chuẩn năng lực tiếng Anh, quy trình xét duyệt hồ sơ và phỏng vấn. |

*(Thư mục `data/landing/legal/` vẫn lưu kèm 2 bản PDF scan gốc bổ sung: `de_an_tuyen_sinh_dai_hoc_2024_vinuni.pdf` và `thong_tin_tuyen_sinh_dai_hoc_2026_vinuni.pdf`).*

---

### 1.2. Bài viết & Tin tức Tuyển sinh (`data/standardized/news/`)
Gồm 5 bài viết từ các báo chính thống, đã qua cào dữ liệu (`data/landing/news/*.json`) có đủ 4 trường metadata (`url`, `title`, `date_crawled`, `content_markdown`):

| Tên file Markdown | Nguồn bài viết (URL) | Tiêu đề bài viết | Nội dung tóm tắt |
| :--- | :--- | :--- | :--- |
| `article_01.md` | Tuổi Trẻ | Ba điểm mới trong kỳ tuyển sinh năm học 2024 - 2025 của VinUni | Các mốc xét tuyển (Early, Regular, Rolling), đổi mới phỏng vấn và chính sách học bổng. |
| `article_02.md` | Dân Trí | Cấp học bổng, hỗ trợ tài chính hào phóng: VinUni liệu có thu hút được nhân tài? | Các mức học bổng tài năng (50%–100%), hỗ trợ Need-based và 4 tiêu chí toàn diện AACC. |
| `article_03.md` | VietNamNet | Vòng tuyển sinh đặc biệt cho học sinh tài năng vào Đại học VinUni | Cơ chế tuyển sinh đặc biệt cho học sinh đạt giải Olympic quốc gia, quốc tế hoặc tài năng kiệt xuất. |
| `article_04.md` | Dân Trí | Dự án Đại học VinUni công bố hướng tuyển sinh năm học | Các viện đào tạo trọng điểm liên kết với Cornell và Pennsylvania, chuẩn đầu vào. |
| `article_05.md` | Dân sinh (Dân Trí) | VinUni mở vòng tuyển sinh đặc biệt thu hút sinh viên quốc tế | Quy trình tuyển sinh quốc tế và quy trình xét tuyển nhanh "5-5-5". |

---

## 2. Thông số Kỹ thuật (Chunking, Embedding, Indexing)

* **Chiến lược Chunking:** `RecursiveCharacterTextSplitter`
  * `CHUNK_SIZE = 500` (ký tự)
  * `CHUNK_OVERLAP = 50` (ký tự, tỷ lệ 10%)
  * `separators = ["\n\n", "\n", ". ", " ", ""]`
* **Quy ước ID & Metadata Chunk:**
  * Mỗi chunk có ID ổn định: `{file_relative_path}::chunk-{chunk_index}` (ví dụ: `legal/quy_dinh_tai_chinh...md::chunk-0`).
  * Metadata bắt buộc: `source`, `title`, `doc_type` (`"legal"` hoặc `"news"`), `url`, `chunk_index`.
* **Tổng số chunks:** **396 chunks** (không rỗng, không trùng lặp ID).
* **Mô hình Embedding:**
  * Model mặc định: `keepitreal/vietnamese-sbert` (hoặc `BAAI/bge-m3` nếu cấu hình trong `.env`).
  * Kích thước vector (Dimension): `768` (với `vietnamese-sbert`) hoặc `1024` (với `bge-m3`).
  * Đã tải sẵn trong cache local, sinh vector nhanh và tối ưu tiếng Việt.
* **Vector Database:** `ChromaDB` (Persistent)
  * Đường dẫn lưu trữ: `chroma_db/`
  * Collection name: `rag_documents`
  * Metric khoảng cách: `cosine` (`{"hnsw:space": "cosine"}`)
  * Trạng thái hiện tại: Đã nạp thành công **396/396 chunks**.

---

## 3. Hướng dẫn dành cho Trọng (Retrieval & RRF)

Theo [docs/MODULE_CONTRACTS.md](docs/MODULE_CONTRACTS.md), các module tiếp theo sử dụng tài nguyên đã chuẩn bị như sau:

### 3.1. Đối với Lexical Search (BM25 - Task 6)
Để BM25 chạy trên **cùng corpus chunks** với Dense Search, import trong `src/task6_lexical_search.py`:
```python
from src.task4_chunking_indexing import chunk_documents, load_documents

# Nạp 396 chunks dùng chung
CORPUS = chunk_documents(load_documents())
```

### 3.2. Đối với Semantic Search (ChromaDB - Task 5)
Sử dụng chung hàm `embed_texts()` và `get_collection()` trong `src/task5_semantic_search.py`:
```python
from src.task4_chunking_indexing import embed_texts, get_collection

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    query_vector = embed_texts([query])[0]
    collection = get_collection()
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        results.append({
            "id": item_id,
            "content": content,
            "score": max(0.0, 1.0 - distance),  # Chuyển cosine distance sang cosine similarity
            "metadata": metadata,
            "retrieval_method": "dense",
        })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
```

---

## 4. Cách Tái tạo Index từ đầu (Re-indexing)

Khi có thêm tài liệu mới hoặc cần xây dựng lại toàn bộ cơ sở dữ liệu vector:

```bash
# 1. Chuyển đổi dữ liệu thô sang Markdown
python -m src.task3_convert_markdown

# 2. Thực hiện chunking, embedding và upsert vào ChromaDB
python -m src.task4_chunking_indexing
```

Do cơ chế `upsert` theo ID ổn định, việc chạy lại lệnh trên hoàn toàn an toàn, tự động cập nhật nội dung mới mà không làm nhân bản các bản ghi cũ.

---

## 5. Kết quả Kiểm thử Nghiệm thu (Acceptance Tests)

Chạy lệnh kiểm tra:
```bash
pytest tests/test_acceptance.py -q
```
Kết quả các bài test phạm vi Data:
* `test_corpus_has_required_legal_documents`: **PASSED** (≥ 3 văn bản pháp lý hợp lệ).
* `test_corpus_has_required_news_with_metadata`: **PASSED** (≥ 5 bài báo JSON đầy đủ metadata).
* `test_standardized_output_covers_both_source_types`: **PASSED** (8 file Markdown chuẩn hóa ≥ 200 ký tự).
* `test_chunk_documents_preserves_identity_and_metadata`: **PASSED** (Chunk bảo toàn ID, thứ tự và metadata).
