"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải hoặc kiểm tra ít nhất 3 PDF/DOCX từ nguồn công khai."""
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sources = {
        "quy_dinh_tai_chinh_va_bieu_phi_vinuni.pdf": (
            "https://policy.vinuni.edu.vn/wp-content/uploads/2026/08/"
            "VU_TS03.VN_Quy-dinh-tai-chinh-va-Bieu-phi_AY26-27_22.7.2026_Student.pdf"
        ),
        "quy_che_dao_tao_dai_hoc_tin_chi_vinuni.pdf": (
            "https://policy.vinuni.edu.vn/wp-content/uploads/2024/05/"
            "VU_HT03.VN_QC-dao-tao-dai-hoc-he-chinh-quy-theo-he-thong-tin-chi.pdf"
        ),
    }

    import requests

    headers = {"User-Agent": "Mozilla/5.0"}
    for filename, url in sources.items():
        target = DATA_DIR / filename
        if not target.exists() or target.stat().st_size < 1024:
            try:
                res = requests.get(url, headers=headers, timeout=20)
                res.raise_for_status()
                target.write_bytes(res.content)
                print(f"Downloaded: {filename} ({len(res.content):,} bytes)")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

    valid_files = [
        path for path in DATA_DIR.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in {".pdf", ".doc", ".docx"}
    ]
    print(f"Total {len(valid_files)} legal documents ready in {DATA_DIR}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
