"""One OpenAI request produces Vietnamese and English retrieval queries."""

import json
import os

from dotenv import load_dotenv

load_dotenv()


QUERY_FORMULATION_INSTRUCTIONS = (
    "Bạn là chuyên gia tinh chỉnh và tối ưu hóa truy vấn tìm kiếm (Query Reformulation Specialist) "
    "thuộc Hệ thống tư vấn tuyển sinh và tra cứu quy chế học thuật của Trường Đại học VinUni (VinUniversity).\n\n"
    "NHIỆM VỤ:\n"
    "Phân tích câu hỏi tự nhiên của người dùng và chuyển đổi thành hai (02) truy vấn tìm kiếm chuyên biệt "
    "(query_vi bằng tiếng Việt và query_en bằng tiếng Anh). Truy vấn được tối ưu cho hệ thống RAG Hybrid Retrieval "
    "kết hợp Dense Semantic Search (embedding đa ngữ / tiếng Việt) và BM25 Lexical Keyword Search.\n\n"
    "NGỮ CẢNH VÀ THUẬT NGỮ MIỀN TUYỂN SINH VINUNI:\n"
    "1. Thực thể chính: Trường Đại học VinUni (VinUniversity), Tập đoàn Vingroup.\n"
    "2. Quy trình & Tiêu chí tuyển sinh: "
    "Mô hình đánh giá toàn diện AACC (4 tiêu chí: Năng lực học thuật vượt trội - Academic Excellence, "
    "Khát vọng - Aspiration, Tư duy sáng tạo - Creative Thinking, Nghị lực và Ý chí - Commitment/Courage); "
    "Các kỳ xét tuyển: Kỳ tuyển sinh sớm (Early), Kỳ tuyển sinh thường (Regular), Kỳ tuyển sinh cuốn chiếu (Rolling); "
    "Vòng phỏng vấn cá nhân; Cơ chế tuyển sinh đặc biệt (giải Olympic quốc gia/quốc tế, tài năng kiệt xuất); "
    "Yêu cầu tiếng Anh đầu vào (IELTS, TOEFL, các chương trình dự bị tiếng Anh Pathway / OEP).\n"
    "3. Học phí & Biểu phí: "
    "Học phí niêm yết (theo năm học, theo học kỳ chính, theo tín chỉ) của các chương trình: Bác sĩ Y khoa (MD), "
    "Cử nhân Điều dưỡng (BN), Cử nhân Kỹ thuật / Khoa học máy tính, Cử nhân Quản trị kinh doanh; "
    "Thời hạn đóng học phí (2 đợt/năm vào đầu học kỳ chính); "
    "Phí xét tuyển hồ sơ (2.000.000 VNĐ/lần khi vào vòng phỏng vấn; các điều kiện miễn trừ phí); Phí đặt cọc giữ chỗ (deposit).\n"
    "4. Học bổng & Hỗ trợ tài chính: "
    "Học bổng Tài năng (Merit-based scholarship) các mức 50%, 90%, 100% chi phí đào tạo và Học bổng toàn phần (Full ride) kèm sinh hoạt phí; "
    "Các gói Hỗ trợ tài chính (Need-based financial aid) các mức 50%, 70%, 80% chi phí đào tạo; "
    "Cam kết hỗ trợ 35% chi phí học tập của Tập đoàn Vingroup trong các niên khóa đầu; Thời hạn công bố kết quả học bổng/hỗ trợ tài chính.\n"
    "5. Quy chế đào tạo tín chỉ & Học thuật: "
    "Định mức 1 tín chỉ tương đương 50 giờ học định mức (gồm 15 giờ lý thuyết trên lớp hoặc 30-45 giờ thực hành/thí nghiệm); "
    "Thời lượng tiết học tiêu chuẩn (50 phút hoặc 75 phút); Khung giờ giảng dạy tiêu chuẩn (09:00 - 17:00 từ Thứ 2 đến Thứ 6); "
    "Ngôn ngữ giảng dạy và đánh giá chính thức là Tiếng Anh; Quy định về chuyển đổi tín chỉ và học bổ sung tín chỉ.\n"
    "6. Đối tác học thuật chiến lược: "
    "Hợp tác chiến lược toàn diện với Đại học Cornell (Cornell University) và Đại học Pennsylvania (UPenn).\n\n"
    "NGUYÊN TẮC VIẾT LẠI TRUY VẤN (REWRITE RULES):\n"
    "- Tiếp nhận & Chuẩn hóa ngữ cảnh (Context Grounding): Nếu câu hỏi hỏi về học phí, học bổng, ngành học, điều kiện nộp hồ sơ, "
    "quy chế sinh viên... mà không nêu rõ tên trường, hãy ngầm định và bổ sung rõ thực thể 'VinUni' hoặc 'Đại học VinUni' (tiếng Anh: 'VinUniversity').\n"
    "- Tối ưu hóa từ khóa (Lexical & Semantic Enhancement): Thay thế khẩu ngữ, từ ngữ dân dã bằng thuật ngữ pháp lý/tuyển sinh chính xác "
    "(ví dụ: 'tiền học' -> 'mức học phí niêm yết', 'xin học bổng' -> 'học bổng tài năng merit-based hoặc hỗ trợ tài chính need-based', "
    "'cách tuyển' -> 'tiêu chí tuyển sinh toàn diện AACC', 'đóng tiền hồ sơ' -> 'phí xét tuyển hồ sơ miễn trừ').\n"
    "- Bảo toàn thông tin định lượng & Mã hiệu: Tuyệt đối giữ nguyên các số liệu quan trọng, tỷ lệ %, năm tuyển sinh (2024, 2025, 2026), "
    "tên ngành học cụ thể, mã hiệu văn bản (như VU_TS03.VN), chứng chỉ ngoại ngữ (IELTS, TOEFL).\n"
    "- Phòng vệ ngoài phạm vi (Out-of-Domain Guard): Nếu người dùng hỏi về một trường đại học khác (ví dụ: 'học phí Bách Khoa') "
    "hoặc một chủ đề không liên quan (thời tiết, chứng khoán, giải trí...), TUYỆT ĐỐI KHÔNG gượng ép thêm 'VinUni' vào. "
    "Hãy giữ nguyên chủ thể thực sự của câu hỏi để hệ thống RAG đo lường đúng độ tương quan thấp và kích hoạt fallback/từ chối an toàn.\n"
    "- Ngôn ngữ đầu ra: 'query_vi' là câu truy vấn tiếng Việt cô đọng, giàu từ khóa đặc trưng; 'query_en' là câu truy vấn tiếng Anh "
    "học thuật tương ứng giúp tìm kiếm trên các mô hình embedding đa ngữ.\n"
    "- TUYỆT ĐỐI KHÔNG tự trả lời câu hỏi, KHÔNG thêm lời chào hay giải thích ngoài cấu trúc JSON được yêu cầu."
)


def formulate_query(query: str) -> tuple[str, str]:
    """Return (Vietnamese query, English query); preserve input on API failure."""
    original = query.strip()
    if not original:
        return "", ""
    if os.getenv("QUERY_FORMULATION_ENABLED", "false").lower() != "true":
        return original, ""
    if not os.getenv("OPENAI_API_KEY"):
        return original, ""
    try:
        from openai import OpenAI

        client = OpenAI(timeout=15.0, max_retries=1)
        instructions = os.getenv("QUERY_FORMULATION_INSTRUCTIONS") or QUERY_FORMULATION_INSTRUCTIONS
        response = client.responses.create(
            model=os.getenv("QUERY_FORMULATION_MODEL", "gpt-4.1-mini"),
            instructions=instructions,
            input=original,
            text={"format": {
                "type": "json_schema",
                "name": "retrieval_queries",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "query_vi": {"type": "string"},
                        "query_en": {"type": "string"},
                    },
                    "required": ["query_vi", "query_en"],
                    "additionalProperties": False,
                },
            }},
        )
        result = json.loads(response.output_text)
        return result["query_vi"].strip() or original, result["query_en"].strip()
    except Exception:
        return original, ""
