"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
from typing import Any

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower().strip()
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""

SAFE_REFUSAL_ANSWER = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (Front-Back reordering).
    
    Không làm thay đổi danh sách truyền vào (non-mutating).
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label rõ ràng cho từng chunk."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Unknown")
        source = metadata.get("source", "Unknown")
        content = chunk.get("content", "")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình LLM_PROVIDER trong .env."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER or "openai").lower().strip()
    model = os.getenv("LLM_MODEL", LLM_MODEL)

    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""

    if provider == "gemini":
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model or "gemini-2.5-flash",
            contents=user_message,
            config={"system_instruction": system_prompt, "temperature": TEMPERATURE},
        )
        return response.text or ""

    if provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model or "claude-3-5-sonnet-20241022",
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            max_tokens=1024,
        )
        return response.content[0].text or ""

    raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Trả về GenerationResult gồm answer, sources và retrieval_source.
    
    Khi context không đủ hoặc provider lỗi, trả safe refusal thay vì bịa thông tin.
    """
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        chunks = []

    if not chunks:
        return {
            "answer": SAFE_REFUSAL_ANSWER,
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        if not answer or not answer.strip():
            answer = SAFE_REFUSAL_ANSWER
    except Exception:
        return {
            "answer": SAFE_REFUSAL_ANSWER,
            "sources": [],
            "retrieval_source": "none",
        }

    primary_method = chunks[0].get("retrieval_method", "hybrid")
    retrieval_source = primary_method if primary_method in {"hybrid", "pageindex"} else "hybrid"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
